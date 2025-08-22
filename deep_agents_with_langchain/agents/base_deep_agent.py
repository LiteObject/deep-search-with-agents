"""
Base Deep Agent Implementation using LangChain

This module provides the foundational deep agent class that implements
hierarchical planning, reflection loops, and multi-step reasoning.
"""

# pylint: disable=import-error,no-name-in-module,too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,unnecessary-pass,broad-exception-caught,logging-fstring-interpolation,bare-except,import-outside-toplevel,too-few-public-methods,ungrouped-imports
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import json
import logging

from langchain.agents import AgentExecutor, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class DeepAgentResult:
    """Result from a deep agent execution with full context"""
    query: str
    plan: List[str]
    execution_steps: List[Dict[str, Any]]
    final_result: str
    reflection: Dict[str, Any]
    confidence_score: float
    execution_time: float
    tokens_used: int
    iterations: int


@dataclass
class PlanStep:
    """Individual step in a deep agent plan"""
    step_id: str
    description: str
    tool_name: str
    parameters: Dict[str, Any]
    expected_output: str
    priority: int = 1


class DeepAgentMemory:
    """Memory system for deep agents to learn from past experiences"""

    def __init__(self, max_memories: int = 100):
        self.max_memories = max_memories
        self.memories: List[Dict[str, Any]] = []
        self.successful_patterns: Dict[str, int] = {}

    def add_memory(self, query: str, plan: List[str], result: str,
                   success_score: float, execution_time: float):
        """Add a new memory from an execution"""
        memory = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'plan': plan,
            'result': result,
            'success_score': success_score,
            'execution_time': execution_time
        }

        self.memories.append(memory)

        # Keep only recent memories
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories:]

        # Track successful patterns
        if success_score > 0.8:
            plan_signature = self._get_plan_signature(plan)
            self.successful_patterns[plan_signature] = \
                self.successful_patterns.get(plan_signature, 0) + 1

    def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to current query"""
        # Simple similarity based on common words
        query_words = set(query.lower().split())

        scored_memories = []
        for memory in self.memories:
            memory_words = set(memory['query'].lower().split())
            similarity = len(query_words.intersection(
                memory_words)) / len(query_words.union(memory_words))
            scored_memories.append((similarity, memory))

        # Sort by similarity and success score
        scored_memories.sort(key=lambda x: (
            x[0], x[1]['success_score']), reverse=True)

        return [memory for _, memory in scored_memories[:limit]]

    def _get_plan_signature(self, plan: List[str]) -> str:
        """Generate a signature for a plan to track successful patterns"""
        return json.dumps(sorted([step.split()[0] if step.split() else '' for step in plan]))


class BaseDeepAgent(ABC):
    """
    Abstract base class for LangChain-powered deep agents

    Features:
    - Hierarchical planning with decomposition
    - Self-reflection and improvement loops
    - Memory of past successful strategies
    - Multi-step reasoning with tool usage
    - Quality assessment and refinement
    """

    def __init__(self, name: str, description: str,
                 model: str = "gpt-4", temperature: float = 0.1):
        self.name = name
        self.description = description
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.memory = DeepAgentMemory()
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Initialize tools and agent
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent_executor()

        # Reflection and planning templates
        self.planning_template = self._get_planning_template()
        self.reflection_template = self._get_reflection_template()
        self.refinement_template = self._get_refinement_template()

    @abstractmethod
    def _create_tools(self) -> List[BaseTool]:
        """Create domain-specific tools for this agent"""
        pass

    @abstractmethod
    def _get_domain_context(self) -> str:
        """Get domain-specific context for planning"""
        pass

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor"""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.conversation_memory,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )

    def deep_search(self, query: str, max_iterations: int = 3) -> DeepAgentResult:
        """
        Execute a deep search with planning, execution, and reflection

        Args:
            query: The search query
            max_iterations: Maximum refinement iterations

        Returns:
            DeepAgentResult with complete execution context
        """
        start_time = datetime.now()

        try:
            # Phase 1: Planning
            logger.info("Starting deep search for query: %s", query)
            plan = self._generate_plan(query)

            # Phase 2: Initial execution
            execution_steps = []
            result = self._execute_plan(plan, query, execution_steps)

            # Phase 3: Reflection and refinement loop
            best_result = result
            best_confidence = 0.0
            reflection = {"confidence_score": 0.0,
                          "feedback": "Initial result"}
            iteration = 0

            for iteration in range(max_iterations):
                reflection = self._reflect_on_result(result, query)
                confidence = reflection.get('confidence_score', 0.0)

                if confidence > best_confidence:
                    best_result = result
                    best_confidence = confidence

                # If quality is good enough, stop refining
                if confidence > 0.85:
                    logger.info("[%s] High confidence achieved: %s",
                                self.name, confidence)
                    break

                # Refine if quality is insufficient
                if confidence < 0.7 and iteration < max_iterations - 1:
                    logger.info(
                        "[%s] Refining result (confidence: %s)", self.name, confidence)
                    refined_plan = self._refine_plan(plan, result, reflection)
                    result = self._execute_plan(
                        refined_plan, query, execution_steps)
                    plan = refined_plan

            # Phase 4: Final processing
            execution_time = (datetime.now() - start_time).total_seconds()

            # Store in memory for future learning
            self.memory.add_memory(
                query=query,
                plan=plan,
                result=best_result,
                success_score=best_confidence,
                execution_time=execution_time
            )

            return DeepAgentResult(
                query=query,
                plan=plan,
                execution_steps=execution_steps,
                final_result=best_result,
                reflection=reflection,
                confidence_score=best_confidence,
                execution_time=execution_time,
                tokens_used=0,  # NOTE: Token tracking not implemented
                iterations=iteration + 1
            )

        except (ValueError, AttributeError, RuntimeError, ConnectionError) as e:
            logger.error("[%s] Error in deep search: %s", self.name, str(e))
            return self._create_error_result(query, str(e))

    def _generate_plan(self, query: str) -> List[str]:
        """Generate a detailed execution plan for the query"""
        # Get relevant past experiences
        relevant_memories = self.memory.get_relevant_memories(query)
        memory_context = self._format_memory_context(relevant_memories)

        planning_prompt = self.planning_template.format(
            query=query,
            domain_context=self._get_domain_context(),
            memory_context=memory_context,
            available_tools=self._format_tools_description()
        )

        plan_response = self.llm.predict(planning_prompt)

        # Parse the plan into steps
        plan_steps = []
        for line in plan_response.strip().split('\n'):
            line = line.strip()
            if line and (line.startswith(tuple('123456789')) or line.startswith('-')):
                # Remove numbering and clean up
                step = line.split(
                    '.', 1)[-1].strip() if '.' in line else line.strip('- ')
                plan_steps.append(step)

        return plan_steps

    def _execute_plan(self, plan: List[str], query: str,
                      execution_steps: List[Dict[str, Any]]) -> str:
        """Execute the plan using the agent executor"""
        context = f"Original query: {query}\n\nExecution plan:\n"
        context += "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        context += "\n\nExecute this plan step by step."

        try:
            result = self.agent_executor.invoke({
                "input": context,
                "chat_history": self.conversation_memory.chat_memory.messages
            })

            execution_steps.append({
                "type": "plan_execution",
                "input": context,
                "output": result.get('output', ''),
                "timestamp": datetime.now().isoformat()
            })

            return result.get('output', 'No output generated')

        except (ValueError, AttributeError, RuntimeError, ConnectionError) as e:
            error_msg = f"Execution failed: {str(e)}"
            execution_steps.append({
                "type": "error",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            return error_msg

    def _reflect_on_result(self, result: str, query: str) -> Dict[str, Any]:
        """Reflect on the quality of the result and suggest improvements"""
        reflection_prompt = self.reflection_template.format(
            query=query,
            result=result,
            domain_context=self._get_domain_context()
        )

        reflection_response = self.llm.predict(reflection_prompt)

        # Parse reflection (simplified - could use structured output)
        confidence_score = 0.5  # Default
        feedback = reflection_response

        # Try to extract confidence score
        for line in reflection_response.split('\n'):
            if 'confidence' in line.lower() or 'score' in line.lower():
                try:
                    # Extract number from line
                    import re
                    numbers = re.findall(r'0\.\d+|\d+\.\d+', line)
                    if numbers:
                        confidence_score = min(float(numbers[0]), 1.0)
                        break
                except:
                    pass

        return {
            'confidence_score': confidence_score,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        }

    def _refine_plan(self, original_plan: List[str], result: str,
                     reflection: Dict[str, Any]) -> List[str]:
        """Refine the plan based on reflection feedback"""
        refinement_prompt = self.refinement_template.format(
            original_plan="\n".join(
                f"{i+1}. {step}" for i, step in enumerate(original_plan)),
            result=result,
            feedback=reflection.get('feedback', ''),
            domain_context=self._get_domain_context()
        )

        refined_response = self.llm.predict(refinement_prompt)

        # Parse refined plan
        refined_steps = []
        for line in refined_response.strip().split('\n'):
            line = line.strip()
            if line and (line.startswith(tuple('123456789')) or line.startswith('-')):
                step = line.split(
                    '.', 1)[-1].strip() if '.' in line else line.strip('- ')
                refined_steps.append(step)

        return refined_steps if refined_steps else original_plan

    def _format_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """Format relevant memories for use in prompts"""
        if not memories:
            return "No relevant past experiences found."

        context = "Relevant past experiences:\n"
        for i, memory in enumerate(memories[:3]):  # Limit to top 3
            context += f"{i+1}. Query: {memory['query']}\n"
            context += f"   Success: {memory['success_score']:.2f}\n"
            context += f"   Plan: {memory['plan'][:2]}...\n\n"

        return context

    def _format_tools_description(self) -> str:
        """Format available tools for planning prompts"""
        tools_desc = "Available tools:\n"
        for tool in self.tools:
            tools_desc += f"- {tool.name}: {tool.description}\n"
        return tools_desc

    def _get_planning_template(self) -> PromptTemplate:
        """Get the planning prompt template"""
        template = """
You are an expert {domain_context} agent tasked with creating a detailed execution plan.

Query: {query}

{memory_context}

{available_tools}

Create a step-by-step plan to thoroughly address this query. Consider:
1. What information needs to be gathered?
2. Which tools are most appropriate?
3. How should results be processed and synthesized?
4. What are potential challenges or edge cases?

Output a numbered list of specific, actionable steps:
"""
        return PromptTemplate(
            template=template,
            input_variables=["query", "domain_context",
                             "memory_context", "available_tools"]
        )

    def _get_reflection_template(self) -> PromptTemplate:
        """Get the reflection prompt template"""
        template = """
Evaluate the quality of this {domain_context} result:

Original Query: {query}

Result: {result}

Assess this result on:
1. Completeness - Does it fully address the query?
2. Accuracy - Is the information correct and up-to-date?
3. Relevance - Is it focused on what was asked?
4. Depth - Does it provide sufficient detail?
5. Clarity - Is it well-organized and understandable?

Provide:
- Confidence score (0.0 to 1.0)
- Specific feedback for improvement
- Missing elements or gaps

Format: Start with "Confidence: X.XX" then provide detailed feedback.
"""
        return PromptTemplate(
            template=template,
            input_variables=["query", "result", "domain_context"]
        )

    def _get_refinement_template(self) -> PromptTemplate:
        """Get the refinement prompt template"""
        template = """
Improve this {domain_context} plan based on the feedback:

Original Plan:
{original_plan}

Current Result:
{result}

Feedback:
{feedback}

Create an improved plan that addresses the identified issues. Keep successful elements and modify or add steps to improve quality.

Output a numbered list of refined steps:
"""
        return PromptTemplate(
            template=template,
            input_variables=["original_plan",
                             "result", "feedback", "domain_context"]
        )

    def _get_agent_prompt(self) -> PromptTemplate:
        """Get the main agent prompt template"""
        template = """
You are a {name}, an expert in {description}.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""
        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "name": self.name,
                "description": self.description,
                "tools": self._format_tools_description(),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )

    def _create_error_result(self, query: str, error_msg: str) -> DeepAgentResult:
        """Create an error result when execution fails"""
        return DeepAgentResult(
            query=query,
            plan=["Error occurred during planning"],
            execution_steps=[{"type": "error", "message": error_msg}],
            final_result=f"Error: {error_msg}",
            reflection={"confidence_score": 0.0,
                        "feedback": "Execution failed"},
            confidence_score=0.0,
            execution_time=0.0,
            tokens_used=0,
            iterations=0
        )
