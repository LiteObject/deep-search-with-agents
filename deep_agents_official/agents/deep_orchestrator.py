"""
Official DeepAgents Orchestrator
Coordinates multiple DeepAgents for complex tasks requiring multiple capabilities.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config.settings import Config
from .analysis_deep_agent import AnalysisDeepAgent
from .coding_deep_agent import CodingDeepAgent
from .research_deep_agent import ResearchDeepAgent

# Add parent directory to path for imports (must be before other local imports)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import configuration and agents

logger = logging.getLogger(__name__)


class DeepAgentsOrchestrator:
    """Orchestrates multiple Official DeepAgents for complex multi-step tasks"""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or Config.DEFAULT_MODEL

        # Initialize specialized agents
        self.research_agent = ResearchDeepAgent(model_name)
        self.coding_agent = CodingDeepAgent(model_name)
        self.analysis_agent = AnalysisDeepAgent(model_name)

        # Track orchestration state
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.task_history = []

        logger.info(
            "DeepAgents Orchestrator initialized with session: %s", self.session_id)

    def execute_multi_agent_task(self,
                                 task_description: str,
                                 task_type: str = "auto",
                                 context: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a complex task that may require multiple agents

        Args:
            task_description: Description of the task to execute
            task_type: Type of task (auto, research, coding, analysis, mixed)
            context: Additional context for the task
        """
        try:
            logger.info("Starting multi-agent task: %s", task_description)

            # Analyze task requirements
            task_plan = self._analyze_task_requirements(
                task_description, task_type, context)

            # Execute task based on plan
            results = []
            for step in task_plan["steps"]:
                step_result = self._execute_task_step(step)
                results.append(step_result)

                # Check if step failed and handle accordingly
                if step_result["status"] == "error":
                    logger.warning("Step failed: %s", step['description'])
                    # Could implement retry logic or alternative approaches here

            # Synthesize final results
            final_result = self._synthesize_results(
                task_description, results, task_plan)

            # Store in task history
            self.task_history.append({
                "timestamp": datetime.now().isoformat(),
                "task_description": task_description,
                "task_type": task_type,
                "task_plan": task_plan,
                "results": results,
                "final_result": final_result
            })

            logger.info("Multi-agent task completed: %s", task_description)
            return final_result

        except (ValueError, TypeError, KeyError) as e:
            logger.error("Multi-agent task failed: %s", e)
            return {
                "status": "error",
                "task_description": task_description,
                "error": str(e),
                "agent_type": "DeepAgentsOrchestrator"
            }

    def _analyze_task_requirements(self,
                                   task_description: str,
                                   task_type: str,
                                   context: Optional[str]) -> Dict[str, Any]:
        """Analyze task and create execution plan"""

        # Determine which agents are needed based on task description
        needs_research = any(keyword in task_description.lower() for keyword in [
            'research', 'find', 'search', 'investigate', 'explore', 'analyze information'
        ])

        needs_coding = any(keyword in task_description.lower() for keyword in [
            'code', 'program', 'implement', 'develop', 'debug', 'test', 'function', 'class'
        ])

        needs_analysis = any(keyword in task_description.lower() for keyword in [
            'analyze', 'compare', 'evaluate', 'assess', 'synthesize', 'insights', 'trends'
        ])

        # Override with explicit task type if provided
        if task_type == "research":
            needs_research = True
            needs_coding = False
            needs_analysis = False
        elif task_type == "coding":
            needs_research = False
            needs_coding = True
            needs_analysis = False
        elif task_type == "analysis":
            needs_research = False
            needs_coding = False
            needs_analysis = True
        elif task_type == "mixed":
            needs_research = True
            needs_coding = True
            needs_analysis = True

        # Create execution plan
        steps = []

        if needs_research:
            steps.append({
                "agent": "research",
                "description": "Conduct research and gather information",
                "priority": 1
            })

        if needs_coding:
            steps.append({
                "agent": "coding",
                "description": "Implement code solutions",
                "priority": 2
            })

        if needs_analysis:
            steps.append({
                "agent": "analysis",
                "description": "Analyze data and generate insights",
                "priority": 3 if needs_research or needs_coding else 1
            })

        # Default to research if no specific needs identified
        if not steps:
            steps.append({
                "agent": "research",
                "description": "General research and information gathering",
                "priority": 1
            })

        # Sort steps by priority
        steps.sort(key=lambda x: x["priority"])

        return {
            "task_description": task_description,
            "task_type": task_type,
            "context": context,
            "needs_research": needs_research,
            "needs_coding": needs_coding,
            "needs_analysis": needs_analysis,
            "steps": steps,
            "estimated_complexity": len(steps)
        }

    def _execute_task_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task step with the appropriate agent"""

        agent_type = step["agent"]
        description = step["description"]

        try:
            if agent_type == "research":
                return self.research_agent.research(
                    query=description,
                    context=f"Part of larger task: {step.get('task_description', '')}"
                )

            elif agent_type == "coding":
                return self.coding_agent.generate_code(
                    requirements=description,
                    context=f"Part of larger task: {step.get('task_description', '')}"
                )

            elif agent_type == "analysis":
                # For analysis, we might need data from previous steps
                return self.analysis_agent.analyze_data(
                    data=description,  # Would normally pass actual data
                    analysis_type="general",
                    context=f"Part of larger task: {step.get('task_description', '')}"
                )

            else:
                return {
                    "status": "error",
                    "error": f"Unknown agent type: {agent_type}",
                    "step": step
                }

        except (AttributeError, ValueError, TypeError) as e:
            logger.error("Task step execution failed: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "step": step
            }

    def _synthesize_results(self,
                            task_description: str,
                            results: List[Dict[str, Any]],
                            task_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from multiple agents"""

        successful_results = [
            r for r in results if r.get("status") == "success"]
        failed_results = [r for r in results if r.get("status") == "error"]

        # Create synthesis
        synthesis = {
            "status": "success" if successful_results else "partial_failure",
            "task_description": task_description,
            "session_id": self.session_id,
            "task_plan": task_plan,
            "execution_summary": {
                "total_steps": len(results),
                "successful_steps": len(successful_results),
                "failed_steps": len(failed_results),
                "agents_used": list(set(r.get("agent_type", "unknown") for r in results))
            },
            "results": {
                "successful": successful_results,
                "failed": failed_results
            },
            "insights": self._generate_insights(successful_results),
            "recommendations": self._generate_recommendations(task_description, results)
        }

        return synthesis

    def _generate_insights(self, successful_results: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from successful results"""
        insights = []

        if not successful_results:
            return ["No successful results to generate insights from"]

        # Analyze patterns in results
        agent_types = [r.get("agent_type", "") for r in successful_results]
        unique_agents = set(agent_types)

        insights.append(
            f"Successfully coordinated {len(unique_agents)} different agent types")

        if "ResearchDeepAgent" in agent_types:
            insights.append("Research phase provided foundational information")

        if "CodingDeepAgent" in agent_types:
            insights.append("Coding phase delivered technical implementation")

        if "AnalysisDeepAgent" in agent_types:
            insights.append("Analysis phase generated data-driven insights")

        return insights

    def _generate_recommendations(self,
                                  task_description: str,
                                  results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on task execution"""
        recommendations = []

        failed_count = len([r for r in results if r.get("status") == "error"])

        if failed_count == 0:
            recommendations.append(
                "Task completed successfully - consider similar orchestration patterns for future complex tasks")
        elif failed_count < len(results):
            recommendations.append(
                "Task partially completed - review failed steps and consider retry mechanisms")
        else:
            recommendations.append(
                "Task failed - review requirements and agent capabilities")

        # Agent-specific recommendations
        if any("research" in str(r).lower() for r in results):
            recommendations.append(
                "Research component identified - ensure sufficient search context for future tasks")

        if any("coding" in str(r).lower() for r in results):
            recommendations.append(
                "Coding component identified - consider code review and testing phases")

        if any("analysis" in str(r).lower() for r in results):
            recommendations.append(
                "Analysis component identified - validate data quality and analysis methodology")

        return recommendations

    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get the history of tasks executed in this session"""
        return self.task_history

    def get_available_agents(self) -> Dict[str, str]:
        """Get information about available agents"""
        return {
            "research": "ResearchDeepAgent - Specialized in research and information gathering",
            "coding": "CodingDeepAgent - Specialized in code analysis, generation, and debugging",
            "analysis": "AnalysisDeepAgent - Specialized in data analysis and insight generation"
        }
