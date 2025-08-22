"""
General Deep Agent Implementation

Specialized deep agent for general-purpose queries using LangChain.
Implements adaptive strategies for diverse information needs.
"""

from typing import List, Dict, Any
import logging
from datetime import datetime

from langchain.tools import Tool, BaseTool
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

from .base_deep_agent import BaseDeepAgent, DeepAgentResult

logger = logging.getLogger(__name__)


class AdaptiveSearchTool(BaseTool):
    """Tool that adapts search strategy based on query type"""

    name = "adaptive_search"
    description = "Adaptively search using the best strategy for the query type"

    def _run(self, query: str) -> str:
        """Execute adaptive search"""
        try:
            search = DuckDuckGoSearchRun()

            # Determine search strategy based on query characteristics
            query_lower = query.lower()

            if any(word in query_lower for word in ['how to', 'tutorial', 'guide', 'steps']):
                # How-to query - search for tutorials and guides
                enhanced_query = f"{query} tutorial guide how-to step by step"
            elif any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning']):
                # Definition query - search for explanations
                enhanced_query = f"{query} definition explanation overview"
            elif any(word in query_lower for word in ['best', 'top', 'review', 'compare']):
                # Comparison query - search for reviews and comparisons
                enhanced_query = f"{query} review comparison best practices"
            elif any(word in query_lower for word in ['price', 'cost', 'buy', 'purchase']):
                # Commercial query - search for pricing and purchasing
                enhanced_query = f"{query} price cost where to buy"
            else:
                # General query - use as-is
                enhanced_query = query

            results = search.run(enhanced_query)
            return f"Adaptive search results for '{query}':\n{results}"

        except (ConnectionError, ValueError, AttributeError, RuntimeError) as e:
            return f"Adaptive search failed: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


class KnowledgeBaseTool(BaseTool):
    """Tool for accessing structured knowledge"""

    name = "knowledge_base"
    description = "Access structured knowledge and encyclopedic information"

    def _run(self, query: str) -> str:
        """Search knowledge base"""
        try:
            # Use Wikipedia as primary knowledge base
            wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
            results = wikipedia.run(query)

            return f"Knowledge base results for '{query}':\n{results}"

        except (ConnectionError, ValueError, AttributeError, RuntimeError) as e:
            return f"Knowledge base search failed: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


class ContextEnrichmentTool(BaseTool):
    """Tool for enriching queries with additional context"""

    name = "context_enrichment"
    description = "Enrich queries with related context and background information"

    def _run(self, query: str) -> str:
        """Enrich query with context"""
        try:
            search = DuckDuckGoSearchRun()

            # Search for background context
            context_query = f"{query} background context overview"
            context_results = search.run(context_query)

            # Search for related topics
            related_query = f"{query} related topics similar"
            related_results = search.run(related_query)

            enrichment = f"Context enrichment for '{query}':\n\n"
            enrichment += f"Background Context:\n{context_results}\n\n"
            enrichment += f"Related Topics:\n{related_results}"

            return enrichment

        except (ConnectionError, ValueError, AttributeError, RuntimeError) as e:
            return f"Context enrichment failed: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


class SynthesisTool(BaseTool):
    """Tool for synthesizing information from multiple sources"""

    name = "synthesis"
    description = "Synthesize and organize information from multiple sources"

    def _run(self, information: str) -> str:
        """Synthesize information"""
        try:
            # This would use LLM for synthesis
            synthesis_template = f"""
Synthesize the following information into a coherent, well-organized response:

{information}

Organize by:
1. Key points and main findings
2. Supporting details and examples
3. Different perspectives or viewpoints
4. Practical applications or implications
5. Limitations or considerations

Provide a clear, comprehensive synthesis.
"""
            return f"Information Synthesis:\n{synthesis_template}"

        except (ValueError, AttributeError, TypeError) as e:
            return f"Synthesis failed: {str(e)}"

    async def _arun(self, information: str) -> str:
        """Async version"""
        return self._run(information)


class GeneralDeepAgent(BaseDeepAgent):
    """
    Deep agent for general-purpose queries

    Features:
    - Adaptive search strategies
    - Multi-source information gathering
    - Context enrichment and synthesis
    - Flexible reasoning patterns
    - Quality assessment across domains
    """

    def __init__(self, model: str = "gpt-4", temperature: float = 0.2):
        super().__init__(
            name="General Deep Agent",
            description="Expert in comprehensive information gathering and synthesis across all domains",
            model=model,
            temperature=temperature
        )

        # General-purpose configuration
        self.adaptability_threshold = 0.5
        self.context_enrichment_enabled = True
        self.synthesis_required = True

    def _create_tools(self) -> List[BaseTool]:
        """Create general-purpose tools"""
        tools = [
            AdaptiveSearchTool(),
            KnowledgeBaseTool(),
            ContextEnrichmentTool(),
            SynthesisTool(),
            DuckDuckGoSearchRun(),
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        ]

        return tools

    def _get_domain_context(self) -> str:
        """Get general domain context"""
        return """comprehensive information analysis across all domains. You specialize in:
- Adapting search strategies to query types
- Gathering information from diverse sources
- Synthesizing complex information
- Providing balanced, well-rounded responses
- Identifying the most relevant and useful information"""

    def comprehensive_query(self, query: str, include_context: bool = True,
                            synthesize_results: bool = True) -> DeepAgentResult:
        """
        Execute a comprehensive general query

        Args:
            query: General information query
            include_context: Whether to enrich with context
            synthesize_results: Whether to synthesize findings

        Returns:
            DeepAgentResult with comprehensive analysis
        """
        logger.info("[GeneralAgent] Starting comprehensive query: %s", query)

        # Enhance query with general instructions
        enhanced_query = self._enhance_general_query(
            query, include_context, synthesize_results)

        # Execute deep search with general parameters
        result = self.deep_search(enhanced_query, max_iterations=3)

        # Post-process for general quality
        result = self._post_process_general_result(result)

        return result

    def _enhance_general_query(self, query: str, include_context: bool,
                               synthesize_results: bool) -> str:
        """Enhance query with general-purpose instructions"""
        enhanced = f"Comprehensive Query: {query}\n\n"
        enhanced += "Instructions:\n"
        enhanced += "1. Determine the best search strategy for this query type\n"
        enhanced += "2. Gather information from multiple relevant sources\n"
        enhanced += "3. Ensure coverage of key aspects and perspectives\n"

        if include_context:
            enhanced += "4. Provide background context and related information\n"

        enhanced += "5. Verify information accuracy across sources\n"
        enhanced += "6. Identify any conflicting information or viewpoints\n"

        if synthesize_results:
            enhanced += "7. Synthesize findings into a coherent, comprehensive response\n"

        enhanced += "8. Ensure the response is practical and actionable\n"
        enhanced += "9. Note any limitations or areas for further exploration\n"

        return enhanced

    def _post_process_general_result(self, result: DeepAgentResult) -> DeepAgentResult:
        """Post-process result for general quality"""
        try:
            # Add general metadata
            metadata = self._generate_general_metadata(result)
            result.final_result = f"{result.final_result}\n\n{metadata}"

            # Add quality indicators
            quality_indicators = self._assess_general_quality(result)
            if quality_indicators:
                result.final_result += f"\n\n{quality_indicators}"

            return result

        except (ValueError, AttributeError, TypeError) as e:
            logger.error("[GeneralAgent] Post-processing failed: %s", str(e))
            return result

    def _generate_general_metadata(self, result: DeepAgentResult) -> str:
        """Generate general metadata for the result"""
        metadata = "--- Analysis Metadata ---\n"
        metadata += f"Query: {result.query}\n"
        metadata += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        metadata += f"Confidence Score: {result.confidence_score:.2f}\n"
        metadata += f"Processing Time: {result.execution_time:.2f}s\n"
        metadata += f"Refinement Iterations: {result.iterations}\n"

        # Add query classification
        query_type = self._classify_query_type(result.query)
        metadata += f"Query Type: {query_type}\n"

        return metadata

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['how to', 'tutorial', 'guide', 'steps']):
            return "How-to/Tutorial"
        elif any(word in query_lower for word in ['what is', 'define', 'definition']):
            return "Definition/Explanation"
        elif any(word in query_lower for word in ['best', 'top', 'review', 'compare']):
            return "Comparison/Review"
        elif any(word in query_lower for word in ['why', 'reason', 'cause']):
            return "Causal/Explanatory"
        elif any(word in query_lower for word in ['when', 'history', 'timeline']):
            return "Historical/Temporal"
        elif any(word in query_lower for word in ['where', 'location', 'place']):
            return "Geographic/Location"
        else:
            return "General Information"

    def _assess_general_quality(self, result: DeepAgentResult) -> str:
        """Assess the quality of general results"""
        quality_assessment = "--- Quality Assessment ---\n"

        # Comprehensive scoring
        if result.confidence_score >= 0.9:
            quality_assessment += "✓ Excellent: High-quality, comprehensive response\n"
        elif result.confidence_score >= 0.8:
            quality_assessment += "✓ Good: Well-researched with solid information\n"
        elif result.confidence_score >= 0.7:
            quality_assessment += "~ Adequate: Basic requirements met, some gaps possible\n"
        else:
            quality_assessment += "⚠ Needs Improvement: Additional research recommended\n"

        # Processing indicators
        if result.iterations > 2:
            quality_assessment += "✓ Thorough: Multiple refinement rounds completed\n"

        if result.execution_time > 15:
            quality_assessment += "✓ Comprehensive: Extensive analysis performed\n"

        # Add recommendations
        if result.confidence_score < 0.8:
            quality_assessment += "\nRecommendations:\n"
            quality_assessment += "• Consider consulting additional primary sources\n"
            quality_assessment += "• Verify key facts independently\n"
            quality_assessment += "• Look for recent updates on this topic\n"

        return quality_assessment

    def adaptive_search_strategy(self, query: str) -> Dict[str, Any]:
        """Determine the optimal search strategy for a query"""
        strategy_prompt = f"""
Analyze this query and determine the optimal search strategy:

Query: {query}

Consider:
1. Query type and intent
2. Required information depth
3. Best sources for this topic
4. Potential challenges or limitations
5. Expected user needs

Recommend:
- Search approach
- Key terms to include/exclude
- Source types to prioritize
- Verification methods needed

Provide a structured search strategy.
"""

        try:
            strategy_result = self.llm.predict(strategy_prompt)

            return {
                "query": query,
                "strategy": strategy_result,
                "query_type": self._classify_query_type(query),
                "timestamp": datetime.now().isoformat()
            }

        except (ValueError, AttributeError, RuntimeError) as e:
            logger.error("[GeneralAgent] Strategy planning failed: %s", str(e))
            return {"error": str(e)}

    def multi_perspective_analysis(self, topic: str) -> Dict[str, Any]:
        """Analyze a topic from multiple perspectives"""
        perspectives_prompt = f"""
Analyze this topic from multiple perspectives:

Topic: {topic}

Provide analysis from these viewpoints:
1. Technical/Scientific perspective
2. Economic/Business perspective
3. Social/Cultural perspective
4. Ethical/Moral perspective
5. Historical perspective
6. Future/Trend perspective

For each perspective:
- Key considerations
- Main arguments or findings
- Potential biases or limitations
- Supporting evidence

Identify areas of agreement and disagreement between perspectives.
"""

        try:
            analysis_result = self.llm.predict(perspectives_prompt)

            return {
                "topic": topic,
                "multi_perspective_analysis": analysis_result,
                "analysis_date": datetime.now().isoformat(),
                "perspectives_covered": 6
            }

        except (ValueError, AttributeError, RuntimeError) as e:
            logger.error(
                "[GeneralAgent] Multi-perspective analysis failed: %s", str(e))
            return {"error": str(e)}
