"""
General Agent - For broad, general-purpose searches.
"""

from agents.base_agent import BaseAgent, SearchResult, SearchSummary
import os
import sys
import time
from typing import List

# Add parent directory to Python path to ensure consistent imports
parent_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class GeneralAgent(BaseAgent):
    """
    General-purpose search agent for broad topics and everyday queries.
    Provides comprehensive results from multiple sources.
    """

    def __init__(self, max_results: int = 15):
        super().__init__(
            "GeneralAgent",
            "General-purpose agent for comprehensive web searches.",
            max_results
        )

        # Initialize components using the parent class method
        self._initialize_common_components()

    def search(self, query: str, **kwargs) -> SearchSummary:
        """
        Perform general-purpose search

        Args:
            query: Search query
            **kwargs: Additional parameters
                - include_wikipedia: Include Wikipedia results (default: True)
                - engines: List of search engines to use
                  (default: ['duckduckgo', 'wikipedia'])
                - max_results_override: Override max_results for this search

        Returns:
            SearchSummary: Comprehensive search summary
        """
        start_time = time.time()

        # Default engines
        engines = kwargs.get('engines', ['duckduckgo'])
        if kwargs.get('include_wikipedia', True):
            engines.append('wikipedia')

        # Use override if provided, otherwise use instance max_results
        max_results = kwargs.get('max_results_override', self.max_results)

        # Ensure search_manager is initialized
        if self.search_manager is None:
            self._initialize_common_components()

        # If still None after initialization, handle gracefully
        if self.search_manager is None:
            # Fallback to simple single-engine search
            from tools.web_search import DuckDuckGoSearch  # pylint: disable=import-outside-toplevel
            ddg_search = DuckDuckGoSearch()
            results = ddg_search.search(query, max_results)
        else:
            # Perform multi-engine search
            results = self.search_manager.multi_search(
                query,
                engines=engines,
                max_results_per_engine=max_results // len(engines)
            )

        # Process and filter results
        processed_results = self._filter_general_results(results)

        # Limit results
        top_results = processed_results[:self.max_results]

        # Generate summary
        if self.summarizer:
            summary = self.summarizer.summarize_results(top_results, query)
            key_points = self._extract_general_insights(top_results)
        else:
            summary = f"Found {len(top_results)} results for: {query}"
            key_points = []

        search_time = time.time() - start_time

        return SearchSummary(
            query=query,
            summary=summary,
            key_points=key_points,
            sources=list(set(r.source for r in top_results)),
            total_results=len(top_results),
            search_time=search_time
        )

    def _filter_general_results(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Filter and rank results for general relevance

        Args:
            results: Raw search results

        Returns:
            List[SearchResult]: Filtered and ranked results
        """
        # Simple scoring based on content length and source diversity
        scored_results = []
        for result in results:
            score = 0

            # Prefer results with substantial content
            if len(result.content) > 100:
                score += 2
            if len(result.content) > 300:
                score += 1

            # Prefer educational and informational sources
            domain = result.url.lower()
            edu_domains = ['.edu', '.org', 'wikipedia']
            if any(edu_domain in domain for edu_domain in edu_domains):
                score += 2

            scored_results.append((score, result))

        # Sort by score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result for score, result in scored_results]

    def _extract_general_insights(
            self, results: List[SearchResult]) -> List[str]:
        """
        Extract general insights from search results

        Args:
            results: Search results

        Returns:
            List[str]: Key insights
        """
        # Extract meaningful sentences
        insights = []

        for result in results[:6]:
            # Split content into sentences
            sentences = [s.strip()
                         for s in result.content.split('.') if s.strip()]

            # Find informative sentences
            for sentence in sentences:
                if len(sentence) > 30 and len(sentence) < 200:
                    insights.append(sentence)
                    break

        return insights[:7]

    def _process_results(self, results: List[SearchResult]) -> str:
        """
        Process results into a summary string

        Args:
            results: Search results

        Returns:
            str: Processed results summary
        """
        if not results:
            return "No results found."

        summary_lines = []
        summary_lines.append(f"Found {len(results)} general search results:")

        # Group by source
        sources = {}
        for result in results:
            if result.source not in sources:
                sources[result.source] = []
            sources[result.source].append(result)

        for source, source_results in sources.items():
            summary_lines.append(
                f"\nFrom {source} ({len(source_results)} results):")
            for result in source_results[:3]:  # Show top 3 per source
                summary_lines.append(f"  • {result.title}")
                summary_lines.append(f"    {result.content[:100]}...")

        return "\n".join(summary_lines)

    def _get_supported_query_types(self) -> List[str]:
        """Get supported query types for this agent"""
        return [
            "general", "information", "definition",
            "explanation", "how_to", "what_is"
        ]

    def get_supported_categories(self) -> List[str]:
        """
        Get list of supported search categories

        Returns:
            List[str]: Supported categories
        """
        return [
            "general_information",
            "definitions",
            "how_to_guides",
            "explanations",
            "comparisons",
            "history",
            "science",
            "technology",
            "education"
        ]

    def quick_search(self, query: str) -> SearchSummary:
        """
        Perform a quick search with limited results

        Args:
            query: Search query

        Returns:
            SearchSummary: Quick search results
        """
        # Use a limited max_results for quick search without modifying the instance
        result = self.search(
            query, engines=['duckduckgo'], max_results_override=5)
        return result

    def comprehensive_search(self, query: str) -> SearchSummary:
        """
        Perform a comprehensive search with extended results

        Args:
            query: Search query

        Returns:
            SearchSummary: Comprehensive search results
        """
        # Use extended max_results for comprehensive search without modifying the instance
        result = self.search(
            query,
            engines=['duckduckgo', 'wikipedia'],
            include_wikipedia=True,
            max_results_override=25
        )
        return result
