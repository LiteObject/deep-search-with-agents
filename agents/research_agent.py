"""
Research Agent - Specialized for academic and research-focused searches.
"""

import time
from typing import List
from agents.base_agent import BaseAgent, SearchResult, SearchSummary  # pylint: disable=import-error


class ResearchAgent(BaseAgent):
    """
    Agent specialized for academic and research content.
    Focuses on scholarly articles, research papers, and authoritative sources.
    """

    def __init__(self, max_results: int = 15):
        super().__init__(
            name="ResearchAgent",
            description=("Specialized agent for academic and "
                         "research content searches"),
            max_results=max_results
        )

        # Initialize components using the parent class method
        self._initialize_common_components()

    def search(self, query: str, **kwargs) -> SearchSummary:
        """
        Perform research-focused search

        Args:
            query: Research query
            **kwargs: Additional parameters
                - include_wikipedia: Include Wikipedia results (default: True)
                - focus_academic: Focus on academic sources (default: True)

        Returns:
            SearchSummary: Comprehensive research summary
        """
        start_time = time.time()

        # Enhance query for research
        enhanced_query = self._enhance_research_query(
            query, kwargs.get('focus_academic', True))

        # Check if search manager is available
        if self.search_manager is None:
            self.logger.error(
                "Search manager not available - cannot perform search")
            return SearchSummary(
                query=query,
                summary="Search functionality not available - missing dependencies",
                key_points=["Search tools not properly initialized"],
                sources=[],
                total_results=0,
                search_time=0.0
            )

        # Search multiple sources
        search_engines = ['duckduckgo']
        if kwargs.get('include_wikipedia', True):
            search_engines.append('wikipedia')

        # Add academic-specific terms
        if kwargs.get('focus_academic', True):
            academic_query = f"{enhanced_query} research study academic paper"
            results = self.search_manager.multi_search(
                academic_query,
                engines=search_engines,
                max_results_per_engine=8
            )
        else:
            results = self.search_manager.multi_search(
                enhanced_query,
                engines=search_engines,
                max_results_per_engine=8
            )

        # Filter and rank results
        filtered_results = self._filter_research_results(results)
        top_results = filtered_results[:self.max_results]

        # Generate summary
        summary_text = self._process_results(top_results)

        # Extract key insights
        key_points = self._extract_research_insights(top_results)

        search_time = time.time() - start_time

        return SearchSummary(
            query=query,
            summary=summary_text,
            key_points=key_points,
            sources=[r.url for r in top_results],
            total_results=len(top_results),
            search_time=search_time
        )

    def _enhance_research_query(self, query: str,
                                academic_focus: bool = True) -> str:
        """
        Enhance query for research purposes

        Args:
            query: Original query
            academic_focus: Whether to add academic terms

        Returns:
            str: Enhanced query
        """
        if academic_focus:
            # Add research-specific terms
            academic_terms = [
                "research", "study", "analysis", "findings",
                "methodology", "peer-reviewed", "academic"
            ]

            # Check if query already contains academic terms
            query_lower = query.lower()
            has_academic_terms = any(
                term in query_lower for term in academic_terms)

            if not has_academic_terms:
                query = f"{query} research study"

        return query

    def _filter_research_results(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Filter results for research quality

        Args:
            results: Raw search results

        Returns:
            List[SearchResult]: Filtered research results
        """
        research_indicators = [
            'research', 'study', 'analysis', 'journal', 'academic',
            'university', 'institute', 'paper', 'findings', 'methodology',
            'peer-reviewed', 'scholar', 'publication', '.edu', 'doi',
            'arxiv', 'pubmed', 'ncbi', 'nature', 'science'
        ]

        scored_results = []

        for result in results:
            # Calculate research relevance score
            content_text = (result.title + " " +
                            result.content + " " + result.url).lower()

            research_score = 0
            for indicator in research_indicators:
                if indicator in content_text:
                    research_score += 1

            # Boost Wikipedia and educational domains
            if 'wikipedia' in result.source or '.edu' in result.url:
                research_score += 2

            # Update relevance score
            result.relevance_score = min(
                1.0, result.relevance_score + (research_score * 0.1))
            scored_results.append(result)

        # Sort by research relevance
        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)

        return scored_results

    def _process_results(self, results: List[SearchResult]) -> str:
        """
        Process search results and generate research summary

        Args:
            results: Search results

        Returns:
            str: Research summary
        """
        if self.summarizer is None:
            # Fallback when summarizer is not available
            if not results:
                return "No research results found."

            summary_lines = [f"Found {len(results)} research results:"]
            for i, result in enumerate(results[:5], 1):
                summary_lines.append(f"{i}. {result.title}")
                summary_lines.append(f"   Source: {result.source}")
                summary_lines.append(f"   {result.content[:150]}...")
                summary_lines.append("")
            return "\n".join(summary_lines)

        if hasattr(self.summarizer, 'summarize_results'):
            return self.summarizer.summarize_results(
                results, "research query")

        return self.summarizer.summarize_results(
            results, "research query")

    def _extract_research_insights(
            self, results: List[SearchResult]) -> List[str]:
        """
        Extract research-specific insights

        Args:
            results: Search results
            query: Original query

        Returns:
            List[str]: Research insights
        """
        # Extract research-relevant points manually
        insights = []
        research_keywords = ['study', 'research',
                             'finding', 'analysis', 'methodology']

        for result in results[:5]:
            sentences = result.content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower()
                       for keyword in research_keywords):
                    if len(sentence) > 30 and len(sentence) < 200:
                        insights.append(sentence)
                        break

        return insights[:5]

    def _get_supported_query_types(self) -> List[str]:
        """
        Get supported query types for research agent

        Returns:
            List[str]: Supported query types
        """
        return [
            "academic_research",
            "scientific_studies",
            "literature_review",
            "methodology_analysis",
            "peer_reviewed_content",
            "educational_content"
        ]

    def search_papers(self, topic: str) -> SearchSummary:
        """
        Search specifically for research papers and academic content

        Args:
            topic: Research topic
            max_results: Maximum results to return

        Returns:
            SearchSummary: Academic search results
        """
        academic_query = f"{topic} research paper study academic journal"
        return self.search(academic_query, focus_academic=True,
                           include_wikipedia=False)

    def literature_review(self, topic: str) -> SearchSummary:
        """
        Conduct a literature review search

        Args:
            topic: Topic for literature review

        Returns:
            SearchSummary: Literature review results
        """
        review_query = (f"{topic} literature review systematic "
                        "review meta-analysis")
        return self.search(review_query, focus_academic=True,
                           include_wikipedia=True)
