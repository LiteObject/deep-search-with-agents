"""
Base Agent class for all search agents.
Provides common functionality and interface for specialized agents.
"""

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import logging


@dataclass
class SearchResult:
    """Data class for search results"""
    title: str
    url: str
    content: str
    source: str
    timestamp: datetime
    relevance_score: float = 0.0


@dataclass
class SearchSummary:
    """Data class for search summary"""
    query: str
    summary: str
    key_points: List[str]
    sources: List[str]
    total_results: int
    search_time: float


class BaseAgent(ABC):
    """
    Abstract base class for all search agents.
    Provides common interface and shared functionality.
    """

    def __init__(self, name: str, description: str, max_results: int = 10):
        self.name = name
        self.description = description
        self.max_results = max_results
        self.logger = logging.getLogger(f"agent.{name}")

        # These will be initialized by subclasses
        self.search_manager = None
        self.summarizer = None

    def _initialize_common_components(self):
        """Initialize common components used by all agents"""
        try:
            # Add parent directory to path for imports
            parent_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            from tools.web_search import WebSearchManager  # pylint: disable=import-outside-toplevel
            from tools.summarizer import LLMSummarizer, SimpleSummarizer  # pylint: disable=import-outside-toplevel

        except ImportError as e:
            # Handle missing modules gracefully
            self.logger.warning("Some tools modules not available: %s", str(e))
            return

        if self.search_manager is None:
            self.search_manager = WebSearchManager()

        if self.summarizer is None:
            # Try to use LLM summarizer, fallback to simple
            try:
                self.summarizer = LLMSummarizer()
            except (ImportError, AttributeError):
                self.summarizer = SimpleSummarizer()

    def _create_search_summary(self, *, query: str, summary: str,
                               results_data: Dict[str, Any]) -> SearchSummary:
        """Create a SearchSummary object with common fields

        Args:
            query: Search query
            summary: Summary text
            results_data: Dict containing key_points, top_results, search_time
        """
        return SearchSummary(
            query=query,
            summary=summary,
            key_points=results_data['key_points'],
            sources=list(set(r.source for r in results_data['top_results'])),
            total_results=len(results_data['top_results']),
            search_time=results_data['search_time']
        )

    def _extract_insights_from_results(self, results: List[SearchResult],
                                       keywords: List[str]) -> List[str]:
        """Common insight extraction logic used by multiple agents"""
        insights = []
        for result in results[:5]:
            sentences = result.content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in keywords):
                    if 20 < len(sentence) < 200:
                        insights.append(sentence)
                        break
        return insights[:7]

    def search(self, query: str, **kwargs) -> SearchSummary:
        """
        Main search method that each agent must implement.

        Args:
            query: The search query string
            **kwargs: Additional search parameters

        Returns:
            SearchSummary: Processed search results with summary
        """
        raise NotImplementedError("Subclasses must implement search method")

    @abstractmethod
    def _process_results(self, results: List[SearchResult]) -> str:
        """
        Process search results and generate summary.

        Args:
            results: List of search results

        Returns:
            str: Generated summary
        """
        raise NotImplementedError(
            "Subclasses must implement _process_results method")

    def _filter_results(
        self, results: List[SearchResult], min_score: float = 0.5
    ) -> List[SearchResult]:
        """
        Filter results based on relevance score.

        Args:
            results: List of search results
            min_score: Minimum relevance score

        Returns:
            List[SearchResult]: Filtered results
        """
        return [r for r in results if r.relevance_score >= min_score]

    def _deduplicate_results(
            self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate results based on URL.

        Args:
            results: List of search results

        Returns:
            List[SearchResult]: Deduplicated results
        """
        seen_urls = set()
        unique_results = []

        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        return unique_results

    def _extract_key_points(self, content: str,
                            max_points: int = 5) -> List[str]:
        """
        Extract key points from content.

        Args:
            content: Content to analyze
            max_points: Maximum number of key points

        Returns:
            List[str]: Key points extracted
        """
        # Simple implementation - can be enhanced with NLP
        sentences = content.split('.')
        # Filter out short sentences and take the most informative ones
        meaningful_sentences = [s.strip()
                                for s in sentences if len(s.strip()) > 50]
        return meaningful_sentences[:max_points]

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get agent capabilities and metadata.

        Returns:
            Dict: Agent capabilities
        """
        return {
            "name": self.name,
            "description": self.description,
            "max_results": self.max_results,
            "supported_queries": self._get_supported_query_types()
        }

    @abstractmethod
    def _get_supported_query_types(self) -> List[str]:
        """
        Get list of supported query types for this agent.

        Returns:
            List[str]: Supported query types
        """
        raise NotImplementedError(
            "Subclasses must implement _get_supported_query_types method")
