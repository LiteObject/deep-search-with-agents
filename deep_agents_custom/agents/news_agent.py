"""
News Agent - Specialized for current events and news searches.
"""

# Standard library imports
import time
from datetime import datetime
from typing import List

# First-party imports
from .base_agent import BaseAgent, SearchResult, SearchSummary

# Local imports with error handling
try:
    from config.settings import Settings
except ImportError:
    Settings = None


class NewsAgent(BaseAgent):
    """
    Agent specialized for news and current events.
    Focuses on recent news, breaking stories, and current developments.
    """

    def __init__(self, max_results: int = 15):
        super().__init__(
            "NewsAgent",
            (
                "Specialized for recent news, breaking stories, and "
                "current events. Prioritizes fresh, timely information."
            ),
            max_results=max_results,
        )

        # Initialize components using the parent class method
        self._initialize_common_components()

    def search(self, query: str, **kwargs) -> SearchSummary:
        """
        Perform news-focused search

        Args:
            query: News query
            **kwargs: Additional parameters
                - time_filter: Time filter for news (default: 'month')
                - include_breaking: Include breaking news sources
                  (default: True)

        Returns:
            SearchSummary: News-focused search summary
        """
        start_time = time.time()

        # Enhance query for news
        enhanced_query = self._enhance_news_query(
            query, kwargs.get("time_filter", "month")
        )

        # Ensure search_manager is initialized
        if self.search_manager is None:
            self._initialize_common_components()

        # If still None after initialization, handle gracefully
        if self.search_manager is None:
            # Fallback to simple single-engine search
            try:
                from ..tools.web_search import (  # pylint: disable=import-outside-toplevel
                    DuckDuckGoSearch,
                )
            except ImportError:
                # Handle case when running as main script
                import sys  # pylint: disable=import-outside-toplevel
                import os  # pylint: disable=import-outside-toplevel

                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                from tools.web_search import (  # pylint: disable=import-outside-toplevel
                    DuckDuckGoSearch,
                )

            ddg_search = DuckDuckGoSearch()
            results = ddg_search.search(enhanced_query, self.max_results)
        else:
            # Search for news with recent content preference
            # Use Tavily for better news results if available
            engines = ["duckduckgo"]
            if (
                Settings
                and hasattr(Settings, "TAVILY_API_KEY")
                and Settings.TAVILY_API_KEY
            ):
                engines = ["tavily", "duckduckgo"]

            results = self.search_manager.multi_search(
                enhanced_query, engines=engines, max_results_per_engine=self.max_results
            )

        # Filter and rank by news relevance
        filtered_results = self._filter_news_results(results)

        # Limit results
        top_results = filtered_results[: self.max_results]

        # Generate summary
        if self.summarizer:
            summary = self.summarizer.summarize_results(top_results, query)
            key_points = self._extract_news_insights(top_results)
        else:
            summary = f"Found {len(top_results)} news results for: {query}"
            key_points = []

        search_time = time.time() - start_time

        # Generate citations and cited summary
        citations = self._generate_citations(top_results)
        cited_summary = self._create_cited_summary(summary, citations, top_results)

        return SearchSummary(
            query=query,
            summary=summary,
            key_points=key_points,
            sources=list(set(r.source for r in top_results)),
            total_results=len(top_results),
            search_time=search_time,
            results=top_results,  # Include full results for better display
            citations=citations,  # Add citations
            cited_summary=cited_summary,  # Add cited summary
        )

    def _enhance_news_query(self, query: str, time_filter: str = "month") -> str:
        """
        Enhance query for news searching

        Args:
            query: Original query
            time_filter: Time filter for results

        Returns:
            str: Enhanced query
        """
        news_terms = [
            "news",
            "latest",
            "recent",
            "today",
            "breaking",
            "announcement",
            "update",
            "development",
        ]

        # Add temporal terms based on filter
        if time_filter == "day":
            news_terms.extend(["today", "daily", "24 hours"])
        elif time_filter == "week":
            news_terms.extend(["weekly", "past week", "7 days"])
        elif time_filter == "month":
            news_terms.extend(["monthly", "past month", "30 days"])

        # Combine with original query
        enhanced = f"{query} {' '.join(news_terms[:3])}"
        return enhanced

    def _filter_news_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Filter and rank results by news relevance

        Args:
            results: Raw search results

        Returns:
            List[SearchResult]: Filtered and ranked results
        """
        news_sources = [
            "reuters",
            "ap",
            "bbc",
            "cnn",
            "npr",
            "wsj",
            "nyt",
            "guardian",
            "bloomberg",
            "cnbc",
            "abc",
            "cbs",
            "nbc",
            "politico",
            "axios",
            "techcrunch",
            "ars-technica",
        ]

        news_keywords = [
            "breaking",
            "announced",
            "confirmed",
            "reported",
            "revealed",
            "update",
            "latest",
            "today",
            "recent",
        ]

        scored_results = []
        for result in results:
            score = 0

            # Check source
            domain = result.url.lower()
            for source in news_sources:
                if source in domain:
                    score += 3
                    break

            # Check for news keywords in title and content
            title_content = (result.title + " " + result.content).lower()
            for keyword in news_keywords:
                if keyword in title_content:
                    score += 1

            # Check for temporal indicators
            current_year = datetime.now().year
            temporal_words = [
                "today",
                "yesterday",
                "hours ago",
                "minutes ago",
                str(current_year),
                str(current_year - 1),
            ]
            for word in temporal_words:
                if word in title_content:
                    score += 2
                    break

            scored_results.append((score, result))

        # Sort by score (descending) and return results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result for score, result in scored_results]

    def _extract_news_insights(self, results: List[SearchResult]) -> List[str]:
        """
        Extract news-specific insights and developments

        Args:
            results: Search results

        Returns:
            List[str]: News insights
        """
        # Extract news-relevant points
        insights = []
        news_keywords = [
            "announced",
            "confirmed",
            "reported",
            "revealed",
            "stated",
            "said",
        ]

        for result in results[:5]:
            sentences = result.content.split(".")
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in news_keywords):
                    if len(sentence) > 20 and len(sentence) < 200:
                        insights.append(sentence)
                        break

        return insights[:7]

    def _process_results(self, results: List[SearchResult]) -> str:
        """
        Process and enhance news results into a summary string

        Args:
            results: Raw results

        Returns:
            str: Processed results summary
        """
        if not results:
            return "No news results found."

        summary_lines = []
        summary_lines.append(f"Found {len(results)} news results:")

        for i, result in enumerate(results[:5], 1):
            temporal_score = self._assess_temporal_relevance(result)
            summary_lines.append(f"{i}. {result.title}")
            summary_lines.append(
                f"   Source: {result.source} "
                f"(Temporal relevance: {temporal_score:.1f})"
            )
            summary_lines.append(f"   {result.content[:100]}...")
            summary_lines.append("")

        return "\n".join(summary_lines)

    def _assess_temporal_relevance(self, result: SearchResult) -> float:
        """
        Assess how recent/relevant the news item is

        Args:
            result: Search result

        Returns:
            float: Temporal relevance score (0-1)
        """
        content = (result.title + " " + result.content).lower()

        # Check for time indicators
        if any(word in content for word in ["today", "breaking", "just announced"]):
            return 1.0
        if any(word in content for word in ["yesterday", "hours ago"]):
            return 0.8
        if any(word in content for word in ["this week", "days ago"]):
            return 0.6
        if any(word in content for word in ["this month", "weeks ago"]):
            return 0.4

        return 0.2

    def _get_supported_query_types(self) -> List[str]:
        """Get supported query types for this agent"""
        return ["news", "current_events", "breaking", "announcement", "update"]

    def get_supported_categories(self) -> List[str]:
        """
        Get list of supported news categories

        Returns:
            List[str]: Supported categories
        """
        return [
            "breaking_news",
            "politics",
            "technology",
            "business",
            "science",
            "health",
            "sports",
            "entertainment",
            "world_news",
            "local_news",
        ]

    def search_by_category(self, topic: str, category: str) -> SearchSummary:
        """
        Search news by specific category

        Args:
            topic: News topic
            category: News category

        Returns:
            SearchSummary: Category-specific news results
        """
        category_query = f"{topic} {category} news"
        return self.search(category_query, time_filter="week")

    def breaking_news_search(self, topic: str) -> SearchSummary:
        """
        Search for breaking news on a topic

        Args:
            topic: Topic for breaking news

        Returns:
            SearchSummary: Breaking news results
        """
        breaking_query = f"breaking news {topic} urgent alert"
        return self.search(breaking_query, time_filter="day", include_breaking=True)
