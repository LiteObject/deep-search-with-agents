"""
Web search implementations using various search engines.
"""

import logging
import os
from datetime import datetime
from typing import List, Optional

import requests  # type: ignore

from utils.types import SearchResult  # type: ignore # pylint: disable=import-error
from config.settings import Settings  # type: ignore # pylint: disable=import-error

logger = logging.getLogger(__name__)


class DuckDuckGoSearch:  # pylint: disable=too-few-public-methods
    """
    DuckDuckGo search implementation

    Note: The duckduckgo-search package has known compatibility issues with httpx versions.
    The package tries to pass a 'proxies' parameter to httpx.Client() but many httpx versions
    don't support this parameter, causing TypeError: Client.__init__() got an unexpected
    keyword argument 'proxies'.

    This implementation includes fallback mechanisms to handle this issue gracefully.

    Potential solutions:
    1. Update httpx to latest version (may break other dependencies)
    2. Use specific duckduckgo-search version that's compatible
    3. Use fallback search method (implemented)
    4. Set DDGS_PROXY environment variable if proxy is needed
    """

    def __init__(self):
        self.base_url = "https://html.duckduckgo.com/html/"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            }
        )

    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Perform DuckDuckGo search

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List[SearchResult]: Search results
        """
        try:
            # Try to use duckduckgo-search library for better results
            from duckduckgo_search import DDGS  # type: ignore  # pylint: disable=import-outside-toplevel

            results = []
            try:
                # Updated initialization for newer versions of duckduckgo-search
                # Handle version compatibility issues with httpx/proxies
                try:
                    # Initialize DDGS with basic configuration
                    # The package supports proxy parameter but we'll start with basic init
                    ddgs = DDGS(timeout=10)
                except TypeError as type_error:
                    if "proxies" in str(type_error):
                        logger.warning(
                            "DuckDuckGo DDGS version incompatibility with httpx: %s",
                            str(type_error),
                        )
                        logger.info(
                            "This is a known issue with duckduckgo-search + httpx compatibility"
                        )
                        logger.info("Falling back to requests-based search")
                        return self._fallback_requests_search(query, max_results)
                    else:
                        raise

                # Use the text() method as recommended in the official docs
                search_results = ddgs.text(query, max_results=max_results)

                for i, result in enumerate(search_results):
                    if i >= max_results:
                        break

                    search_result = SearchResult(
                        title=result.get("title") or "",
                        url=result.get("href") or "",
                        content=result.get("body") or "",
                        source="duckduckgo",
                        timestamp=datetime.now(),
                        relevance_score=1.0 - (i * 0.1),  # Simple scoring
                    )
                    results.append(search_result)
            except Exception as ddgs_error:  # pylint: disable=broad-exception-caught
                logger.error("DuckDuckGo DDGS search error: %s", str(ddgs_error))
                # Return empty results instead of fallback
                logger.info("DuckDuckGo search failed, returning empty results")
                return []

            return results

        except ImportError as import_error:
            logger.warning(
                "DuckDuckGo duckduckgo-search library not available: %s",
                str(import_error),
            )
            logger.info("DuckDuckGo search disabled, other engines will be used")
            return []
        except (ConnectionError, TimeoutError) as e:
            logger.error("DuckDuckGo search connection error: %s", str(e))
            return []

    def _fallback_requests_search(
        self, query: str, max_results: int = 10
    ) -> List[SearchResult]:
        """
        Fallback search method using requests when DDGS fails

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List[SearchResult]: Search results
        """
        try:
            # Simple DuckDuckGo search using requests
            params = {
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
                "skip_disambig": "1",
            }

            response = self.session.get(
                "https://api.duckduckgo.com/", params=params, timeout=10
            )
            response.raise_for_status()

            data = response.json()
            results = []

            # Parse DuckDuckGo API response
            related_topics = data.get("RelatedTopics", [])
            for i, topic in enumerate(related_topics):
                if i >= max_results:
                    break

                if isinstance(topic, dict) and "Text" in topic and "FirstURL" in topic:
                    result = SearchResult(
                        title=(
                            topic.get("Text", "")[:100] + "..."
                            if len(topic.get("Text", "")) > 100
                            else topic.get("Text", "")
                        ),
                        url=topic.get("FirstURL", ""),
                        content=topic.get("Text", ""),
                        source="duckduckgo",
                        timestamp=datetime.now(),
                        relevance_score=1.0 - (i * 0.1),
                    )
                    results.append(result)

            logger.info("DuckDuckGo fallback search found %d results", len(results))
            return results

        except (requests.RequestException, ValueError, KeyError) as e:
            logger.error("DuckDuckGo fallback search failed: %s", str(e))
            return []

    def _fallback_search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Fallback search method - returns empty results instead of demo data

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List[SearchResult]: Empty search results
        """
        logger.info("DuckDuckGo search failed, returning empty results")
        return []


class TavilySearch:  # pylint: disable=too-few-public-methods
    """Tavily search implementation"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Settings.TAVILY_API_KEY
        if not self.api_key:
            logger.warning("Tavily API key not found. Tavily search will not work.")

    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Perform Tavily search

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List[SearchResult]: Search results
        """
        if not self.api_key:
            logger.warning("Tavily API key not available")
            return []

        try:
            # type: ignore  # pylint: disable=import-outside-toplevel
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.api_key)
            response = client.search(
                query=query, search_depth="advanced", max_results=max_results
            )

            results = []
            # Check if response is valid and has results
            if response and isinstance(response, dict):
                for result in response.get("results", []):
                    search_result = SearchResult(
                        title=result.get("title") or "",
                        url=result.get("url") or "",
                        content=result.get("content") or "",
                        source="tavily",
                        timestamp=datetime.now(),
                        relevance_score=result.get("score", 0.5),
                    )
                    results.append(search_result)

            return results

        except ImportError:
            logger.error("Tavily library not available")
            return []
        except (ConnectionError, ValueError) as e:
            logger.error("Tavily search error: %s", str(e))
            return []


class WikipediaSearch:  # pylint: disable=too-few-public-methods
    """Wikipedia search implementation"""

    def __init__(self):
        self.base_url = "https://en.wikipedia.org/api/rest_v1"

    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Perform Wikipedia search

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List[SearchResult]: Search results
        """
        try:
            import wikipedia  # type: ignore  # pylint: disable=import-outside-toplevel

            # Set language to English
            wikipedia.set_lang("en")

            results = []

            # Search for pages
            search_results = wikipedia.search(query, results=max_results)

            for i, title in enumerate(search_results):
                try:
                    # Get page summary
                    summary = wikipedia.summary(title, sentences=3)
                    page = wikipedia.page(title)

                    search_result = SearchResult(
                        title=title,
                        url=page.url,
                        content=summary,
                        source="wikipedia",
                        timestamp=datetime.now(),
                        relevance_score=1.0 - (i * 0.1),
                    )
                    results.append(search_result)

                except wikipedia.exceptions.DisambiguationError as e:
                    # Take the first option from disambiguation
                    try:
                        page = wikipedia.page(e.options[0])
                        summary = wikipedia.summary(e.options[0], sentences=3)

                        search_result = SearchResult(
                            title=e.options[0],
                            url=page.url,
                            content=summary,
                            source="wikipedia",
                            timestamp=datetime.now(),
                            relevance_score=0.8 - (i * 0.1),
                        )
                        results.append(search_result)
                    except (AttributeError, ValueError):
                        continue

                except wikipedia.exceptions.PageError:
                    # Skip pages that don't exist
                    continue

            return results

        except ImportError:
            logger.error("Wikipedia library not available")
            return []
        except (ConnectionError, ValueError) as e:
            logger.error("Wikipedia search error: %s", str(e))
            return []


class WebSearchManager:
    """Manages multiple search engines"""

    def __init__(self):
        self.engines = {
            "duckduckgo": DuckDuckGoSearch(),
            "tavily": TavilySearch(),
            "wikipedia": WikipediaSearch(),
        }
        self.default_engine = os.getenv("DEFAULT_SEARCH_ENGINE", "duckduckgo")

    def search(
        self, query: str, engine: Optional[str] = None, max_results: int = 10
    ) -> List[SearchResult]:
        """
        Perform search using specified engine

        Args:
            query: Search query
            engine: Search engine to use
            max_results: Maximum number of results

        Returns:
            List[SearchResult]: Search results
        """
        engine = engine or self.default_engine

        if engine not in self.engines:
            logger.error("Unknown search engine: %s", engine)
            return []

        logger.info("Searching with %s: %s", engine, query)
        return self.engines[engine].search(query, max_results)

    def multi_search(
        self,
        query: str,
        engines: Optional[List[str]] = None,
        max_results_per_engine: int = 5,
    ) -> List[SearchResult]:
        """
        Search across multiple engines and combine results

        Args:
            query: Search query
            engines: List of engines to use
            max_results_per_engine: Max results per engine

        Returns:
            List[SearchResult]: Combined search results
        """
        if engines is None:
            # Use all available engines by default, prioritizing Tavily if available
            engines = ["duckduckgo", "wikipedia"]
            if Settings.TAVILY_API_KEY:
                engines.insert(0, "tavily")  # Add Tavily first for better results

        all_results = []

        for engine in engines:
            if engine in self.engines:
                results = self.search(query, engine, max_results_per_engine)
                all_results.extend(results)

        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []

        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        # Sort by relevance score
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)

        return unique_results
