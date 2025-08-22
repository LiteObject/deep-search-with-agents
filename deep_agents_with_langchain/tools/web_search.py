"""
Enhanced web search tool for LangChain Deep Agents
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

import requests  # type: ignore

from langchain.tools import BaseTool
from langchain.tools.base import ToolException

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSearchResult:
    """Enhanced search result with additional metadata"""
    title: str
    url: str
    content: str
    source: str
    timestamp: datetime
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SerperWebSearchTool(BaseTool):
    """Enhanced web search tool using Serper API"""

    name = "serper_web_search"
    description = "Search the web using Serper API for current information"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name=self.name,
            description=self.description
        )
        self.api_key = api_key or os.getenv('SERPER_API_KEY')
        if not self.api_key:
            raise ValueError("SERPER_API_KEY is required for web search")

        self.base_url = "https://google.serper.dev"
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        })

    def _run(self, query: str, num_results: int = 10, search_type: str = "search",
             **kwargs) -> str:
        """Run web search"""
        try:
            # Prepare search parameters
            search_params = {
                "q": query,
                "num": min(num_results, 10),  # Serper limits to 10
                "gl": "us",  # Geolocation
                "hl": "en"   # Language
            }

            # Choose endpoint based on search type
            endpoint = f"{self.base_url}/{search_type}"

            response = self.session.post(
                endpoint, json=search_params, timeout=30)
            response.raise_for_status()

            data = response.json()
            results = self._parse_search_results(data, search_type)

            # Format results for LLM consumption
            return self._format_results_for_llm(results)

        except requests.RequestException as e:
            logger.error("Serper search error: %s", str(e))
            raise ToolException(f"Web search failed: {str(e)}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error("Unexpected search error: %s", str(e))
            raise ToolException(f"Search error: {str(e)}") from e

    async def _arun(self, query: str, num_results: int = 10, search_type: str = "search",
                    **kwargs) -> str:
        """Async version of web search"""
        return self._run(query, num_results, search_type, **kwargs)

    def _parse_search_results(self, data: Dict[str, Any],
                              search_type: str) -> List[EnhancedSearchResult]:
        """Parse search results from API response"""
        results = []

        # Handle different search types
        if search_type == "search":
            organic_results = data.get("organic", [])

            for i, result in enumerate(organic_results):
                enhanced_result = EnhancedSearchResult(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    content=result.get("snippet", ""),
                    source="serper_web",
                    timestamp=datetime.now(),
                    relevance_score=1.0 - (i * 0.1),
                    metadata={
                        "position": result.get("position", i + 1),
                        "displayed_link": result.get("displayedLink", ""),
                        "date": result.get("date"),
                        "search_type": search_type
                    }
                )
                results.append(enhanced_result)

        elif search_type == "news":
            news_results = data.get("news", [])

            for i, result in enumerate(news_results):
                enhanced_result = EnhancedSearchResult(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    content=result.get("snippet", ""),
                    source="serper_news",
                    timestamp=datetime.now(),
                    # News gets higher relevance
                    relevance_score=1.0 - (i * 0.05),
                    metadata={
                        "position": i + 1,
                        "date": result.get("date"),
                        "source_name": result.get("source", ""),
                        "image_url": result.get("imageUrl"),
                        "search_type": search_type
                    }
                )
                results.append(enhanced_result)

        return results

    def _format_results_for_llm(self, results: List[EnhancedSearchResult]) -> str:
        """Format search results for LLM consumption"""
        if not results:
            return "No search results found."

        formatted_results = []

        for i, result in enumerate(results, 1):
            formatted_result = f"""
Result {i}:
Title: {result.title}
URL: {result.url}
Content: {result.content}
Source: {result.source}
Relevance: {result.relevance_score:.2f}
"""
            if result.metadata.get("date"):
                formatted_result += f"Date: {result.metadata['date']}\n"

            formatted_results.append(formatted_result.strip())

        return "\n\n".join(formatted_results)


class DuckDuckGoSearchTool(BaseTool):
    """DuckDuckGo search tool as fallback"""

    name = "duckduckgo_search"
    description = "Search using DuckDuckGo when other search engines are unavailable"

    def _run(self, query: str, max_results: int = 10, **kwargs) -> str:
        """Run DuckDuckGo search"""
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)

                for i, result in enumerate(search_results):
                    if i >= max_results:
                        break

                    enhanced_result = EnhancedSearchResult(
                        title=result.get('title') or '',
                        url=result.get('href') or '',
                        content=result.get('body') or '',
                        source='duckduckgo',
                        timestamp=datetime.now(),
                        relevance_score=1.0 - (i * 0.1),
                        metadata={'position': i + 1}
                    )
                    results.append(enhanced_result)

            return self._format_results_for_llm(results)

        except ImportError as exc:
            raise ToolException(
                "DuckDuckGo search library not available") from exc
        except (ValueError, AttributeError, ConnectionError) as e:
            raise ToolException(f"DuckDuckGo search failed: {str(e)}") from e

    async def _arun(self, query: str, max_results: int = 10, **kwargs) -> str:
        """Async version"""
        return self._run(query, max_results, **kwargs)

    def _format_results_for_llm(self, results: List[EnhancedSearchResult]) -> str:
        """Format results for LLM"""
        if not results:
            return "No search results found."

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = f"""
Result {i}:
Title: {result.title}
URL: {result.url}
Content: {result.content}
Source: {result.source}
"""
            formatted_results.append(formatted_result.strip())

        return "\n\n".join(formatted_results)


class WikipediaSearchTool(BaseTool):
    """Wikipedia search tool for encyclopedic information"""

    name = "wikipedia_search"
    description = "Search Wikipedia for encyclopedic and factual information"

    def _run(self, query: str, max_results: int = 5, **kwargs) -> str:
        """Run Wikipedia search"""
        try:
            import wikipedia

            wikipedia.set_lang("en")
            results = []

            search_results = wikipedia.search(query, results=max_results)

            for i, title in enumerate(search_results):
                try:
                    summary = wikipedia.summary(title, sentences=3)
                    page = wikipedia.page(title)

                    enhanced_result = EnhancedSearchResult(
                        title=title,
                        url=page.url,
                        content=summary,
                        source='wikipedia',
                        timestamp=datetime.now(),
                        relevance_score=1.0 - (i * 0.1),
                        metadata={
                            'page_id': page.pageid,
                            'categories': page.categories[:5] if hasattr(page, 'categories') else []
                        }
                    )
                    results.append(enhanced_result)

                except wikipedia.exceptions.DisambiguationError as e:
                    if e.options:
                        try:
                            page = wikipedia.page(e.options[0])
                            summary = wikipedia.summary(
                                e.options[0], sentences=3)

                            enhanced_result = EnhancedSearchResult(
                                title=e.options[0],
                                url=page.url,
                                content=summary,
                                source='wikipedia',
                                timestamp=datetime.now(),
                                relevance_score=0.8 - (i * 0.1),
                                metadata={'disambiguation': True}
                            )
                            results.append(enhanced_result)
                        except Exception:
                            continue

                except wikipedia.exceptions.PageError:
                    continue

            return self._format_results_for_llm(results)

        except ImportError as exc:
            raise ToolException("Wikipedia library not available") from exc
        except (ValueError, AttributeError, ConnectionError) as e:
            raise ToolException(f"Wikipedia search failed: {str(e)}") from e

    async def _arun(self, query: str, max_results: int = 5, **kwargs) -> str:
        """Async version"""
        return self._run(query, max_results, **kwargs)

    def _format_results_for_llm(self, results: List[EnhancedSearchResult]) -> str:
        """Format results for LLM"""
        if not results:
            return "No Wikipedia articles found."

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = f"""
Wikipedia Result {i}:
Title: {result.title}
URL: {result.url}
Summary: {result.content}
"""
            formatted_results.append(formatted_result.strip())

        return "\n\n".join(formatted_results)


class NewsSearchTool(BaseTool):
    """News search tool for current events"""

    name = "news_search"
    description = "Search for recent news articles and current events"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name=self.name,
            description=self.description
        )
        self.api_key = api_key or os.getenv('SERPER_API_KEY')
        if not self.api_key:
            logger.warning(
                "SERPER_API_KEY not found. News search may be limited.")

        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            })

    def _run(self, query: str, num_results: int = 10, **kwargs) -> str:
        """Run news search"""
        if self.api_key:
            return self._serper_news_search(query, num_results)
        return self._fallback_news_search(query, num_results)

    async def _arun(self, query: str, num_results: int = 10, **kwargs) -> str:
        """Async version"""
        return self._run(query, num_results, **kwargs)

    def _serper_news_search(self, query: str, num_results: int) -> str:
        """Search news using Serper API"""
        try:
            search_params = {
                "q": query,
                "num": min(num_results, 10),
                "gl": "us",
                "hl": "en",
                "tbm": "nws"  # News search
            }

            response = self.session.post(
                "https://google.serper.dev/news",
                json=search_params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            news_results = data.get("news", [])

            if not news_results:
                return "No recent news found for this query."

            formatted_results = []
            for i, result in enumerate(news_results, 1):
                formatted_result = f"""
News Result {i}:
Title: {result.get('title', 'No title')}
Source: {result.get('source', 'Unknown source')}
Date: {result.get('date', 'No date')}
URL: {result.get('link', 'No URL')}
Summary: {result.get('snippet', 'No summary available')}
"""
                formatted_results.append(formatted_result.strip())

            return "\n\n".join(formatted_results)

        except (requests.RequestException, ValueError, KeyError) as e:
            logger.error("Serper news search error: %s", str(e))
            return f"News search failed: {str(e)}"

    def _fallback_news_search(self, query: str, num_results: int) -> str:
        """Fallback news search method"""
        # num_results not used in fallback but required for interface compatibility
        _ = num_results
        return f"News search for '{query}' requires API key. Please configure SERPER_API_KEY."


class ArxivSearchTool(BaseTool):
    """ArXiv search tool for academic papers"""

    name = "arxiv_search"
    description = "Search ArXiv for academic papers and research"

    def _run(self, query: str, max_results: int = 5, **kwargs) -> str:
        """Run ArXiv search"""
        try:
            import arxiv

            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            results = []
            for i, paper in enumerate(client.results(search)):
                result = f"""
ArXiv Paper {i+1}:
Title: {paper.title}
Authors: {', '.join([author.name for author in paper.authors[:3]])}
Published: {paper.published.strftime('%Y-%m-%d')}
URL: {paper.entry_id}
Summary: {paper.summary[:300]}...
Categories: {', '.join(paper.categories)}
"""
                results.append(result.strip())

            return "\n\n".join(results) if results else "No ArXiv papers found."

        except ImportError as exc:
            raise ToolException("ArXiv library not available") from exc
        except (ValueError, AttributeError, ConnectionError) as e:
            raise ToolException(f"ArXiv search failed: {str(e)}") from e

    async def _arun(self, query: str, max_results: int = 5, **kwargs) -> str:
        """Async version"""
        return self._run(query, max_results, **kwargs)


def create_web_search_tools() -> List[BaseTool]:
    """Create a list of web search tools"""
    tools = []

    try:
        # Primary search tool
        serper_tool = SerperWebSearchTool()
        tools.append(serper_tool)
    except ValueError:
        logger.warning("Serper API key not available")

    # Fallback tools
    try:
        ddg_tool = DuckDuckGoSearchTool()
        tools.append(ddg_tool)
    except Exception:
        logger.warning("DuckDuckGo search not available")

    try:
        wiki_tool = WikipediaSearchTool()
        tools.append(wiki_tool)
    except Exception:
        logger.warning("Wikipedia search not available")

    try:
        news_tool = NewsSearchTool()
        tools.append(news_tool)
    except Exception:
        logger.warning("News search not available")

    try:
        arxiv_tool = ArxivSearchTool()
        tools.append(arxiv_tool)
    except Exception:
        logger.warning("ArXiv search not available")

    if not tools:
        logger.error("No search tools available")

    return tools
