"""
Search tools for the Official DeepAgents implementation.
These tools integrate with the virtual file system and planning tools.
"""

import json
import logging
from typing import List

import requests
from langchain.tools import Tool

logger = logging.getLogger(__name__)


class InternetSearchTool:
    """Internet search tool compatible with official DeepAgents"""

    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        # Try to import DuckDuckGo search functionality
        try:
            # pylint: disable=import-outside-toplevel
            from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
            self.search_wrapper = DuckDuckGoSearchAPIWrapper()
        except ImportError:
            logger.warning("DuckDuckGo search not available, using fallback")
            self.search_wrapper = None

    def search(self, query: str) -> str:
        """Perform internet search and return formatted results"""
        if self.search_wrapper is None:
            return (f"Search not available - DuckDuckGo search dependencies "
                    f"missing for query: '{query}'")

        try:
            results = self.search_wrapper.run(query)
            return f"Search results for '{query}':\n{results}"
        except (ConnectionError, requests.RequestException, ValueError, AttributeError) as e:
            logger.error("Internet search failed: %s", e)
            return f"Search failed: {str(e)}"


class AcademicSearchTool:
    """Academic search tool for research papers and scholarly content"""

    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(self, query: str, limit: int = 5) -> str:
        """Search for academic papers"""
        try:
            params = {
                'query': query,
                'limit': limit,
                'fields': 'title,authors,abstract,url,year,citationCount'
            }

            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            papers = data.get('data', [])

            if not papers:
                return f"No academic papers found for query: {query}"

            results = [f"Academic search results for '{query}':"]
            for paper in papers:
                title = paper.get('title', 'No title')
                authors = [author.get('name', '')
                           for author in paper.get('authors', [])]
                abstract = paper.get('abstract', 'No abstract available')[
                    :300] + '...'
                year = paper.get('year', 'Unknown year')
                citations = paper.get('citationCount', 0)
                url = paper.get('url', '')

                results.append(f"\nTitle: {title}")
                results.append(f"Authors: {', '.join(authors)}")
                results.append(f"Year: {year} | Citations: {citations}")
                results.append(f"Abstract: {abstract}")
                if url:
                    results.append(f"URL: {url}")
                results.append("-" * 50)

            return '\n'.join(results)

        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Academic search failed: %s", e)
            return f"Academic search failed: {str(e)}"


class WebContentTool:
    """Tool to fetch and summarize web page content"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_content(self, url: str) -> str:
        """Fetch and return web page content"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # pylint: disable=import-outside-toplevel
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text content
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip()
                      for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            # Limit content length
            if len(text) > 5000:
                text = text[:5000] + "... [Content truncated]"

            return f"Content from {url}:\n{text}"

        except (requests.RequestException, ImportError, AttributeError, ValueError) as e:
            logger.error("Failed to fetch content from %s: %s", url, e)
            return f"Failed to fetch content from {url}: {str(e)}"


def create_search_tools() -> List[Tool]:
    """Create search tools for DeepAgents"""

    internet_search = InternetSearchTool()
    academic_search = AcademicSearchTool()
    web_content = WebContentTool()

    tools = [
        Tool(
            name="internet_search",
            description=("Search the internet for current information on any topic. "
                         "Use this for general web searches, news, and current events."),
            func=internet_search.search
        ),
        Tool(
            name="academic_search",
            description=("Search for academic papers and scholarly research. "
                         "Use this for in-depth research on scientific, technical, "
                         "or academic topics."),
            func=academic_search.search
        ),
        Tool(
            name="web_content",
            description=("Fetch and analyze content from a specific web page URL. "
                         "Use this to get detailed information from specific websites."),
            func=web_content.fetch_content
        )
    ]

    return tools
