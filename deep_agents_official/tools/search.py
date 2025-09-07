"""
This module provides tools for the DeepAgents implementation.
"""

import os
from langchain_core.tools import tool
from tavily import TavilyClient


@tool
def web_search(query: str) -> str:
    """
    Search the web for information on a given query using the Tavily search engine.

    Args:
        query: The search query.

    Returns:
        A string containing the search results.
    """
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return "Error: TAVILY_API_KEY environment variable not set."

        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(query=query, search_depth="advanced")

        # Check if response is valid and contains results
        if not response or not isinstance(response, dict):
            return "No results found or invalid response from search API."

        # Format the results into a readable string
        result_string = ""
        for result in response.get("results", []):
            if isinstance(result, dict):
                title = result.get("title", "No Title")
                url = result.get("url", "No URL")
                content = result.get("content", "No Content")
                result_string += f"Title: {title}\n"
                result_string += f"URL: {url}\n"
                result_string += f"Content: {content}\n\n"

        return result_string if result_string else "No results found."
    except (KeyError, AttributeError, TypeError, ValueError, OSError) as e:
        return f"An error occurred during web search: {e}"
