"""
AI-powered content summarization tools.
"""

import os
import logging
from typing import List, Optional
from agents.base_agent import SearchResult

logger = logging.getLogger(__name__)


class LLMSummarizer:
    """LLM-based content summarizer using OpenAI"""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = None

        if self.api_key:
            try:
                from openai import OpenAI  # type: ignore  # pylint: disable=import-outside-toplevel
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.error(
                    "OpenAI library not installed. "
                    "Please install: pip install openai"
                )
        else:
            logger.warning(
                "OpenAI API key not found. "
                "LLM summarization will not work."
            )

    def summarize_results(
            self, results: List[SearchResult], query: str) -> str:
        """
        Summarize search results using LLM

        Args:
            results: List of search results
            query: Original search query

        Returns:
            str: Generated summary
        """
        if not self.client:
            return self._fallback_summary(results, query)

        try:
            # Prepare content for summarization
            content = self._prepare_content(results)

            # Create prompt
            prompt = (
                f'Based on the following search results for the query '
                f'"{query}", provide a comprehensive summary:\n\n{content}\n\n'
                f'Please provide:\n'
                f'1. A concise overview of the topic\n'
                f'2. Key findings from the search results\n'
                f'3. Important details and facts\n'
                f'4. Any notable trends or patterns\n\n'
                f'Keep the summary informative but concise '
                f'(around 200-300 words).'
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert "
                     "content summarizer and research analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except (ImportError, AttributeError, KeyError) as e:
            logger.error("LLM summarization error: %s", e)
            return self._fallback_summary(results, query)

    def extract_key_insights(self, results: List[SearchResult],
                             query: str) -> List[str]:
        """
        Extract key insights from search results

        Args:
            results: List of search results
            query: Original search query

        Returns:
            List[str]: Key insights
        """
        if not self.client:
            return self._fallback_insights(results)

        try:
            content = self._prepare_content(results)

            prompt = (
                f'Based on the following search results for "{query}", '
                f'extract 3-5 key insights as a numbered list:\n\n{content}'
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": (
                         "You are a research analyst extracting "
                         "key insights from web search results.")},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )

            # Parse numbered list
            insights_text = response.choices[0].message.content.strip()
            insights = []

            for line in insights_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering/bullets
                    insight = line.split('.', 1)[-1].strip()
                    if insight:
                        insights.append(insight)

            return insights[:5]  # Limit to 5 insights

        except (ImportError, AttributeError, KeyError) as e:
            logger.error("Insight extraction error: %s", e)
            return self._fallback_insights(results)

    def _prepare_content(self, results: List[SearchResult],
                         max_length: int = 3000) -> str:
        """
        Prepare content from search results for summarization

        Args:
            results: List of search results
            max_length: Maximum character length

        Returns:
            str: Formatted content
        """
        content_parts = []
        current_length = 0

        for i, result in enumerate(results):
            part = (
                f"Result {i+1}: {result.title}\n"
                f"Source: {result.source}\n"
                f"URL: {result.url}\n"
                f"Content: {result.content[:500]}...\n\n"
            )

            if current_length + len(part) <= max_length:
                content_parts.append(part)
                current_length += len(part)
            else:
                # Try to fit partial content
                remaining = max_length - current_length
                # Only if meaningful content can fit
                if remaining > 100:
                    content_parts.append(part[:remaining] + "...")
                break

        return "\n".join(content_parts)

    def _fallback_summary(self, results: List[SearchResult],
                          query: str) -> str:
        """
        Fallback summary when LLM is not available

        Args:
            results: List of search results
            query: Original search query

        Returns:
            str: Basic summary
        """
        if not results:
            return f"No results found for query: {query}"

        summary_parts = [
            f"Search Results Summary for: {query}",
            (f"Found {len(results)} results from sources: "
             f"{', '.join(set(r.source for r in results))}"),
            "",
            "Top Results:"
        ]

        for i, result in enumerate(results[:3]):
            summary_parts.extend([
                f"{i+1}. {result.title}",
                f"   Source: {result.source}",
                f"   {result.content[:150]}...",
                ""
            ])

        return "\n".join(summary_parts)

    def _fallback_insights(self, results: List[SearchResult]) -> List[str]:
        """
        Extract basic insights without LLM

        Args:
            results: List of search results

        Returns:
            List[str]: Basic insights
        """
        insights = []

        # Extract first sentences from content as basic insights
        for result in results[:5]:
            if result.content:
                # Get first sentence
                sentences = result.content.split('.')
                if sentences:
                    first_sentence = sentences[0].strip()
                    # Meaningful sentence
                    if len(first_sentence) > 20:
                        insights.append(first_sentence)

        return insights


class SimpleSummarizer:  # pylint: disable=too-few-public-methods
    """Simple rule-based summarizer as fallback"""

    def summarize_results(self, results: List[SearchResult],
                          query: str) -> str:
        """
        Create a simple summary of search results

        Args:
            results: List of search results
            query: Original search query

        Returns:
            str: Simple summary
        """
        if not results:
            return f"No results found for query: {query}"

        # Count sources
        sources = {}
        for result in results:
            sources[result.source] = sources.get(result.source, 0) + 1

        summary_lines = [
            f"Search Summary for: '{query}'",
            f"Total Results: {len(results)}",
            ("Sources: " +
             ', '.join(f'{source} ({count})'
                       for source, count in sources.items())),
            "",
            "Top Results:"
        ]

        # Add top 5 results
        for i, result in enumerate(results[:5]):
            summary_lines.append(f"{i+1}. {result.title}")
            summary_lines.append(f"   Source: {result.source}")
            summary_lines.append(f"   {result.content[:200]}...")
            summary_lines.append("")

        return "\n".join(summary_lines)
