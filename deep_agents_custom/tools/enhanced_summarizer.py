"""
Enhanced AI-powered content summarization tools using the new LLM adapter system.
"""

import logging
import os
import sys
from typing import List, Optional, Dict, Any

# Add the llm module to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# pylint: disable=import-error,wrong-import-position
# Import from local modules (pylint false positives)
from agents.base_agent import SearchResult
from llm import get_llm, get_best_llm, LLMInterface, LLMProvider

logger = logging.getLogger(__name__)


class EnhancedLLMSummarizer:
    """Enhanced LLM-based content summarizer using the new adapter system"""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        llm_adapter: Optional[LLMInterface] = None,
        **config,
    ):
        """
        Initialize enhanced LLM summarizer

        Args:
            provider: Specific LLM provider to use (e.g., "ollama", "openai")
            model: Specific model to use (optional)
            llm_adapter: Pre-configured LLM adapter (optional)
            **config: Additional configuration for the provider
        """
        if llm_adapter:
            # Use provided adapter
            self.llm = llm_adapter
        elif provider:
            # Create adapter for specific provider
            self.llm = get_llm(provider, model, **config)
        else:
            # Use best available adapter
            self.llm = get_best_llm(**config)

        self.available = self.llm is not None

        if self.available and self.llm:
            model_info = self.llm.get_model_info()
            logger.info(
                "Enhanced LLM Summarizer initialized with %s model: %s",
                model_info["provider"],
                model_info["name"],
            )
        else:
            logger.warning("No LLM available. Summarizer will use fallback methods.")

    def summarize_results(
        self,
        results: List[SearchResult],
        query: str,
        max_length: int = 2000,
        style: str = "comprehensive",
    ) -> str:
        """
        Summarize search results using available LLM

        Args:
            results: List of search results
            query: Original search query
            max_length: Maximum summary length in characters
            style: Summary style ("comprehensive", "brief", "technical", "casual")

        Returns:
            str: Generated summary
        """
        if not self.available or not results or not self.llm:
            return self._fallback_summarize(results, query)

        try:
            # Prepare content for summarization
            content_chunks = self._prepare_content(results)

            # Create summarization prompt
            prompt = self._create_summary_prompt(
                query=query,
                content_chunks=content_chunks,
                max_length=max_length,
                style=style,
            )

            # Generate summary using LLM
            summary = self.llm.generate_with_system(
                system_prompt=self._get_system_prompt(style),
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=min(max_length // 2, 1000),  # Rough estimate
            )

            return (
                summary.strip() if summary else self._fallback_summarize(results, query)
            )

        except (AttributeError, TypeError, ValueError, OSError, ConnectionError) as e:
            logger.error("Failed to generate LLM summary: %s", e)
            return self._fallback_summarize(results, query)

    def summarize_with_key_points(
        self, results: List[SearchResult], query: str, max_points: int = 5
    ) -> Dict[str, Any]:
        """
        Generate summary with structured key points

        Args:
            results: List of search results
            query: Original search query
            max_points: Maximum number of key points

        Returns:
            Dict containing summary and key points
        """
        if not self.available or not results or not self.llm:
            return {
                "summary": self._fallback_summarize(results, query),
                "key_points": self._extract_key_points_fallback(results, max_points),
                "method": "fallback",
            }

        try:
            # Prepare content
            content_chunks = self._prepare_content(results)

            # Create structured prompt
            prompt = f"""
Based on the search query "{query}" and the following content, provide a structured analysis:

Content:
{chr(10).join(content_chunks[:5])}

Please provide:
1. A comprehensive summary (200-300 words)
2. {max_points} key points or insights
3. Any important trends or patterns

Format as JSON with keys: "summary", "key_points", "trends"
"""

            # Generate structured response
            response_format = {
                "summary": "string",
                "key_points": ["string"],
                "trends": ["string"],
            }

            structured_response = self.llm.generate_structured(
                prompt=prompt, response_format=response_format, temperature=0.4
            )

            # Ensure we have the required fields
            return {
                "summary": structured_response.get(
                    "summary", self._fallback_summarize(results, query)
                ),
                "key_points": structured_response.get("key_points", [])[:max_points],
                "trends": structured_response.get("trends", []),
                "method": "llm_structured",
            }

        except (AttributeError, TypeError, ValueError, OSError, ConnectionError) as e:
            logger.error("Failed to generate structured summary: %s", e)
            return {
                "summary": self._fallback_summarize(results, query),
                "key_points": self._extract_key_points_fallback(results, max_points),
                "method": "fallback",
            }

    def _prepare_content(self, results: List[SearchResult]) -> List[str]:
        """Prepare content chunks from search results"""
        chunks = []
        for result in results[:10]:  # Limit to top 10 results
            # Create a content chunk with title and content
            chunk = f"Title: {result.title}\n"
            if hasattr(result, "content") and result.content:
                # Truncate content to reasonable length
                content = (
                    result.content[:500] + "..."
                    if len(result.content) > 500
                    else result.content
                )
                chunk += f"Content: {content}\n"
            if hasattr(result, "source"):
                chunk += f"Source: {result.source}\n"
            chunks.append(chunk)
        return chunks

    def _create_summary_prompt(
        self, query: str, content_chunks: List[str], max_length: int, style: str
    ) -> str:
        """Create summarization prompt"""
        style_instructions = {
            "comprehensive": "Provide a detailed, thorough analysis",
            "brief": "Keep it concise and to the point",
            "technical": "Focus on technical details and specifications",
            "casual": "Write in a conversational, easy-to-understand tone",
        }

        instruction = style_instructions.get(style, "Provide a comprehensive analysis")

        return f"""
Please summarize the following search results for the query: "{query}"

{instruction}. The summary should be approximately {max_length} characters.

Search Results:
{chr(10).join(content_chunks[:8])}

Please provide a well-structured summary that captures the key information and insights from these results.
"""

    def _get_system_prompt(self, style: str) -> str:
        """Get system prompt based on style"""
        base_prompt = (
            "You are an expert research analyst who creates clear, accurate "
            "summaries of search results."
        )

        style_additions = {
            "comprehensive": " Focus on providing thorough coverage of all important aspects.",
            "brief": " Focus on brevity while maintaining accuracy.",
            "technical": " Focus on technical accuracy and detailed specifications.",
            "casual": " Write in a friendly, accessible tone for general audiences.",
        }

        return base_prompt + style_additions.get(style, "")

    def _fallback_summarize(self, results: List[SearchResult], query: str) -> str:
        """Fallback summarization when LLM is not available"""
        if not results:
            return f"No results found for query: {query}"

        # Simple extractive summary
        summary_parts = []
        summary_parts.append(f"Found {len(results)} results for '{query}'.")

        # Add top result information
        if results:
            top_result = results[0]
            summary_parts.append(f"Top result: {top_result.title}")
            if hasattr(top_result, "content") and top_result.content:
                # Extract first sentence or two
                content = (
                    top_result.content[:200] + "..."
                    if len(top_result.content) > 200
                    else top_result.content
                )
                summary_parts.append(content)

        # Add source diversity information
        sources = set()
        for result in results[:5]:
            if hasattr(result, "source"):
                sources.add(result.source)

        if sources:
            summary_parts.append(f"Sources include: {', '.join(list(sources)[:3])}")

        return " ".join(summary_parts)

    def _extract_key_points_fallback(
        self, results: List[SearchResult], max_points: int
    ) -> List[str]:
        """Extract key points using simple heuristics"""
        points = []

        for result in results[:max_points]:
            if hasattr(result, "title"):
                points.append(result.title)

        return points[:max_points]

    def get_summarizer_info(self) -> Dict[str, Any]:
        """Get information about the summarizer"""
        info = {"available": self.available, "type": "enhanced_llm"}

        if self.available and self.llm:
            model_info = self.llm.get_model_info()
            info.update(
                {
                    "provider": model_info.get("provider", "unknown"),
                    "model": model_info.get("name", "unknown"),
                    "features": model_info.get("features", []),
                    "context_window": model_info.get("context_window", 0),
                }
            )

        return info


# Maintain backward compatibility
class LLMSummarizer(EnhancedLLMSummarizer):
    """Backward compatible LLM summarizer"""

    def __init__(self, provider=None, model=None, base_url=None, **kwargs):
        """Initialize with backward compatible parameters"""
        # Map old parameters to new system
        if provider and hasattr(LLMProvider, provider.upper()):
            provider_str = provider.lower()
        else:
            provider_str = None

        # Handle base_url for Ollama
        config = kwargs
        if base_url:
            config["base_url"] = base_url

        super().__init__(provider=provider_str, model=model, **config)


class SimpleSummarizer:
    """Simple rule-based summarizer as fallback"""

    def summarize_results(self, results: List[SearchResult], query: str) -> str:
        """Simple rule-based summarization"""
        if not results:
            return f"No results found for query: {query}"

        summary = f"Found {len(results)} results for '{query}'. "

        # Add top few results
        for i, result in enumerate(results[:3], 1):
            summary += f"{i}. {result.title}. "
            if hasattr(result, "content") and result.content:
                # Get first sentence
                first_sentence = result.content.split(".")[0]
                if len(first_sentence) < 150:
                    summary += first_sentence + ". "

        return summary
