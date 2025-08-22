"""
Enhanced summarizer tool for LangChain Deep Agents
"""

import logging
import re
from collections import Counter
from typing import Dict, Any
from langchain.tools import BaseTool
from langchain.tools.base import ToolException

logger = logging.getLogger(__name__)


class TextSummarizerTool(BaseTool):
    """Tool for summarizing text content"""

    name = "text_summarizer"
    description = "Summarize text content to extract key information"

    def __init__(self):
        super().__init__(
            name=self.name,
            description=self.description
        )

    def _run(self, text: str, max_length: int = 500, summary_type: str = "extractive") -> str:
        """Summarize text content"""
        try:
            if summary_type == "extractive":
                return self._extractive_summary(text, max_length)
            return self._simple_summary(text, max_length)

        except (ValueError, AttributeError, TypeError) as e:
            logger.error("Text summarization error: %s", str(e))
            raise ToolException(f"Summarization failed: {str(e)}") from e

    async def _arun(self, text: str, max_length: int = 500,
                    summary_type: str = "extractive") -> str:
        """Async version"""
        return self._run(text, max_length, summary_type)

    def _extractive_summary(self, text: str, max_length: int) -> str:
        """Create extractive summary by selecting key sentences"""
        if len(text) <= max_length:
            return text

        # Simple sentence extraction based on position and length
        sentences = text.split('. ')
        if not sentences:
            return text[:max_length] + "..."

        # Score sentences
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Position score (earlier sentences get higher scores)
            position_score = 1.0 - (i / len(sentences)) * 0.5

            # Length score (prefer moderate length sentences)
            length_score = min(len(sentence.split()) / 20, 1.0)

            # Keyword score (simple implementation)
            keyword_score = self._calculate_keyword_score(sentence)

            total_score = position_score * 0.4 + length_score * 0.3 + keyword_score * 0.3
            scored_sentences.append((sentence, total_score))

        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        summary = ""
        for sentence, _ in scored_sentences:
            if len(summary) + len(sentence) + 2 <= max_length:
                summary += sentence + ". "
            else:
                break

        return summary.strip() or text[:max_length] + "..."

    def _simple_summary(self, text: str, max_length: int) -> str:
        """Simple truncation summary"""
        if len(text) <= max_length:
            return text

        # Find the last complete sentence within the limit
        truncated = text[:max_length]
        last_period = truncated.rfind('.')

        if last_period > max_length * 0.7:  # If we found a period reasonably close to the end
            return truncated[:last_period + 1]
        return truncated + "..."

    def _calculate_keyword_score(self, sentence: str) -> float:
        """Calculate keyword score for sentence importance"""
        # Simple keyword scoring based on common important words
        important_words = [
            'important', 'significant', 'key', 'main', 'primary', 'essential',
            'critical', 'major', 'notable', 'substantial', 'crucial', 'vital',
            'research', 'study', 'analysis', 'findings', 'results', 'conclusion'
        ]

        sentence_lower = sentence.lower()
        score = 0.0

        for word in important_words:
            if word in sentence_lower:
                score += 0.1

        return min(score, 1.0)


class KeywordExtractorTool(BaseTool):
    """Tool for extracting keywords from text"""

    name = "keyword_extractor"
    description = "Extract important keywords and phrases from text"

    def __init__(self):
        super().__init__(
            name=self.name,
            description=self.description
        )

    def _run(self, text: str, max_keywords: int = 10) -> str:
        """Extract keywords from text"""
        try:
            keywords = self._extract_keywords(text, max_keywords)
            return ", ".join(keywords)

        except (ValueError, AttributeError, TypeError) as e:
            logger.error("Keyword extraction error: %s", str(e))
            raise ToolException(f"Keyword extraction failed: {str(e)}") from e

    async def _arun(self, text: str, max_keywords: int = 10) -> str:
        """Async version"""
        return self._run(text, max_keywords)

    def _extract_keywords(self, text: str, max_keywords: int) -> list:
        """Extract keywords using simple frequency analysis"""
        # Clean text and extract words
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text_clean.split()

        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its',
            'our', 'their'
        }

        # Filter words
        filtered_words = [
            word for word in words
            if len(word) > 3 and word not in stop_words
        ]

        # Count frequency
        word_freq = Counter(filtered_words)

        # Get top keywords
        top_keywords = [word for word,
                        _ in word_freq.most_common(max_keywords)]

        return top_keywords


class ContentAnalyzerTool(BaseTool):
    """Tool for analyzing content structure and quality"""

    name = "content_analyzer"
    description = "Analyze content structure, readability, and quality metrics"

    def __init__(self):
        super().__init__(
            name=self.name,
            description=self.description
        )

    def _run(self, text: str) -> str:
        """Analyze content"""
        try:
            analysis = self._analyze_content(text)
            return self._format_analysis(analysis)

        except (ValueError, AttributeError, TypeError) as e:
            logger.error("Content analysis error: %s", str(e))
            raise ToolException(f"Content analysis failed: {str(e)}") from e

    async def _arun(self, text: str) -> str:
        """Async version"""
        return self._run(text)

    def _analyze_content(self, text: str) -> Dict[str, Any]:
        """Perform content analysis"""
        # Basic metrics
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(re.split(r'[.!?]+', text))
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])

        # Calculate averages
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        avg_chars_per_word = char_count / word_count if word_count > 0 else 0

        # Reading level (simplified)
        reading_level = self._calculate_reading_level(
            avg_words_per_sentence, avg_chars_per_word)

        # Structure analysis
        has_headers = bool(re.search(r'^#{1,6}\s', text, re.MULTILINE))
        has_lists = bool(re.search(r'^\s*[-*+]\s', text, re.MULTILINE))
        has_links = bool(re.search(r'https?://', text))

        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'avg_words_per_sentence': round(avg_words_per_sentence, 1),
            'avg_chars_per_word': round(avg_chars_per_word, 1),
            'reading_level': reading_level,
            'has_structure': {
                'headers': has_headers,
                'lists': has_lists,
                'links': has_links
            }
        }

    def _calculate_reading_level(self, avg_words_per_sentence: float,
                                 avg_chars_per_word: float) -> str:
        """Calculate reading level (simplified)"""
        if avg_words_per_sentence > 20 or avg_chars_per_word > 6:
            return "Advanced"
        if avg_words_per_sentence > 15 or avg_chars_per_word > 5:
            return "Intermediate"
        return "Basic"

    def _format_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format analysis results"""
        return f"""
Content Analysis:
- Word Count: {analysis['word_count']}
- Character Count: {analysis['char_count']}
- Sentences: {analysis['sentence_count']}
- Paragraphs: {analysis['paragraph_count']}
- Average Words per Sentence: {analysis['avg_words_per_sentence']}
- Average Characters per Word: {analysis['avg_chars_per_word']}
- Reading Level: {analysis['reading_level']}
- Structure Features:
  - Headers: {'Yes' if analysis['has_structure']['headers'] else 'No'}
  - Lists: {'Yes' if analysis['has_structure']['lists'] else 'No'}
  - Links: {'Yes' if analysis['has_structure']['links'] else 'No'}
"""


def create_summarizer_tools() -> list:
    """Create summarizer and analysis tools"""
    return [
        TextSummarizerTool(),
        KeywordExtractorTool(),
        ContentAnalyzerTool()
    ]
