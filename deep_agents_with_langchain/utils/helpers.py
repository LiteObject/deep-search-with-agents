"""
Enhanced utilities for LangChain Deep Agents
"""

import re
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class QueryAnalyzer:  # pylint: disable=too-few-public-methods
    """Advanced query analysis utilities for deep agents"""

    @staticmethod
    def classify_query_complexity(query: str) -> Dict[str, Any]:
        """
        Classify query complexity for routing decisions

        Returns:
            Dict containing complexity metrics and recommendations
        """
        # Basic metrics
        word_count = len(query.split())
        question_words = ['what', 'who', 'when',
                          'where', 'why', 'how', 'which']
        question_count = sum(
            1 for word in question_words if word.lower() in query.lower())

        # Complexity indicators
        comparative_words = ['compare', 'versus', 'vs', 'difference', 'better']
        temporal_words = ['recent', 'latest', 'current', 'trend', 'history']
        analytical_words = ['analyze', 'evaluate',
                            'assess', 'impact', 'effect']
        comprehensive_words = ['comprehensive',
                               'detailed', 'thorough', 'complete']
        causal_words = ['why', 'cause', 'reason', 'because', 'due to']

        complexity_indicators = {
            'multi_part': len(re.findall(r'[.!?;]', query)) > 1,
            'comparative': any(word in query.lower() for word in comparative_words),
            'temporal': any(word in query.lower() for word in temporal_words),
            'analytical': any(word in query.lower() for word in analytical_words),
            'comprehensive': any(word in query.lower() for word in comprehensive_words),
            'causal': any(word in query.lower() for word in causal_words),
            # Acronyms/technical terms
            'technical': bool(re.search(r'\b[A-Z]{2,}\b', query))
        }

        # Calculate complexity score
        complexity_score = (
            min(word_count / 10, 1.0) * 0.2 +
            min(question_count / 3, 1.0) * 0.1 +
            sum(complexity_indicators.values()) /
            len(complexity_indicators) * 0.7
        )

        # Determine complexity level
        if complexity_score >= 0.7:
            complexity_level = "high"
        elif complexity_score >= 0.4:
            complexity_level = "medium"
        else:
            complexity_level = "low"

        return {
            'complexity_score': complexity_score,
            'complexity_level': complexity_level,
            'word_count': word_count,
            'question_count': question_count,
            'indicators': complexity_indicators,
            'recommended_agents': QueryAnalyzer._recommend_agents(query, complexity_indicators),
            'estimated_time': QueryAnalyzer._estimate_processing_time(complexity_score),
            'suggested_approach': QueryAnalyzer._suggest_approach(
                complexity_level, complexity_indicators)
        }

    @staticmethod
    def _recommend_agents(query: str, indicators: Dict[str, bool]) -> List[str]:
        """Recommend agents based on query analysis"""
        recommended = []

        # Research agent indicators
        research_conditions = (indicators.get('analytical') or
                               indicators.get('technical') or
                               'research' in query.lower())
        if research_conditions:
            recommended.append('research')

        # News agent indicators
        news_keywords = ['news', 'recent', 'current', 'latest']
        if indicators.get('temporal') or any(word in query.lower() for word in news_keywords):
            recommended.append('news')

        # General agent as fallback
        if not recommended or indicators.get('comprehensive'):
            recommended.append('general')

        # Reflection agent for complex queries
        if indicators.get('comparative') or indicators.get('analytical'):
            recommended.append('reflection')

        return recommended

    @staticmethod
    def _estimate_processing_time(complexity_score: float) -> int:
        """Estimate processing time in seconds"""
        base_time = 10  # Base processing time
        return int(base_time + (complexity_score * 30))

    @staticmethod
    def _suggest_approach(complexity_level: str, indicators: Dict[str, bool]) -> str:
        """Suggest processing approach"""
        if complexity_level == "high":
            if indicators.get('multi_part'):
                return "hierarchical"
            if indicators.get('comparative'):
                return "parallel"
            return "sequential"
        if complexity_level == "medium":
            return "parallel" if indicators.get('comparative') else "sequential"
        return "direct"


class ContentProcessor:
    """Utilities for processing and enhancing content"""

    @staticmethod
    def extract_key_information(content: str) -> Dict[str, Any]:
        """Extract key information from content"""
        # Extract URLs
        url_pattern = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       r'[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        urls = re.findall(url_pattern, content)

        # Extract dates
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\w+ \d{1,2}, \d{4}\b'  # Month DD, YYYY
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, content))

        # Extract numbers and statistics
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', content)

        # Extract potential entities (capitalized words)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)

        # Calculate readability metrics
        sentences = len(re.split(r'[.!?]+', content))
        words = len(content.split())
        avg_words_per_sentence = words / sentences if sentences > 0 else 0

        return {
            'urls': urls,
            'dates': dates,
            'numbers': numbers,
            'entities': entities[:20],  # Limit entities
            'word_count': words,
            'sentence_count': sentences,
            'avg_words_per_sentence': avg_words_per_sentence,
            'readability_score': ContentProcessor._calculate_readability(content)
        }

    @staticmethod
    def _calculate_readability(content: str) -> float:
        """Calculate a simple readability score"""
        sentences = len(re.split(r'[.!?]+', content))
        words = len(content.split())
        syllables = ContentProcessor._count_syllables(content)

        if sentences == 0 or words == 0:
            return 0.0

        # Simplified Flesch Reading Ease score
        score = 206.835 - (1.015 * (words / sentences)) - \
            (84.6 * (syllables / words))
        return max(0.0, min(100.0, score))

    @staticmethod
    def _count_syllables(text: str) -> int:
        """Count syllables in text (simplified)"""
        vowels = 'aeiouyAEIOUY'
        syllable_count = 0
        prev_char_was_vowel = False

        for char in text:
            if char in vowels:
                if not prev_char_was_vowel:
                    syllable_count += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False

        return max(1, syllable_count)  # At least one syllable per word

    @staticmethod
    def summarize_content(content: str, max_length: int = 500) -> str:
        """Create a summary of content"""
        if len(content) <= max_length:
            return content

        # Simple extractive summarization
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return content[:max_length] + "..."

        # Score sentences by position and length
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Earlier sentences score higher
            position_score = 1.0 - (i / len(sentences)) * 0.5
            length_score = min(len(sentence.split()) / 20,
                               1.0)  # Prefer moderate length
            total_score = position_score + length_score
            scored_sentences.append((sentence, total_score))

        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        summary = ""
        for sentence, _ in scored_sentences:
            if len(summary) + len(sentence) + 2 <= max_length:
                summary += sentence + ". "
            else:
                break

        return summary.strip()


class ValidationUtils:
    """Utilities for validating content and results"""

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate if URL is properly formed"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_date_format(date_str: str) -> bool:
        """Validate date format"""
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%B %d, %Y',
            '%b %d, %Y'
        ]

        for fmt in date_formats:
            try:
                datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue
        return False

    @staticmethod
    def check_content_quality(content: str) -> Dict[str, Any]:
        """Check content quality metrics"""
        word_count = len(content.split())
        char_count = len(content)

        # Quality indicators
        has_structure = bool(re.search(r'[.!?]', content))
        has_details = word_count >= 50
        not_too_repetitive = ValidationUtils._check_repetition(content) < 0.3
        proper_formatting = ValidationUtils._check_formatting(content)

        quality_score = sum([
            has_structure * 0.25,
            has_details * 0.25,
            not_too_repetitive * 0.25,
            proper_formatting * 0.25
        ])

        return {
            'quality_score': quality_score,
            'word_count': word_count,
            'char_count': char_count,
            'has_structure': has_structure,
            'has_details': has_details,
            'not_too_repetitive': not_too_repetitive,
            'proper_formatting': proper_formatting,
            'issues': ValidationUtils._identify_quality_issues(content)
        }

    @staticmethod
    def _check_repetition(content: str) -> float:
        """Check for repetitive content"""
        words = content.lower().split()
        if len(words) < 10:
            return 0.0

        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Calculate repetition ratio
        total_repetitions = sum(max(0, freq - 1)
                                for freq in word_freq.values())
        return total_repetitions / len(words)

    @staticmethod
    def _check_formatting(content: str) -> bool:
        """Check if content has proper formatting"""
        # Check for basic formatting elements
        has_sentences = bool(re.search(r'[.!?]', content))
        proper_capitalization = bool(re.search(r'^[A-Z]', content))

        return has_sentences and proper_capitalization

    @staticmethod
    def _identify_quality_issues(content: str) -> List[str]:
        """Identify specific quality issues"""
        issues = []

        if len(content.split()) < 20:
            issues.append("Content too short")

        if not re.search(r'[.!?]', content):
            issues.append("No sentence structure")

        if ValidationUtils._check_repetition(content) > 0.4:
            issues.append("Highly repetitive content")

        if not re.search(r'^[A-Z]', content):
            issues.append("Improper capitalization")

        return issues


class CacheUtils:
    """Utilities for caching functionality"""

    @staticmethod
    def generate_cache_key(query: str, agent_type: str = "", **kwargs) -> str:
        """Generate a cache key for query results"""
        # Create a string representation of the query and parameters
        key_components = [query, agent_type]

        # Add sorted kwargs to ensure consistent ordering
        for k, v in sorted(kwargs.items()):
            key_components.append(f"{k}:{v}")

        key_string = "|".join(key_components)

        # Generate hash
        return hashlib.md5(key_string.encode()).hexdigest()

    @staticmethod
    def is_cache_valid(timestamp: datetime, ttl_seconds: int = 3600) -> bool:
        """Check if cached result is still valid"""
        expiry_time = timestamp + timedelta(seconds=ttl_seconds)
        return datetime.now() < expiry_time

    @staticmethod
    def create_cache_entry(data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a cache entry with metadata"""
        return {
            'data': data,
            'timestamp': datetime.now(),
            'metadata': metadata or {},
            'access_count': 0
        }


class DataSanitizer:
    """Utilities for sanitizing and cleaning data"""

    @staticmethod
    def sanitize_query(query: str) -> str:
        """Sanitize user query"""
        # Remove excessive whitespace
        query = re.sub(r'\s+', ' ', query.strip())

        # Remove potentially harmful characters
        query = re.sub(r'[<>\"\'`]', '', query)

        # Limit length
        if len(query) > 1000:
            query = query[:1000] + "..."

        return query

    @staticmethod
    def sanitize_content(content: str) -> str:
        """Sanitize content for processing"""
        # Remove or replace problematic characters
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)

        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)

        # Remove excessive punctuation
        content = re.sub(r'([.!?]){3,}', r'\1\1\1', content)

        return content.strip()

    @staticmethod
    def extract_clean_text(html_content: str) -> str:
        """Extract clean text from HTML content"""
        # Simple HTML tag removal (for basic cases)
        text = re.sub(r'<[^>]+>', '', html_content)

        # Decode HTML entities (basic ones)
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")

        return DataSanitizer.sanitize_content(text)


# Helper functions for common operations
def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format timestamp for logging"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely load JSON with fallback"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries safely"""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result
