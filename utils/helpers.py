"""
Helper functions and utilities for the deep search agents application.
"""

import re
from typing import List, Dict, Any
from urllib.parse import urlparse
import hashlib


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.

    Args:
        text: Raw text to clean

    Returns:
        str: Cleaned text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)

    # Remove multiple consecutive punctuation
    text = re.sub(r'[\.]{2,}', '.', text)
    text = re.sub(r'[\!\?]{2,}', '!', text)

    return text.strip()


def extract_domain(url: str) -> str:
    """
    Extract domain from URL.

    Args:
        url: URL to parse

    Returns:
        str: Domain name
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except ValueError:
        return ""


def truncate_text(text: str, max_length: int = 200,
                  add_ellipsis: bool = True) -> str:
    """
    Truncate text to specified length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        add_ellipsis: Whether to add ... at the end

    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]

    # Try to cut at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # If space is reasonably close to end
        truncated = truncated[:last_space]

    if add_ellipsis:
        truncated += "..."

    return truncated


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using simple word overlap.

    Args:
        text1: First text
        text2: Second text

    Returns:
        float: Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0

    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union if union > 0 else 0.0


def generate_content_hash(content: str) -> str:
    """
    Generate hash for content to detect duplicates.

    Args:
        content: Content to hash

    Returns:
        str: Content hash
    """
    # Normalize content for hashing
    normalized = clean_text(content).lower()

    # Generate SHA-256 hash
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis.

    Args:
        text: Text to analyze
        max_keywords: Maximum number of keywords to return

    Returns:
        List[str]: Extracted keywords
    """
    if not text:
        return []

    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
        'its', 'our', 'their'
    }

    # Extract words and count frequency
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_freq = {}

    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]


def format_search_time(seconds: float) -> str:
    """
    Format search time in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.1f}s"


def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL.

    Args:
        url: URL to validate

    Returns:
        bool: True if valid URL
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def create_search_summary_dict(summary) -> Dict[str, Any]:
    """
    Convert SearchSummary object to dictionary for JSON serialization.

    Args:
        summary: SearchSummary object

    Returns:
        Dict[str, Any]: Dictionary representation
    """
    return {
        'query': summary.query,
        'summary': summary.summary,
        'key_points': summary.key_points,
        'sources': summary.sources,
        'total_results': summary.total_results,
        'search_time': summary.search_time,
        'search_time_formatted': format_search_time(summary.search_time)
    }
