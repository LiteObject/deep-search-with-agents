"""
Utilities package - Contains helper functions and utilities
"""

from .helpers import (
    clean_text,
    extract_domain,
    truncate_text,
    calculate_text_similarity,
    generate_content_hash,
    extract_keywords,
    format_search_time,
    validate_url,
    create_search_summary_dict
)
from .logger import setup_logging, get_logger

__all__ = [
    'clean_text',
    'extract_domain',
    'truncate_text',
    'calculate_text_similarity',
    'generate_content_hash',
    'extract_keywords',
    'format_search_time',
    'validate_url',
    'create_search_summary_dict',
    'setup_logging',
    'get_logger'
]
