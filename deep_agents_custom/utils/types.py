"""
Common data types and structures used across the application.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class SearchResult:
    """Data class for search results"""

    title: str
    url: str
    content: str
    source: str
    timestamp: datetime
    relevance_score: float = 0.0


@dataclass
class SearchSummary:
    """Data class for search summary"""

    query: str
    summary: str
    key_points: List[str]
    sources: List[str]
    total_results: int
    search_time: float
    # Add full results for better display
    results: Optional[List[SearchResult]] = None
    # Add citations for better referencing
    citations: Optional[Dict[str, Dict[str, str]]] = None
    cited_summary: Optional[str] = None
