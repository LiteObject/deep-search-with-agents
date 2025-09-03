"""
LLM Package
Provides unified interface for multiple LLM providers using adapter and factory patterns.
"""

from .factory.llm_factory import LLMFactory, LLMProvider, get_llm, get_best_llm
from .interfaces.llm_interface import LLMInterface

__all__ = ["LLMFactory", "LLMProvider", "LLMInterface", "get_llm", "get_best_llm"]

# Version info
__version__ = "1.0.0"
