"""
LLM Interface Definition
Defines the contract that all LLM adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Iterator


class LLMInterface(ABC):
    """Abstract interface for all LLM providers"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate text completion

        Args:
            prompt: The input prompt
            temperature: Randomness level (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional provider-specific parameters

        Returns:
            str: Generated text
        """

    @abstractmethod
    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate completion with system prompt

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User input prompt
            temperature: Randomness level
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            str: Generated text
        """

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_format: Dict[str, Any],
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured output (JSON mode)

        Args:
            prompt: Input prompt
            response_format: Expected JSON structure
            temperature: Randomness level
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Structured response
        """

    @abstractmethod
    def stream_generate(
        self, prompt: str, temperature: float = 0.7, **kwargs
    ) -> Iterator[str]:
        """
        Stream text generation

        Args:
            prompt: Input prompt
            temperature: Randomness level
            **kwargs: Additional parameters

        Yields:
            str: Text chunks
        """

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model

        Returns:
            Dict[str, Any]: Model information
        """

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the provider is available and healthy

        Returns:
            bool: True if available
        """

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        Args:
            text: Input text

        Returns:
            int: Estimated token count
        """

    @abstractmethod
    def get_context_window(self) -> int:
        """
        Get maximum context window size

        Returns:
            int: Context window size in tokens
        """
