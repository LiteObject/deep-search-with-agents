"""
Base Adapter Implementation
Provides common functionality for all LLM adapters.
"""

import logging
import os
import sys
from abc import ABC
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# pylint: disable=import-error,wrong-import-position
# Import from the interfaces module (after path modification)
from ..interfaces.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class BaseLLMAdapter(LLMInterface, ABC):
    """Base adapter with common functionality for all LLM providers"""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        retry_count: int = 3,
        **kwargs
    ):
        """
        Initialize base adapter

        Args:
            model_name: Name of the model to use
            api_key: API key for authentication (if required)
            base_url: Base URL for the provider
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.retry_count = retry_count
        self.additional_params = kwargs

        # Common attributes
        self._context_window = 4096  # Default, override in subclasses
        self._tokenizer = None
        self._available = None  # Cache availability check

        logger.info(
            "Initialized %s with model: %s", self.__class__.__name__, model_name
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using simple heuristic if no tokenizer available"""
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text))
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning("Tokenizer failed, using fallback: %s", e)

        # Simple estimation: ~4 characters per token for most models
        return max(1, len(text) // 4)

    def get_context_window(self) -> int:
        """Get context window size"""
        return self._context_window

    def health_check(self) -> bool:
        """
        Check if provider is available (cached result)

        Returns:
            bool: True if provider is healthy
        """
        if self._available is None:
            self._available = self._perform_health_check()
        return self._available

    def _perform_health_check(self) -> bool:
        """
        Perform actual health check (to be implemented by subclasses)

        Returns:
            bool: True if provider is healthy
        """
        return True

    def _handle_error(self, error: Exception, context: str) -> None:
        """
        Common error handling

        Args:
            error: The exception that occurred
            context: Context where the error occurred
        """
        logger.error(
            "Error in %s - %s: %s", self.__class__.__name__, context, str(error)
        )

    def _validate_response(self, response: Any) -> bool:
        """
        Validate response format

        Args:
            response: Response to validate

        Returns:
            bool: True if response is valid
        """
        return response is not None and bool(str(response).strip())

    def _format_messages(
        self, system_prompt: Optional[str], user_prompt: str
    ) -> List[Dict[str, str]]:
        """
        Format messages for chat-based models

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User input prompt

        Returns:
            List[Dict[str, str]]: Formatted messages
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _extract_text_from_response(self, response: Any) -> str:
        """
        Extract text from provider response

        Args:
            response: Provider response

        Returns:
            str: Extracted text
        """
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict):
            # Common response formats
            for key in ["text", "content", "message", "response", "output"]:
                if key in response:
                    return str(response[key]).strip()
        return str(response).strip()

    def _apply_retry_logic(self, func, *args, **kwargs):
        """
        Apply retry logic to function calls

        Args:
            func: Function to retry
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        last_error = None

        for attempt in range(self.retry_count):
            try:
                return func(*args, **kwargs)
            except (ConnectionError, TimeoutError, ValueError) as e:
                last_error = e
                if attempt < self.retry_count - 1:
                    logger.warning(
                        "Attempt %d failed, retrying: %s", attempt + 1, str(e)
                    )
                else:
                    logger.error("All %d attempts failed: %s", self.retry_count, str(e))

        # If we get here, all attempts failed
        if last_error:
            raise last_error

        # This should not happen but ensures consistent return behavior
        raise RuntimeError("No attempts were made")
