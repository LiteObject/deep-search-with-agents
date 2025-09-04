"""
Anthropic Adapter Implementation
Adapter for Anthropic Claude models.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Iterator

from .base_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)

# Try to import Anthropic
try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False
    logger.warning(
        "Anthropic library not available. Install with: pip install anthropic"
    )


class AnthropicAdapter(BaseLLMAdapter):
    """Adapter for Anthropic Claude models"""

    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize Anthropic adapter"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic library not installed. Install with: pip install anthropic"
            )

        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            **kwargs,
        )

        self.client = None

        # Set context windows for different models
        self._context_windows = {
            "claude-3-sonnet-20240229": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-2.1": 200000,
            "claude-2.0": 100000,
            "claude-instant-1.2": 100000,
        }

        self._context_window = self._context_windows.get(model_name, 100000)
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Anthropic client"""
        try:
            if anthropic and self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
        except (AttributeError, ValueError, ImportError) as e:
            logger.error("Failed to initialize Anthropic client: %s", e)
            self.client = None

    def _perform_health_check(self) -> bool:
        """Check if Anthropic API is available"""
        if not self.client or not self.api_key:
            return False

        try:
            # Simple test call
            self.client.messages.create(
                model=self.model_name,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except (AttributeError, ValueError, ConnectionError, OSError) as e:
            logger.warning("Anthropic health check failed: %s", e)
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Generate completion using Anthropic"""
        if not self.client:
            return ""

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or 1000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text if response.content else ""

        except (AttributeError, ValueError, ConnectionError, OSError, TypeError) as e:
            self._handle_error(e, "generate")
            return ""

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate with system prompt"""
        if not self.client:
            return ""

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or 1000,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            return response.content[0].text if response.content else ""

        except (AttributeError, ValueError, ConnectionError, OSError, TypeError) as e:
            self._handle_error(e, "generate_with_system")
            return ""

    def generate_structured(
        self,
        prompt: str,
        response_format: Dict[str, Any],
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate structured output"""
        # Add JSON instruction to prompt
        json_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this format: "
            f"{json.dumps(response_format, indent=2)}"
        )

        response = self.generate(prompt=json_prompt, temperature=temperature, **kwargs)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON response: %s", e)
            return {"error": str(e), "text": response}

    def stream_generate(
        self, prompt: str, temperature: float = 0.7, **kwargs
    ) -> Iterator[str]:
        """Stream generation"""
        if not self.client:
            return

        try:
            stream = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            for event in stream:
                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    yield event.delta.text

        except (AttributeError, ValueError, ConnectionError, OSError, TypeError) as e:
            self._handle_error(e, "stream_generate")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.model_name,
            "provider": "anthropic",
            "context_window": self._context_window,
            "available": self.health_check(),
            "features": ["cloud", "safety_focused", "rate_limited", "paid"],
            "supports_streaming": True,
            "supports_system_prompt": True,
            "api_key_configured": bool(self.api_key),
        }
