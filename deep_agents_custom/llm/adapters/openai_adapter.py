"""
OpenAI Adapter Implementation
Adapter for OpenAI GPT models.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Iterator

from .base_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")

# Try to import tiktoken for accurate token counting
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI models"""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs,
    ):
        """Initialize OpenAI adapter"""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            **kwargs,
        )

        self.organization = organization
        self.client = None

        # Set context windows for different models
        self._context_windows = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }

        self._context_window = self._context_windows.get(model_name, 4096)
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            if openai and self.api_key:
                self.client = openai.OpenAI(
                    api_key=self.api_key, organization=self.organization
                )
        except (AttributeError, ValueError, TypeError, OSError) as e:
            logger.error("Failed to initialize OpenAI client: %s", e)
            self.client = None

    def _perform_health_check(self) -> bool:
        """Check if OpenAI API is available"""
        if not self.client or not self.api_key:
            return False

        try:
            # Simple test call
            self.client.models.list()
            return True
        except (AttributeError, ValueError, ConnectionError, OSError) as e:
            logger.warning("OpenAI health check failed: %s", e)
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Generate completion using OpenAI"""
        if not self.client:
            return ""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )

            return response.choices[0].message.content or ""

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
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content or ""

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
        """Generate structured output using JSON mode"""
        if not self.client:
            return {}

        try:
            # Add JSON instruction
            json_prompt = f"{prompt}\n\nRespond with valid JSON only."

            # Use JSON mode if available
            extra_params = {}
            if "gpt-3.5-turbo" in self.model_name or "gpt-4" in self.model_name:
                extra_params["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": json_prompt}],
                temperature=temperature,
                **extra_params,
            )

            content = response.choices[0].message.content or "{}"

            return json.loads(content)

        except (
            AttributeError,
            ValueError,
            ConnectionError,
            OSError,
            TypeError,
            json.JSONDecodeError,
        ) as e:
            self._handle_error(e, "generate_structured")
            return {"error": str(e)}

    def stream_generate(
        self, prompt: str, temperature: float = 0.7, **kwargs
    ) -> Iterator[str]:
        """Stream generation"""
        if not self.client:
            return

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except (AttributeError, ValueError, ConnectionError, OSError, TypeError) as e:
            self._handle_error(e, "stream_generate")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.model_name,
            "provider": "openai",
            "context_window": self._context_window,
            "available": self.health_check(),
            "features": ["cloud", "high_quality", "rate_limited", "paid"],
            "supports_functions": True,
            "supports_json_mode": True,
            "supports_streaming": True,
            "api_key_configured": bool(self.api_key),
        }

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using tiktoken if available"""
        try:
            if TIKTOKEN_AVAILABLE and tiktoken:
                encoding = tiktoken.encoding_for_model(self.model_name)
                return len(encoding.encode(text))
        except (AttributeError, ValueError, TypeError, KeyError) as e:
            logger.debug("Tiktoken estimation failed: %s", e)

        return super().estimate_tokens(text)
