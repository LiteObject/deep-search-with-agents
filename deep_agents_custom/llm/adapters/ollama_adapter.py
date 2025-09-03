"""
Ollama Adapter Implementation
Adapter for local Ollama models.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Iterator

import requests

from .base_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)


class OllamaAdapter(BaseLLMAdapter):
    """Adapter for Ollama local models"""

    def __init__(
        self,
        model_name: str = "gpt-oss:latest",
        base_url: str = "http://localhost:11434",
        **kwargs,
    ):
        """
        Initialize Ollama adapter

        Args:
            model_name: Name of the Ollama model
            base_url: Ollama server URL
            **kwargs: Additional parameters
        """
        super().__init__(model_name=model_name, base_url=base_url.rstrip("/"), **kwargs)

        # Set context windows for common models
        self._context_windows = {
            "llama2": 4096,
            "llama3": 8192,
            "mistral": 8192,
            "codellama": 16384,
            "gpt-oss": 4096,
            "phi": 2048,
            "neural-chat": 4096,
        }

        # Determine context window based on model name
        for model_pattern, window_size in self._context_windows.items():
            if model_pattern in model_name.lower():
                self._context_window = window_size
                break

    def _perform_health_check(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False

            # Check if model is available
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]

            # Check if our model exists
            model_available = any(self.model_name in name for name in model_names)

            if model_available:
                logger.info(
                    "Ollama model %s is available at %s", self.model_name, self.base_url
                )
            else:
                logger.warning(
                    "Model %s not found in Ollama. Available: %s",
                    self.model_name,
                    model_names[:5],
                )

            return model_available

        except requests.exceptions.RequestException as e:
            logger.warning("Could not connect to Ollama at %s: %s", self.base_url, e)
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Generate completion using Ollama"""
        if not self.health_check():
            raise RuntimeError(f"Ollama is not available at {self.base_url}")

        try:
            # Prepare options
            options = {
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.9),
                "num_ctx": self._context_window,
                **kwargs.get("options", {}),
            }

            if max_tokens:
                options["num_predict"] = max_tokens

            if stop:
                options["stop"] = stop

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": options,
            }

            response = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "").strip()

        except requests.exceptions.RequestException as e:
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
        """Generate with system prompt using chat format"""
        if not self.health_check():
            raise RuntimeError(f"Ollama is not available at {self.base_url}")

        try:
            messages = self._format_messages(system_prompt, user_prompt)

            options = {
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.9),
                "num_ctx": self._context_window,
                **kwargs.get("options", {}),
            }

            if max_tokens:
                options["num_predict"] = max_tokens

            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": options,
            }

            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            return result.get("message", {}).get("content", "").strip()

        except requests.exceptions.RequestException as e:
            self._handle_error(e, "generate_with_system")
            return ""

    def generate_structured(
        self,
        prompt: str,
        response_format: Dict[str, Any],
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate structured output (JSON mode)"""
        # Add JSON instruction to prompt
        json_instruction = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this format: "
            f"{json.dumps(response_format, indent=2)}\n"
            f"Important: Return only valid JSON, no additional text."
        )

        response = self.generate(
            prompt=json_instruction, temperature=temperature, **kwargs
        )

        try:
            # Try to parse as JSON
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON response: %s", e)
            # Return structured fallback
            return {
                "text": response,
                "format": "text",
                "error": f"JSON parsing failed: {str(e)}",
            }

    def stream_generate(
        self, prompt: str, temperature: float = 0.7, **kwargs
    ) -> Iterator[str]:
        """Stream generation"""
        if not self.health_check():
            raise RuntimeError(f"Ollama is not available at {self.base_url}")

        try:
            options = {
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.9),
                "num_ctx": self._context_window,
                **kwargs.get("options", {}),
            }

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": options,
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data and data["response"]:
                            yield data["response"]
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.RequestException as e:
            self._handle_error(e, "stream_generate")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()

            models = response.json().get("models", [])
            model_info = next(
                (m for m in models if self.model_name in m.get("name", "")), None
            )

            return {
                "name": self.model_name,
                "provider": "ollama",
                "base_url": self.base_url,
                "context_window": self._context_window,
                "available": model_info is not None,
                "features": ["local", "private", "offline", "customizable"],
                "details": model_info,
                "supports_streaming": True,
                "supports_chat": True,
            }

        except requests.exceptions.RequestException as e:
            logger.warning("Failed to get model info: %s", e)
            return {
                "name": self.model_name,
                "provider": "ollama",
                "base_url": self.base_url,
                "context_window": self._context_window,
                "available": False,
                "error": str(e),
            }
