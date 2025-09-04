"""
LLM Factory Implementation

This module implements the Factory pattern for creating and managing LLM (Large Language Model) adapters.
It provides a unified interface for working with different LLM providers while abstracting away the
complexity of provider-specific initialization and configuration.

Purpose:
- Centralized creation and management of LLM adapters for multiple providers (Ollama, OpenAI, Anthropic)
- Auto-detection of available providers based on environment configuration
- Caching mechanism to reuse adapter instances for improved performance
- Fallback strategies when preferred providers are unavailable
- Environment-based configuration management for API keys and endpoints

How it works:
1. The LLMFactory class maintains a registry of available providers and their corresponding adapter classes
2. When creating an adapter, it auto-detects the best available provider or uses the specified one
3. Configuration is automatically retrieved from environment variables (API keys, endpoints)
4. Adapters are cached to avoid repeated initialization overhead
5. If a provider fails, it automatically falls back to Ollama as the local default
6. All adapters implement the LLMInterface contract, ensuring consistent behavior across providers

Key Components:
- LLMProvider enum: Defines available providers (OLLAMA, OPENAI, ANTHROPIC, AUTO)
- LLMFactory class: Main factory implementation with adapter creation and management
- Provider registry: Maps providers to their adapter classes and default models
- Configuration management: Handles environment-based setup for each provider
- Caching system: Stores created adapters for reuse based on provider, model, and config
"""

import logging
import os
import sys
from enum import Enum
from typing import Any, Dict, Optional, Type, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# pylint: disable=wrong-import-position
# Import adapters and interfaces (after sys.path modification)
from ..adapters.ollama_adapter import OllamaAdapter
from ..adapters.openai_adapter import OpenAIAdapter
from ..adapters.anthropic_adapter import AnthropicAdapter
from ..interfaces.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Available LLM providers"""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AUTO = "auto"


class LLMFactory:
    """Enhanced factory for creating LLM adapters"""

    # Registry of providers and their adapters
    _providers: Dict[LLMProvider, Type[Any]] = {
        LLMProvider.OLLAMA: OllamaAdapter,
        LLMProvider.OPENAI: OpenAIAdapter,
        LLMProvider.ANTHROPIC: AnthropicAdapter,
    }

    # Default models for each provider
    _default_models = {
        LLMProvider.OLLAMA: "gpt-oss:latest",
        LLMProvider.OPENAI: "gpt-3.5-turbo",
        LLMProvider.ANTHROPIC: "claude-3-sonnet-20240229",
    }

    # Cache for created adapters
    _adapter_cache: Dict[str, LLMInterface] = {}

    @classmethod
    def create_adapter(
        cls,
        provider: LLMProvider = LLMProvider.AUTO,
        model_name: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> LLMInterface:
        """
        Create an LLM adapter

        Args:
            provider: The LLM provider to use
            model_name: Model name (uses default if not specified)
            use_cache: Whether to use cached adapters
            **kwargs: Additional arguments for the adapter

        Returns:
            LLMInterface: The created adapter
        """
        # Auto-detect provider if requested
        if provider == LLMProvider.AUTO:
            provider = cls._detect_best_provider()

        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}")

        # Use default model if not specified
        if not model_name:
            model_name = cls._default_models.get(provider)

        # Create cache key
        cache_key = f"{provider.value}:{model_name}:{hash(frozenset(kwargs.items()))}"

        # Return cached adapter if available
        if use_cache and cache_key in cls._adapter_cache:
            logger.debug("Using cached adapter for %s", provider.value)
            return cls._adapter_cache[cache_key]

        # Get adapter class
        adapter_class = cls._providers[provider]

        # Get configuration from environment
        config = cls._get_provider_config(provider)
        config.update(kwargs)

        # Create adapter
        logger.info("Creating %s adapter with model: %s", provider.value, model_name)

        try:
            adapter = adapter_class(model_name=model_name, **config)

            # Cache the adapter
            if use_cache:
                cls._adapter_cache[cache_key] = adapter

            return adapter

        except (ImportError, ConnectionError, ValueError, OSError) as e:
            logger.error("Failed to create adapter for %s: %s", provider.value, e)
            # Try fallback to Ollama if not already trying Ollama
            if provider != LLMProvider.OLLAMA:
                logger.info("Falling back to Ollama")
                return cls.create_adapter(
                    provider=LLMProvider.OLLAMA,
                    model_name="gpt-oss:latest",
                    use_cache=use_cache,
                    **kwargs,
                )
            raise

    @classmethod
    def get_adapter(
        cls,
        provider: LLMProvider = LLMProvider.AUTO,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> LLMInterface:
        """
        Get or create a cached LLM adapter (alias for create_adapter with caching)

        Args:
            provider: The LLM provider to use
            model_name: Model name
            **kwargs: Additional arguments

        Returns:
            LLMInterface: The adapter
        """
        return cls.create_adapter(
            provider=provider, model_name=model_name, use_cache=True, **kwargs
        )

    @classmethod
    def _detect_best_provider(cls) -> LLMProvider:
        """Detect the best available provider based on environment"""
        # Check for API keys in environment
        if os.getenv("OPENAI_API_KEY"):
            logger.info("Auto-detected OpenAI (API key found)")
            return LLMProvider.OPENAI
        if os.getenv("ANTHROPIC_API_KEY"):
            logger.info("Auto-detected Anthropic (API key found)")
            return LLMProvider.ANTHROPIC

        # Default to Ollama for local deployment
        logger.info("Auto-detected Ollama (local deployment)")
        return LLMProvider.OLLAMA

    @classmethod
    def _get_provider_config(cls, provider: LLMProvider) -> Dict[str, Any]:
        """Get configuration for a provider from environment"""
        config = {}

        if provider == LLMProvider.OPENAI:
            config["api_key"] = os.getenv("OPENAI_API_KEY")
            config["organization"] = os.getenv("OPENAI_ORGANIZATION")

        elif provider == LLMProvider.ANTHROPIC:
            config["api_key"] = os.getenv("ANTHROPIC_API_KEY")

        elif provider == LLMProvider.OLLAMA:
            config["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        return config

    @classmethod
    def get_available_providers(cls) -> Dict[str, Dict[str, Any]]:
        """Check which providers are available and get their info"""
        availability = {}

        for provider in LLMProvider:
            if provider == LLMProvider.AUTO:
                continue

            try:
                adapter = cls.create_adapter(provider, use_cache=False)
                model_info = adapter.get_model_info()
                model_info["health_status"] = adapter.health_check()
                availability[provider.value] = model_info
            except (ImportError, ConnectionError, ValueError, OSError) as e:
                availability[provider.value] = {
                    "available": False,
                    "error": str(e),
                    "provider": provider.value,
                }

        return availability

    @classmethod
    def get_best_available_adapter(
        cls, preference_order: Optional[List[LLMProvider]] = None, **kwargs
    ) -> Optional[LLMInterface]:
        """
        Get the best available adapter based on preference order

        Args:
            preference_order: List of providers in order of preference
            **kwargs: Additional arguments for adapter creation

        Returns:
            LLMInterface: Best available adapter or None
        """
        if not preference_order:
            preference_order = [
                LLMProvider.OLLAMA,  # Local first
                LLMProvider.OPENAI,
                LLMProvider.ANTHROPIC,
            ]

        for provider in preference_order:
            try:
                adapter = cls.create_adapter(provider, use_cache=True, **kwargs)
                if adapter.health_check():
                    logger.info("Using LLM provider: %s", provider.value)
                    return adapter

                logger.debug("Provider %s not healthy", provider.value)
            except (ImportError, ConnectionError, ValueError, OSError) as e:
                logger.debug("Provider %s not available: %s", provider.value, e)
                continue

        logger.warning("No LLM providers available")
        return None

    @classmethod
    def clear_cache(cls):
        """Clear the adapter cache"""
        cls._adapter_cache.clear()
        logger.info("Adapter cache cleared")

    @classmethod
    def register_provider(
        cls,
        provider: LLMProvider,
        adapter_class: Type[Any],
        default_model: str,
    ):
        """
        Register a new provider

        Args:
            provider: Provider enum
            adapter_class: Adapter class
            default_model: Default model name
        """
        cls._providers[provider] = adapter_class
        cls._default_models[provider] = default_model
        logger.info("Registered new provider: %s", provider.value)

    @classmethod
    def list_models(cls, provider: LLMProvider) -> List[str]:
        """
        List available models for a provider

        Args:
            provider: The provider to query

        Returns:
            List[str]: Available model names
        """
        try:
            # Verify provider is available by creating adapter
            cls.create_adapter(provider, use_cache=False)
            # For now, return default model as most providers don't expose model lists
            # This could be enhanced in the future to query provider APIs
            return [cls._default_models[provider]]

        except (ImportError, ConnectionError, ValueError, OSError) as e:
            logger.error("Failed to list models for %s: %s", provider.value, e)
            return []


# Global factory instance for backward compatibility
llm_factory = LLMFactory()


def get_llm(
    provider: str = "auto", model: Optional[str] = None, **kwargs
) -> LLMInterface:
    """
    Convenience function to get an LLM adapter

    Args:
        provider: Provider name
        model: Model name
        **kwargs: Additional arguments

    Returns:
        LLMInterface: LLM adapter
    """
    provider_enum = (
        LLMProvider(provider.lower()) if provider != "auto" else LLMProvider.AUTO
    )
    return LLMFactory.create_adapter(provider_enum, model, **kwargs)


def get_best_llm(**kwargs) -> Optional[LLMInterface]:
    """
    Convenience function to get the best available LLM adapter

    Args:
        **kwargs: Additional arguments

    Returns:
        LLMInterface: Best available adapter or None
    """
    return LLMFactory.get_best_available_adapter(**kwargs)
