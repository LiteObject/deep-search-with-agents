"""
LLM Factory for managing different AI model providers and implementations.
"""

import os
import logging
import requests
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs
        self.available = self._check_availability()
    
    @abstractmethod
    def _check_availability(self) -> bool:
        """Check if the LLM provider is available"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate text using the LLM"""
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider"""
        pass


class OllamaClient(BaseLLMClient):
    """Ollama LLM client implementation"""
    
    def __init__(self, model: str = "gpt-oss:latest", 
                 base_url: str = "http://localhost:11434", **kwargs):
        self.base_url = base_url.rstrip('/')
        super().__init__(model, base_url=base_url, **kwargs)
    
    def _check_availability(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                if any(self.model in name for name in model_names):
                    logger.info(f"Ollama model {self.model} is available at {self.base_url}")
                    return True
                else:
                    logger.warning(f"Model {self.model} not found. Available: {model_names}")
                    return False
            return False
        except Exception as e:
            logger.warning(f"Could not connect to Ollama at {self.base_url}: {e}")
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate text using Ollama"""
        if not self.available:
            raise RuntimeError("Ollama client is not available")
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Merge kwargs with default options
            options = {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_ctx": 4096,
                **kwargs.get('options', {})
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": options
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=kwargs.get('timeout', 60)
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '').strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return ""
        
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return ""
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get Ollama provider information"""
        return {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "available": self.available,
            "features": ["local", "private", "offline", "customizable"]
        }


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client implementation"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", 
                 api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        super().__init__(model, api_key=self.api_key, **kwargs)
    
    def _check_availability(self) -> bool:
        """Check if OpenAI client is available"""
        if not self.api_key:
            logger.warning("OpenAI API key not available")
            return False
        
        try:
            # Try to import OpenAI
            import openai  # pylint: disable=import-outside-toplevel
            self.client = openai.OpenAI(api_key=self.api_key)
            return True
        except ImportError:
            logger.error("OpenAI library not installed. Install with: pip install openai")
            return False
        except Exception as e:
            logger.error(f"OpenAI client initialization error: {e}")
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate text using OpenAI"""
        if not self.available:
            raise RuntimeError("OpenAI client is not available")
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.3),
                max_tokens=kwargs.get('max_tokens', 1000),
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return ""
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get OpenAI provider information"""
        return {
            "provider": "openai",
            "model": self.model,
            "available": self.available,
            "features": ["cloud", "high_quality", "rate_limited", "paid"]
        }


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude LLM client implementation"""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", 
                 api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        super().__init__(model, api_key=self.api_key, **kwargs)
    
    def _check_availability(self) -> bool:
        """Check if Anthropic client is available"""
        if not self.api_key:
            logger.warning("Anthropic API key not available")
            return False
        
        try:
            import anthropic  # pylint: disable=import-outside-toplevel
            self.client = anthropic.Anthropic(api_key=self.api_key)
            return True
        except ImportError:
            logger.error("Anthropic library not installed. Install with: pip install anthropic")
            return False
        except Exception as e:
            logger.error(f"Anthropic client initialization error: {e}")
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate text using Anthropic Claude"""
        if not self.available:
            raise RuntimeError("Anthropic client is not available")
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.3),
                system=system_prompt or "You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
        
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            return ""
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get Anthropic provider information"""
        return {
            "provider": "anthropic",
            "model": self.model,
            "available": self.available,
            "features": ["cloud", "safety_focused", "rate_limited", "paid"]
        }


class LLMFactory:
    """Factory class for creating and managing LLM clients"""
    
    def __init__(self):
        self._clients: Dict[str, BaseLLMClient] = {}
        self._default_configs = {
            LLMProvider.OLLAMA: {
                "model": "gpt-oss:latest",
                "base_url": "http://localhost:11434"
            },
            LLMProvider.OPENAI: {
                "model": "gpt-3.5-turbo",
                "api_key": None
            },
            LLMProvider.ANTHROPIC: {
                "model": "claude-3-sonnet-20240229",
                "api_key": None
            }
        }
    
    def create_client(self, provider: LLMProvider, **config) -> BaseLLMClient:
        """Create an LLM client for the specified provider"""
        
        # Merge with default config
        final_config = {**self._default_configs.get(provider, {}), **config}
        
        if provider == LLMProvider.OLLAMA:
            return OllamaClient(**final_config)
        elif provider == LLMProvider.OPENAI:
            return OpenAIClient(**final_config)
        elif provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(**final_config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def get_client(self, provider: LLMProvider, **config) -> BaseLLMClient:
        """Get or create a cached LLM client"""
        cache_key = f"{provider.value}:{hash(frozenset(config.items()))}"
        
        if cache_key not in self._clients:
            self._clients[cache_key] = self.create_client(provider, **config)
        
        return self._clients[cache_key]
    
    def get_available_clients(self) -> List[Dict[str, Any]]:
        """Get information about all available LLM providers"""
        available = []
        
        for provider in LLMProvider:
            try:
                client = self.create_client(provider)
                if client.available:
                    available.append(client.get_provider_info())
            except Exception as e:
                logger.debug(f"Provider {provider.value} not available: {e}")
        
        return available
    
    def get_best_available_client(self, preference_order: Optional[List[LLMProvider]] = None) -> Optional[BaseLLMClient]:
        """Get the best available client based on preference order"""
        if not preference_order:
            preference_order = [LLMProvider.OLLAMA, LLMProvider.OPENAI, LLMProvider.ANTHROPIC]
        
        for provider in preference_order:
            try:
                client = self.create_client(provider)
                if client.available:
                    logger.info(f"Using LLM provider: {provider.value}")
                    return client
            except Exception as e:
                logger.debug(f"Provider {provider.value} not available: {e}")
                continue
        
        logger.warning("No LLM providers available")
        return None


# Global factory instance
llm_factory = LLMFactory()


def get_llm_client(provider: LLMProvider = LLMProvider.OLLAMA, **config) -> Optional[BaseLLMClient]:
    """Convenience function to get an LLM client"""
    try:
        return llm_factory.get_client(provider, **config)
    except Exception as e:
        logger.error(f"Failed to get LLM client for {provider.value}: {e}")
        return None


def get_best_llm_client(**config) -> Optional[BaseLLMClient]:
    """Convenience function to get the best available LLM client"""
    return llm_factory.get_best_available_client()
