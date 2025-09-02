"""
Application settings and configuration.
"""

import os
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv  # type: ignore

# Load environment variables from root directory
# Get the root directory (parent of deep_agents_custom)
current_dir = Path(__file__).parent.parent.parent
env_path = current_dir / '.env'
load_dotenv(env_path)


class Settings:
    """Application settings"""

    # API Keys
    TAVILY_API_KEY: str = os.getenv('TAVILY_API_KEY', '')

    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL: str = os.getenv('OLLAMA_MODEL', 'gpt-oss:latest')

    # Search Engine Settings
    DEFAULT_SEARCH_ENGINE: str = os.getenv(
        'DEFAULT_SEARCH_ENGINE', 'duckduckgo')
    MAX_SEARCH_RESULTS: int = int(os.getenv('MAX_SEARCH_RESULTS', '10'))
    SEARCH_TIMEOUT: int = int(os.getenv('SEARCH_TIMEOUT', '30'))

    # Agent Settings
    MAX_ITERATIONS: int = int(os.getenv('MAX_ITERATIONS', '5'))
    VERBOSE_MODE: bool = os.getenv('VERBOSE_MODE', 'true').lower() == 'true'

    # Streamlit Configuration
    STREAMLIT_PORT: int = int(os.getenv('STREAMLIT_PORT', '8501'))
    STREAMLIT_HOST: str = os.getenv('STREAMLIT_HOST', 'localhost')

    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """Get search-related configuration"""
        return {
            'default_engine': cls.DEFAULT_SEARCH_ENGINE,
            'max_results': cls.MAX_SEARCH_RESULTS,
            'timeout': cls.SEARCH_TIMEOUT,
            'verbose': cls.VERBOSE_MODE
        }

    @classmethod
    def get_agent_config(cls) -> Dict[str, Any]:
        """Get agent-related configuration"""
        return {
            'max_iterations': cls.MAX_ITERATIONS,
            'verbose': cls.VERBOSE_MODE,
            'ollama_base_url': cls.OLLAMA_BASE_URL,
            'ollama_model': cls.OLLAMA_MODEL,
            'tavily_key': cls.TAVILY_API_KEY
        }

    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate configuration and return True if all required settings
        are present
        """
        required_for_basic = []  # No required settings for basic functionality

        # Check if any required settings are missing
        missing = [
            setting for setting in required_for_basic
            if not getattr(cls, setting)]

        if missing:
            print(f"Warning: Missing required configuration: "
                  f"{', '.join(missing)}")
            return False

        # Check optional but recommended settings
        try:
            import requests
            response = requests.get(f"{cls.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code != 200:
                print(f"Warning: Could not connect to Ollama at {cls.OLLAMA_BASE_URL}. "
                      f"AI summarization will not be available.")
        except Exception:
            print(f"Warning: Could not connect to Ollama at {cls.OLLAMA_BASE_URL}. "
                  f"AI summarization will not be available.")

        if not cls.TAVILY_API_KEY:
            print("Warning: TAVILY_API_KEY not set. "
                  "Tavily search will not be available.")

        return True


# Global settings instance
settings = Settings()
