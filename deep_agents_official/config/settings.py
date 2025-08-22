"""
Configuration settings for Official DeepAgents Implementation
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for Official DeepAgents"""

    # API Keys - set these in your .env file
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

    # Model configurations
    DEFAULT_MODEL = "gpt-4"
    FAST_MODEL = "gpt-3.5-turbo"
    ANTHROPIC_MODEL = "claude-3-sonnet-20240229"

    # DeepAgent settings
    MAX_PLANNING_ITERATIONS = 5
    MAX_REFLECTION_DEPTH = 3
    FILE_SYSTEM_ROOT = "./workspace"

    # Search settings
    MAX_SEARCH_RESULTS = 10
    SEARCH_TIMEOUT = 30

    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "deep_agents.log"

    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        required_keys = [cls.OPENAI_API_KEY]
        missing_keys = [key for key in required_keys if not key]

        if missing_keys:
            raise ValueError(f"Missing required API keys: {missing_keys}")

        return True
