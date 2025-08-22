"""
Configuration settings for LangChain Deep Agents implementation
"""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    primary_model: str = "gpt-4"
    fallback_model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 30


@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    max_iterations: int = 5
    confidence_threshold: float = 0.8
    memory_limit: int = 100
    reflection_enabled: bool = True
    planning_depth: int = 3


@dataclass
class SearchConfig:
    """Configuration for search operations"""
    max_results: int = 10
    search_timeout: int = 30
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour


@dataclass
class OrchestrationConfig:
    """Configuration for agent orchestration"""
    default_collaboration_mode: str = "hierarchical"
    consensus_threshold: float = 0.7
    max_parallel_agents: int = 3
    cross_validation_enabled: bool = True


class Settings:
    """Main settings class for LangChain Deep Agents"""

    def __init__(self):
        # API Keys
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.SERPER_API_KEY = os.getenv("SERPER_API_KEY")
        self.HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

        # Configuration instances
        self.model_config = ModelConfig()
        self.agent_config = AgentConfig()
        self.search_config = SearchConfig()
        self.orchestration_config = OrchestrationConfig()

        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = "deep_agents.log"

        # Agent-specific configurations
        self.AGENT_CONFIGS = {
            "research": {
                "model": "gpt-4",
                "temperature": 0.1,
                "max_iterations": 5,
                "specialization": "academic_research",
                "tools": ["academic_search", "citation_analysis", "research_synthesis"]
            },
            "news": {
                "model": "gpt-4",
                "temperature": 0.2,
                "max_iterations": 4,
                "specialization": "news_analysis",
                "tools": ["realtime_news", "fact_check", "trend_analysis", "source_credibility"]
            },
            "general": {
                "model": "gpt-4",
                "temperature": 0.3,
                "max_iterations": 4,
                "specialization": "general_search",
                "tools": ["adaptive_search", "knowledge_base", "context_enrichment"]
            },
            "reflection": {
                "model": "gpt-4",
                "temperature": 0.1,
                "max_iterations": 3,
                "specialization": "meta_analysis",
                "tools": ["quality_assessment", "bias_detection", "gap_analysis", "improvement_suggestions"]
            }
        }

        # Deep agent specific settings
        self.DEEP_AGENT_SETTINGS = {
            "enable_hierarchical_planning": True,
            "enable_reflection_loops": True,
            "enable_meta_reasoning": True,
            "enable_collaborative_reasoning": True,
            "planning_horizon": 5,
            "reflection_frequency": 2,
            "memory_consolidation_interval": 10,
            "cross_agent_communication": True
        }

        # Search orchestration settings
        self.ORCHESTRATION_SETTINGS = {
            "query_analysis_model": "gpt-4",
            "route_planning_enabled": True,
            "adaptive_routing": True,
            "quality_gates_enabled": True,
            "consensus_voting_enabled": True,
            "meta_agent_enabled": True,
            "learning_enabled": True
        }

        # Tool configurations
        self.TOOL_CONFIGS = {
            "web_search": {
                "api_key_env": "SERPER_API_KEY",
                "endpoint": "https://google.serper.dev/search",
                "max_results": 10,
                "timeout": 30
            },
            "academic_search": {
                "sources": ["arxiv", "pubmed", "google_scholar"],
                "max_papers": 20,
                "citation_depth": 2
            },
            "news_search": {
                "sources": ["newsapi", "bing_news", "google_news"],
                "max_articles": 15,
                "time_range": "24h"
            },
            "fact_check": {
                "sources": ["snopes", "factcheck.org", "politifact"],
                "confidence_threshold": 0.7
            }
        }

        # Validation
        self._validate_config()

    def _validate_config(self):
        """Validate configuration settings"""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        if not self.SERPER_API_KEY:
            print(
                "Warning: SERPER_API_KEY not found. Web search functionality may be limited.")

    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for specific agent type"""
        return self.AGENT_CONFIGS.get(agent_type, self.AGENT_CONFIGS["general"])

    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """Get configuration for specific tool"""
        return self.TOOL_CONFIGS.get(tool_name, {})

    def update_agent_config(self, agent_type: str, config_updates: Dict[str, Any]):
        """Update configuration for specific agent"""
        if agent_type in self.AGENT_CONFIGS:
            self.AGENT_CONFIGS[agent_type].update(config_updates)

    def is_deep_agent_feature_enabled(self, feature: str) -> bool:
        """Check if a deep agent feature is enabled"""
        return self.DEEP_AGENT_SETTINGS.get(feature, False)

    def get_orchestration_setting(self, setting: str) -> Any:
        """Get orchestration setting"""
        return self.ORCHESTRATION_SETTINGS.get(setting)


# Global settings instance
settings = Settings()

# Environment-specific configurations
DEVELOPMENT_CONFIG = {
    "LOG_LEVEL": "DEBUG",
    "enable_caching": False,
    "max_iterations": 3,
    "timeout": 60
}

PRODUCTION_CONFIG = {
    "LOG_LEVEL": "INFO",
    "enable_caching": True,
    "max_iterations": 5,
    "timeout": 30
}


def get_environment_config():
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return PRODUCTION_CONFIG
    else:
        return DEVELOPMENT_CONFIG


# Apply environment-specific config
env_config = get_environment_config()
for key, value in env_config.items():
    if hasattr(settings, key):
        setattr(settings, key, value)
