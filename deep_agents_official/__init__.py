"""
Official DeepAgents Implementation
A complete implementation using the official deepagents package patterns.

This package provides:
- ResearchDeepAgent: Specialized in research and information gathering
- CodingDeepAgent: Specialized in code analysis, generation, and debugging  
- AnalysisDeepAgent: Specialized in data analysis and insight generation
- DeepAgentsOrchestrator: Coordinates multiple agents for complex tasks

The implementation follows the official deepagents patterns:
1. Planning tools (TodoWrite) for structured task breakdown
2. Virtual file system (ls, edit_file, read_file, write_file) for workspace management
3. Sub-agents with context quarantine for specialized tasks
4. Claude Code-inspired detailed prompts for better performance

Usage:
    from deep_agents_official import ResearchDeepAgent, DeepAgentsOrchestrator
    
    # Use individual agent
    research_agent = ResearchDeepAgent()
    result = research_agent.research("What are the latest trends in AI?")
    
    # Use orchestrator for complex tasks
    orchestrator = DeepAgentsOrchestrator()
    result = orchestrator.execute_multi_agent_task(
        "Research AI trends and create a summary report", 
        task_type="mixed"
    )
"""

from .agents import (
    ResearchDeepAgent,
    CodingDeepAgent,
    AnalysisDeepAgent,
    DeepAgentsOrchestrator
)

from .config import Config
from .tools import create_search_tools

__version__ = "0.1.0"
__author__ = "DeepSearch Project"

__all__ = [
    'ResearchDeepAgent',
    'CodingDeepAgent',
    'AnalysisDeepAgent',
    'DeepAgentsOrchestrator',
    'Config',
    'create_search_tools'
]
