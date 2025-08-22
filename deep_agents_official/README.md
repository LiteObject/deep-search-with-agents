# Official DeepAgents Implementation

This directory contains a complete implementation using the official `deepagents` package patterns. It demonstrates how to build DeepAgents that follow the four core components identified in the official documentation:

## Core Components

1. **Planning Tool (TodoWrite)**: Structured task breakdown and planning
2. **Virtual File System**: Workspace management with ls, edit_file, read_file, write_file tools
3. **Sub-agents with Context Quarantine**: Specialized agents for specific tasks
4. **Claude Code-inspired Detailed Prompts**: Comprehensive instructions for better performance

## Architecture

### Agents

- **ResearchDeepAgent** (`agents/research_deep_agent.py`): Specialized in research and information gathering
- **CodingDeepAgent** (`agents/coding_deep_agent.py`): Specialized in code analysis, generation, and debugging
- **AnalysisDeepAgent** (`agents/analysis_deep_agent.py`): Specialized in data analysis and insight generation
- **DeepAgentsOrchestrator** (`agents/deep_orchestrator.py`): Coordinates multiple agents for complex tasks

### Tools

- **Search Tools** (`tools/search_tools.py`): Internet search, academic search, and web content analysis

### Configuration

- **Settings** (`config/settings.py`): Configuration management for API keys and model settings

## Usage

### Individual Agent Usage

```python
from deep_agents_official import ResearchDeepAgent

# Create and use research agent
research_agent = ResearchDeepAgent()
result = research_agent.research("What are the latest trends in AI?")
```

### Orchestrated Multi-Agent Tasks

```python
from deep_agents_official import DeepAgentsOrchestrator

# Create orchestrator
orchestrator = DeepAgentsOrchestrator()

# Execute complex task requiring multiple agents
result = orchestrator.execute_multi_agent_task(
    "Research AI trends and create a summary report", 
    task_type="mixed"
)
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API keys:
```bash
export OPENAI_API_KEY='your-key-here'
export ANTHROPIC_API_KEY='your-key-here'
export SERPAPI_API_KEY='your-key-here'
```

3. Install the official deepagents package (when available):
```bash
pip install deepagents>=0.1.0
```

## Examples

Run the example script to see all agents in action:

```bash
python example_usage.py
```

## Implementation Notes

This implementation currently uses placeholder patterns for the `create_deep_agent()` function since the actual deepagents package may not be available. To use with the real package:

1. Uncomment the `from deepagents import create_deep_agent` imports
2. Uncomment the actual `create_deep_agent()` calls in each agent
3. Remove the placeholder implementations

## Key Features

- **Virtual File System Integration**: Each agent maintains its own workspace
- **Hierarchical Planning**: Uses TodoWrite planning tool for structured task breakdown
- **Sub-agent Specialization**: Context quarantine for specialized tasks
- **Multi-agent Orchestration**: Coordinate multiple agents for complex workflows
- **Comprehensive Tooling**: Search, analysis, and coding tools
- **Error Handling**: Robust error handling and recovery mechanisms

## Comparison with Custom Implementation

This official implementation differs from the custom implementation in `deep_agents_with_langchain/` by:

1. Using the official `create_deep_agent()` function
2. Implementing virtual file system integration
3. Following official sub-agent patterns with context quarantine
4. Using TodoWrite planning tool specifically
5. Following Claude Code-inspired prompt engineering

## Next Steps

1. Test with real scenarios once the deepagents package is available
2. Extend with additional specialized agents as needed
3. Integrate with existing search infrastructure
4. Add monitoring and logging capabilities
5. Implement advanced orchestration patterns
