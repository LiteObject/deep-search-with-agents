# LLM System Architecture Guide

This guide covers the LLM system architecture for the Deep Search Agents project, focusing on the enhanced adapter and factory pattern implementation.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Enhanced System Implementation](#enhanced-system-implementation)
4. [Usage Examples](#usage-examples)
5. [Configuration](#configuration)
6. [Best Practices](#best-practices)
7. [Future Roadmap](#future-roadmap)

## Overview

The Deep Search Agents project implements a sophisticated LLM integration system using key design patterns:

- **Factory Pattern**: Abstracts the creation and management of LLM provider clients
- **Adapter Pattern**: Provides a unified interface across different LLM providers
- **Strategy Pattern**: Enables runtime switching between different LLM providers

The system provides a robust, extensible foundation for AI integration with support for multiple LLM providers while maintaining consistent interfaces and intelligent fallback mechanisms.

## System Architecture

The LLM system is built on a clean, modular architecture that separates concerns and provides flexibility:

```
deep_agents_custom/llm/
├── __init__.py                      # Main package exports
├── interfaces/
│   ├── __init__.py
│   └── llm_interface.py             # Abstract interface definition
├── adapters/
│   ├── __init__.py
│   ├── base_adapter.py              # Common adapter functionality
│   ├── ollama_adapter.py            # Ollama implementation
│   ├── openai_adapter.py            # OpenAI implementation
│   └── anthropic_adapter.py         # Anthropic implementation
└── factory/
    ├── __init__.py
    └── llm_factory.py               # Factory pattern implementation
```

### Core Benefits
- **Better Abstraction**: Enhanced interface design
- **Improved Error Handling**: Robust fallback mechanisms
- **Enhanced Caching**: More efficient adapter reuse
- **Extensibility**: Easier to add new providers
- **Modularity**: Clean separation of concerns

## Enhanced System Implementation

The enhanced system in `deep_agents_custom/llm/` provides a comprehensive, production-ready LLM integration solution.

### Architecture

```
deep_agents_custom/
├── llm/                          # Enhanced LLM system
│   ├── __init__.py              # Main package exports
│   ├── interfaces/
│   │   ├── __init__.py
│   │   └── llm_interface.py     # Abstract interface definition
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base_adapter.py      # Common adapter functionality
│   │   ├── ollama_adapter.py    # Ollama implementation
│   │   ├── openai_adapter.py    # OpenAI implementation
│   │   └── anthropic_adapter.py # Anthropic implementation
│   └── factory/
│       ├── __init__.py
│       └── llm_factory.py       # Enhanced factory implementation
```

### Key Components

#### 1. LLM Interface (`llm/interfaces/llm_interface.py`)
Comprehensive contract that all adapters must implement:
- `generate()` - Simple text generation
- `generate_with_system()` - Generation with system prompt
- `generate_structured()` - JSON mode generation
- `stream_generate()` - Streaming generation
- `health_check()` - Provider availability check
- `get_model_info()` - Model metadata
- `estimate_tokens()` - Token estimation
- `get_context_window()` - Context window size

#### 2. Base Adapter (`llm/adapters/base_adapter.py`)
Common functionality for all adapters:
- Error handling and logging
- Retry logic with exponential backoff
- Token estimation fallbacks
- Health check caching
- Message formatting utilities
- Rate limiting awareness

#### 3. Provider Adapters
Enhanced implementations with provider-specific optimizations:
- **`OllamaAdapter`**: Local models with advanced error handling
- **`OpenAIAdapter`**: Cloud models with cost optimization
- **`AnthropicAdapter`**: Safety-focused models with content filtering

#### 4. Enhanced Factory (`llm/factory/llm_factory.py`)
Advanced adapter lifecycle management:
- Intelligent provider auto-detection
- Environment-based configuration
- Sophisticated adapter caching
- Provider health monitoring
- Fallback chain management

### Enhanced Usage Examples

#### Basic Usage
```python
from llm import get_llm, get_best_llm

# Auto-select best available provider
llm = get_best_llm()

# Use specific provider
ollama_llm = get_llm("ollama", "gpt-oss:latest")
openai_llm = get_llm("openai", "gpt-4")

# Generate text
response = llm.generate("Explain quantum computing")
```

#### Advanced Usage
```python
from llm import LLMFactory, LLMProvider

# Check available providers
providers = LLMFactory.get_available_providers()
print(providers)

# Create adapter with custom configuration
adapter = LLMFactory.create_adapter(
    provider=LLMProvider.OLLAMA,
    model_name="llama2",
    base_url="http://custom-ollama:11434",
    temperature=0.7,
    max_tokens=2000
)

# Structured generation
result = adapter.generate_structured(
    prompt="Analyze this data",
    response_format={
        "summary": "string", 
        "insights": ["string"],
        "confidence": "number"
    }
)

# Streaming generation
for chunk in adapter.stream_generate("Write a story"):
    print(chunk, end="", flush=True)
```

#### Integration with Agents
```python
from agents.base_agent import BaseAgent
from agents.general_agent import GeneralAgent

# Create agent with enhanced LLM system
agent = GeneralAgent(max_results=15)

# The agent automatically uses EnhancedLLMSummarizer with auto-selected LLM
result = agent.search("latest AI developments")

# Access the summary and citations
print(result.summary)
print(result.cited_summary)  # Summary with citations
```

---

## Configuration

### Environment Variables

Both systems use the same environment variables:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key
OPENAI_ORGANIZATION=your_org_id

# Anthropic  
ANTHROPIC_API_KEY=your_anthropic_key

# Ollama (optional, defaults to localhost)
OLLAMA_BASE_URL=http://localhost:11434
```

### Provider Auto-Detection Order

1. **OpenAI** (if API key is available)
2. **Anthropic** (if API key is available)  
3. **Ollama** (default fallback for local deployment)

### Custom Configuration

```python
adapter = LLMFactory.create_adapter(
    provider="ollama",
    model_name="custom-model",
    base_url="http://remote-ollama:11434",
    temperature=0.5,
    max_tokens=2000,
    retry_attempts=3
)
```

---

## Best Practices

### 1. Provider Selection
- **Local Development**: Use Ollama for privacy and cost efficiency
- **Production**: Use OpenAI/Anthropic for reliability and performance
- **Hybrid**: Auto-detection for maximum flexibility and fallback support

### 2. Error Handling
```python
try:
    llm = get_best_llm()
    if llm and llm.health_check():
        response = llm.generate("prompt")
    else:
        # Fallback logic
        response = "LLM not available"
except Exception as e:
    logger.error(f"LLM error: {e}")
    # Handle gracefully
```

### 3. Token Management
```python
# Check context window before processing
if llm.estimate_tokens(prompt) > llm.get_context_window():
    # Split or truncate prompt
    prompt = truncate_prompt(prompt, llm.get_context_window())
```

### 4. Performance Optimization
```python
# Reuse adapters
llm = get_llm("ollama", "model-name")  # Cached after first call

# Use appropriate model for task
summary_llm = get_llm("ollama", "fast-model")
analysis_llm = get_llm("openai", "gpt-4")
```

### 5. Configuration Management
```python
# Environment-specific configuration
if os.getenv("ENVIRONMENT") == "production":
    llm = get_llm("openai", "gpt-4")
else:
    llm = get_llm("ollama", "development-model")
```

---

## Future Roadmap

### Short Term (Next Release)
- [ ] Complete migration guide automation
- [ ] Performance monitoring dashboard
- [ ] Additional provider support (Azure OpenAI, Google PaLM)
- [ ] Enhanced error recovery mechanisms

### Medium Term
- [ ] Load balancing across multiple instances
- [ ] A/B testing framework for providers
- [ ] Cost optimization algorithms
- [ ] Automatic model selection based on task type

### Long Term
- [ ] Plugin architecture for custom providers
- [ ] Machine learning-based provider selection
- [ ] Distributed LLM orchestration
- [ ] Real-time performance analytics

## Conclusion

The LLM system architecture provides a robust, flexible foundation for AI integration in the Deep Search Agents project. The enhanced system in `deep_agents_custom/llm/` is the sole implementation after successful migration and legacy code cleanup.

All agents in the `deep_agents_custom` implementation automatically use the enhanced LLM system with intelligent fallbacks. The architecture supports local deployment with Ollama, cloud integration with OpenAI and Anthropic, and automatic provider selection for optimal performance.

**Current Status**: ✅ Production-ready enhanced system with comprehensive adapter support.

---

## Related Documentation

- **Enhanced System**: `deep_agents_custom/llm/` (current implementation)
- **Agent Examples**: `deep_agents_custom/agents/` (general_agent.py, research_agent.py, news_agent.py)
- **Usage Examples**: See the enhanced system implementation above
- **Configuration Guide**: Environment variables and provider setup sections
