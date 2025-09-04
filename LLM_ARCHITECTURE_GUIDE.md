# LLM System Architecture Guide

This comprehensive guide covers the complete LLM system architecture for the Deep Search Agents project, including both legacy and current implementations.

## Table of Contents

1. [Overview](#overview)
2. [System Evolution](#system-evolution)
3. [Legacy System (tools/llm_factory.py)](#legacy-system)
4. [Enhanced System (deep_agents_custom/llm/)](#enhanced-system)
5. [Migration Guide](#migration-guide)
6. [Comparison](#comparison)
7. [Configuration](#configuration)
8. [Best Practices](#best-practices)
9. [Future Roadmap](#future-roadmap)

## Overview

The Deep Search Agents project implements a sophisticated LLM integration system using key design patterns:

- **Factory Pattern**: Abstracts the creation and management of LLM provider clients
- **Adapter Pattern**: Provides a unified interface across different LLM providers
- **Strategy Pattern**: Enables runtime switching between different LLM providers

The system has evolved from a basic factory implementation to an enhanced adapter-factory hybrid that provides better flexibility, maintainability, and extensibility.

## System Evolution

### Timeline
1. **Initial Implementation**: Basic factory pattern in `tools/llm_factory.py`
2. **Enhanced Version**: Comprehensive adapter system in `deep_agents_custom/llm/`
3. **Current State**: Both systems coexist with migration path available

### Key Improvements
- **Better Abstraction**: Enhanced interface design
- **Improved Error Handling**: Robust fallback mechanisms
- **Enhanced Caching**: More efficient adapter reuse
- **Extensibility**: Easier to add new providers
- **Modularity**: Clean separation of concerns

---

## Legacy System

> **⚠️ DEPRECATED**: The legacy system in `tools/llm_factory.py` is maintained for backward compatibility but new development should use the enhanced system.

### Architecture

```
tools/llm_factory.py
├── BaseLLMClient (Abstract Base Class)
├── OllamaClient (Local AI)
├── OpenAIClient (Cloud AI)
├── AnthropicClient (Cloud AI)
└── LLMFactory (Factory Class)
```

### Core Components

#### `BaseLLMClient` (ABC)
Abstract interface that all LLM providers must implement:
- `generate(prompt, system_prompt, **kwargs)` - Generate text
- `get_provider_info()` - Provider metadata
- `_check_availability()` - Availability validation

#### Provider Implementations
- **`OllamaClient`**: Local Ollama integration with HTTP API
- **`OpenAIClient`**: OpenAI API integration with official SDK
- **`AnthropicClient`**: Anthropic Claude integration with official SDK

#### `LLMFactory`
Factory class for creating and managing clients:
- Provider instantiation
- Client caching
- Automatic provider selection

### Legacy Usage Examples

```python
# Auto-select best provider
from tools.llm_factory import get_best_llm_client

client = get_best_llm_client()
if client:
    response = client.generate("Hello world", "You are helpful.")

# Specific provider selection
from tools.llm_factory import LLMFactory, LLMProvider

factory = LLMFactory()
ollama_client = factory.create_client(
    LLMProvider.OLLAMA,
    model="gpt-oss:latest",
    base_url="http://localhost:11434"
)

# Smart summarizer
from tools.summarizer import LLMSummarizer

summarizer = LLMSummarizer(
    provider=LLMProvider.OLLAMA,
    model="gpt-oss:latest"
)
```

---

## Enhanced System

> **✅ RECOMMENDED**: The enhanced system in `deep_agents_custom/llm/` is the current recommended approach for all new development.

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
from enhanced_agent_example import EnhancedBaseAgent

# Create agent with auto-selected LLM
agent = EnhancedBaseAgent(
    name="Research Agent",
    description="Enhanced research capabilities",
    llm_provider="auto"  # or "ollama", "openai", etc.
)

# Process queries with LLM enhancement
result = agent.process_query(
    query="latest AI developments",
    enhanced_search=True,
    summary_style="comprehensive"
)
```

---

## Migration Guide

### From Legacy to Enhanced System

#### Step 1: Update Imports
```python
# OLD WAY
from tools.llm_factory import get_best_llm_client, LLMProvider

# NEW WAY
from llm import get_best_llm, LLMProvider
```

#### Step 2: Update Client Creation
```python
# OLD WAY
client = get_best_llm_client()
response = client.generate("prompt", system_prompt="system")

# NEW WAY
llm = get_best_llm()
response = llm.generate_with_system("system", "prompt")
```

#### Step 3: Update Agent Integration
```python
# OLD WAY
class MyAgent:
    def __init__(self):
        self.llm_client = get_best_llm_client()
    
    def process(self, query):
        return self.llm_client.generate(query, "You are helpful")

# NEW WAY
class MyAgent:
    def __init__(self, llm_provider="auto"):
        self.llm = get_llm(llm_provider)
    
    def process(self, query):
        return self.llm.generate_with_system("You are helpful", query)
```

#### Step 4: Update Summarizers
```python
# OLD WAY
from tools.summarizer import LLMSummarizer
summarizer = LLMSummarizer(provider=LLMProvider.OLLAMA)

# NEW WAY  
from tools.enhanced_summarizer import EnhancedLLMSummarizer
summarizer = EnhancedLLMSummarizer(provider="ollama")
```

### Migration Checklist

- [ ] Update imports to use new `llm` package
- [ ] Replace `generate(prompt, system_prompt)` with `generate_with_system(system, user)`
- [ ] Update provider specification from enum to string
- [ ] Test with existing functionality
- [ ] Update configuration files
- [ ] Update documentation references

---

## Comparison

| Feature | Legacy System | Enhanced System |
|---------|---------------|-----------------|
| **Architecture** | Basic Factory | Adapter + Factory |
| **Interface** | Simple | Comprehensive |
| **Error Handling** | Basic | Advanced with retries |
| **Caching** | Simple | Sophisticated |
| **Streaming** | Limited | Full support |
| **Token Estimation** | Basic | Advanced |
| **Health Monitoring** | Simple | Continuous |
| **Configuration** | Limited | Flexible |
| **Extensibility** | Moderate | High |
| **Performance** | Good | Optimized |
| **Maintenance** | Higher effort | Lower effort |

### When to Use Each System

#### Use Legacy System When:
- Working with existing code that hasn't been migrated
- Need simple, basic LLM integration
- Backward compatibility is critical

#### Use Enhanced System When:
- Starting new development
- Need advanced features (streaming, structured output)
- Want better error handling and reliability
- Planning to add new providers
- Performance optimization is important

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

#### Legacy System
```python
factory = LLMFactory()
client = factory.create_client(
    LLMProvider.OLLAMA,
    model="custom-model",
    base_url="http://remote-ollama:11434"
)
```

#### Enhanced System
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
- **Local Development**: Use Ollama for privacy and cost
- **Production**: Use OpenAI/Anthropic for reliability
- **Hybrid**: Auto-detection for flexibility

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

The LLM system architecture provides a robust, flexible foundation for AI integration in the Deep Search Agents project. The enhanced system represents the current state-of-the-art implementation, while the legacy system ensures backward compatibility during the transition period.

For new development, always use the enhanced system (`deep_agents_custom/llm/`). For existing code, plan migration according to the provided guide. The architecture supports both local deployment with Ollama and cloud integration with OpenAI and Anthropic, providing flexibility for different deployment scenarios and requirements.

---

## Related Documentation

- **Legacy System**: `tools/llm_factory.py` (deprecated)
- **Enhanced System**: `deep_agents_custom/llm/` (current)
- **Migration Examples**: `enhanced_agent_example.py`
- **Test Suite**: `test_llm_system.py`
