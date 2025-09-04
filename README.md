# Deep Search with Agents

A comprehensive intelligent web search system with **three distinct implementations** of AI agents that perform specialized, multi-step web searches. The system automatically selects optimal agents based on query type and provides deep insights across various domains.

## What Makes This "Deep" Search?

Unlike traditional search that returns raw results, **Deep Search Agents** employ sophisticated multi-step reasoning:

- **Intelligent Agent Selection**: Automatically chooses the best agent (Research, News, or General) based on query analysis
- **Multi-Source Synthesis**: Combines results from multiple search engines (DuckDuckGo, Tavily, Wikipedia)
- **Concurrent Processing**: Parallel searches with intelligent result ranking and deduplication
- **Context-Aware Summarization**: Extracts key insights and creates coherent summaries
- **Temporal Intelligence**: Understands time-sensitive queries and prioritizes recent content
- **Deep Analysis**: Goes beyond simple keyword matching to provide contextual understanding

## Key Features

- **Multi-Agent Architecture**: Specialized agents for research, news, and general searches
- **Three Implementation Approaches**: Choose from custom, LangChain, or official patterns
- **Multiple Search Engines**: DuckDuckGo, Tavily, Wikipedia integration
- **Enhanced LLM Integration**: Advanced adapter system supporting Ollama, OpenAI, and Anthropic
- **Interactive Interfaces**: Both Streamlit web app and CLI available
- **Privacy-Focused**: Works locally without external APIs, all processing on your machine
- **High Performance**: Concurrent searches with intelligent result ranking
- **Extensible Architecture**: Easy to add new agents, search sources, and LLM providers

> **For detailed LLM architecture information, see [LLM_ARCHITECTURE_GUIDE.md](LLM_ARCHITECTURE_GUIDE.md)**

## Project Structure

```
deep-search-with-agents/
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
├── .pylintrc                        # Pylint configuration
├── README.md                        # This documentation
├── requirements.txt                 # Python dependencies
├── logs/                            # Log files directory (auto-created)
├── deep_agents_custom/              # ⭐ Custom implementation (production-ready)
│   ├── __init__.py                  # Package initialization
│   ├── agents/                      # Custom agents and orchestrator
│   │   ├── __init__.py              # Package exports
│   │   ├── base_agent.py            # Base agent class with SearchResult/SearchSummary
│   │   ├── research_agent.py        # Academic/research focused agent
│   │   ├── news_agent.py            # News and current events agent  
│   │   ├── general_agent.py         # General-purpose web search agent
│   │   └── search_orchestrator.py   # Multi-agent coordination and auto-selection
│   ├── llm/                         # Enhanced LLM integration system
│   │   ├── __init__.py              # LLM package exports
│   │   ├── interfaces/              # LLM interface definitions
│   │   │   ├── __init__.py          # Interface exports
│   │   │   └── llm_interface.py     # Abstract LLM interface
│   │   ├── adapters/                # LLM provider adapters
│   │   │   ├── __init__.py          # Adapter exports
│   │   │   ├── base_adapter.py      # Base adapter functionality
│   │   │   ├── ollama_adapter.py    # Ollama integration
│   │   │   ├── openai_adapter.py    # OpenAI integration
│   │   │   └── anthropic_adapter.py # Anthropic Claude integration
│   │   └── factory/                 # LLM factory pattern implementation
│   │       ├── __init__.py          # Factory exports
│   │       └── llm_factory.py       # Enhanced LLM factory
│   ├── tools/                       # Search and processing tools
│   │   ├── __init__.py              # Package exports
│   │   ├── web_search.py            # WebSearchManager, DuckDuckGo, Tavily, Wikipedia
│   │   └── enhanced_summarizer.py   # Enhanced LLM-powered summarization
│   ├── config/                      # Configuration management
│   │   ├── __init__.py              # Package initialization
│   │   └── settings.py              # Application settings and validation
│   ├── utils/                       # Helper utilities
│   │   ├── __init__.py              # Package initialization
│   │   ├── logger.py                # Logging configuration
│   │   └── helpers.py               # Utility functions
│   ├── app.py                       # ⭐ Streamlit web interface (MAIN ENTRY POINT)
│   ├── main.py                      # CLI interface
│   └── basic_test.py                # Basic functionality test
├── deep_agents_with_langchain/      # LangChain-based implementation
│   ├── agents/                      # LangChain agents with advanced features
│   │   ├── base_deep_agent.py       # LangChain base agent
│   │   ├── research_deep_agent.py   # Research with reflection loops
│   │   ├── news_deep_agent.py       # News with hierarchical planning
│   │   ├── general_deep_agent.py    # General with meta-reasoning
│   │   ├── reflection_agent.py      # Self-reflection capabilities
│   │   └── deep_orchestrator.py     # Advanced orchestration
│   ├── tools/                       # LangChain-compatible tools
│   ├── config/                      # LangChain configuration
│   ├── utils/                       # LangChain utilities
│   ├── app.py                       # LangChain Streamlit app
│   └── main.py                      # LangChain CLI
└── deep_agents_official/            # Official DeepAgents package patterns
    ├── agents/                      # Official implementation agents
    │   ├── research_deep_agent.py   # Research agent using official patterns
    │   ├── analysis_deep_agent.py   # Analysis agent
    │   ├── coding_deep_agent.py     # Coding agent
    │   └── deep_orchestrator.py     # Official orchestrator
    ├── tools/                       # Official tools
    ├── config/                      # Official configuration
    ├── demo.py                      # Official demo
    └── example_usage.py             # Usage examples
```

## Quick Start

> **Primary Entry Point:** `cd deep_agents_custom && streamlit run app.py`  
> The main production-ready implementation is in the `deep_agents_custom/` directory.

### Installation
```bash
# Clone repository
git clone https://github.com/LiteObject/deep-search-with-agents.git
cd deep-search-with-agents

# Create virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Set up environment variables for enhanced features
cp .env.example .env
# Edit .env with your configuration (OLLAMA_BASE_URL, OLLAMA_MODEL, TAVILY_API_KEY)
```

### Running the Application

#### **Custom Implementation**
The custom implementation provides lightweight, fast performance without heavy dependencies:

**Web Interface (Recommended)**
```bash
cd deep_agents_custom
streamlit run app.py
```
*Uses: `deep_agents_custom/` - Custom implementation*  
Access at: http://localhost:8501

**Command Line Interface**
```bash
cd deep_agents_custom

# Basic search with auto-agent selection
python main.py search "artificial intelligence trends 2024"

# Use specific agent
python main.py search "machine learning research" --agent research
python main.py search "tech news today" --agent news  
python main.py search "python programming" --agent general

# Multi-agent comparison
python main.py multi "quantum computing" --agents research news general

# Comprehensive search with all agents
python main.py comprehensive "renewable energy technologies"

# Show system capabilities
python main.py capabilities
```

#### **Alternative Implementations**

**LangChain Implementation (Advanced Features)**
```bash
cd deep_agents_with_langchain
streamlit run app.py
# OR
python main.py search "your query here"
```
*Features: Reflection loops, hierarchical planning, advanced coordination*

**Official DeepAgents Implementation**
```bash
cd deep_agents_official
python demo.py
# OR
python example_usage.py
```
*Features: Official package patterns, community standards*

**Test Basic Functionality**
```bash
cd deep_agents_custom
python basic_test.py
```

## Usage Examples

> **Important:** All Python code examples below assume you're either:
> 1. **Running from within the `deep_agents_custom` directory** (recommended for CLI/scripts)
> 2. **Using the path manipulation shown** (for standalone scripts)
> 3. **Installing as a package** (for advanced users)

### Basic Search with Auto-Agent Selection
```python
# Import from the deep_agents_custom implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_agents_custom'))
from agents.search_orchestrator import SearchOrchestrator

# OR if running from within deep_agents_custom directory:
# from agents.search_orchestrator import SearchOrchestrator

# Initialize orchestrator
orchestrator = SearchOrchestrator()

# Automatic agent selection based on query
result = orchestrator.search("artificial intelligence trends 2024")

# Access SearchSummary attributes
print(f"Query: {result.query}")
print(f"Summary: {result.summary}")
print(f"Key Points: {result.key_points}")
print(f"Sources: {result.sources}")
print(f"Total Results: {result.total_results}")
print(f"Search Time: {result.search_time:.2f}s")
```

### Using Specific Agents
```python
# Import from the deep_agents_custom implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_agents_custom'))
from agents.research_agent import ResearchAgent
from agents.news_agent import NewsAgent
from agents.general_agent import GeneralAgent

# OR if running from within deep_agents_custom directory:
# from agents.research_agent import ResearchAgent
# from agents.news_agent import NewsAgent
# from agents.general_agent import GeneralAgent

# Academic research (papers, studies, scholarly content)
research_agent = ResearchAgent()
research_result = research_agent.search("machine learning transformer architecture")
print(f"Research Summary: {research_result.summary}")

# Current news and events
news_agent = NewsAgent()
news_result = news_agent.search("AI regulation updates 2024")
print(f"News Summary: {news_result.summary}")

# General web search
general_agent = GeneralAgent()
general_result = general_agent.search("how to learn python programming")
print(f"General Summary: {general_result.summary}")
```

### Multi-Agent Comparison
```python
# Import from the deep_agents_custom implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_agents_custom'))
from agents.search_orchestrator import SearchOrchestrator, SearchType

# OR if running from within deep_agents_custom directory:
# from agents.search_orchestrator import SearchOrchestrator, SearchType

orchestrator = SearchOrchestrator()

# Compare results from multiple agents
results = orchestrator.multi_agent_search(
    query="renewable energy",
    agents=[SearchType.RESEARCH, SearchType.NEWS, SearchType.GENERAL],
    max_results_per_agent=5
)

# results is a dictionary with agent results
for agent_type, agent_result in results.items():
    print(f"\n{agent_type.value.upper()} Agent:")
    print(f"Summary: {agent_result.summary}")
    print(f"Sources: {len(agent_result.sources)}")
```

### Comprehensive Search
```python
# Import from the deep_agents_custom implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_agents_custom'))
from agents.search_orchestrator import SearchOrchestrator

# OR if running from within deep_agents_custom directory:
# from agents.search_orchestrator import SearchOrchestrator

orchestrator = SearchOrchestrator()

# Deep analysis with all agents
results = orchestrator.comprehensive_search(
    query="climate change solutions",
    max_results_per_agent=8
)

# Access comprehensive results
print(f"Overall Summary: {results.get('summary', 'N/A')}")
print(f"Combined Key Points: {results.get('key_points', [])}")
print(f"All Sources: {results.get('all_sources', [])}")

# Individual agent results
agent_results = results.get('agent_results', {})
for agent_name, result in agent_results.items():
    print(f"\n{agent_name} Focus:")
    print(f"  {result.summary}")
```

## Core Agents

### Research Agent
**Specialization**: Academic papers, studies, scholarly content, research methodologies
- Prioritizes peer-reviewed sources
- Focuses on scientific and academic content
- Enhanced keyword matching for research terms

### News Agent  
**Specialization**: Current events, breaking news, recent developments
- Temporal relevance scoring (prioritizes recent content)
- News source credibility ranking
- Real-time event tracking

### General Agent
**Specialization**: Comprehensive web search across all content types
- Broad domain coverage
- Balanced source diversity
- General-purpose optimization

### Search Orchestrator
**Capabilities**: Intelligent agent selection and coordination
- Auto-detects query type and selects optimal agent
- Multi-agent comparison and synthesis
- Performance analytics and source overlap analysis

## Search Engines & Tools

### Search Engines
- **DuckDuckGo**: Privacy-focused web search (default, no API key required)
- **Tavily**: Professional search API with enhanced results (requires API key)
- **Wikipedia**: Encyclopedia and reference content (no API key required)

### AI Summarization
- **EnhancedLLMSummarizer**: Advanced AI-powered summarization with multiple LLM provider support
- **Smart Provider Selection**: Automatically chooses best available LLM (Ollama, OpenAI, Anthropic)
- **Fallback Summarization**: Graceful degradation when LLMs are unavailable

### Key Classes
```python
# Core data structures
@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    source: str
    timestamp: datetime
    relevance_score: float = 0.0

@dataclass  
class SearchSummary:
    query: str
    summary: str
    key_points: List[str]
    sources: List[str]
    total_results: int
    search_time: float
```

## Configuration

### Environment Variables (Optional)
```bash
# .env file (copy from .env.example)
OLLAMA_BASE_URL=http://localhost:11434      # Ollama server URL
OLLAMA_MODEL=gpt-oss:latest                 # Ollama model for AI summarization
OPENAI_API_KEY=your_openai_key_here         # OpenAI API key (optional)
ANTHROPIC_API_KEY=your_anthropic_key_here   # Anthropic API key (optional)
TAVILY_API_KEY=your_tavily_api_key_here     # For enhanced search results
LOG_LEVEL=INFO                              # Logging level (DEBUG, INFO, WARNING, ERROR)
```

### Configuration Features
- **Enhanced LLM Integration**: Advanced adapter system with multiple provider support
- **Graceful Fallbacks**: System works without external LLMs using built-in summarization
- **Multiple Search Engines**: Configurable search engine selection and priority
- **Automatic Provider Selection**: Intelligent LLM provider detection and fallback
- **Local AI Processing**: Uses Ollama for private, local AI summarization
- **Customizable Agents**: Easy to modify agent behavior and specializations
- **Logging**: Comprehensive logging with configurable levels
- **Settings Validation**: Automatic configuration validation on startup

### LLM Provider Support

The system includes an enhanced LLM adapter system that supports multiple providers:

- **Ollama** (Default): Local, private, offline AI processing
- **OpenAI**: Cloud-based GPT models (API key required)  
- **Anthropic**: Cloud-based Claude models (API key required)

**Easy Provider Switching:**
```python
from llm import get_llm, get_best_llm
from tools.enhanced_summarizer import EnhancedLLMSummarizer

# Auto-select best available provider
llm = get_best_llm()
summarizer = EnhancedLLMSummarizer()

# Or specify a provider
ollama_llm = get_llm("ollama", "gpt-oss:latest")
openai_llm = get_llm("openai", "gpt-4")
anthropic_llm = get_llm("anthropic", "claude-3-sonnet-20240229")

# Provider-specific summarizer
summarizer = EnhancedLLMSummarizer(provider="ollama")
```

## Implementation Choices & Deep Agents Explained

This project offers **three distinct implementations** of deep search agents, each designed for different use cases and complexity levels. Here's a detailed comparison:

### 1. **Custom Implementation** (`/agents`, `/tools`, `/config`, `/utils`)
**Philosophy**: Lightweight, fast, production-ready
**Best for**: Production deployments, direct control, maximum performance

**Key Characteristics:**
- **Zero Dependencies**: No LangChain or heavy frameworks required
- **Direct API Control**: Raw integrations with search engines and LLMs
- **Fast Startup**: Minimal overhead, quick initialization
- **Production Optimized**: Clean imports, no path manipulation
- **Enhanced Features**: Dynamic year detection, comprehensive error handling
- **Privacy-First**: Works completely offline with DuckDuckGo
- **Resource Efficient**: Low memory footprint, fast execution
- **Latest Architecture**: Enhanced LLM adapter system with multiple provider support

**What makes it "Deep":**
- Multi-agent orchestration with intelligent agent selection
- Concurrent multi-source search (DuckDuckGo, Tavily, Wikipedia)
- Intelligent result ranking and deduplication
- Context-aware summarization with key insight extraction
- Temporal relevance scoring for news content

**Entry point**: `cd deep_agents_custom && streamlit run app.py`

### 2. **LangChain Implementation** (`/deep_agents_with_langchain`)
**Philosophy**: Advanced AI capabilities, experimental features
**Best for**: Research, experimentation, complex AI workflows

**Key Characteristics:**
- **Advanced Planning**: Hierarchical task decomposition
- **Reflection Loops**: Self-evaluating and improving search strategies
- **Sophisticated Coordination**: Multi-agent collaboration patterns
- **Rich Tooling**: Built-in LangChain tools and integrations
- **Experimental Features**: Cutting-edge AI capabilities
- **Framework Integration**: Leverages LangChain ecosystem

**What makes it "Deep":**
- **Meta-reasoning**: Agents that reason about their own reasoning
- **Reflection mechanisms**: Self-evaluation and strategy adjustment
- **Hierarchical planning**: Breaking complex queries into sub-tasks
- **Advanced memory**: Conversation and search history retention
- **Tool composition**: Dynamic tool selection and chaining

**Entry point**: `cd deep_agents_with_langchain && streamlit run app.py`

### 3. **Official DeepAgents** (`/deep_agents_official`)
**Philosophy**: Community standards, established patterns
**Best for**: Learning, following best practices, compatibility

**Key Characteristics:**
- **Package Standards**: Uses official deepagents conventions
- **Virtual File System**: Structured data management
- **TodoWrite Planning**: Systematic task planning tools
- **Community Patterns**: Follows established architectural patterns
- **Ecosystem Compatibility**: Works with other deepagents tools

**What makes it "Deep":**
- **Structured planning**: TodoWrite-based task decomposition
- **Virtual file management**: Organized data handling
- **Standard interfaces**: Compatible with deepagents ecosystem
- **Best practices**: Implements community-established patterns

**Entry point**: `cd deep_agents_official && python demo.py`

---

### **"Deep" vs Traditional Search - What's the Difference?**

- **Traditional Search**: Single query → Single engine → Raw results
- **Deep Search Agents**: Multi-step reasoning → Multi-source → Intelligent synthesis

**Deep Agent Capabilities:**
1. **Multi-Agent Orchestration**: Different agents for different domains
2. **Intelligent Source Selection**: Chooses optimal search engines per query
3. **Concurrent Processing**: Parallel searches across multiple sources
4. **Context Understanding**: Interprets query intent and selects appropriate agent
5. **Result Synthesis**: Combines and summarizes information from multiple sources
6. **Temporal Intelligence**: Understands time-sensitive queries (news vs research)
7. **Quality Ranking**: Intelligent relevance scoring and deduplication

### **Which Implementation Should You Choose?**

| Use Case | Recommended Implementation | Why |
|----------|---------------------------|-----|
| **Production Deployment** | Custom Implementation | Fast, reliable, no heavy dependencies |
| **Research & Experimentation** | LangChain Implementation | Advanced AI features, reflection capabilities |
| **Learning Deep Agents** | Official Implementation | Standard patterns, community best practices |
| **Quick Start/Demo** | Custom Implementation | Fastest setup, works out of the box |
| **Complex AI Workflows** | LangChain Implementation | Hierarchical planning, meta-reasoning |
| **Community Standards** | Official Implementation | Uses official deepagents package conventions, virtual file system integration, TodoWrite planning tools |

---

## Testing & Development

### Run Tests
```bash
# Basic functionality test (main implementation)
cd deep_agents_custom
python basic_test.py

# Test import functionality
cd deep_agents_custom
python -c "from agents.search_orchestrator import SearchOrchestrator; print('✅ Imports working')"

# Note: pytest tests would be run from the respective implementation directories
# if test files exist in the future
```

### Development Setup
```bash
# Install development tools
pip install pylint black flake8 pytest

# Run code quality checks
pylint agents/ tools/ config/ utils/ *.py

# Format code
black agents/ tools/ config/ utils/ *.py
```

### Code Quality Standards
- **Pylint**: Maintaining 10.00/10 scores across all modules
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Detailed docstrings for all public methods
- **Error Handling**: Graceful degradation and comprehensive logging

## Troubleshooting

### Common Issues & Solutions

#### **ImportError: attempted relative import beyond top-level package**
If you get this error when running CLI commands:

**Solution**: Set PYTHONPATH when running CLI:
```bash
# Windows PowerShell
cd deep_agents_custom
$env:PYTHONPATH="."; python main.py search "your query"

# Windows Command Prompt
cd deep_agents_custom
set PYTHONPATH=. && python main.py search "your query"

# Linux/Mac
cd deep_agents_custom
PYTHONPATH=. python main.py search "your query"
```

#### **"ddgs library not available" or "WebSearchManager not available"**
This indicates missing search dependencies.

**Solution**: Install missing packages:
```bash
pip install ddgs tavily-python anthropic
# Or reinstall all requirements
pip install -r requirements.txt
```

#### **"Could not connect to Ollama" Warning**
This is expected if you don't have Ollama running locally.

**Options**:
1. **Install and run Ollama** (recommended for full AI features):
   ```bash
   # Download from https://ollama.ai
   ollama serve
   ollama pull gpt-oss:latest
   ```

2. **Use OpenAI/Anthropic instead**:
   ```bash
   # Set API keys in .env file
   OPENAI_API_KEY=your_key_here
   # OR
   ANTHROPIC_API_KEY=your_key_here
   ```

3. **Continue without AI summarization** (basic search still works)

#### **"No results found" or Empty Search Results**
**Potential causes**:
- Network connectivity issues
- Search engine rate limiting
- Missing API keys for enhanced search

**Solutions**:
1. **Check internet connection**
2. **Wait a moment and retry** (rate limiting)
3. **Try different search engines**:
   ```bash
   python main.py search "query" --engines duckduckgo wikipedia
   ```
4. **Add Tavily API key** for enhanced results:
   ```bash
   # In .env file
   TAVILY_API_KEY=your_tavily_key_here
   ```

#### **Streamlit App Not Loading**
**Solution**:
```bash
cd deep_agents_custom
streamlit run app.py --server.port 8501 --server.address localhost
```

#### **Virtual Environment Issues**
**Solution**: Recreate virtual environment:
```bash
# Remove existing environment
rmdir .venv /s  # Windows
rm -rf .venv    # Linux/Mac

# Create new environment
python -m venv .venv
# Activate
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Performance Tips

1. **Use specific agents** for better performance:
   ```bash
   python main.py search "academic topic" --agent research
   python main.py search "current news" --agent news
   ```

2. **Limit results** for faster searches:
   ```bash
   python main.py search "query" --max-results 5
   ```

3. **Use local Ollama** for private AI processing:
   ```bash
   # Better than cloud APIs for privacy and speed
   ollama serve
   ```

4. **Configure engines** for optimal results:
   ```bash
   # In .env file
   TAVILY_API_KEY=your_key  # For enhanced search
   OLLAMA_MODEL=gpt-oss:latest  # For local AI
   ```

### Debug Mode

Enable detailed logging for troubleshooting:
```bash
# In .env file
LOG_LEVEL=DEBUG

# Or set environment variable
export LOG_LEVEL=DEBUG  # Linux/Mac
$env:LOG_LEVEL="DEBUG"  # Windows PowerShell
```

## Privacy & Security

- **Privacy-First**: Uses DuckDuckGo by default (no user tracking)
- **No Data Persistence**: Search results are not permanently stored
- **Optional APIs**: Core functionality works without external API keys
- **Configurable Data Flow**: Control what information is sent to external services
- **Local Processing**: AI processing can be done locally with fallbacks

## Performance Features

- **Concurrent Processing**: Multiple searches executed in parallel
- **Smart Caching**: Avoids duplicate requests and API calls
- **Intelligent Ranking**: Relevance scoring and result deduplication
- **Graceful Fallbacks**: Continues working when external services are unavailable
- **Analytics**: Performance metrics and source overlap analysis
- **Optimized Parsing**: Efficient content extraction and processing

## License

MIT License - see LICENSE file for details.

---

## Quick Commands Reference

### **Custom Implementation (Primary)**
```bash
# Web interface - production-ready custom implementation
cd deep_agents_custom
streamlit run app.py

# CLI examples - custom implementation
cd deep_agents_custom
python main.py search "your query here"
python main.py search "research topic" --agent research
python main.py multi "topic" --agents research news general
python main.py comprehensive "deep analysis topic"
python main.py capabilities

# Test custom implementation
cd deep_agents_custom
python -c "from agents.search_orchestrator import SearchOrchestrator; print('Test successful')"
```

### **Alternative Implementations**
```bash
# LangChain implementation (advanced features)
cd deep_agents_with_langchain
streamlit run app.py                    # LangChain web interface
python main.py search "query"           # LangChain CLI

# Official patterns implementation
cd deep_agents_official
python demo.py                          # Official demo
python example_usage.py                 # Official examples
```

## **Implementation Summary**

| Command | Implementation Used | Features |
|---------|-------------------|----------|
| `cd deep_agents_custom && streamlit run app.py` | **Custom Primary** | Production-ready, optimized, enhanced features |
| `cd deep_agents_custom && python main.py` | **Custom Primary** | Fast, stable, dynamic year detection |
| `cd deep_agents_with_langchain && streamlit run app.py` | **LangChain** | Advanced AI, reflection loops |
| `cd deep_agents_official && python demo.py` | **Official** | Package standards |
