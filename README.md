# Deep Search with Agents

A comprehensive intelligent web search system with **three distinct implementations** of AI agents that perform specialized, multi-step web searches. The system automatically selects optimal agents based on query type and provides deep insights across various domains.

## Key Features

- **Multi-Agent Architecture**: Specialized agents for research, news, and general searches
- **Three Implementation Approaches**: Choose from custom, LangChain, or official patterns
- **Multiple Search Engines**: DuckDuckGo, Tavily, Wikipedia integration
- **AI-Powered Intelligence**: LLM-based summarization and insight extraction
- **Interactive Interfaces**: Both Streamlit web app and CLI available
- **Privacy-Focused**: Works without external APIs using DuckDuckGo by default
- **High Performance**: Concurrent searches with intelligent result ranking
- **Extensible Architecture**: Easy to add new agents and search sources

## Project Structure

```
deep-search-with-agents/
├── agents/                           # Main implementation (lightweight)
│   ├── base_agent.py                # Base agent class with SearchResult/SearchSummary
│   ├── research_agent.py            # Academic/research focused agent
│   ├── news_agent.py                # News and current events agent  
│   ├── general_agent.py             # General-purpose web search agent
│   └── search_orchestrator.py       # Multi-agent coordination and auto-selection
├── tools/                           # Search and processing tools
│   ├── web_search.py                # WebSearchManager, DuckDuckGo, Tavily, Wikipedia
│   └── summarizer.py                # LLMSummarizer and SimpleSummarizer
├── config/                          # Configuration management
│   └── settings.py                  # Application settings and validation
├── utils/                           # Helper utilities
│   ├── logger.py                    # Logging configuration
│   └── helpers.py                   # Utility functions
├── deep_agents_without_langchain/   # Standalone copy of main implementation
│   ├── agents/                      # Mirror of main agents
│   ├── tools/                       # Mirror of main tools
│   ├── config/                      # Mirror of main config
│   ├── utils/                       # Mirror of main utils
│   ├── app.py                       # Standalone Streamlit app
│   └── main.py                      # Standalone CLI
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
├── deep_agents_official/            # Official DeepAgents package patterns
│   ├── agents/                      # Official implementation agents
│   │   ├── research_deep_agent.py   # Research agent using official patterns
│   │   ├── analysis_deep_agent.py   # Analysis agent
│   │   ├── coding_deep_agent.py     # Coding agent
│   │   └── deep_orchestrator.py     # Official orchestrator
│   ├── tools/                       # Official tools
│   ├── config/                      # Official configuration
│   ├── demo.py                      # Official demo
│   └── example_usage.py             # Usage examples
├── app.py                           # Main Streamlit web interface
├── main.py                          # Main CLI interface
├── basic_test.py                    # Basic functionality test
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
└── README.md                        # This documentation
```

## Quick Start

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
# Edit .env with your API keys (OPENAI_API_KEY, TAVILY_API_KEY)
```

### Running the Application

**Web Interface (Recommended)**
```bash
streamlit run app.py
```
Access at: http://localhost:8501

**Command Line Interface**
```bash
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

**Test Basic Functionality**
```bash
python basic_test.py
```

## Usage Examples

### Basic Search with Auto-Agent Selection
```python
from agents.search_orchestrator import SearchOrchestrator

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
from agents.research_agent import ResearchAgent
from agents.news_agent import NewsAgent
from agents.general_agent import GeneralAgent

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
from agents.search_orchestrator import SearchOrchestrator, SearchType

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
from agents.search_orchestrator import SearchOrchestrator

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
- **LLMSummarizer**: OpenAI GPT-powered intelligent summarization (requires API key)
- **SimpleSummarizer**: Fallback text-based summarization (no API key required)

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
OPENAI_API_KEY=your_openai_api_key_here     # For AI summarization
TAVILY_API_KEY=your_tavily_api_key_here     # For enhanced search results
LOG_LEVEL=INFO                              # Logging level (DEBUG, INFO, WARNING, ERROR)
```

### Configuration Features
- **Graceful Fallbacks**: System works without API keys using free alternatives
- **Multiple Search Engines**: Configurable search engine selection and priority
- **Customizable Agents**: Easy to modify agent behavior and specializations
- **Logging**: Comprehensive logging with configurable levels
- **Settings Validation**: Automatic configuration validation on startup

## Implementation Choices

### 1. Main Implementation (`/agents`, `/tools`, `/config`, `/utils`)
**Best for**: Production use, minimal dependencies, performance-focused
- Lightweight with no LangChain dependency
- Direct API integrations for maximum control
- Fast startup and execution
- Comprehensive error handling

### 2. LangChain Implementation (`/deep_agents_with_langchain`)
**Best for**: Advanced AI features, complex workflows, experimentation
- Advanced hierarchical planning and reflection loops
- Multi-agent collaboration with sophisticated coordination
- Built-in LangChain tools and integrations
- Experimental features and cutting-edge capabilities

### 3. Official DeepAgents (`/deep_agents_official`)
**Best for**: Following established patterns, community compatibility
- Uses official deepagents package conventions
- Virtual file system integration
- TodoWrite planning tools
- Community-standard implementations

## Testing & Development

### Run Tests
```bash
# Basic functionality test
python basic_test.py

# Test specific implementation
cd deep_agents_without_langchain
python basic_test.py

# Test with pytest (if test files exist)
python -m pytest tests/ -v
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

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Add tests** for new functionality
4. **Ensure code quality**: Run pylint and maintain 10.00/10 scores
5. **Update documentation** as needed
6. **Submit a pull request**

### Contribution Guidelines
- Follow existing code style and patterns
- Add comprehensive docstrings and type hints
- Include error handling and logging
- Write tests for new functionality
- Update README if adding new features

## License

MIT License - see LICENSE file for details.

---

## Quick Commands Reference

```bash
# Start web interface
streamlit run app.py

# CLI search examples  
python main.py search "your query here"
python main.py search "research topic" --agent research
python main.py multi "topic" --agents research news general
python main.py comprehensive "deep analysis topic"
python main.py capabilities

# Test functionality
python basic_test.py

# Alternative implementations
cd deep_agents_with_langchain && python main.py search "query"
cd deep_agents_official && python demo.py
```
