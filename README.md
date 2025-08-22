# Deep Search with Agents

A comprehensive web search system using specialized AI agents to perform intelligent, multi-step web searches on any topic. This system leverages multiple search engines and AI-powered analysis to provide deep insights across various domains.

## Features

- **Multi-Agent Search System**: Specialized agents for research, news, and general searches
- **Multiple Search Engines**: DuckDuckGo, Tavily, and Wikipedia integration
- **AI-Powered Intelligence**: LLM-based content summarization and insight extraction
- **Interactive Dashboard**: Streamlit web interface with real-time results and analytics
- **High Performance**: Concurrent searches with intelligent result ranking
- **Extensible Architecture**: Easy to add new search sources and custom agents
- **Privacy-Focused**: Works without external APIs, uses DuckDuckGo by default

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Environment Variables (Optional)
```bash
cp .env.example .env
# Edit .env with your API keys for enhanced features
```

### 3. Run the Application

**Web Interface (Recommended)**
```bash
streamlit run app.py
```

**Command Line Interface**
```bash
# Basic search with auto-agent selection
python main.py search "artificial intelligence trends 2024"

# Use specific agent
python main.py search "machine learning research" --agent research
python main.py search "tech news today" --agent news

# Multi-agent comprehensive search
python main.py comprehensive "quantum computing developments"
```

## Core Agents

### Research Agent
Specialized for academic papers, studies, and scholarly content
```python
from agents.research_agent import ResearchAgent

research_agent = ResearchAgent()
results = research_agent.search("machine learning transformer architecture")
```

### News Agent  
Focused on current events and breaking news
```python
from agents.news_agent import NewsAgent

news_agent = NewsAgent()
results = news_agent.search("AI regulation updates 2024")
```

### General Agent
Comprehensive search across all content types
```python
from agents.general_agent import GeneralAgent

general_agent = GeneralAgent()
results = general_agent.search("how to learn python programming")
```

### Search Orchestrator
Manages and coordinates multiple agents intelligently
```python
from agents.search_orchestrator import SearchOrchestrator

orchestrator = SearchOrchestrator()
results = orchestrator.search("renewable energy technologies")
print(results.summary)
```

## Project Structure

```
deep-search-with-agents/
├── agents/                 # AI agent implementations
│   ├── base_agent.py      # Base agent class and interfaces
│   ├── research_agent.py  # Academic/research focused agent
│   ├── news_agent.py      # News and current events agent
│   ├── general_agent.py   # General-purpose agent
│   └── search_orchestrator.py  # Multi-agent coordination
├── tools/                 # Search tools and utilities
│   ├── web_search.py      # Web search implementations
│   └── summarizer.py      # AI-powered summarization
├── config/                # Configuration management
│   └── settings.py        # Application settings
├── utils/                 # Helper utilities
│   ├── logger.py          # Logging configuration
│   └── helpers.py         # Utility functions
├── app.py                 # Streamlit web interface
├── main.py                # CLI interface
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This comprehensive documentation
```

## Configuration

The system supports multiple search engines and AI providers:

### Search Engines
- **DuckDuckGo**: Privacy-focused web search (default, no API key required)
- **Tavily**: Professional search API (requires API key for enhanced results)
- **Wikipedia**: Encyclopedia and reference content (no API key required)

### AI Providers
- **OpenAI**: For advanced LLM capabilities (GPT-3.5/4) - requires API key
- **Fallback Options**: Simple text-based summarization when AI is unavailable

### Environment Variables
```bash
# Optional - for enhanced features
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# System works without these - fallbacks are provided
```

## Advanced Usage Examples

### Multi-Agent Comparison
```bash
# Compare results from different agents
python main.py multi "climate change" --agents research news general
```

### Comprehensive Analysis
```bash
# Deep dive with all available agents and sources
python main.py comprehensive "artificial intelligence ethics"
```

### Programmatic Usage
```python
from agents.search_orchestrator import SearchOrchestrator
from config.settings import Settings

# Initialize with custom settings
settings = Settings()
orchestrator = SearchOrchestrator(settings)

# Perform intelligent search
results = orchestrator.search(
    query="machine learning best practices",
    max_results=10,
    include_summary=True
)

# Access structured results
print(f"Summary: {results.summary}")
print(f"Sources: {len(results.sources)}")
for source in results.sources:
    print(f"- {source.title}: {source.url}")
```

## Customization & Extension

### Adding Custom Agents
```python
from agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.name = "Custom Agent"
        self.description = "Specialized for custom domain"
    
    def _get_search_strategy(self, query: str) -> dict:
        # Implement custom search logic
        pass
```

### Adding New Search Engines
Edit `tools/web_search.py` to integrate new search providers:
```python
class CustomSearchProvider:
    def search(self, query: str, max_results: int = 5):
        # Implement custom search logic
        pass
```

### Custom Summarization
Modify `tools/summarizer.py` for domain-specific summarization:
```python
class DomainSpecificSummarizer:
    def summarize(self, content: str, context: str = None):
        # Implement specialized summarization
        pass
```

## Performance Features

- **Concurrent Processing**: Multiple searches run in parallel
- **Smart Caching**: Avoid duplicate requests and API calls
- **Result Ranking**: Intelligent relevance scoring and deduplication
- **Graceful Fallbacks**: Continues working when external services are unavailable
- **Analytics**: Performance metrics and source overlap analysis

## Privacy & Security

- **Privacy-First**: Uses DuckDuckGo by default (no tracking)
- **No Data Storage**: Results are not permanently stored
- **Optional APIs**: Core functionality works without external API keys
- **Configurable**: Control what data is sent to external services
- **Local Processing**: AI processing can be done locally with fallbacks

## Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/test_agents.py
python -m pytest tests/test_search.py
```

Basic functionality test:
```bash
python basic_test.py
```

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Add tests for new functionality**
4. **Ensure all tests pass**
   ```bash
   python -m pytest
   ```
5. **Submit a pull request**

### Development Setup
```bash
# Clone the repository
git clone https://github.com/LiteObject/deep-search-with-agents.git
cd deep-search-with-agents

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 pylint

# Run linting
pylint agents/ tools/ config/ utils/ *.py
```

## License

MIT License - see LICENSE file for details.

## Project Status

✅ **Production Ready** - Comprehensive error handling, logging, and documentation  
✅ **Highly Extensible** - Easy to customize and extend for specific needs  
✅ **Multi-Interface** - Both CLI and web interfaces ready to use  
✅ **AI-Powered** - Smart agent selection and content processing  
✅ **Privacy Focused** - Works without external APIs, respects user privacy  

---

**Ready to start intelligent searching?**

```bash
streamlit run app.py
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Usage

### Basic Search
```python
from agents.search_orchestrator import SearchOrchestrator

orchestrator = SearchOrchestrator()
results = orchestrator.search("artificial intelligence trends 2024")
print(results.summary)
```

### Advanced Search with Specific Agents
```python
from agents.research_agent import ResearchAgent
from agents.news_agent import NewsAgent

# Research academic papers
research_agent = ResearchAgent()
academic_results = research_agent.search("machine learning algorithms")

# Search for latest news
news_agent = NewsAgent()
news_results = news_agent.search("AI regulations")
```

## Project Structure

```
deep-search-with-agents/
├── agents/                 # Agent implementations
│   ├── base_agent.py      # Base agent class
│   ├── research_agent.py  # Academic/research focused agent
│   ├── news_agent.py      # News and current events agent
│   ├── general_agent.py   # General web search agent
│   └── search_orchestrator.py  # Main orchestration logic
├── tools/                 # Search tools and utilities
│   ├── web_search.py      # Web search implementations
│   ├── content_parser.py  # Content parsing utilities
│   └── summarizer.py      # AI summarization tools
├── config/                # Configuration files
│   └── settings.py        # Application settings
├── utils/                 # Utility functions
│   ├── logger.py          # Logging configuration
│   └── helpers.py         # Helper functions
├── tests/                 # Test files
├── app.py                 # Streamlit web interface
├── main.py                # CLI interface
├── requirements.txt       # Python dependencies
└── .env.example           # Environment variables template
```

## Configuration

The system supports multiple search engines and AI providers:

- **OpenAI**: For LLM capabilities (GPT-3.5/4)
- **Tavily**: Professional web search API
- **DuckDuckGo**: Free web search
- **Wikipedia**: Encyclopedia search

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
