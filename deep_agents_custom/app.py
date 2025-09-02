"""
Streamlit web interface for the Deep Search Agents application.
"""

import time
from typing import Any, Dict, List

import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore  # pylint: disable=import-error
import streamlit as st  # type: ignore

from agents.search_orchestrator import SearchOrchestrator, SearchType  # type: ignore # pylint: disable=import-error
from config.settings import settings  # type: ignore # pylint: disable=import-error
from utils.logger import setup_logging, get_logger  # type: ignore # pylint: disable=import-error
from utils.helpers import format_search_time  # type: ignore # pylint: disable=import-error

# Configure page
st.set_page_config(
    page_title="Deep Search Agents",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging (don't log to file in Streamlit)
setup_logging(settings.LOG_LEVEL, log_to_file=False)
logger = get_logger(__name__)

# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = SearchOrchestrator()


def main():
    """Main Streamlit application"""

    # Header
    st.title("🔬 Deep Research Agents")
    st.markdown("*AI-powered research agents for comprehensive topic analysis*")

    # Research description
    st.info("""
    **How it works:** Enter a research topic and our specialized AI agents will:
    • Search multiple sources for relevant information
    • Analyze and synthesize findings from various databases
    • Generate a comprehensive 250-word summary
    • Provide reference links for further reading
    """)

    # Sidebar
    create_sidebar()

    # Main content
    create_main_interface()

    # Search history
    if st.session_state.search_history:
        create_search_history()


def create_sidebar():
    """Create sidebar with configuration and information"""

    st.sidebar.header("⚙️ Configuration")

    # Search mode selection
    search_mode = st.sidebar.selectbox(
        "Search Mode:",
        ["Auto-Select Agent", "Single Agent", "Multi-Agent", "Comprehensive"],
        help="Choose how to perform the search"
    )

    # Agent selection (for single agent mode)
    selected_agent = None
    if search_mode == "Single Agent":
        selected_agent = st.sidebar.selectbox(
            "Select Agent:",
            ["Research", "News", "General"],
            help="Choose which specialized agent to use"
        )

    # Multi-agent selection (for multi-agent mode)
    selected_agents = None
    if search_mode == "Multi-Agent":
        selected_agents = st.sidebar.multiselect(
            "Select Agents:",
            ["Research", "News", "General"],
            default=["Research", "News", "General"],
            help="Choose which agents to use"
        )

    # Store in session state
    st.session_state.search_mode = search_mode
    st.session_state.selected_agent = selected_agent
    st.session_state.selected_agents = selected_agents

    # Advanced options
    st.sidebar.header("🔧 Advanced Options")

    max_results = st.sidebar.slider(
        "Max Results per Agent:",
        min_value=5,
        max_value=50,
        value=10,
        help="Maximum number of results per agent"
    )
    st.session_state.max_results = max_results

    # Information section
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("""
    **Deep Research Agents** uses specialized AI agents for comprehensive research:

    - 🔬 **Research Agent**: Academic papers, scientific studies
    - 📰 **News Agent**: Current events, recent developments
    - 🌐 **General Agent**: Comprehensive web research

    Get 250-word research summaries with reference links for any topic.
    """)

    # Agent capabilities
    if st.sidebar.expander("View Agent Capabilities"):
        capabilities = st.session_state.orchestrator.get_agent_capabilities()
        for agent_name, caps in capabilities.items():
            st.sidebar.write(f"**{agent_name.title()}**")
            st.sidebar.write(f"- {caps['description']}")
            st.sidebar.write(
                f"- Supports: {', '.join(caps['supported_queries'][:3])}...")


def create_main_interface():
    """Create main search interface"""

    # Search input with proper alignment
    with st.container():
        st.markdown("**Research Topic:**")
        col1, col2 = st.columns([4, 1])

        with col1:
            query = st.text_input(
                "search_input",
                placeholder="e.g., quantum computing breakthroughs, sustainable energy solutions",
                key="search_query",
                label_visibility="collapsed"
            )

        with col2:
            search_button = st.button(
                "🔍 Research", type="primary", use_container_width=True)

    # Example research topics
    st.markdown("""
    **Example research topics:**
    - 🧬 CRISPR gene editing latest developments
    - 🌱 Renewable energy storage technologies
    - 🤖 Machine learning in healthcare applications
    - � Space exploration recent missions
    - 🧠 Neuroplasticity and brain research
    - 🌊 Ocean plastic pollution solutions
    """)

    # Perform search
    if search_button and query:
        perform_search(query)
    elif search_button and not query:
        st.warning("Please enter a research topic.")


def perform_search(query: str):
    """Perform research based on selected mode"""

    search_mode = st.session_state.search_mode

    with st.spinner(f"🔍 Researching {query} using {search_mode.lower()}..."):
        start_time = time.time()

        try:
            if search_mode == "Auto-Select Agent":
                result = st.session_state.orchestrator.search(
                    query, SearchType.AUTO)
                display_single_result(result, "Auto-Selected Research Agent")

            elif search_mode == "Single Agent":
                agent_map = {
                    "Research": SearchType.RESEARCH,
                    "News": SearchType.NEWS,
                    "General": SearchType.GENERAL
                }
                search_type = agent_map[st.session_state.selected_agent]
                result = st.session_state.orchestrator.search(
                    query, search_type)
                display_single_result(
                    result, f"{st.session_state.selected_agent} Research Agent")

            elif search_mode == "Multi-Agent":
                agent_map = {
                    "Research": SearchType.RESEARCH,
                    "News": SearchType.NEWS,
                    "General": SearchType.GENERAL
                }
                agent_types = [agent_map[agent]
                               for agent in st.session_state.selected_agents]
                results = st.session_state.orchestrator.multi_agent_search(
                    query, agent_types)
                display_multi_results(results)

            elif search_mode == "Comprehensive":
                results = st.session_state.orchestrator.comprehensive_search(
                    query)
                display_comprehensive_results(results)

            # Add to search history
            search_time = time.time() - start_time
            st.session_state.search_history.append({
                'query': query,
                'mode': search_mode,
                'time': search_time,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })

        except (ValueError, ConnectionError) as e:
            st.error(f"Search failed: {str(e)}")
            logger.error("Search error: %s", e)


def display_single_result(result, agent_name: str):
    """Display single agent search result with research focus"""

    st.header(f"🔬 Research Results: {result.query}")

    # Research metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Sources Found", result.total_results)
    with col2:
        st.metric("⏱️ Research Time", format_search_time(result.search_time))
    with col3:
        st.metric("🔍 Agent Used", agent_name)

    # Main research summary (250 words)
    st.subheader("📋 Research Summary")

    # Generate or display a research-focused summary
    if hasattr(result, 'summary') and result.summary:
        # Ensure summary is approximately 250 words
        words = result.summary.split()
        if len(words) > 280:
            # Truncate to approximately 250 words and add proper ending
            summary_250 = ' '.join(words[:250])
            # Find the last complete sentence
            last_period = summary_250.rfind('.')
            if last_period > 0:
                summary_250 = summary_250[:last_period + 1]
            else:
                summary_250 += "..."
        else:
            summary_250 = result.summary

        st.write(summary_250)

        # Word count display
        word_count = len(summary_250.split())
        st.caption(f"📝 Summary: {word_count} words")
    else:
        st.warning("No research summary available for this topic.")

    # Key insights (if available)
    if hasattr(result, 'key_points') and result.key_points:
        st.subheader("🔍 Key Research Findings")
        for i, point in enumerate(result.key_points[:5], 1):
            st.write(f"• {point}")

    # Reference links for further reading
    st.subheader("🔗 Reference Links for Further Information")

    if hasattr(result, 'results') and result.results:
        # Display top sources with titles and links
        st.write("**Primary Sources:**")
        for i, source in enumerate(result.results[:8], 1):
            if hasattr(source, 'url') and hasattr(source, 'title'):
                st.markdown(f"{i}. **[{source.title}]({source.url})**")
                if hasattr(source, 'source'):
                    st.caption(f"   Source: {source.source}")
            elif hasattr(source, 'url'):
                st.markdown(f"{i}. **[View Source]({source.url})**")

    elif hasattr(result, 'sources') and result.sources:
        # Fallback to basic source links
        st.write("**Source Links:**")
        for i, source_url in enumerate(result.sources[:8], 1):
            st.markdown(f"{i}. [Research Source {i}]({source_url})")

    else:
        st.info("No reference links available for this research topic.")

    # Research completion indicator
    st.success("✅ Research completed successfully!")


def display_multi_results(results: Dict[str, Any]):
    """Display multi-agent search results"""

    st.header("🤖 Multi-Agent Search Results")

    # Overview metrics
    total_results = sum(result.total_results for result in results.values())
    total_time = sum(result.search_time for result in results.values())
    unique_sources = set()
    for result in results.values():
        unique_sources.update(result.sources)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Results", total_results)
    with col2:
        st.metric("Total Time", format_search_time(total_time))
    with col3:
        st.metric("Unique Sources", len(unique_sources))
    with col4:
        st.metric("Agents Used", len(results))

    # Individual agent results
    for agent_name, result in results.items():
        expander = st.expander(
            f"🔍 {agent_name.title()} Agent Results", expanded=True)
        with expander:
            display_single_result(result, f"{agent_name.title()} Agent")


def display_comprehensive_results(results: Dict[str, Any]):
    """Display comprehensive search results with analysis"""

    st.header("🎯 Comprehensive Search Results")

    # Overall metrics
    analysis = results['analysis']
    _display_overall_metrics(analysis, results['total_search_time'])

    # Analysis chart
    if len(results['agent_results']) > 1:
        _display_performance_analysis(analysis)

    # Combined insights
    if analysis['combined_insights']:
        st.subheader("🎯 Combined Key Insights")
        for i, insight in enumerate(analysis['combined_insights'], 1):
            st.write(f"{i}. {insight}")

    # Individual agent results
    _display_agent_tabs(results['agent_results'])


def _display_overall_metrics(analysis: Dict[str, Any], total_search_time: float):
    """Display overall metrics for comprehensive results"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sources", analysis['total_sources'])
    with col2:
        st.metric("Unique Sources", analysis['unique_source_count'])
    with col3:
        st.metric("Search Time", format_search_time(total_search_time))
    with col4:
        overlap_pct = analysis['source_overlap_ratio'] * 100
        st.metric("Source Overlap", f"{overlap_pct:.1f}%")


def _display_performance_analysis(analysis: Dict[str, Any]):
    """Display performance analysis charts"""
    st.subheader("📊 Agent Performance Analysis")

    # Create performance data
    perf_data = _create_performance_data(analysis['agent_performance'])
    df = pd.DataFrame(perf_data)

    # Create charts
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(df, x='Agent', y='Results', title='Results by Agent')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(df, x='Agent', y='Time (s)',
                      title='Search Time by Agent')
        st.plotly_chart(fig2, use_container_width=True)


def _create_performance_data(agent_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create performance data for charts"""
    perf_data = []
    for agent_name, perf in agent_performance.items():
        perf_data.append({
            'Agent': agent_name.title(),
            'Results': perf['result_count'],
            'Time (s)': perf['search_time'],
            'Results/Time': (
                perf['result_count'] / max(perf['search_time'], 0.1))
        })
    return perf_data


def _display_agent_tabs(agent_results: Dict[str, Any]):
    """Display individual agent results in tabs"""
    st.subheader("🤖 Detailed Agent Results")

    tabs = st.tabs([f"{name.title()} Agent"
                   for name in agent_results.keys()])

    for tab, (agent_name, result) in zip(tabs, agent_results.items()):
        with tab:
            display_single_result(result, f"{agent_name.title()} Agent")


def create_search_history():
    """Create search history section"""

    st.header("📊 Search History")

    # History metrics
    total_searches = len(st.session_state.search_history)
    avg_time = sum(h['time']
                   for h in st.session_state.search_history) / total_searches

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Searches", total_searches)
    with col2:
        st.metric("Average Time", format_search_time(avg_time))
    with col3:
        if st.button("Clear History"):
            st.session_state.search_history = []
            st.rerun()

    # History table
    if st.session_state.search_history:
        history_df = pd.DataFrame(st.session_state.search_history)
        history_df['Search Time'] = history_df['time'].apply(
            format_search_time)

        st.dataframe(
            history_df[['timestamp', 'query', 'mode', 'Search Time']],
            column_config={
                'timestamp': 'Timestamp',
                'query': 'Query',
                'mode': 'Search Mode',
                'Search Time': 'Duration'
            },
            use_container_width=True
        )


if __name__ == "__main__":
    main()
