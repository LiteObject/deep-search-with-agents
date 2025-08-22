"""
Search Orchestrator - Manages and coordinates multiple search agents.
"""

import time
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

from agents.base_agent import SearchSummary  # pylint: disable=import-error
from agents.research_agent import ResearchAgent  # pylint: disable=import-error
from agents.news_agent import NewsAgent  # pylint: disable=import-error
from agents.general_agent import GeneralAgent  # pylint: disable=import-error

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Enum for different search types"""
    RESEARCH = "research"
    NEWS = "news"
    GENERAL = "general"
    AUTO = "auto"


class SearchOrchestrator:
    """
    Orchestrates searches across multiple specialized agents.
    Automatically determines the best agent for a query or allows manual
    selection.
    """

    def __init__(self):
        """Initialize the search orchestrator with all agents"""
        self.agents = {
            SearchType.RESEARCH: ResearchAgent(),
            SearchType.NEWS: NewsAgent(),
            SearchType.GENERAL: GeneralAgent()
        }

        # Keywords for automatic agent selection
        self.agent_keywords = {
            SearchType.RESEARCH: [
                'research', 'study', 'academic', 'paper', 'journal',
                'methodology', 'analysis', 'peer-reviewed', 'scholar',
                'literature', 'findings', 'scientific', 'university'
            ],
            SearchType.NEWS: [
                'news', 'breaking', 'latest', 'update', 'current',
                'today', 'recent', 'happening', 'announced', 'report',
                'development', 'politics', 'election', 'government'
            ]
        }

    def search(
        self, query: str, search_type: SearchType = SearchType.AUTO, **kwargs
    ) -> SearchSummary:
        """
        Perform search using appropriate agent

        Args:
            query: Search query
            search_type: Type of search (auto-detects if AUTO)
            **kwargs: Additional parameters for specific agents

        Returns:
            SearchSummary: Search results from appropriate agent
        """
        # Determine which agent to use
        if search_type == SearchType.AUTO:
            search_type = self._detect_search_type(query)

        logger.info("Using %s agent for query: %s", search_type.value, query)

        # Get the appropriate agent and perform search
        agent = self.agents[search_type]
        return agent.search(query, **kwargs)

    def multi_agent_search(
        self, query: str, agents: Optional[List[SearchType]] = None
    ) -> Dict[str, SearchSummary]:
        """
        Search using multiple agents and combine results

        Args:
            query: Search query
            agents: List of agents to use (default: all)

        Returns:
            Dict[str, SearchSummary]: Results from each agent
        """
        if agents is None:
            agents = [SearchType.RESEARCH, SearchType.NEWS, SearchType.GENERAL]

        logger.info("Starting multi-agent search for query: %s", query)
        results = {}

        for agent_type in agents:
            if agent_type in self.agents:
                try:
                    result = self.agents[agent_type].search(query)
                    results[agent_type.value] = result
                except (ConnectionError, TimeoutError, RuntimeError) as e:
                    logger.error("Error with %s agent: %s",
                                 agent_type.value, str(e))

        return results

    def comprehensive_search(self, query: str) -> Dict[str, Any]:
        """
        Perform comprehensive search across all agents with analysis

        Args:
            query: Search query

        Returns:
            Dict[str, Any]: Comprehensive search results with analysis
        """
        start_time = time.time()

        # Get results from all agents
        agent_results = self.multi_agent_search(query)

        # Analyze and combine results
        analysis = self._analyze_multi_agent_results(agent_results)

        total_time = time.time() - start_time

        return {
            'query': query,
            'agent_results': agent_results,
            'analysis': analysis,
            'total_search_time': total_time,
            'agents_used': list(agent_results.keys())
        }

    def _detect_search_type(self, query: str) -> SearchType:
        """
        Automatically detect the best search type for a query

        Args:
            query: Search query

        Returns:
            SearchType: Detected search type
        """
        query_lower = query.lower()

        # Score each agent type
        scores = {}

        for agent_type, keywords in self.agent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            scores[agent_type] = score

        # Find the agent type with highest score
        if scores:
            best_agent = max(scores.keys(), key=lambda x: scores[x])
            if scores[best_agent] > 0:
                return best_agent

        # Default to general search
        return SearchType.GENERAL

    def _analyze_multi_agent_results(
        self, results: Dict[str, SearchSummary]
    ) -> Dict[str, Any]:
        """
        Analyze results from multiple agents

        Args:
            results: Results from different agents
            query: Original query

        Returns:
            Dict[str, Any]: Analysis of combined results
        """
        analysis = {
            'total_sources': 0,
            'unique_sources': set(),
            'agent_performance': {},
            'combined_insights': [],
            'source_overlap': {},
            'best_performing_agent': None
        }

        # Analyze each agent's results
        for agent_name, result in results.items():
            agent_analysis = {
                'result_count': result.total_results,
                'search_time': result.search_time,
                'source_count': len(result.sources),
                'avg_time_per_result': (
                    result.search_time / max(result.total_results, 1))
            }

            analysis['agent_performance'][agent_name] = agent_analysis
            analysis['total_sources'] += len(result.sources)
            analysis['unique_sources'].update(result.sources)
            analysis['combined_insights'].extend(result.key_points)

        # Find best performing agent (by results per time)
        if analysis['agent_performance']:
            best_agent = max(
                analysis['agent_performance'].items(),
                key=lambda x: x[1]['result_count'] /
                max(x[1]['search_time'], 0.1)
            )
            analysis['best_performing_agent'] = best_agent[0]

        # Calculate source overlap
        analysis['unique_source_count'] = len(analysis['unique_sources'])
        total_sources = analysis['total_sources']
        unique_sources = analysis['unique_source_count']
        analysis['source_overlap_ratio'] = (
            total_sources - unique_sources) / max(total_sources, 1)

        # Deduplicate insights
        unique_insights = []
        seen_insights = set()

        for insight in analysis['combined_insights']:
            insight_lower = insight.lower().strip()
            if insight_lower not in seen_insights and len(insight) > 20:
                seen_insights.add(insight_lower)
                unique_insights.append(insight)

        # Top 10 unique insights
        analysis['combined_insights'] = unique_insights[:10]

        return analysis

    def get_agent_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """
        Get capabilities of all available agents

        Returns:
            Dict[str, Dict[str, Any]]: Agent capabilities
        """
        capabilities = {}

        for agent_type, agent in self.agents.items():
            capabilities[agent_type.value] = agent.get_capabilities()

        return capabilities

    def suggest_query_improvements(self, query: str) -> List[str]:
        """
        Suggest improvements to the search query

        Args:
            query: Original query

        Returns:
            List[str]: Suggested query improvements
        """
        suggestions = []

        # Detect search type and suggest enhancements
        search_type = self._detect_search_type(query)

        if search_type == SearchType.RESEARCH:
            suggestions.extend([
                f"{query} research study",
                f"{query} academic paper",
                f"{query} methodology analysis",
                f"{query} peer-reviewed literature"
            ])
        elif search_type == SearchType.NEWS:
            suggestions.extend([
                f"{query} latest news",
                f"{query} recent updates",
                f"{query} breaking news",
                f"{query} current developments"
            ])
        else:  # GENERAL
            suggestions.extend([
                f"{query} overview",
                f"{query} guide tutorial",
                f"{query} explanation",
                f"{query} comprehensive information"
            ])

        return suggestions[:3]  # Return top 3 suggestions
