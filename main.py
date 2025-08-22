"""
Main CLI interface for the Deep Search Agents application.
"""

import argparse
import json
import sys

from agents.search_orchestrator import SearchOrchestrator, SearchType
from config.settings import settings
from utils.helpers import create_search_summary_dict, format_search_time
from utils.logger import get_logger, setup_logging


logger = get_logger(__name__)


def main():
    """Main CLI entry point"""

    # Setup logging
    setup_logging(settings.LOG_LEVEL, log_to_file=True)

    # Validate configuration
    if not settings.validate_config():
        logger.warning(
            "Configuration validation failed. Some features may not work.")

    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Handle different commands
    if args.command == 'search':
        handle_search_command(args)
    elif args.command == 'multi':
        handle_multi_search_command(args)
    elif args.command == 'comprehensive':
        handle_comprehensive_search_command(args)
    elif args.command == 'capabilities':
        handle_capabilities_command()
    else:
        parser.print_help()


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""

    parser = argparse.ArgumentParser(
        description=(
            "Deep Search Agents - Intelligent web search using "
            "specialized AI agents"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s search "artificial intelligence trends"
  %(prog)s search "machine learning research" --agent research
  %(prog)s search "latest tech news" --agent news
  %(prog)s multi "climate change impact" --agents research news
  %(prog)s comprehensive "quantum computing developments"
  %(prog)s capabilities
        """
    )

    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')

    # Search command
    search_parser = subparsers.add_parser(
        'search', help='Perform a search using a specific agent')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument(
        '--agent', choices=['auto', 'research', 'news', 'general'],
        default='auto', help='Agent to use (default: auto)')
    search_parser.add_argument(
        '--format', choices=['text', 'json'], default='text',
        help='Output format (default: text)')
    search_parser.add_argument('--max-results', type=int, default=10,
                               help='Maximum number of results (default: 10)')

    # Multi-agent search command
    multi_parser = subparsers.add_parser(
        'multi', help='Search using multiple agents')
    multi_parser.add_argument('query', help='Search query')
    multi_parser.add_argument(
        '--agents', nargs='+', choices=['research', 'news', 'general'],
        default=['research', 'news', 'general'], help='Agents to use')
    multi_parser.add_argument(
        '--format', choices=['text', 'json'], default='text',
        help='Output format (default: text)')

    # Comprehensive search command
    comp_parser = subparsers.add_parser(
        'comprehensive',
        help='Perform comprehensive search with analysis')
    comp_parser.add_argument('query', help='Search query')
    comp_parser.add_argument(
        '--format', choices=['text', 'json'], default='text',
        help='Output format (default: text)')    # Capabilities command
    subparsers.add_parser('capabilities', help='Show agent capabilities')

    return parser


def handle_search_command(args):
    """Handle single agent search command"""

    logger.info("Performing search: %s", args.query)

    # Create orchestrator
    orchestrator = SearchOrchestrator()

    # Convert agent string to enum
    search_type = SearchType.AUTO
    if args.agent == 'research':
        search_type = SearchType.RESEARCH
    elif args.agent == 'news':
        search_type = SearchType.NEWS
    elif args.agent == 'general':
        search_type = SearchType.GENERAL

    try:
        # Perform search
        result = orchestrator.search(args.query, search_type=search_type)

        # Output results
        if args.format == 'json':
            output_json_result(result)
        else:
            output_text_result(result, f"{args.agent.title()} Agent")

    except KeyboardInterrupt as e:
        logger.error("Search failed: %s", e)
        sys.exit(1)


def handle_multi_search_command(args):
    """Handle multi-agent search command"""

    logger.info("Performing multi-agent search: %s", args.query)

    orchestrator = SearchOrchestrator()

    # Convert agent strings to enums
    agent_types = []
    for agent in args.agents:
        if agent == 'research':
            agent_types.append(SearchType.RESEARCH)
        elif agent == 'news':
            agent_types.append(SearchType.NEWS)
        elif agent == 'general':
            agent_types.append(SearchType.GENERAL)

    try:
        # Perform multi-agent search
        results = orchestrator.multi_agent_search(args.query, agent_types)

        # Output results
        if args.format == 'json':
            output_json_multi_results(results)
        else:
            output_text_multi_results(results)

    except KeyboardInterrupt as e:
        logger.error("Multi-agent search failed: %s", e)
        sys.exit(1)


def handle_comprehensive_search_command(args):
    """Handle comprehensive search command"""

    logger.info("Performing comprehensive search: %s", args.query)

    orchestrator = SearchOrchestrator()

    try:
        # Perform comprehensive search
        results = orchestrator.comprehensive_search(args.query)

        # Output results
        if args.format == 'json':
            # Convert to JSON-serializable format
            json_results = {
                'query': results['query'],
                'agent_results': {
                    agent: create_search_summary_dict(summary)
                    for agent, summary in results['agent_results'].items()
                },
                'analysis': results['analysis'],
                'total_search_time': results['total_search_time'],
                'agents_used': results['agents_used']
            }
            print(json.dumps(json_results, indent=2))
        else:
            output_comprehensive_results(results)

    except KeyboardInterrupt as e:
        logger.error("Comprehensive search failed: %s", e)
        sys.exit(1)


def handle_capabilities_command():
    """Handle capabilities command"""

    orchestrator = SearchOrchestrator()
    capabilities = orchestrator.get_agent_capabilities()

    print("=== AGENT CAPABILITIES ===\n")

    for agent_name, caps in capabilities.items():
        print(f"🤖 {agent_name.upper()} AGENT")
        print(f"   Description: {caps['description']}")
        print(f"   Max Results: {caps['max_results']}")
        print(f"   Supported Queries: {', '.join(caps['supported_queries'])}")
        print()


def output_text_result(result, agent_name: str):
    """Output single search result in text format"""

    print(f"\n=== {agent_name.upper()} SEARCH RESULTS ===")
    print(f"Query: {result.query}")
    print(f"Results: {result.total_results}")
    print(f"Search Time: {format_search_time(result.search_time)}")
    print()

    print("📋 SUMMARY:")
    print(result.summary)
    print()

    if result.key_points:
        print("🔍 KEY INSIGHTS:")
        for i, point in enumerate(result.key_points, 1):
            print(f"{i}. {point}")
        print()

    if result.sources:
        print("🔗 SOURCES:")
        # Show first 5 sources
        for i, source in enumerate(result.sources[:5], 1):
            print(f"{i}. {source}")
        if len(result.sources) > 5:
            print(f"   ... and {len(result.sources) - 5} more sources")


def output_text_multi_results(results):
    """Output multi-agent search results in text format"""

    print("\n=== MULTI-AGENT SEARCH RESULTS ===\n")

    for agent_name, result in results.items():
        output_text_result(result, f"{agent_name} Agent")
        print("\n" + "="*50 + "\n")


def output_comprehensive_results(results):
    """Output comprehensive search results in text format"""

    print("\n=== COMPREHENSIVE SEARCH RESULTS ===")
    print(f"Query: {results['query']}")
    print(f"Total Search Time: "
          f"{format_search_time(results['total_search_time'])}")
    print(f"Agents Used: {', '.join(results['agents_used'])}")
    print()

    # Show analysis
    analysis = results['analysis']
    print("📊 SEARCH ANALYSIS:")
    print(f"   Total Sources: {analysis['total_sources']}")
    print(f"   Unique Sources: {analysis['unique_source_count']}")
    print(f"   Source Overlap: {analysis['source_overlap_ratio']:.1%}")
    if analysis['best_performing_agent']:
        print(f"   Best Agent: {analysis['best_performing_agent']}")
    print()

    # Show combined insights
    if analysis['combined_insights']:
        print("🎯 COMBINED INSIGHTS:")
        for i, insight in enumerate(analysis['combined_insights'], 1):
            print(f"{i}. {insight}")
        print()

    # Show individual agent results
    print("🤖 INDIVIDUAL AGENT RESULTS:")
    print()

    for agent_name, result in results['agent_results'].items():
        output_text_result(result, f"{agent_name} Agent")
        print("\n" + "-"*40 + "\n")


def output_json_result(result):
    """Output single search result in JSON format"""
    print(json.dumps(create_search_summary_dict(result), indent=2))


def output_json_multi_results(results):
    """Output multi-agent search results in JSON format"""
    json_results = {
        agent: create_search_summary_dict(summary)
        for agent, summary in results.items()
    }
    print(json.dumps(json_results, indent=2))


if __name__ == "__main__":
    main()
