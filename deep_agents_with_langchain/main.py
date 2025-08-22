"""
Main application for LangChain Deep Agents
"""

from datetime import datetime

from config.settings import settings
from agents.deep_orchestrator import DeepSearchOrchestrator, AgentCollaborationMode
from utils.logger import get_orchestration_logger, setup_logging


def setup_application():
    """Setup the application environment"""
    # Setup logging
    setup_logging(level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)

    # Validate configuration
    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is required. Please set it in your environment.")

    logger = get_orchestration_logger()
    logger.info("LangChain Deep Agents application starting...")

    return logger


def main():
    """Main application entry point"""
    logger = setup_application()

    try:
        # Initialize the orchestrator
        orchestrator = DeepSearchOrchestrator()

        # Example queries to demonstrate the system
        test_queries = [
            {
                "query": ("What are the latest developments in quantum computing "
                          "and their potential impact on cryptography?"),
                "mode": AgentCollaborationMode.HIERARCHICAL,
                "description": "Complex research query requiring multiple perspectives"
            },
            {
                "query": "Recent news about climate change policies in the European Union",
                "mode": AgentCollaborationMode.PARALLEL,
                "description": "News-focused query with time sensitivity"
            },
            {
                "query": "Compare the effectiveness of renewable energy sources",
                "mode": AgentCollaborationMode.CONSENSUS,
                "description": "Comparative analysis requiring consensus"
            }
        ]

        print("\n" + "="*80)
        print("LangChain Deep Agents - Interactive Search System")
        print("="*80)

        # Interactive mode
        while True:
            print("\nOptions:")
            print("1. Enter a custom search query")
            print("2. Run demo queries")
            print("3. Exit")

            choice = input("\nEnter your choice (1-3): ").strip()

            if choice == "1":
                handle_custom_query(orchestrator, logger)
            elif choice == "2":
                run_demo_queries(orchestrator, test_queries, logger)
            elif choice == "3":
                print("\nThank you for using LangChain Deep Agents!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        print("\nApplication stopped.")
    except (ValueError, AttributeError, RuntimeError, ConnectionError) as e:
        logger.error(f"Application error: {str(e)}")
        print(f"An error occurred: {str(e)}")
    finally:
        logger.info("LangChain Deep Agents application shutting down...")


def handle_custom_query(orchestrator: DeepSearchOrchestrator, logger):
    """Handle custom user query"""
    print("\n" + "-"*60)
    print("Custom Query Mode")
    print("-"*60)

    query = input("\nEnter your search query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    # Get collaboration mode preference
    print("\nCollaboration modes:")
    print("1. Hierarchical (default) - Structured, step-by-step analysis")
    print("2. Parallel - Multiple agents work simultaneously")
    print("3. Sequential - Agents work one after another")
    print("4. Consensus - Agents collaborate to reach agreement")

    mode_choice = input(
        "\nSelect collaboration mode (1-4, default=1): ").strip()

    mode_map = {
        "1": AgentCollaborationMode.HIERARCHICAL,
        "2": AgentCollaborationMode.PARALLEL,
        "3": AgentCollaborationMode.SEQUENTIAL,
        "4": AgentCollaborationMode.CONSENSUS
    }

    collaboration_mode = mode_map.get(
        mode_choice, AgentCollaborationMode.HIERARCHICAL)

    execute_search(orchestrator, query, collaboration_mode, logger)


def run_demo_queries(orchestrator: DeepSearchOrchestrator, test_queries: list, logger):
    """Run demonstration queries"""
    print("\n" + "-"*60)
    print("Demo Queries Mode")
    print("-"*60)

    for i, query_info in enumerate(test_queries, 1):
        print(f"\n{i}. {query_info['description']}")
        print(f"   Query: {query_info['query']}")

    choice = input(
        f"\nSelect a demo query (1-{len(test_queries)}) or 'all' to run all: ").strip().lower()

    if choice == "all":
        for i, query_info in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"Running Demo Query {i}: {query_info['description']}")
            print(f"{'='*80}")
            execute_search(
                orchestrator, query_info['query'], query_info['mode'], logger)

            if i < len(test_queries):
                input("\nPress Enter to continue to the next query...")
    else:
        try:
            query_index = int(choice) - 1
            if 0 <= query_index < len(test_queries):
                query_info = test_queries[query_index]
                print(f"\n{'='*80}")
                print(f"Running Demo Query: {query_info['description']}")
                print(f"{'='*80}")
                execute_search(
                    orchestrator, query_info['query'], query_info['mode'], logger)
            else:
                print("Invalid query number.")
        except ValueError:
            print("Invalid input. Please enter a number or 'all'.")


def display_search_header(query: str, collaboration_mode: AgentCollaborationMode, start_time: datetime):
    """Display search execution header"""
    print("\nExecuting search:")
    print(f"Query: {query}")
    print(f"Mode: {collaboration_mode.value}")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "."*60 + "\n")


def display_search_summary(query: str, collaboration_mode: AgentCollaborationMode,
                           duration: float, overall_confidence: float):
    """Display search results summary"""
    print("="*80)
    print("SEARCH RESULTS")
    print("="*80)
    print(f"\nQuery: {query}")
    print(f"Collaboration Mode: {collaboration_mode.value}")
    print(f"Execution Time: {duration:.2f} seconds")
    print(f"Overall Confidence: {overall_confidence:.2f}")


def display_final_result(result: dict):
    """Display the final search result"""
    if 'final_result' in result:
        print(f"\n{'-'*60}")
        print("FINAL RESULT")
        print(f"{'-'*60}")
        print(result['final_result'])


def display_agent_results(result: dict):
    """Display individual agent results"""
    if 'agent_results' in result:
        print(f"\n{'-'*60}")
        print("INDIVIDUAL AGENT RESULTS")
        print(f"{'-'*60}")

        for agent_name, agent_result in result['agent_results'].items():
            print(f"\n[{agent_name.upper()}]")
            if isinstance(agent_result, dict):
                print(f"Confidence: {agent_result.get('confidence', 0.0):.2f}")
                print(
                    f"Result: {agent_result.get('final_result', 'No result')}")
            else:
                print(f"Result: {agent_result}")


def display_search_metadata(result: dict):
    """Display search metadata and performance metrics"""
    if 'search_metadata' in result:
        metadata = result['search_metadata']
        print(f"\n{'-'*60}")
        print("SEARCH METADATA")
        print(f"{'-'*60}")
        print(f"Agents Used: {', '.join(metadata.get('agents_used', []))}")
        print(f"Strategy: {metadata.get('strategy', 'unknown')}")
        print(f"Total Iterations: {metadata.get('total_iterations', 0)}")

        if 'performance_metrics' in metadata:
            metrics = metadata['performance_metrics']
            print("Performance Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")


def display_search_error(error: Exception, duration: float, logger):
    """Display search error information"""
    print("="*80)
    print("SEARCH ERROR")
    print("="*80)
    print("An error occurred during search execution:")
    print(f"Error: {str(error)}")
    print(f"Duration: {duration:.2f} seconds")
    print("="*80)
    logger.error(f"Search failed after {duration:.2f}s: {str(error)}")


def execute_search(orchestrator: DeepSearchOrchestrator, query: str,
                   collaboration_mode: AgentCollaborationMode, logger):
    """Execute a search query"""
    start_time = datetime.now()

    display_search_header(query, collaboration_mode, start_time)

    try:
        # Execute the search
        result = orchestrator.deep_orchestrated_search(
            query, collaboration_mode=collaboration_mode)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Display results
        display_search_summary(
            query, collaboration_mode, duration,
            result.get('overall_confidence', 0.0)
        )

        display_final_result(result)
        display_agent_results(result)
        display_search_metadata(result)

        print("\n" + "="*80)
        logger.info(f"Search completed successfully in {duration:.2f}s")

    except (ValueError, AttributeError, RuntimeError, ConnectionError) as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        display_search_error(e, duration, logger)


def run_sync():
    """Synchronous wrapper for the main function"""
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication stopped.")
    except (ValueError, AttributeError, RuntimeError, ConnectionError) as e:
        print(f"Application error: {str(e)}")


if __name__ == "__main__":
    run_sync()
