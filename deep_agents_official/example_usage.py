#!/usr/bin/env python3
# pylint: disable=import-error,no-name-in-module
"""
Example usage of the Official DeepAgents implementation.
Demonstrates how to use individual agents and the orchestrator.
"""

import os
import sys

from agents.analysis_deep_agent import AnalysisDeepAgent
from agents.coding_deep_agent import CodingDeepAgent
from agents.deep_orchestrator import DeepAgentsOrchestrator
from agents.research_deep_agent import ResearchDeepAgent
from config.settings import Config

# Add the current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def example_research_agent():
    """Example of using the ResearchDeepAgent"""
    print("=== ResearchDeepAgent Example ===")

    try:
        # Validate configuration
        Config.validate()

        # Create research agent
        research_agent = ResearchDeepAgent()

        # Conduct research
        query = "What are the latest developments in large language models?"
        result = research_agent.research(
            query, context="Focus on 2024 developments")

        print(f"Research Query: {query}")
        print(f"Status: {result['status']}")
        print(f"Agent Type: {result['agent_type']}")

        if result['status'] == 'success':
            print("Research completed successfully!")
            # In a real implementation, you could access workspace files
            # files = research_agent.get_workspace_files()
            # print(f"Generated files: {files}")
        else:
            print(f"Research failed: {result.get('error', 'Unknown error')}")

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please ensure your API keys are set in the environment variables.")
    except (ImportError, RuntimeError, ConnectionError) as e:
        print(f"Unexpected error: {e}")


def example_coding_agent():
    """Example of using the CodingDeepAgent"""
    print("\n=== CodingDeepAgent Example ===")

    try:
        # Create coding agent
        coding_agent = CodingDeepAgent()

        # Generate code
        requirements = "Create a Python function that calculates the factorial of a number"
        result = coding_agent.generate_code(
            requirements, context="Include error handling and documentation")

        print(f"Code Requirements: {requirements}")
        print(f"Status: {result['status']}")
        print(f"Agent Type: {result['agent_type']}")

        if result['status'] == 'success':
            print("Code generation completed successfully!")
        else:
            print(
                f"Code generation failed: {result.get('error', 'Unknown error')}")

        # Example of code analysis
        sample_code = """
def factorial(n):
    if n < 0:
        return None
    elif n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""

        analysis_result = coding_agent.code_analysis(
            sample_code, "Analyze factorial function")
        print(f"\nCode Analysis Status: {analysis_result['status']}")

    except (ImportError, RuntimeError, ConnectionError, ValueError) as e:
        print(f"Error in coding example: {e}")


def example_analysis_agent():
    """Example of using the AnalysisDeepAgent"""
    print("\n=== AnalysisDeepAgent Example ===")

    try:
        # Create analysis agent
        analysis_agent = AnalysisDeepAgent()

        # Analyze sample data
        sample_data = """
        Month,Sales,Revenue
        Jan,100,10000
        Feb,120,12000
        Mar,110,11000
        Apr,140,14000
        May,160,16000
        """

        result = analysis_agent.analyze_data(
            sample_data,
            "sales_trend_analysis",
            context="Monthly sales data for Q1-Q2"
        )

        print("Analysis Type: sales_trend_analysis")
        print(f"Status: {result['status']}")
        print(f"Agent Type: {result['agent_type']}")

        if result['status'] == 'success':
            print("Data analysis completed successfully!")
        else:
            print(
                f"Data analysis failed: {result.get('error', 'Unknown error')}")

    except (ImportError, RuntimeError, ValueError) as e:
        print(f"Error in analysis example: {e}")


def example_orchestrator():
    """Example of using the DeepAgentsOrchestrator"""
    print("\n=== DeepAgentsOrchestrator Example ===")

    try:
        # Create orchestrator
        orchestrator = DeepAgentsOrchestrator()

        # Execute complex multi-agent task
        task_description = ("Research the latest Python web frameworks, "
                            "analyze their features, and provide code examples")

        result = orchestrator.execute_multi_agent_task(
            task_description=task_description,
            task_type="mixed",  # Uses multiple agent types
            context="Focus on frameworks suitable for enterprise applications"
        )

        print(f"Task: {task_description}")
        print(f"Status: {result['status']}")
        print(f"Session ID: {result['session_id']}")

        execution_summary = result.get('execution_summary', {})
        print(f"Total Steps: {execution_summary.get('total_steps', 0)}")
        print(
            f"Successful Steps: {execution_summary.get('successful_steps', 0)}")
        print(f"Agents Used: {execution_summary.get('agents_used', [])}")

        insights = result.get('insights', [])
        if insights:
            print("\nKey Insights:")
            for insight in insights:
                print(f"- {insight}")

        recommendations = result.get('recommendations', [])
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"- {rec}")

        # Show available agents
        available_agents = orchestrator.get_available_agents()
        print(f"\nAvailable Agents: {list(available_agents.keys())}")

    except (ImportError, RuntimeError, ConnectionError, ValueError) as e:
        print(f"Error in orchestrator example: {e}")


def main():
    """Run all examples"""
    print("Official DeepAgents Implementation Examples")
    print("=" * 50)

    # Note about configuration
    print("\nNote: This example uses placeholder implementations.")
    print("To use the actual deepagents package, install it with:")
    print("pip install deepagents>=0.1.0")
    print("\nAlso ensure your API keys are set:")
    print("export OPENAI_API_KEY='your-key-here'")
    print("export ANTHROPIC_API_KEY='your-key-here'")
    print("=" * 50)

    # Run examples
    example_research_agent()
    example_coding_agent()
    example_analysis_agent()
    example_orchestrator()

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNext steps:")
    print("1. Install the actual deepagents package")
    print("2. Set up your API keys")
    print("3. Uncomment the create_deep_agent calls in the agent implementations")
    print("4. Test with real scenarios")


if __name__ == "__main__":
    main()
