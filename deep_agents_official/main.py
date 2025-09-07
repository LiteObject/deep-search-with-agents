"""
Main application file for the proper LangChain DeepAgents implementation.
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from .tools.search import web_search

# Load environment variables from .env file
load_dotenv()


async def main():
    """
    Main function to run the DeepAgent.
    """
    # --- Model Configuration ---
    # Default to Anthropic's Sonnet model, which is recommended for DeepAgents
    # You can switch to OpenAI's GPT-4 by uncommenting the other model line
    # and ensuring OPENAI_API_KEY is set in your .env file.

    # model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    model = ChatOpenAI(model="gpt-4-turbo")

    # --- Agent Creation ---
    # Create a DeepAgent using the official factory function.
    # This sets up the agent with the necessary LangGraph components,
    # including the planning tool, file system tools, and sub-agent support.
    agent = create_deep_agent(
        model=model,
        tools=[web_search],
        prompt="""
        You are a world-class research assistant. 
        
        Your goal is to provide accurate, well-researched, and comprehensive answers.
        
        1.  **Plan**: First, create a step-by-step plan to address the user's request. Use the `todo_write` tool for this.
        2.  **Execute**: Carry out the plan, using the `web_search` tool to find relevant information.
        3.  **Synthesize**: Combine the information into a coherent and well-structured final answer.
        4.  **File**: Write the final answer to a file using the `write_file` tool.
        """,
    )

    # --- Agent Invocation ---
    # Define the research task for the agent
    task = {
        "messages": [
            (
                "user",
                "What are the latest advancements in artificial intelligence as of late 2025, "
                "focusing on large language models and their real-world applications?",
            )
        ]
    }

    print("🚀 Starting DeepAgent task...")

    # Use astream_events to get a stream of events from the agent's execution
    # This provides detailed insight into the agent's internal state and actions.
    async for event in agent.astream_events(task, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Print the tokens as they are streamed from the model
                print(content, end="")
        elif kind == "on_tool_start":
            print(f"\n--- 🛠️ Tool Started: {event['name']} ---")
            print(f"Input: {event['data'].get('input')}")
        elif kind == "on_tool_end":
            print(f"--- ✅ Tool Ended: {event['name']} ---")
            print(f"Output: {event['data'].get('output')}")
            print("---")

    print("\n\n✅ DeepAgent task finished.")


if __name__ == "__main__":
    # Ensure API keys are set
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "Error: Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file."
        )
    else:
        asyncio.run(main())
