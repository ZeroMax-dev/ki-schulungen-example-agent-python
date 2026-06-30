# Import relevant functionality
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create the agent
memory = InMemorySaver()
model = ChatOpenAI(model="gpt-4o-mini")
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="web-search",
        func=search.run,
        description="Useful for searching the web for current information"
    )
]

# create_agent builds a LangGraph ReAct-style agent (LangChain v1). This
# replaces langgraph.prebuilt.create_react_agent from the 0.x examples.
# Docs: https://docs.langchain.com/oss/python/langchain/agents
agent_executor = create_agent(model, tools, checkpointer=memory)

# Example usage
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "abc123"}}
    
    # First example: introduction
    print("\n=== Example 1: Introduction ===\n")
    for step in agent_executor.stream(
        {"messages": [HumanMessage(content="Hey ich heiße Andy und komme aus Berlin.")]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
    
    # Second example: asking about weather
    print("\n=== Example 2: Weather Question ===\n")
    for step in agent_executor.stream(
        {"messages": [HumanMessage(content="Wie ist das Wetter gerade, da wo ich her komme?")]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
