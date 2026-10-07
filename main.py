# Import relevant functionality
import os

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables from .env file
load_dotenv()


# A custom web-search tool built on the Serper API (https://serper.dev).
# The @tool decorator turns a plain function into a tool: the function name is
# the tool name, the docstring is the description the model sees, and the type
# hints become the input schema.
# Docs: https://docs.langchain.com/oss/python/langchain/tools
@tool
def web_search(query: str) -> str:
    """Search the web (Google, via Serper) for current information.

    Use this for anything that may require up-to-date facts.
    """
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": os.environ["SERPER_API_KEY"]},
        json={"q": query},
        timeout=10,
    )
    if not response.ok:
        return f"Web search failed with status {response.status_code}"

    data = response.json()

    # Prefer a direct answer if Serper provides one, otherwise summarize the
    # top organic results into a compact, model-friendly string.
    answer_box = data.get("answerBox", {})
    if answer_box.get("answer"):
        return str(answer_box["answer"])
    if answer_box.get("snippet"):
        return str(answer_box["snippet"])

    results = [
        f"{r.get('title', '')}\n{r.get('snippet', '')}\n{r.get('link', '')}"
        for r in data.get("organic", [])[:5]
    ]
    return "\n\n".join(results) or "No results found."


# create_agent builds a LangGraph ReAct-style agent (LangChain v1).
# The model is given as a "provider:model" string; LangChain resolves it to
# ChatOpenAI for you. The checkpointer gives the agent short-term memory: pass
# the same `thread_id` across invocations to keep the conversation.
# Docs: https://docs.langchain.com/oss/python/langchain/agents
agent = create_agent(
    model="openai:gpt-5.4-mini",
    tools=[web_search],
    system_prompt="You are a helpful assistant.",
    checkpointer=InMemorySaver(),
)

# Example usage
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "abc123"}}

    # First example: introduction
    print("\n=== Example 1: Introduction ===\n")
    for step in agent.stream(
        {"messages": [{"role": "user", "content": "Hey ich heiße Andy und komme aus Berlin."}]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

    # Second example: asking about weather (uses memory + the web_search tool)
    print("\n=== Example 2: Weather Question ===\n")
    for step in agent.stream(
        {"messages": [{"role": "user", "content": "Wie ist das Wetter gerade, da wo ich her komme?"}]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
