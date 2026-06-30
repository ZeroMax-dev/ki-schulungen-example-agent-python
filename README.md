# LangChain Agent Example (Python)

A minimal LangChain **v1** agent with web-search capabilities, built with
`create_agent` (LangGraph) and conversational memory via `InMemorySaver`.

Docs: https://docs.langchain.com/oss/python/langchain/agents

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env_example` to `.env` and fill in your keys:

```
OPENAI_API_KEY=...
SERPER_API_KEY=...   # from https://serper.dev
```

## Run

```bash
python main.py
```

The agent introduces itself, remembers facts across turns (same `thread_id`),
and uses the `web-search` tool (Google via Serper) for current information.
