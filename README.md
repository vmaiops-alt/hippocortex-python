# hippocortex

Official Python SDK for [Hippocortex](https://hippocortex.dev) — persistent memory for AI agents that learns from experience.

[![PyPI](https://img.shields.io/pypi/v/hippocortex)](https://pypi.org/project/hippocortex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Install

```bash
pip install hippocortex
```

With framework adapters:

```bash
pip install hippocortex[openai-agents]   # OpenAI Agents SDK
pip install hippocortex[langgraph]       # LangGraph
pip install hippocortex[crewai]          # CrewAI
pip install hippocortex[autogen]         # AutoGen
pip install hippocortex[all]             # Everything
```

## Quick Start

### Auto-Instrumentation (Recommended)

One import. Every OpenAI or Anthropic call gets persistent memory automatically.

```python
import hippocortex.auto
from openai import OpenAI

client = OpenAI()

# Memory context is injected, conversation is captured automatically
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Deploy payments to staging"}]
)
```

### Explicit Wrap

```python
from hippocortex import wrap
from openai import OpenAI

client = wrap(OpenAI())
# Only this client has memory
```

### Manual Client

```python
import asyncio
from hippocortex import Hippocortex

async def main():
    hx = Hippocortex(api_key="hx_live_...")

    # Capture an event
    await hx.capture(CaptureEvent(
        type="message",
        session_id="session-1",
        payload={"role": "user", "content": "Deploy the service"}
    ))

    # Retrieve relevant context
    context = await hx.synthesize("How do I deploy?")

    # Trigger knowledge compilation
    result = await hx.learn()

    # Search the vault
    secrets = await hx.vault_query("database password")

asyncio.run(main())
```

### Sync Client

```python
from hippocortex import HippocortexSync

hx = HippocortexSync(api_key="hx_live_...")
context = hx.synthesize("How do I deploy?")
```

## Framework Adapters

### OpenAI Agents SDK

```python
from hippocortex.adapters.openai_agents import OpenAIAgentsAdapter

adapter = OpenAIAgentsAdapter(hx, session_id="sess-1")
context = await adapter.get_context("deploy the API")
```

### LangGraph

```python
from hippocortex.adapters.langgraph import LangGraphAdapter

adapter = LangGraphAdapter(hx)
wrapped_graph = adapter.wrap(compiled_graph)
```

### CrewAI

```python
from hippocortex.adapters.crewai import CrewAIAdapter

adapter = CrewAIAdapter(hx)
enhanced_crew = adapter.wrap(crew)
```

### AutoGen

```python
from hippocortex.adapters.autogen import AutoGenAdapter

adapter = AutoGenAdapter(hx)
enhanced_agent = adapter.wrap(agent)
```

## Configuration

```python
hx = Hippocortex(
    api_key="hx_live_...",                          # or set HIPPOCORTEX_API_KEY
    base_url="https://api.hippocortex.dev/v1",      # default
    session_id="my-session",                         # optional
)
```

## API Reference

### `Hippocortex` (async)

| Method | Description |
|--------|-------------|
| `capture(event)` | Capture a single event |
| `capture_batch(events)` | Capture multiple events |
| `synthesize(query, **kwargs)` | Retrieve relevant context |
| `learn(**kwargs)` | Trigger knowledge compilation |
| `vault_query(query, **kwargs)` | Search vault (metadata only) |
| `vault_reveal(item_id)` | Decrypt a vault secret |
| `list_artifacts(**kwargs)` | List knowledge artifacts |
| `get_metrics()` | Get usage metrics |

### `HippocortexSync` (sync)

Same methods, synchronous interface.

## Requirements

- Python 3.9+

## Links

- [Documentation](https://hippocortex.dev/docs)
- [Dashboard](https://dashboard.hippocortex.dev)
- [Gateway Guide](https://hippocortex.dev/docs/gateway/GATEWAY)
- [JavaScript SDK](https://github.com/vmaiops-alt/hippocortex-js)
- [Examples](https://github.com/vmaiops-alt/hippocortex-examples)

## License

MIT
