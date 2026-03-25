"""Hippocortex adapter for LangGraph.

Wraps a LangGraph ``CompiledGraph`` to automatically:
    1. Inject synthesized context at the entry node.
    2. Capture state transitions and node outputs.

Usage::

    from hippocortex.adapters import langgraph as hx_langgraph

    graph = hx_langgraph.wrap(graph, api_key="hx_live_...")
    result = await graph.ainvoke({"messages": [...]})
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from ._base import HippocortexAdapter

logger = logging.getLogger("hippocortex.adapters.langgraph")


def wrap(
    graph: Any,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    session_id: Optional[str] = None,
    capture_nodes: bool = True,
    inject_memory: bool = True,
    input_key: str = "messages",
    **kwargs: Any,
) -> Any:
    """Wrap a LangGraph CompiledGraph with Hippocortex auto-memory.

    Args:
        graph: A LangGraph ``CompiledGraph`` instance (from ``graph.compile()``).
        api_key: Hippocortex API key (falls back to ``HIPPOCORTEX_API_KEY``).
        base_url: API base URL override.
        session_id: Explicit session ID.
        capture_nodes: Whether to capture node state transitions.
        inject_memory: Whether to inject synthesized context.
        input_key: Key in the state dict that contains messages (default: "messages").

    Returns:
        A ``HippocortexGraph`` wrapper with the same invoke/ainvoke interface.
    """
    adapter = HippocortexAdapter(
        api_key=api_key,
        base_url=base_url,
        session_id=session_id,
    )

    if not adapter.enabled:
        logger.info("Hippocortex disabled (no API key). Returning graph unchanged.")
        return graph

    return HippocortexGraph(
        graph=graph,
        adapter=adapter,
        capture_nodes=capture_nodes,
        inject_memory=inject_memory,
        input_key=input_key,
    )


class HippocortexGraph:
    """Wrapper around a LangGraph CompiledGraph with memory hooks.

    Proxies all attribute access to the underlying graph, but intercepts
    ``invoke`` and ``ainvoke`` to add memory capture/injection.
    """

    def __init__(
        self,
        graph: Any,
        adapter: HippocortexAdapter,
        capture_nodes: bool = True,
        inject_memory: bool = True,
        input_key: str = "messages",
    ):
        self._graph = graph
        self._adapter = adapter
        self._capture_nodes = capture_nodes
        self._inject_memory = inject_memory
        self._input_key = input_key

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying graph."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._graph, name)

    async def ainvoke(
        self,
        input: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Async invoke with memory hooks."""
        # Extract query from input
        query = _extract_query(input, self._input_key)

        # Inject memory context
        if self._inject_memory and query:
            input = await self._inject_context(input, query)

        # Capture user input
        if query:
            await self._adapter.capture(
                "message",
                {"role": "user", "content": query},
                {"source": "langgraph", "session_id": self._adapter.session_id},
            )

        # Run the graph with node capture callback
        if self._capture_nodes and config is None:
            config = {}

        result = await self._graph.ainvoke(input, config=config, **kwargs)

        # Capture output
        output_text = _extract_output(result, self._input_key)
        if output_text:
            await self._adapter.capture(
                "message",
                {"role": "assistant", "content": output_text},
                {"source": "langgraph", "session_id": self._adapter.session_id},
            )

        return result

    def invoke(
        self,
        input: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronous invoke with memory hooks."""
        query = _extract_query(input, self._input_key)

        # Inject memory context (sync)
        if self._inject_memory and query:
            input = self._inject_context_sync(input, query)

        # Capture user input
        if query:
            self._adapter.capture_sync(
                "message",
                {"role": "user", "content": query},
                {"source": "langgraph", "session_id": self._adapter.session_id},
            )

        result = self._graph.invoke(input, config=config, **kwargs)

        # Capture output
        output_text = _extract_output(result, self._input_key)
        if output_text:
            self._adapter.capture_sync(
                "message",
                {"role": "assistant", "content": output_text},
                {"source": "langgraph", "session_id": self._adapter.session_id},
            )

        return result

    async def astream(
        self,
        input: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Async stream with memory hooks on entry/exit."""
        query = _extract_query(input, self._input_key)

        if self._inject_memory and query:
            input = await self._inject_context(input, query)

        if query:
            await self._adapter.capture(
                "message",
                {"role": "user", "content": query},
                {"source": "langgraph", "session_id": self._adapter.session_id},
            )

        collected_output: List[str] = []

        async for chunk in self._graph.astream(input, config=config, **kwargs):
            if self._capture_nodes and isinstance(chunk, dict):
                for node_name, node_output in chunk.items():
                    await self._adapter.capture(
                        "tool_call",
                        {
                            "tool_name": f"node:{node_name}",
                            "output": _truncate(str(node_output), 2000),
                        },
                        {"source": "langgraph"},
                    )
                    # Collect output messages
                    output_text = _extract_output(node_output, self._input_key)
                    if output_text:
                        collected_output.append(output_text)

            yield chunk

        # Capture final output
        if collected_output:
            await self._adapter.capture(
                "message",
                {"role": "assistant", "content": collected_output[-1]},
                {"source": "langgraph", "session_id": self._adapter.session_id},
            )

    async def _inject_context(self, input: Any, query: str) -> Any:
        """Inject synthesized context into the input state."""
        entries = await self._adapter.synthesize(query)
        if not entries:
            return input

        context_text = _build_context_text(entries)
        return _prepend_context(input, context_text, self._input_key)

    def _inject_context_sync(self, input: Any, query: str) -> Any:
        """Synchronous context injection."""
        entries = self._adapter.synthesize_sync(query)
        if not entries:
            return input

        context_text = _build_context_text(entries)
        return _prepend_context(input, context_text, self._input_key)


def _extract_query(input: Any, input_key: str) -> str:
    """Extract the user query from the graph input."""
    if isinstance(input, str):
        return input
    if isinstance(input, dict):
        messages = input.get(input_key, [])
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    return content if isinstance(content, str) else str(content)
                if hasattr(msg, "type") and getattr(msg, "type", None) == "human":
                    return getattr(msg, "content", "")
        # Try 'input' key
        if "input" in input:
            return str(input["input"])
        # Try 'question' key
        if "question" in input:
            return str(input["question"])
    return ""


def _extract_output(result: Any, input_key: str) -> str:
    """Extract assistant output text from graph result."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        messages = result.get(input_key, [])
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                return last.get("content", "")
            if hasattr(last, "content"):
                return str(last.content)
        if "output" in result:
            return str(result["output"])
    return ""


def _prepend_context(input: Any, context_text: str, input_key: str) -> Any:
    """Prepend a system message with context to the input."""
    if isinstance(input, dict):
        messages = input.get(input_key, [])
        if isinstance(messages, list):
            # Try to create a LangChain SystemMessage if available
            system_msg = _make_system_message(context_text)
            new_input = dict(input)
            new_input[input_key] = [system_msg] + list(messages)
            return new_input
    return input


def _make_system_message(content: str) -> Any:
    """Create a system message, using LangChain types if available."""
    try:
        from langchain_core.messages import SystemMessage  # type: ignore[import-untyped]
        return SystemMessage(content=content)
    except ImportError:
        return {"role": "system", "content": content}


def _build_context_text(entries: list) -> str:
    """Build context injection string from synthesis entries."""
    parts = []
    for entry in entries:
        parts.append(f"[{entry.section}] (confidence: {entry.confidence:.2f})\n{entry.content}")

    return (
        "# Hippocortex Memory Context\n"
        "The following is synthesized context from past experience. "
        "Use it to inform your responses.\n\n"
        + "\n\n".join(parts)
    )


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max length."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
