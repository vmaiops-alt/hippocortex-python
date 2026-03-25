"""Hippocortex adapter for the OpenAI Agents SDK.

Wraps an OpenAI ``Agent`` instance to automatically:
    1. Synthesize past context before each run (injected as system message).
    2. Capture user messages, assistant responses, and tool calls.

Usage::

    from hippocortex import auto_memory

    agent = auto_memory(agent, api_key="hx_live_...")
    # Or with env var: agent = auto_memory(agent)

    result = await Runner.run(agent, "Deploy to staging")

The wrapped agent is the same object with its ``run()`` / ``Runner.run()``
behavior augmented via lifecycle hooks.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ._base import HippocortexAdapter

logger = logging.getLogger("hippocortex.adapters.openai_agents")


def wrap(
    agent: Any,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    session_id: Optional[str] = None,
    capture_tools: bool = True,
    inject_memory: bool = True,
    **kwargs: Any,
) -> Any:
    """Wrap an OpenAI Agents SDK ``Agent`` with Hippocortex auto-memory.

    Args:
        agent: An ``agents.Agent`` instance.
        api_key: Hippocortex API key (falls back to ``HIPPOCORTEX_API_KEY``).
        base_url: API base URL override.
        session_id: Explicit session ID.
        capture_tools: Whether to capture tool calls (default: True).
        inject_memory: Whether to inject synthesized context (default: True).

    Returns:
        The same agent object, with memory hooks attached.
    """
    adapter = HippocortexAdapter(
        api_key=api_key,
        base_url=base_url,
        session_id=session_id,
    )

    if not adapter.enabled:
        logger.info("Hippocortex disabled (no API key). Returning agent unchanged.")
        return agent

    # Store adapter reference on the agent for external access
    agent._hippocortex = adapter

    # Hook into the agent's hooks system if available
    _install_hooks(agent, adapter, capture_tools=capture_tools, inject_memory=inject_memory)

    return agent


def _install_hooks(
    agent: Any,
    adapter: HippocortexAdapter,
    capture_tools: bool = True,
    inject_memory: bool = True,
) -> None:
    """Install Hippocortex lifecycle hooks on the agent.

    The OpenAI Agents SDK uses a hooks-based system. We wrap the agent's
    existing hooks (if any) and add memory capture/injection behavior.
    """
    try:
        from agents import AgentHooks  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "openai-agents package not installed. Install with: "
            "pip install hippocortex[openai-agents]"
        )
        return

    original_hooks = getattr(agent, "hooks", None)

    class HippocortexAgentHooks(AgentHooks):  # type: ignore[misc]
        """Agent hooks that capture events and inject memory context."""

        def __init__(self) -> None:
            super().__init__()
            self._adapter = adapter
            self._original_hooks = original_hooks
            self._inject_memory = inject_memory
            self._capture_tools = capture_tools

        async def on_start(self, context: Any, agent: Any, input_data: Any) -> None:
            """Called before the agent processes input."""
            # Forward to original hooks
            if self._original_hooks and hasattr(self._original_hooks, "on_start"):
                await self._original_hooks.on_start(context, agent, input_data)

            # Extract user message text
            user_text = _extract_user_text(input_data)

            if user_text:
                # Capture user message
                await self._adapter.capture(
                    "message",
                    {"role": "user", "content": user_text},
                    {"agent_name": getattr(agent, "name", "unknown")},
                )

                # Inject synthesized context
                if self._inject_memory:
                    entries = await self._adapter.synthesize(user_text)
                    if entries:
                        context_text = _build_context_text(entries)
                        # Prepend context to agent instructions
                        original_instructions = getattr(agent, "instructions", "") or ""
                        if callable(original_instructions):
                            # Instructions is a function — wrap it
                            orig_fn = original_instructions

                            async def augmented_instructions(ctx: Any, ag: Any) -> str:
                                base = orig_fn
                                if callable(base):
                                    import asyncio
                                    if asyncio.iscoroutinefunction(base):
                                        base = await base(ctx, ag)
                                    else:
                                        base = base(ctx, ag)
                                return context_text + "\n\n" + str(base)

                            agent._hx_original_instructions = orig_fn
                            agent.instructions = augmented_instructions
                        else:
                            agent._hx_original_instructions = original_instructions
                            agent.instructions = context_text + "\n\n" + original_instructions

        async def on_end(self, context: Any, agent: Any, output: Any) -> None:
            """Called after the agent produces output."""
            # Restore original instructions if we modified them
            if hasattr(agent, "_hx_original_instructions"):
                agent.instructions = agent._hx_original_instructions
                del agent._hx_original_instructions

            # Capture assistant output
            output_text = _extract_output_text(output)
            if output_text:
                await self._adapter.capture(
                    "message",
                    {"role": "assistant", "content": output_text},
                    {"agent_name": getattr(agent, "name", "unknown")},
                )

            # Forward to original hooks
            if self._original_hooks and hasattr(self._original_hooks, "on_end"):
                await self._original_hooks.on_end(context, agent, output)

        async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
            """Called before a tool is executed."""
            if self._original_hooks and hasattr(self._original_hooks, "on_tool_start"):
                await self._original_hooks.on_tool_start(context, agent, tool)

            if self._capture_tools:
                tool_name = getattr(tool, "name", None) or str(type(tool).__name__)
                tool_input = {}
                if hasattr(tool, "model_dump"):
                    try:
                        tool_input = tool.model_dump()
                    except Exception:
                        pass
                elif hasattr(tool, "__dict__"):
                    tool_input = {
                        k: v for k, v in tool.__dict__.items()
                        if not k.startswith("_") and _is_serializable(v)
                    }

                await self._adapter.capture(
                    "tool_call",
                    {
                        "tool_name": tool_name,
                        "input": tool_input,
                    },
                    {"agent_name": getattr(agent, "name", "unknown")},
                )

        async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
            """Called after a tool produces a result."""
            if self._capture_tools:
                tool_name = getattr(tool, "name", None) or str(type(tool).__name__)
                result_str = str(result)[:2000] if result else ""

                await self._adapter.capture(
                    "tool_result",
                    {
                        "tool_name": tool_name,
                        "output": result_str,
                    },
                    {"agent_name": getattr(agent, "name", "unknown")},
                )

            if self._original_hooks and hasattr(self._original_hooks, "on_tool_end"):
                await self._original_hooks.on_tool_end(context, agent, tool, result)

        async def on_handoff(self, context: Any, agent: Any, source: Any) -> None:
            """Called when the agent is handed off to."""
            if self._original_hooks and hasattr(self._original_hooks, "on_handoff"):
                await self._original_hooks.on_handoff(context, agent, source)

    agent.hooks = HippocortexAgentHooks()


def _extract_user_text(input_data: Any) -> str:
    """Extract user text from various input formats."""
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, list):
        # List of messages — find the last user message
        for msg in reversed(input_data):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
            if hasattr(msg, "role") and msg.role == "user":
                return getattr(msg, "content", "")
        # If no user role found, concatenate text
        parts = []
        for msg in input_data:
            if isinstance(msg, str):
                parts.append(msg)
            elif isinstance(msg, dict):
                parts.append(msg.get("content", ""))
        return " ".join(parts)
    if hasattr(input_data, "content"):
        return str(input_data.content)
    return str(input_data) if input_data else ""


def _extract_output_text(output: Any) -> str:
    """Extract text from agent output."""
    if isinstance(output, str):
        return output
    if hasattr(output, "final_output"):
        return str(output.final_output) if output.final_output else ""
    if hasattr(output, "output"):
        return str(output.output) if output.output else ""
    if hasattr(output, "content"):
        return str(output.content)
    return ""


def _build_context_text(entries: list) -> str:
    """Build a context injection string from synthesis entries."""
    parts = []
    for entry in entries:
        parts.append(f"[{entry.section}] (confidence: {entry.confidence:.2f})\n{entry.content}")

    return (
        "# Hippocortex Memory Context\n"
        "The following is synthesized context from past experience. "
        "Use it to inform your responses.\n\n"
        + "\n\n".join(parts)
    )


def _is_serializable(value: Any) -> bool:
    """Check if a value is JSON-serializable (rough check)."""
    return isinstance(value, (str, int, float, bool, list, dict, type(None)))
