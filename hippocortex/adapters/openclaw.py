"""Hippocortex adapter for OpenClaw agents.

Provides middleware that captures agent events and injects synthesized
context into the system prompt.

Usage::

    from hippocortex.adapters import openclaw as hx_openclaw

    memory = hx_openclaw.create_middleware(api_key="hx_live_...")

    # In your OpenClaw skill or agent config
    # Use memory.on_message() and memory.get_context() in your hooks

    # Or use the convenience wrapper
    skill = hx_openclaw.create_skill(api_key="hx_live_...")
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from ._base import HippocortexAdapter

logger = logging.getLogger("hippocortex.adapters.openclaw")


class OpenClawMemoryMiddleware:
    """Middleware for OpenClaw agents that provides auto-memory.

    Intercepts messages, captures events, and injects synthesized
    context into the system prompt.
    """

    def __init__(
        self,
        adapter: HippocortexAdapter,
        inject_memory: bool = True,
        capture_messages: bool = True,
        capture_tools: bool = True,
    ):
        self._adapter = adapter
        self._inject_memory = inject_memory
        self._capture_messages = capture_messages
        self._capture_tools = capture_tools

    @property
    def session_id(self) -> str:
        return self._adapter.session_id

    async def on_message(
        self,
        message: str,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Process an inbound message.

        Captures the message and returns synthesized context to inject
        into the system prompt (or None if no context available).
        """
        if self._capture_messages:
            await self._adapter.capture(
                "message",
                {"role": role, "content": message},
                {"source": "openclaw", **(metadata or {})},
            )

        if self._inject_memory and role == "user":
            return await self.get_context(message)
        return None

    async def get_context(self, query: str, max_tokens: int = 4000) -> Optional[str]:
        """Get synthesized context for a query.

        Returns a formatted string suitable for injection into a system
        prompt, or None if no relevant context is available.
        """
        entries = await self._adapter.synthesize(query, max_tokens)
        if not entries:
            return None

        parts = []
        for entry in entries:
            parts.append(
                f"[{entry.section}] (confidence: {entry.confidence:.2f})\n{entry.content}"
            )

        return (
            "# Hippocortex Memory Context\n"
            "The following is synthesized context from past experience. "
            "Use it to inform your responses.\n\n"
            + "\n\n".join(parts)
        )

    async def on_response(
        self,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture an outbound response."""
        if self._capture_messages:
            await self._adapter.capture(
                "message",
                {"role": "assistant", "content": response[:2000]},
                {"source": "openclaw", **(metadata or {})},
            )

    async def on_tool_call(
        self,
        tool_name: str,
        tool_input: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture a tool call."""
        if self._capture_tools:
            input_str = tool_input
            if not isinstance(input_str, str):
                try:
                    input_str = json.dumps(tool_input, default=str)[:2000]
                except Exception:
                    input_str = str(tool_input)[:2000]

            await self._adapter.capture(
                "tool_call",
                {"tool_name": tool_name, "input": input_str},
                {"source": "openclaw", **(metadata or {})},
            )

    async def on_tool_result(
        self,
        tool_name: str,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture a tool result."""
        if self._capture_tools:
            result_str = str(result)[:2000] if result else ""
            await self._adapter.capture(
                "tool_result",
                {"tool_name": tool_name, "output": result_str},
                {"source": "openclaw", **(metadata or {})},
            )

    async def inject_into_messages(
        self,
        messages: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """Inject synthesized context into a messages array.

        Prepends a system message with memory context.
        """
        return await self._adapter.inject_context(messages, query)

    async def close(self) -> None:
        """Close underlying HTTP connections."""
        await self._adapter.close()


def create_middleware(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    session_id: Optional[str] = None,
    inject_memory: bool = True,
    capture_messages: bool = True,
    capture_tools: bool = True,
    **kwargs: Any,
) -> OpenClawMemoryMiddleware:
    """Create an OpenClaw memory middleware instance.

    Args:
        api_key: Hippocortex API key (falls back to ``HIPPOCORTEX_API_KEY``).
        base_url: API base URL override.
        session_id: Explicit session ID.
        inject_memory: Whether to inject synthesized context.
        capture_messages: Whether to capture message events.
        capture_tools: Whether to capture tool events.

    Returns:
        An ``OpenClawMemoryMiddleware`` instance.
    """
    adapter = HippocortexAdapter(
        api_key=api_key,
        base_url=base_url,
        session_id=session_id,
    )

    return OpenClawMemoryMiddleware(
        adapter=adapter,
        inject_memory=inject_memory,
        capture_messages=capture_messages,
        capture_tools=capture_tools,
    )


def create_skill(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create an OpenClaw skill definition for Hippocortex memory.

    Returns a skill configuration dict that can be included in an
    OpenClaw agent's skills list.

    Returns:
        A dict with ``name``, ``description``, and ``middleware`` keys.
    """
    middleware = create_middleware(
        api_key=api_key,
        base_url=base_url,
        session_id=session_id,
        **kwargs,
    )

    return {
        "name": "hippocortex-memory",
        "description": "Auto-memory powered by Hippocortex — captures events and injects context",
        "middleware": middleware,
    }
