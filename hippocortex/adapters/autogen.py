"""Hippocortex adapter for AutoGen.

Wraps an AutoGen ``ConversableAgent`` to capture messages and inject
synthesized context.

Usage::

    from hippocortex.adapters import autogen as hx_autogen

    agent = hx_autogen.wrap(agent, api_key="hx_live_...")
    agent.initiate_chat(other_agent, message="Hello")

Status: Beta — AutoGen's API is actively evolving (v0.2 → v0.4).
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional

from ._base import HippocortexAdapter

logger = logging.getLogger("hippocortex.adapters.autogen")


def wrap(
    agent: Any,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    session_id: Optional[str] = None,
    inject_memory: bool = True,
    **kwargs: Any,
) -> Any:
    """Wrap an AutoGen ConversableAgent with Hippocortex auto-memory.

    Args:
        agent: An AutoGen ``ConversableAgent`` (or ``AssistantAgent``/``UserProxyAgent``).
        api_key: Hippocortex API key (falls back to ``HIPPOCORTEX_API_KEY``).
        base_url: API base URL override.
        session_id: Explicit session ID.
        inject_memory: Whether to inject synthesized context.

    Returns:
        The same agent with memory hooks installed.
    """
    adapter = HippocortexAdapter(
        api_key=api_key,
        base_url=base_url,
        session_id=session_id,
    )

    if not adapter.enabled:
        logger.info("Hippocortex disabled (no API key). Returning agent unchanged.")
        return agent

    agent._hippocortex = adapter
    _install_hooks(agent, adapter, inject_memory=inject_memory)
    return agent


def _install_hooks(
    agent: Any,
    adapter: HippocortexAdapter,
    inject_memory: bool = True,
) -> None:
    """Install message capture hooks on an AutoGen agent.

    Uses AutoGen's ``register_hook`` / reply function registration
    to intercept messages.
    """

    # Hook: register a reply function that captures messages and injects context
    # AutoGen calls registered reply functions in order; returning None lets
    # the next handler proceed.

    def _hippocortex_hook(
        recipient: Any,
        messages: Optional[List[Dict[str, Any]]] = None,
        sender: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> tuple:
        """AutoGen reply hook — captures messages and injects context.

        Returns (False, None) to allow the conversation to proceed.
        """
        try:
            if messages:
                # Capture the latest message
                last_msg = messages[-1] if messages else None
                if last_msg and isinstance(last_msg, dict):
                    role = last_msg.get("role", "user")
                    content = last_msg.get("content", "")
                    if content:
                        adapter.capture_sync(
                            "message",
                            {"role": role, "content": str(content)[:2000]},
                            {
                                "source": "autogen",
                                "sender": getattr(sender, "name", "unknown"),
                                "recipient": getattr(recipient, "name", "unknown"),
                            },
                        )

                # Inject synthesized context if this is the first message
                if inject_memory and len(messages) == 1:
                    first_content = ""
                    if isinstance(messages[0], dict):
                        first_content = messages[0].get("content", "")
                    if first_content:
                        entries = adapter.synthesize_sync(str(first_content))
                        if entries:
                            context_text = _build_context_text(entries)
                            # Prepend context as a system message
                            system_msg = {
                                "role": "system",
                                "content": context_text,
                            }
                            # Insert before the first message
                            messages.insert(0, system_msg)

        except Exception as exc:
            logger.warning("Hippocortex autogen hook failed (swallowed): %s", exc)

        # Return (False, None) to let conversation proceed
        return False, None

    # Register the hook using AutoGen's API
    try:
        if hasattr(agent, "register_reply"):
            # AutoGen v0.2 API
            agent.register_reply(
                trigger=[None],  # Trigger on all messages
                reply_func=_hippocortex_hook,
                position=0,  # Run before other reply functions
            )
            logger.debug("Installed Hippocortex hook via register_reply")
        else:
            logger.warning(
                "AutoGen agent does not support register_reply. "
                "Hippocortex hooks not installed."
            )
    except Exception as exc:
        logger.warning("Failed to install AutoGen hooks: %s", exc)

    # Also wrap generate_reply to capture outgoing messages
    original_generate = getattr(agent, "generate_reply", None)
    if original_generate:
        @functools.wraps(original_generate)
        def wrapped_generate(
            messages: Optional[List[Dict[str, Any]]] = None,
            sender: Optional[Any] = None,
            **kw: Any,
        ) -> Any:
            result = original_generate(messages=messages, sender=sender, **kw)
            try:
                if result and isinstance(result, (str, dict)):
                    content = result if isinstance(result, str) else result.get("content", "")
                    if content:
                        adapter.capture_sync(
                            "message",
                            {"role": "assistant", "content": str(content)[:2000]},
                            {
                                "source": "autogen",
                                "agent_name": getattr(agent, "name", "unknown"),
                            },
                        )
            except Exception as exc:
                logger.warning("Hippocortex capture on generate_reply failed: %s", exc)
            return result

        agent.generate_reply = wrapped_generate


def _build_context_text(entries: list) -> str:
    """Build context text from synthesis entries."""
    parts = []
    for entry in entries:
        parts.append(f"[{entry.section}] (confidence: {entry.confidence:.2f})\n{entry.content}")

    return (
        "# Memory Context (from Hippocortex)\n"
        "Use the following learned context from past experience:\n\n"
        + "\n\n".join(parts)
    )
