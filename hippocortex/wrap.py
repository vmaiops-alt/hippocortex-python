"""Hippocortex SDK -- Transparent wrap() for OpenAI and Anthropic clients."""

from __future__ import annotations

import logging
import time
import random
import string
from typing import Any, Optional

from .config import resolve_config
from .extract import extract_memories_sync

logger = logging.getLogger("hippocortex.wrap")

DEFAULT_BASE_URL = "https://api.hippocortex.dev/v1"


def _generate_session_id() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"hx_{int(time.time() * 1000)}_{suffix}"


def _extract_user_message(messages: list) -> Optional[str]:
    """Extract the last user message content from a messages list."""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        # Multimodal content (list of parts)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        return text
    return None


def _build_context_text(result: Any) -> str:
    """Build a context string from a SynthesizeResult."""
    entries = getattr(result, "entries", [])
    if not entries:
        return ""
    parts = [f"[{e.section}] {e.content}" for e in entries]
    return f"[Hippocortex Memory Context]\n" + "\n".join(parts)


def _is_openai_client(client: Any) -> bool:
    return (
        hasattr(client, "chat")
        and hasattr(client.chat, "completions")
        and hasattr(client.chat.completions, "create")
    )


def _is_anthropic_client(client: Any) -> bool:
    return hasattr(client, "messages") and hasattr(client.messages, "create")


def _wrap_openai(client: Any, hx: Any, session_id: str, enable_extract: bool = True) -> Any:
    """Wrap an OpenAI client's chat.completions.create."""
    original_create = client.chat.completions.create

    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        # Merge positional arg into kwargs if present
        if args:
            params = args[0] if isinstance(args[0], dict) else kwargs
            remaining_args = args[1:]
        else:
            params = kwargs
            remaining_args = ()

        messages = list(params.get("messages", []))
        user_msg = _extract_user_message(messages)

        # 1. Synthesize context (fault-tolerant)
        if user_msg:
            try:
                ctx = hx.synthesize(user_msg)
                context_text = _build_context_text(ctx)
                if context_text:
                    system_msg = {"role": "system", "content": context_text}
                    if isinstance(params, dict):
                        params = {**params, "messages": [system_msg] + messages}
                    else:
                        params.messages = [system_msg] + messages
            except Exception:
                logger.debug("Hippocortex synthesize failed, proceeding without context")

        # 2. Call original
        if args and isinstance(args[0], dict):
            response = original_create(params, *remaining_args)
        else:
            response = original_create(**params) if not remaining_args else original_create(params, *remaining_args)

        # 3. Capture (fault-tolerant, sync fire-and-forget)
        if user_msg:
            try:
                assistant_content = ""
                if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]
                    if hasattr(choice, "message") and hasattr(choice.message, "content"):
                        assistant_content = choice.message.content or ""
                elif isinstance(response, dict):
                    choices = response.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        assistant_content = msg.get("content", "")

                if assistant_content:
                    try:
                        from .types import CaptureEvent

                        hx.capture(CaptureEvent(
                            type="message",
                            session_id=session_id,
                            payload={"role": "user", "content": user_msg},
                        ))
                        hx.capture(CaptureEvent(
                            type="message",
                            session_id=session_id,
                            payload={"role": "assistant", "content": assistant_content},
                        ))

                        # Client-side memory extraction (fire-and-forget)
                        # Uses original_create (unpatched) to avoid infinite recursion!
                        if enable_extract:
                            logger.info("extracting memories (client-side)")
                            try:
                                facts = extract_memories_sync(
                                    messages, assistant_content, client,
                                    create_fn=original_create,
                                )
                                if facts:
                                    hx.capture(CaptureEvent(
                                        type="message",
                                        session_id=session_id,
                                        payload={"role": "system", "content": "extracted_memories"},
                                        metadata={"extractedMemories": facts},
                                    ))
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass

        return response

    client.chat.completions.create = wrapped_create
    return client


def _wrap_anthropic(client: Any, hx: Any, session_id: str, enable_extract: bool = True) -> Any:
    """Wrap an Anthropic client's messages.create."""
    original_create = client.messages.create

    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        if args:
            params = args[0] if isinstance(args[0], dict) else kwargs
            remaining_args = args[1:]
        else:
            params = kwargs
            remaining_args = ()

        messages = params.get("messages", []) if isinstance(params, dict) else getattr(params, "messages", [])
        user_msg = _extract_user_message(list(messages))

        # 1. Synthesize context
        if user_msg:
            try:
                ctx = hx.synthesize(user_msg)
                context_text = _build_context_text(ctx)
                if context_text:
                    existing_system = params.get("system", "") if isinstance(params, dict) else getattr(params, "system", "")
                    if existing_system:
                        new_system = f"{context_text}\n\n{existing_system}"
                    else:
                        new_system = context_text
                    if isinstance(params, dict):
                        params = {**params, "system": new_system}
                    else:
                        params.system = new_system
            except Exception:
                logger.debug("Hippocortex synthesize failed, proceeding without context")

        # 2. Call original
        if args and isinstance(args[0], dict):
            response = original_create(params, *remaining_args)
        else:
            response = original_create(**params) if not remaining_args else original_create(params, *remaining_args)

        # 3. Capture
        if user_msg:
            try:
                assistant_content = ""
                content_list = getattr(response, "content", None)
                if isinstance(content_list, list):
                    text_parts = [
                        getattr(b, "text", "") if hasattr(b, "text") else b.get("text", "")
                        for b in content_list
                        if (hasattr(b, "type") and b.type == "text") or (isinstance(b, dict) and b.get("type") == "text")
                    ]
                    assistant_content = "".join(text_parts)

                if assistant_content:
                    try:
                        from .types import CaptureEvent

                        hx.capture(CaptureEvent(
                            type="message",
                            session_id=session_id,
                            payload={"role": "user", "content": user_msg},
                        ))
                        hx.capture(CaptureEvent(
                            type="message",
                            session_id=session_id,
                            payload={"role": "assistant", "content": assistant_content},
                        ))

                        # Client-side memory extraction (fire-and-forget)
                        # Uses original_create (unpatched) to avoid infinite recursion!
                        if enable_extract:
                            logger.info("extracting memories (client-side)")
                            try:
                                facts = extract_memories_sync(
                                    list(messages), assistant_content, client,
                                    create_fn=original_create,
                                )
                                if facts:
                                    hx.capture(CaptureEvent(
                                        type="message",
                                        session_id=session_id,
                                        payload={"role": "system", "content": "extracted_memories"},
                                        metadata={"extractedMemories": facts},
                                    ))
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass

        return response

    client.messages.create = wrapped_create
    return client


def wrap(
    client: Any,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    session_id: Optional[str] = None,
    extract: bool = True,
) -> Any:
    """Transparently wrap an OpenAI or Anthropic client for auto-memory.

    Usage::

        from hippocortex import wrap
        from openai import OpenAI

        client = wrap(OpenAI(), api_key="hx_live_...")
        # Every chat.completions.create() now has memory!

    If Hippocortex is unreachable, all calls pass through transparently.
    """
    config = resolve_config(api_key=api_key, base_url=base_url)
    if not config:
        return client

    from .client import SyncHippocortex

    hx = SyncHippocortex(
        api_key=config["apiKey"],
        base_url=config.get("baseUrl", DEFAULT_BASE_URL),
    )

    sid = session_id or _generate_session_id()

    if _is_openai_client(client):
        return _wrap_openai(client, hx, sid, enable_extract=extract)

    if _is_anthropic_client(client):
        return _wrap_anthropic(client, hx, sid, enable_extract=extract)

    # Unknown client type, return unchanged
    return client
