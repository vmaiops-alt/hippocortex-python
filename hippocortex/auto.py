"""Hippocortex auto-instrumentation -- Sentry-style monkey-patching.

Import this module to automatically patch OpenAI and Anthropic SDKs:

    import hippocortex.auto

All LLM calls will be captured and context-injected automatically.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from typing import Any, Dict, Generator, Iterator, List, Optional

from .config import resolve_config
from .extract import extract_memories_sync

logger = logging.getLogger("hippocortex.auto")

DEFAULT_BASE_URL = "https://api.hippocortex.dev/v1"

_PATCHED_ATTR = "_hippocortex_auto_patched"
_initialized = False


# -- Session ID --

def _generate_session_id() -> str:
    host = socket.gethostname().replace(".", "_")[:32]
    pid = os.getpid()
    start = int(time.time())
    return f"hx_auto_{host}_{pid}_{start}"


# -- Logging --

def _log(msg: str) -> None:
    if os.environ.get("HIPPOCORTEX_SILENT") == "1":
        return
    print(f"[hippocortex] {msg}")


def _warn(msg: str) -> None:
    if os.environ.get("HIPPOCORTEX_SILENT") == "1":
        return
    print(f"[hippocortex] {msg}")


# -- Helpers --

def _extract_user_message(messages: List[Any]) -> Optional[str]:
    """Extract the last user message content from a messages list."""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
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
    return "[Hippocortex Memory Context]\n" + "\n".join(parts)


def _fire_and_forget(hx: Any, event: Any) -> None:
    """Capture an event without letting errors propagate."""
    try:
        hx.capture(event)
    except Exception:
        pass


# -- OpenAI Patching --

def _patch_openai(hx: Any, session_id: str) -> bool:
    """Monkey-patch openai.resources.chat.completions.Completions.create."""
    try:
        import openai.resources.chat.completions as oai_mod
    except ImportError:
        return False

    cls = oai_mod.Completions
    if getattr(cls, _PATCHED_ATTR, False):
        return False

    original_create = cls.create

    def patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        from .types import CaptureEvent

        # Merge first positional arg into kwargs if it's the params
        messages = list(kwargs.get("messages", []))
        user_msg = _extract_user_message(messages)
        is_stream = kwargs.get("stream", False)

        # 1. Inject context
        if user_msg:
            try:
                ctx = hx.synthesize(user_msg)
                context_text = _build_context_text(ctx)
                if context_text:
                    system_msg = {"role": "system", "content": context_text}
                    kwargs["messages"] = [system_msg] + messages
            except Exception:
                pass

        # 2. Call original
        result = original_create(self, *args, **kwargs)

        # 3. Capture
        if user_msg:
            if is_stream:
                return _wrap_openai_stream(result, hx, session_id, user_msg)
            else:
                try:
                    assistant_content = ""
                    if hasattr(result, "choices") and result.choices:
                        choice = result.choices[0]
                        if hasattr(choice, "message") and hasattr(choice.message, "content"):
                            assistant_content = choice.message.content or ""
                    if assistant_content:
                        _fire_and_forget(hx, CaptureEvent(
                            type="message",
                            session_id=session_id,
                            payload={"role": "user", "content": user_msg},
                        ))
                        _fire_and_forget(hx, CaptureEvent(
                            type="message",
                            session_id=session_id,
                            payload={"role": "assistant", "content": assistant_content},
                        ))

                        # Client-side memory extraction (fire-and-forget)
                        # Uses original_create (unpatched) to avoid infinite recursion!
                        if os.environ.get("HIPPOCORTEX_EXTRACT") != "false":
                            _log("extracting memories (client-side)")
                            try:
                                facts = extract_memories_sync(
                                    messages,
                                    assistant_content,
                                    self,  # pass the instance
                                    create_fn=lambda **kw: original_create(self, **kw),
                                )
                                if facts:
                                    _fire_and_forget(hx, CaptureEvent(
                                        type="message",
                                        session_id=session_id,
                                        payload={"role": "system", "content": "extracted_memories"},
                                        metadata={"extractedMemories": facts},
                                    ))
                            except Exception:
                                pass
                except Exception:
                    pass

        return result

    cls.create = patched_create  # type: ignore[assignment]
    setattr(cls, _PATCHED_ATTR, True)
    return True


def _wrap_openai_stream(stream: Any, hx: Any, session_id: str, user_msg: str) -> Any:
    """Wrap an OpenAI streaming response to collect chunks while passing through."""
    from .types import CaptureEvent

    chunks: List[str] = []

    class WrappedStream:
        """Proxy that wraps the original stream, collecting text deltas."""

        def __init__(self, original: Any) -> None:
            self._original = original

        def __iter__(self) -> "WrappedStream":
            self._iter = iter(self._original)
            return self

        def __next__(self) -> Any:
            try:
                chunk = next(self._iter)
                try:
                    if hasattr(chunk, "choices") and chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and isinstance(delta.content, str):
                            chunks.append(delta.content)
                except Exception:
                    pass
                return chunk
            except StopIteration:
                # Stream finished, capture
                full_content = "".join(chunks)
                if full_content:
                    _fire_and_forget(hx, CaptureEvent(
                        type="message",
                        session_id=session_id,
                        payload={"role": "user", "content": user_msg},
                    ))
                    _fire_and_forget(hx, CaptureEvent(
                        type="message",
                        session_id=session_id,
                        payload={"role": "assistant", "content": full_content},
                    ))
                raise

        def __getattr__(self, name: str) -> Any:
            return getattr(self._original, name)

    return WrappedStream(stream)


# -- Anthropic Patching --

def _patch_anthropic(hx: Any, session_id: str) -> bool:
    """Monkey-patch anthropic.resources.messages.Messages.create."""
    try:
        import anthropic.resources.messages as anth_mod
    except ImportError:
        return False

    cls = anth_mod.Messages
    if getattr(cls, _PATCHED_ATTR, False):
        return False

    original_create = cls.create

    def patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        from .types import CaptureEvent

        messages = list(kwargs.get("messages", []))
        user_msg = _extract_user_message(messages)
        is_stream = kwargs.get("stream", False)

        # 1. Inject context
        if user_msg:
            try:
                ctx = hx.synthesize(user_msg)
                context_text = _build_context_text(ctx)
                if context_text:
                    existing_system = kwargs.get("system", "")
                    if existing_system:
                        kwargs["system"] = f"{context_text}\n\n{existing_system}"
                    else:
                        kwargs["system"] = context_text
            except Exception:
                pass

        # 2. Call original
        result = original_create(self, *args, **kwargs)

        # 3. Capture
        if user_msg:
            if is_stream:
                return _wrap_anthropic_stream(result, hx, session_id, user_msg)
            else:
                try:
                    assistant_content = ""
                    content_list = getattr(result, "content", None)
                    if isinstance(content_list, list):
                        text_parts = [
                            getattr(b, "text", "") if hasattr(b, "text") else b.get("text", "")
                            for b in content_list
                            if (hasattr(b, "type") and b.type == "text")
                            or (isinstance(b, dict) and b.get("type") == "text")
                        ]
                        assistant_content = "".join(text_parts)
                    if assistant_content:
                        _fire_and_forget(hx, CaptureEvent(
                            type="message",
                            session_id=session_id,
                            payload={"role": "user", "content": user_msg},
                        ))
                        _fire_and_forget(hx, CaptureEvent(
                            type="message",
                            session_id=session_id,
                            payload={"role": "assistant", "content": assistant_content},
                        ))

                        # Client-side memory extraction (fire-and-forget)
                        # Uses original_create (unpatched) to avoid infinite recursion!
                        if os.environ.get("HIPPOCORTEX_EXTRACT") != "false":
                            _log("extracting memories (client-side)")
                            try:
                                facts = extract_memories_sync(
                                    messages,
                                    assistant_content,
                                    self,  # pass the instance
                                    create_fn=lambda **kw: original_create(self, **kw),
                                )
                                if facts:
                                    _fire_and_forget(hx, CaptureEvent(
                                        type="message",
                                        session_id=session_id,
                                        payload={"role": "system", "content": "extracted_memories"},
                                        metadata={"extractedMemories": facts},
                                    ))
                            except Exception:
                                pass
                except Exception:
                    pass

        return result

    cls.create = patched_create  # type: ignore[assignment]
    setattr(cls, _PATCHED_ATTR, True)
    return True


def _wrap_anthropic_stream(stream: Any, hx: Any, session_id: str, user_msg: str) -> Any:
    """Wrap an Anthropic streaming response to collect chunks while passing through."""
    from .types import CaptureEvent

    chunks: List[str] = []

    class WrappedStream:
        """Proxy that wraps the original stream, collecting text deltas."""

        def __init__(self, original: Any) -> None:
            self._original = original

        def __iter__(self) -> "WrappedStream":
            self._iter = iter(self._original)
            return self

        def __next__(self) -> Any:
            try:
                event = next(self._iter)
                try:
                    ev_type = getattr(event, "type", None)
                    if ev_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta and hasattr(delta, "text") and isinstance(delta.text, str):
                            chunks.append(delta.text)
                except Exception:
                    pass
                return event
            except StopIteration:
                full_content = "".join(chunks)
                if full_content:
                    _fire_and_forget(hx, CaptureEvent(
                        type="message",
                        session_id=session_id,
                        payload={"role": "user", "content": user_msg},
                    ))
                    _fire_and_forget(hx, CaptureEvent(
                        type="message",
                        session_id=session_id,
                        payload={"role": "assistant", "content": full_content},
                    ))
                raise

        def __getattr__(self, name: str) -> Any:
            return getattr(self._original, name)

    return WrappedStream(stream)


# -- Init --

def _init() -> None:
    global _initialized
    if _initialized:
        return
    _initialized = True

    config = resolve_config()
    if not config:
        _warn(
            "no API key found. Set HIPPOCORTEX_API_KEY or create .hippocortex.json. "
            "Auto-instrumentation disabled."
        )
        return

    from .client import SyncHippocortex

    hx = SyncHippocortex(
        api_key=config["apiKey"],
        base_url=config.get("baseUrl", DEFAULT_BASE_URL),
    )

    session_id = _generate_session_id()

    openai_patched = _patch_openai(hx, session_id)
    anthropic_patched = _patch_anthropic(hx, session_id)

    if openai_patched or anthropic_patched:
        tenant_hint = config["apiKey"][:12] + "..."
        sdks = ", ".join(
            name
            for name, patched in [("OpenAI", openai_patched), ("Anthropic", anthropic_patched)]
            if patched
        )
        _log(f"auto-instrumentation active for {sdks} (capturing to {tenant_hint})")
    else:
        _log("auto-instrumentation loaded but no supported SDKs found (OpenAI/Anthropic).")


# Self-executing initialization on import
try:
    _init()
except Exception:
    # Never crash the host process
    pass
