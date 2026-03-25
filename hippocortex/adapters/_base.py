"""Hippocortex adapter core — shared logic for all framework adapters.

Key behaviors:
    - All capture calls are fire-and-forget (never block the agent).
    - All errors are swallowed with logging.warning (agent must never crash
      because of Hippocortex).
    - Synthesize returns empty list on any failure.
    - Session IDs auto-generated if not provided.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

import httpx

from ..types import CaptureEvent, SynthesisEntry, SynthesizeResult

logger = logging.getLogger("hippocortex.adapter")

DEFAULT_BASE_URL = "https://api.hippocortex.dev/v1"
DEFAULT_TIMEOUT_S = 10.0  # Lower timeout for adapters — fail fast


class HippocortexAdapter:
    """Shared adapter core for framework integrations.

    Provides fire-and-forget capture, safe synthesize, and context injection.
    Designed to never interfere with the host agent's execution.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        session_id: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ):
        self._api_key = api_key or os.environ.get("HIPPOCORTEX_API_KEY", "")
        if not self._api_key:
            logger.warning(
                "No Hippocortex API key provided. Set HIPPOCORTEX_API_KEY or pass api_key. "
                "Memory features will be disabled."
            )
        self._base_url = (base_url or os.environ.get("HIPPOCORTEX_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self._session_id = session_id or f"auto-{uuid.uuid4().hex[:12]}"
        self._timeout = timeout
        self._async_client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None
        self._lock = threading.Lock()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    # ── Async API ──

    async def capture(
        self,
        event_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fire-and-forget capture. Never raises, never blocks."""
        if not self.enabled:
            return
        try:
            event = CaptureEvent(
                type=event_type,  # type: ignore[arg-type]
                session_id=self._session_id,
                payload=payload,
                metadata=metadata,
            )
            client = self._get_async_client()
            resp = await client.post("/capture", json=event.to_dict())
            if resp.status_code >= 400:
                logger.warning(
                    "Hippocortex capture returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception as exc:
            logger.warning("Hippocortex capture failed (swallowed): %s", exc)

    async def synthesize(self, query: str, max_tokens: int = 4000) -> List[SynthesisEntry]:
        """Synthesize context. Returns empty list on any failure."""
        if not self.enabled:
            return []
        try:
            client = self._get_async_client()
            resp = await client.post(
                "/synthesize",
                json={
                    "query": query,
                    "options": {"maxTokens": max_tokens},
                },
            )
            if resp.status_code >= 400:
                logger.warning("Hippocortex synthesize returned %d", resp.status_code)
                return []
            data = resp.json()
            if not data.get("ok"):
                return []
            result = SynthesizeResult.from_dict(data["data"])
            return result.entries
        except Exception as exc:
            logger.warning("Hippocortex synthesize failed (swallowed): %s", exc)
            return []

    async def inject_context(
        self,
        messages: List[Dict[str, Any]],
        query: str,
        max_tokens: int = 4000,
    ) -> List[Dict[str, Any]]:
        """Synthesize context and prepend as a system message.

        Returns the original messages list (possibly with a prepended system
        message) — never modifies the input in-place.
        """
        entries = await self.synthesize(query, max_tokens)
        if not entries:
            return messages

        context_parts = []
        for entry in entries:
            context_parts.append(f"[{entry.section}] (confidence: {entry.confidence:.2f})\n{entry.content}")

        context_text = (
            "# Hippocortex Memory Context\n"
            "The following is synthesized context from past experience. "
            "Use it to inform your responses.\n\n"
            + "\n\n".join(context_parts)
        )

        system_msg = {"role": "system", "content": context_text}
        return [system_msg] + list(messages)

    # ── Sync API (for frameworks that don't use asyncio) ──

    def capture_sync(
        self,
        event_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Synchronous fire-and-forget capture."""
        if not self.enabled:
            return
        try:
            event = CaptureEvent(
                type=event_type,  # type: ignore[arg-type]
                session_id=self._session_id,
                payload=payload,
                metadata=metadata,
            )
            client = self._get_sync_client()
            resp = client.post("/capture", json=event.to_dict())
            if resp.status_code >= 400:
                logger.warning(
                    "Hippocortex capture returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception as exc:
            logger.warning("Hippocortex capture failed (swallowed): %s", exc)

    def synthesize_sync(self, query: str, max_tokens: int = 4000) -> List[SynthesisEntry]:
        """Synchronous synthesize. Returns empty list on any failure."""
        if not self.enabled:
            return []
        try:
            client = self._get_sync_client()
            resp = client.post(
                "/synthesize",
                json={
                    "query": query,
                    "options": {"maxTokens": max_tokens},
                },
            )
            if resp.status_code >= 400:
                logger.warning("Hippocortex synthesize returned %d", resp.status_code)
                return []
            data = resp.json()
            if not data.get("ok"):
                return []
            result = SynthesizeResult.from_dict(data["data"])
            return result.entries
        except Exception as exc:
            logger.warning("Hippocortex synthesize failed (swallowed): %s", exc)
            return []

    def inject_context_sync(
        self,
        messages: List[Dict[str, Any]],
        query: str,
        max_tokens: int = 4000,
    ) -> List[Dict[str, Any]]:
        """Synchronous context injection."""
        entries = self.synthesize_sync(query, max_tokens)
        if not entries:
            return messages

        context_parts = []
        for entry in entries:
            context_parts.append(f"[{entry.section}] (confidence: {entry.confidence:.2f})\n{entry.content}")

        context_text = (
            "# Hippocortex Memory Context\n"
            "The following is synthesized context from past experience. "
            "Use it to inform your responses.\n\n"
            + "\n\n".join(context_parts)
        )

        system_msg = {"role": "system", "content": context_text}
        return [system_msg] + list(messages)

    # ── HTTP Client Management ──

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=self._timeout,
            )
        return self._async_client

    def _get_sync_client(self) -> httpx.Client:
        with self._lock:
            if self._sync_client is None or self._sync_client.is_closed:
                self._sync_client = httpx.Client(
                    base_url=self._base_url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    timeout=self._timeout,
                )
        return self._sync_client

    async def close(self) -> None:
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()
