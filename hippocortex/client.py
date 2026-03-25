"""Hippocortex SDK — Python API client.

Mirrors the TypeScript SDK for API compatibility.
Thread-safe, async-first with sync fallback.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from .types import (
    Artifact,
    ArtifactListResult,
    ArtifactStatus,
    ArtifactType,
    BatchCaptureResult,
    CaptureEvent,
    CaptureResult,
    LearnOptions,
    LearnResult,
    MetricsResult,
    SynthesizeOptions,
    SynthesizeResult,
    VaultQueryResult,
    VaultRevealResult,
)

logger = logging.getLogger("hippocortex")

DEFAULT_BASE_URL = "https://api.hippocortex.dev/v1"
DEFAULT_TIMEOUT_S = 30.0

# Current SDK version
SDK_VERSION = "1.2.0"

# Module-level flag: only warn about updates ONCE per process
_update_warning_emitted = False


def _is_newer_version(current: str, latest: str) -> bool:
    """Compare two semver strings. Returns True if latest > current."""
    def parse(v: str) -> List[int]:
        return [int(x) for x in v.lstrip("v").split(".")]

    try:
        c = parse(current)
        l = parse(latest)
        # Pad to 3 elements
        while len(c) < 3:
            c.append(0)
        while len(l) < 3:
            l.append(0)
        return l > c
    except (ValueError, IndexError):
        return False


def _check_update_warning(
    response: httpx.Response,
    suppress: bool,
) -> None:
    """Check response headers for SDK update info and warn once."""
    global _update_warning_emitted
    if _update_warning_emitted or suppress:
        return
    latest = response.headers.get("x-hippocortex-latest-sdk-python")
    if latest and _is_newer_version(SDK_VERSION, latest):
        _update_warning_emitted = True
        logger.warning(
            "\u26a0\ufe0f hippocortex v%s is outdated. Latest: v%s\n"
            "  Update: pip install --upgrade hippocortex",
            SDK_VERSION,
            latest,
        )


class HippocortexError(Exception):
    """Error from the Hippocortex API."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        details: Optional[List[Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = details


class Hippocortex:
    """Hippocortex Python client.

    Usage::

        from hippocortex import Hippocortex

        hx = Hippocortex(api_key="hx_live_...")
        result = await hx.capture(CaptureEvent(...))

    For sync usage, use ``hx.capture_sync(...)`` variants or use
    ``hippocortex.SyncHippocortex``.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_S,
        suppress_update_warning: bool = False,
    ):
        if not api_key:
            raise ValueError(
                "API key is required. Get one at https://dashboard.hippocortex.dev"
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._suppress_update_warning = suppress_update_warning
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-Hippocortex-SDK-Version": f"python/{SDK_VERSION}",
                },
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "Hippocortex":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ── Core Primitives ──

    async def capture(self, event: CaptureEvent) -> CaptureResult:
        """Capture an agent event into Hippocortex memory."""
        data = await self._post("/capture", event.to_dict())
        return CaptureResult.from_dict(data)

    async def capture_batch(self, events: List[CaptureEvent]) -> BatchCaptureResult:
        """Capture multiple events in a single request."""
        data = await self._post(
            "/capture/batch", {"events": [e.to_dict() for e in events]}
        )
        return BatchCaptureResult.from_dict(data)

    async def learn(self, options: Optional[LearnOptions] = None) -> LearnResult:
        """Trigger the Memory Compiler."""
        opts = options or LearnOptions()
        body: Dict[str, Any] = {"scope": opts.scope}
        opt_inner: Dict[str, Any] = {}
        if opts.min_pattern_strength is not None:
            opt_inner["minPatternStrength"] = opts.min_pattern_strength
        if opts.artifact_types is not None:
            opt_inner["artifactTypes"] = opts.artifact_types
        if opt_inner:
            body["options"] = opt_inner
        data = await self._post("/learn", body)
        return LearnResult.from_dict(data)

    async def synthesize(
        self, query: str, options: Optional[SynthesizeOptions] = None
    ) -> SynthesizeResult:
        """Synthesize compressed context from all memory layers."""
        opts = options or SynthesizeOptions()
        body: Dict[str, Any] = {
            "query": query,
            "options": {
                "maxTokens": opts.max_tokens,
                "minConfidence": opts.min_confidence,
                "includeProvenance": opts.include_provenance,
            },
        }
        if opts.sections:
            body["options"]["sections"] = opts.sections
        data = await self._post("/synthesize", body)
        return SynthesizeResult.from_dict(data)

    # ── Artifacts & Metrics ──

    async def list_artifacts(
        self,
        type: Optional[ArtifactType] = None,
        status: Optional[ArtifactStatus] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> ArtifactListResult:
        """List compiled knowledge artifacts."""
        params: Dict[str, str] = {}
        if type:
            params["type"] = type
        if status:
            params["status"] = status
        if sort:
            params["sort"] = sort
        if order:
            params["order"] = order
        if limit is not None:
            params["limit"] = str(limit)
        if cursor:
            params["cursor"] = cursor
        qs = f"?{urlencode(params)}" if params else ""
        data = await self._get(f"/artifacts{qs}")
        return ArtifactListResult.from_dict(data)

    async def get_artifact(self, artifact_id: str) -> Artifact:
        """Get a single artifact by ID."""
        data = await self._get(f"/artifacts/{artifact_id}")
        return Artifact.from_dict(data)

    async def get_metrics(
        self,
        period: Optional[str] = None,
        granularity: Optional[str] = None,
    ) -> MetricsResult:
        """Get usage and performance metrics."""
        params: Dict[str, str] = {}
        if period:
            params["period"] = period
        if granularity:
            params["granularity"] = granularity
        qs = f"?{urlencode(params)}" if params else ""
        data = await self._get(f"/usage-metrics{qs}")
        return MetricsResult.from_dict(data)

    # ── Vault ──

    async def vault_query(
        self,
        query: str,
        *,
        tags: Optional[List[str]] = None,
        item_type: Optional[str] = None,
        limit: int = 20,
    ) -> VaultQueryResult:
        """Search the vault for secrets by natural language query.

        Returns metadata only (titles, types, tags) — never decrypted values.
        """
        body: Dict[str, Any] = {"query": query, "limit": limit}
        if tags:
            body["tags"] = tags
        if item_type:
            body["itemType"] = item_type
        data = await self._post("/vault/query", body)
        return VaultQueryResult.from_dict(data)

    async def vault_reveal(self, item_id: str) -> VaultRevealResult:
        """Reveal (decrypt) a specific vault secret by item ID.

        Requires reveal permission. All access is audited and rate-limited.
        """
        data = await self._post(f"/vault/query/{item_id}/reveal", {})
        return VaultRevealResult.from_dict(data)

    # ── HTTP Layer ──

    async def _request(
        self, method: str, path: str, body: Optional[Dict[str, Any]] = None
    ) -> Any:
        try:
            if method == "GET":
                resp = await self._http.get(path)
            else:
                resp = await self._http.post(path, json=body)

            # Check for SDK update (once per process)
            _check_update_warning(resp, self._suppress_update_warning)

            json_data = resp.json()

            if not json_data.get("ok") or json_data.get("error"):
                err = json_data.get("error", {})
                raise HippocortexError(
                    code=err.get("code", "unknown_error"),
                    message=err.get("message", f"HTTP {resp.status_code}"),
                    status_code=resp.status_code,
                    details=err.get("details"),
                )

            return json_data.get("data")
        except httpx.HTTPError as exc:
            raise HippocortexError(
                code="network_error",
                message=str(exc),
                status_code=0,
            ) from exc

    async def _get(self, path: str) -> Any:
        return await self._request("GET", path)

    async def _post(self, path: str, body: Dict[str, Any]) -> Any:
        return await self._request("POST", path, body)


class SyncHippocortex:
    """Synchronous wrapper around the async Hippocortex client.

    Useful for non-async contexts (scripts, notebooks, adapters).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_S,
        suppress_update_warning: bool = False,
    ):
        if not api_key:
            raise ValueError(
                "API key is required. Get one at https://dashboard.hippocortex.dev"
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._suppress_update_warning = suppress_update_warning
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Hippocortex-SDK-Version": f"python/{SDK_VERSION}",
            },
            timeout=self._timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SyncHippocortex":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def capture(self, event: CaptureEvent) -> CaptureResult:
        data = self._post("/capture", event.to_dict())
        return CaptureResult.from_dict(data)

    def capture_batch(self, events: List[CaptureEvent]) -> BatchCaptureResult:
        data = self._post(
            "/capture/batch", {"events": [e.to_dict() for e in events]}
        )
        return BatchCaptureResult.from_dict(data)

    def learn(self, options: Optional[LearnOptions] = None) -> LearnResult:
        opts = options or LearnOptions()
        body: Dict[str, Any] = {"scope": opts.scope}
        opt_inner: Dict[str, Any] = {}
        if opts.min_pattern_strength is not None:
            opt_inner["minPatternStrength"] = opts.min_pattern_strength
        if opts.artifact_types is not None:
            opt_inner["artifactTypes"] = opts.artifact_types
        if opt_inner:
            body["options"] = opt_inner
        data = self._post("/learn", body)
        return LearnResult.from_dict(data)

    def synthesize(
        self, query: str, options: Optional[SynthesizeOptions] = None
    ) -> SynthesizeResult:
        opts = options or SynthesizeOptions()
        body: Dict[str, Any] = {
            "query": query,
            "options": {
                "maxTokens": opts.max_tokens,
                "minConfidence": opts.min_confidence,
                "includeProvenance": opts.include_provenance,
            },
        }
        if opts.sections:
            body["options"]["sections"] = opts.sections
        data = self._post("/synthesize", body)
        return SynthesizeResult.from_dict(data)

    def vault_query(
        self,
        query: str,
        *,
        tags: Optional[List[str]] = None,
        item_type: Optional[str] = None,
        limit: int = 20,
    ) -> VaultQueryResult:
        """Search the vault for secrets by natural language query (sync)."""
        body: Dict[str, Any] = {"query": query, "limit": limit}
        if tags:
            body["tags"] = tags
        if item_type:
            body["itemType"] = item_type
        data = self._post("/vault/query", body)
        return VaultQueryResult.from_dict(data)

    def vault_reveal(self, item_id: str) -> VaultRevealResult:
        """Reveal (decrypt) a specific vault secret by item ID (sync)."""
        data = self._post(f"/vault/query/{item_id}/reveal", {})
        return VaultRevealResult.from_dict(data)

    def _request(
        self, method: str, path: str, body: Optional[Dict[str, Any]] = None
    ) -> Any:
        try:
            if method == "GET":
                resp = self._client.get(path)
            else:
                resp = self._client.post(path, json=body)

            # Check for SDK update (once per process)
            _check_update_warning(resp, self._suppress_update_warning)

            json_data = resp.json()

            if not json_data.get("ok") or json_data.get("error"):
                err = json_data.get("error", {})
                raise HippocortexError(
                    code=err.get("code", "unknown_error"),
                    message=err.get("message", f"HTTP {resp.status_code}"),
                    status_code=resp.status_code,
                    details=err.get("details"),
                )

            return json_data.get("data")
        except httpx.HTTPError as exc:
            raise HippocortexError(
                code="network_error",
                message=str(exc),
                status_code=0,
            ) from exc

    def _get(self, path: str) -> Any:
        return self._request("GET", path)

    def _post(self, path: str, body: Dict[str, Any]) -> Any:
        return self._request("POST", path, body)
