"""Hippocortex SDK — Type definitions.

Mirrors the TypeScript SDK types for API compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


# ── Capture ──

CaptureEventType = Literal[
    "message",
    "tool_call",
    "tool_result",
    "file_edit",
    "test_run",
    "command_exec",
    "browser_action",
    "api_result",
]


@dataclass
class CaptureEvent:
    """An event to be captured into Hippocortex memory."""

    type: CaptureEventType
    session_id: str
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": self.type,
            "sessionId": self.session_id,
            "payload": self.payload,
        }
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d


@dataclass
class CaptureResult:
    event_id: str
    status: Literal["ingested", "duplicate"]
    salience_score: Optional[float] = None
    trace_id: Optional[str] = None
    reason: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CaptureResult":
        return cls(
            event_id=d["eventId"],
            status=d["status"],
            salience_score=d.get("salienceScore"),
            trace_id=d.get("traceId"),
            reason=d.get("reason"),
        )


@dataclass
class BatchCaptureResult:
    results: List[CaptureResult]
    total: int
    ingested: int
    duplicates: int
    errors: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BatchCaptureResult":
        summary = d.get("summary", {})
        return cls(
            results=[CaptureResult.from_dict(r) for r in d.get("results", [])],
            total=summary.get("total", 0),
            ingested=summary.get("ingested", 0),
            duplicates=summary.get("duplicates", 0),
            errors=summary.get("errors", 0),
        )


# ── Learn ──

ArtifactType = Literal[
    "task_schema",
    "failure_playbook",
    "causal_pattern",
    "decision_policy",
]


@dataclass
class LearnOptions:
    scope: Literal["full", "incremental"] = "incremental"
    min_pattern_strength: Optional[float] = None
    artifact_types: Optional[List[ArtifactType]] = None


@dataclass
class LearnResult:
    run_id: str
    status: Literal["completed", "partial", "failed"]
    artifacts_created: int
    artifacts_updated: int
    artifacts_unchanged: int
    artifacts_by_type: Dict[str, int]
    memories_processed: int
    patterns_found: int
    compilation_ms: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LearnResult":
        artifacts = d.get("artifacts", {})
        stats = d.get("stats", {})
        return cls(
            run_id=d["runId"],
            status=d["status"],
            artifacts_created=artifacts.get("created", 0),
            artifacts_updated=artifacts.get("updated", 0),
            artifacts_unchanged=artifacts.get("unchanged", 0),
            artifacts_by_type=artifacts.get("byType", {}),
            memories_processed=stats.get("memoriesProcessed", 0),
            patterns_found=stats.get("patternsFound", 0),
            compilation_ms=stats.get("compilationMs", 0),
        )


# ── Synthesize ──

ReasoningSection = Literal[
    "procedures",
    "failures",
    "decisions",
    "facts",
    "causal",
    "context",
]


@dataclass
class SynthesizeOptions:
    max_tokens: int = 4000
    sections: Optional[List[ReasoningSection]] = None
    min_confidence: float = 0.3
    include_provenance: bool = True


@dataclass
class ProvenanceRef:
    source_type: str
    source_id: str
    artifact_type: Optional[str] = None
    evidence_count: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProvenanceRef":
        return cls(
            source_type=d["sourceType"],
            source_id=d["sourceId"],
            artifact_type=d.get("artifactType"),
            evidence_count=d.get("evidenceCount"),
        )


@dataclass
class SynthesisEntry:
    section: ReasoningSection
    content: str
    confidence: float
    provenance: Optional[List[ProvenanceRef]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SynthesisEntry":
        prov = d.get("provenance")
        return cls(
            section=d["section"],
            content=d["content"],
            confidence=d["confidence"],
            provenance=[ProvenanceRef.from_dict(p) for p in prov] if prov else None,
        )


@dataclass
class SynthesizeBudget:
    limit: int
    used: int
    compression_ratio: float
    entries_included: int
    entries_dropped: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SynthesizeBudget":
        return cls(
            limit=d["limit"],
            used=d["used"],
            compression_ratio=d["compressionRatio"],
            entries_included=d["entriesIncluded"],
            entries_dropped=d["entriesDropped"],
        )


@dataclass
class SynthesizeResult:
    pack_id: str
    entries: List[SynthesisEntry]
    budget: SynthesizeBudget

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SynthesizeResult":
        return cls(
            pack_id=d["packId"],
            entries=[SynthesisEntry.from_dict(e) for e in d.get("entries", [])],
            budget=SynthesizeBudget.from_dict(d["budget"]),
        )


# ── Artifacts ──

ArtifactStatus = Literal["active", "deprecated", "superseded"]


@dataclass
class Artifact:
    id: str
    type: ArtifactType
    status: ArtifactStatus
    title: str
    content: Dict[str, Any]
    confidence: float
    evidence_count: int
    created_at: str
    updated_at: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Artifact":
        return cls(
            id=d["id"],
            type=d["type"],
            status=d["status"],
            title=d["title"],
            content=d["content"],
            confidence=d["confidence"],
            evidence_count=d["evidenceCount"],
            created_at=d["createdAt"],
            updated_at=d["updatedAt"],
        )


@dataclass
class ArtifactListResult:
    artifacts: List[Artifact]
    has_more: bool
    cursor: Optional[str]
    total: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArtifactListResult":
        pag = d.get("pagination", {})
        return cls(
            artifacts=[Artifact.from_dict(a) for a in d.get("artifacts", [])],
            has_more=pag.get("hasMore", False),
            cursor=pag.get("cursor"),
            total=pag.get("total", 0),
        )


# ── Metrics ──

@dataclass
class VaultQueryMatch:
    """A single vault search result (metadata only, no decrypted value)."""

    id: str
    vault_id: str
    vault_name: str
    title: str
    item_type: str
    service_name: Optional[str]
    tags: List[str]
    sensitivity: str
    description: Optional[str]
    created_at: str
    relevance: float

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VaultQueryMatch":
        return cls(
            id=d["id"],
            vault_id=d["vault_id"],
            vault_name=d["vault_name"],
            title=d["title"],
            item_type=d["item_type"],
            service_name=d.get("service_name"),
            tags=d.get("tags", []),
            sensitivity=d.get("sensitivity", "medium"),
            description=d.get("description"),
            created_at=d.get("created_at", ""),
            relevance=d.get("relevance", 0.0),
        )


@dataclass
class VaultQueryResult:
    """Result of a vault query — metadata only, never decrypted values."""

    matches: List[VaultQueryMatch]
    total: int
    query: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VaultQueryResult":
        return cls(
            matches=[VaultQueryMatch.from_dict(m) for m in d.get("matches", [])],
            total=d.get("total", 0),
            query=d.get("query", ""),
        )


@dataclass
class VaultRevealResult:
    """Result of a vault reveal — the decrypted secret value."""

    value: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VaultRevealResult":
        return cls(value=d["value"])


@dataclass
class MetricsResult:
    period_start: str
    period_end: str
    granularity: str
    events_total: int
    events_ingested: int
    quota_plan: str
    quota_events_used: int
    quota_events_remaining: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricsResult":
        period = d.get("period", {})
        usage = d.get("usage", {})
        events = usage.get("events", {})
        quota = d.get("quota", {})
        return cls(
            period_start=period.get("start", ""),
            period_end=period.get("end", ""),
            granularity=period.get("granularity", ""),
            events_total=events.get("total", 0),
            events_ingested=events.get("ingested", 0),
            quota_plan=quota.get("plan", ""),
            quota_events_used=quota.get("eventsUsed", 0),
            quota_events_remaining=quota.get("eventsRemaining", 0),
        )
