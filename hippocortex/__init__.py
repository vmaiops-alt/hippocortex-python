"""Hippocortex — AI agent memory that learns from experience.

Usage::

    from hippocortex import Hippocortex, auto_memory

    # Direct client usage
    hx = Hippocortex(api_key="hx_live_...")

    # One-line auto-memory for OpenAI Agents
    agent = auto_memory(agent, api_key="hx_live_...")
"""

from .client import Hippocortex, HippocortexError, SyncHippocortex, SDK_VERSION
from .config import load_config, resolve_config
from .extract import extract_memories, extract_memories_sync
from .wrap import wrap
from .types import (
    Artifact,
    ArtifactListResult,
    ArtifactStatus,
    ArtifactType,
    BatchCaptureResult,
    CaptureEvent,
    CaptureEventType,
    CaptureResult,
    LearnOptions,
    LearnResult,
    MetricsResult,
    ProvenanceRef,
    ReasoningSection,
    SynthesisEntry,
    SynthesizeBudget,
    SynthesizeOptions,
    SynthesizeResult,
    VaultQueryMatch,
    VaultQueryResult,
    VaultRevealResult,
)


def auto_memory(agent, api_key=None, base_url=None, session_id=None, **kwargs):
    """One-line auto-memory wrapper for OpenAI Agents SDK.

    Wraps an agent to automatically capture events and inject synthesized
    context from Hippocortex memory.

    Args:
        agent: An OpenAI Agents SDK ``Agent`` instance.
        api_key: Hippocortex API key. Falls back to ``HIPPOCORTEX_API_KEY`` env var.
        base_url: API base URL override.
        session_id: Explicit session ID (auto-generated if omitted).
        **kwargs: Passed to the OpenAI adapter.

    Returns:
        The wrapped agent (same type, with memory hooks attached).
    """
    from .adapters.openai_agents import wrap

    return wrap(agent, api_key=api_key, base_url=base_url, session_id=session_id, **kwargs)


__version__ = SDK_VERSION

__all__ = [
    "Hippocortex",
    "HippocortexError",
    "SyncHippocortex",
    "SDK_VERSION",
    "auto_memory",
    "extract_memories",
    "extract_memories_sync",
    "wrap",
    "load_config",
    "resolve_config",
    "Artifact",
    "ArtifactListResult",
    "ArtifactStatus",
    "ArtifactType",
    "BatchCaptureResult",
    "CaptureEvent",
    "CaptureEventType",
    "CaptureResult",
    "LearnOptions",
    "LearnResult",
    "MetricsResult",
    "ProvenanceRef",
    "ReasoningSection",
    "SynthesisEntry",
    "SynthesizeBudget",
    "SynthesizeOptions",
    "SynthesizeResult",
    "VaultQueryMatch",
    "VaultQueryResult",
    "VaultRevealResult",
]
