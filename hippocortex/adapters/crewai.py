"""Hippocortex adapter for CrewAI.

Wraps a CrewAI ``Crew`` to automatically capture task executions
and inject synthesized context into agent backstories.

Usage::

    from hippocortex.adapters import crewai as hx_crewai

    crew = hx_crewai.wrap(crew, api_key="hx_live_...")
    result = crew.kickoff()

Status: Beta — CrewAI's internal API evolves frequently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ._base import HippocortexAdapter

logger = logging.getLogger("hippocortex.adapters.crewai")


def wrap(
    crew: Any,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    session_id: Optional[str] = None,
    inject_memory: bool = True,
    **kwargs: Any,
) -> Any:
    """Wrap a CrewAI Crew with Hippocortex auto-memory.

    Args:
        crew: A ``crewai.Crew`` instance.
        api_key: Hippocortex API key (falls back to ``HIPPOCORTEX_API_KEY``).
        base_url: API base URL override.
        session_id: Explicit session ID.
        inject_memory: Whether to inject synthesized context into agent backstories.

    Returns:
        A ``HippocortexCrew`` wrapper with the same ``kickoff()`` interface.
    """
    adapter = HippocortexAdapter(
        api_key=api_key,
        base_url=base_url,
        session_id=session_id,
    )

    if not adapter.enabled:
        logger.info("Hippocortex disabled (no API key). Returning crew unchanged.")
        return crew

    return HippocortexCrew(
        crew=crew,
        adapter=adapter,
        inject_memory=inject_memory,
    )


class HippocortexCrew:
    """Wrapper around a CrewAI Crew with memory hooks.

    Captures task descriptions and results; injects memory context
    into agent backstories before kickoff.
    """

    def __init__(
        self,
        crew: Any,
        adapter: HippocortexAdapter,
        inject_memory: bool = True,
    ):
        self._crew = crew
        self._adapter = adapter
        self._inject_memory = inject_memory

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._crew, name)

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """Run the crew with memory hooks."""
        # Build context query from task descriptions
        query = self._build_query(inputs)

        # Inject memory into agent backstories
        original_backstories: Dict[int, str] = {}
        if self._inject_memory and query:
            original_backstories = self._inject_backstories(query)

        # Capture task descriptions
        self._capture_tasks()

        try:
            result = self._crew.kickoff(inputs=inputs, **kwargs)
        finally:
            # Restore original backstories
            self._restore_backstories(original_backstories)

        # Capture crew result
        self._capture_result(result)

        return result

    async def akickoff(self, inputs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """Async kickoff with memory hooks."""
        query = self._build_query(inputs)

        original_backstories: Dict[int, str] = {}
        if self._inject_memory and query:
            entries = await self._adapter.synthesize(query)
            if entries:
                original_backstories = self._apply_backstories(entries)

        self._capture_tasks()

        try:
            if hasattr(self._crew, "akickoff"):
                result = await self._crew.akickoff(inputs=inputs, **kwargs)
            else:
                result = self._crew.kickoff(inputs=inputs, **kwargs)
        finally:
            self._restore_backstories(original_backstories)

        self._capture_result(result)
        return result

    def _build_query(self, inputs: Optional[Dict[str, Any]] = None) -> str:
        """Build a synthesize query from crew tasks."""
        parts = []
        try:
            tasks = getattr(self._crew, "tasks", []) or []
            for task in tasks:
                desc = getattr(task, "description", "")
                if desc:
                    # Interpolate inputs if provided
                    if inputs:
                        try:
                            desc = desc.format(**inputs)
                        except (KeyError, IndexError):
                            pass
                    parts.append(desc)
        except Exception:
            pass
        return " | ".join(parts) if parts else ""

    def _inject_backstories(self, query: str) -> Dict[int, str]:
        """Inject synthesized context into agent backstories (sync)."""
        entries = self._adapter.synthesize_sync(query)
        if not entries:
            return {}
        return self._apply_backstories(entries)

    def _apply_backstories(self, entries: list) -> Dict[int, str]:
        """Apply memory context to agent backstories."""
        context_text = _build_context_text(entries)
        original: Dict[int, str] = {}

        try:
            agents = getattr(self._crew, "agents", []) or []
            for i, agent in enumerate(agents):
                backstory = getattr(agent, "backstory", "") or ""
                original[i] = backstory
                agent.backstory = context_text + "\n\n" + backstory
        except Exception as exc:
            logger.warning("Failed to inject backstories: %s", exc)

        return original

    def _restore_backstories(self, original: Dict[int, str]) -> None:
        """Restore original agent backstories."""
        if not original:
            return
        try:
            agents = getattr(self._crew, "agents", []) or []
            for i, backstory in original.items():
                if i < len(agents):
                    agents[i].backstory = backstory
        except Exception:
            pass

    def _capture_tasks(self) -> None:
        """Capture task descriptions."""
        try:
            tasks = getattr(self._crew, "tasks", []) or []
            for task in tasks:
                desc = getattr(task, "description", "")
                agent_role = ""
                if hasattr(task, "agent") and task.agent:
                    agent_role = getattr(task.agent, "role", "")
                if desc:
                    self._adapter.capture_sync(
                        "tool_call",
                        {
                            "tool_name": f"crewai_task:{agent_role}",
                            "input": {"description": desc[:1000]},
                        },
                        {"source": "crewai"},
                    )
        except Exception as exc:
            logger.warning("Failed to capture tasks: %s", exc)

    def _capture_result(self, result: Any) -> None:
        """Capture crew execution result."""
        try:
            result_text = ""
            if hasattr(result, "raw"):
                result_text = str(result.raw)[:2000]
            elif isinstance(result, str):
                result_text = result[:2000]
            else:
                result_text = str(result)[:2000]

            if result_text:
                self._adapter.capture_sync(
                    "message",
                    {"role": "assistant", "content": result_text},
                    {"source": "crewai", "session_id": self._adapter.session_id},
                )
        except Exception as exc:
            logger.warning("Failed to capture crew result: %s", exc)


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
