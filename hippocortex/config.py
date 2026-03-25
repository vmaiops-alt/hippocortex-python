"""Hippocortex SDK -- Zero-config: load .hippocortex.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(cwd: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Search for ``.hippocortex.json`` from *cwd* upward.

    Returns a dict with ``apiKey`` and optionally ``baseUrl``, or ``None``.
    """
    directory = Path(cwd) if cwd else Path.cwd()

    for _ in range(64):
        config_path = directory / ".hippocortex.json"
        if config_path.is_file():
            try:
                data: Dict[str, Any] = json.loads(config_path.read_text("utf-8"))
                api_key = data.get("apiKey")
                if isinstance(api_key, str) and api_key:
                    result: Dict[str, str] = {"apiKey": api_key}
                    base_url = data.get("baseUrl")
                    if isinstance(base_url, str) and base_url:
                        result["baseUrl"] = base_url
                    return result
            except (json.JSONDecodeError, OSError):
                pass

        parent = directory.parent
        if parent == directory:
            break
        directory = parent

    return None


def resolve_config(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Resolve Hippocortex config from options, env vars, or config file.

    Priority:
    1. Explicit arguments
    2. ``HIPPOCORTEX_API_KEY`` / ``HIPPOCORTEX_BASE_URL`` env vars
    3. ``.hippocortex.json`` file (cwd and parent dirs)
    """
    resolved_key = api_key or os.environ.get("HIPPOCORTEX_API_KEY") or None
    resolved_url = base_url or os.environ.get("HIPPOCORTEX_BASE_URL") or None

    if resolved_key:
        result: Dict[str, str] = {"apiKey": resolved_key}
        if resolved_url:
            result["baseUrl"] = resolved_url
        return result

    file_config = load_config()
    if file_config:
        if resolved_url:
            file_config["baseUrl"] = resolved_url
        return file_config

    return None
