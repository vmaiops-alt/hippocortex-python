"""Hippocortex SDK -- Client-side memory extraction.

Uses the user's own LLM client to extract memorable facts from conversations.
This eliminates server-side LLM costs entirely.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger("hippocortex.extract")

EXTRACTION_PROMPT = """Extract key facts, preferences, decisions, and important information from this conversation turn. Return as a JSON array of short strings. Only include genuinely memorable facts. If nothing notable, return [].

User: {user_message}
Assistant: {assistant_response}"""

EXTRACTION_TIMEOUT_S = 3.0

# Cheaper models preferred for extraction to minimize token overhead
OPENAI_EXTRACTION_MODEL = "gpt-4o-mini"
ANTHROPIC_EXTRACTION_MODEL = "claude-3-haiku-20240307"


def _is_openai_client(client: Any) -> bool:
    return (
        hasattr(client, "chat")
        and hasattr(client.chat, "completions")
        and hasattr(client.chat.completions, "create")
    )


def _is_anthropic_client(client: Any) -> bool:
    return hasattr(client, "messages") and hasattr(client.messages, "create")


def _build_extraction_prompt(user_message: str, assistant_response: str) -> str:
    return EXTRACTION_PROMPT.replace(
        "{user_message}", user_message
    ).replace("{assistant_response}", assistant_response)


def _parse_extracted_facts(text: str) -> List[str]:
    """Parse a JSON array of strings from LLM response text."""
    try:
        import re
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return []
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, str) and item.strip()]
    except (json.JSONDecodeError, Exception):
        return []


def _extract_last_user_message(messages: List[Any]) -> Optional[str]:
    """Extract the last user message content from a messages list."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
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


def _extract_via_openai_sync(
    client: Any,
    user_message: str,
    assistant_response: str,
    create_fn: Any = None,
) -> List[str]:
    """Extract memories using an OpenAI-compatible client (synchronous)."""
    prompt = _build_extraction_prompt(user_message, assistant_response)
    create = create_fn or client.chat.completions.create

    response = create(
        model=OPENAI_EXTRACTION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )

    # Handle both object and dict responses
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            content = choice.message.content or ""
            return _parse_extracted_facts(content)
    elif isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", "")
            return _parse_extracted_facts(content)

    return []


def _extract_via_anthropic_sync(
    client: Any,
    user_message: str,
    assistant_response: str,
    create_fn: Any = None,
) -> List[str]:
    """Extract memories using an Anthropic client (synchronous)."""
    prompt = _build_extraction_prompt(user_message, assistant_response)
    create = create_fn or client.messages.create

    response = create(
        model=ANTHROPIC_EXTRACTION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )

    # Handle both object and dict responses
    content_list = getattr(response, "content", None)
    if isinstance(content_list, list):
        text_parts = []
        for b in content_list:
            b_type = getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else None)
            if b_type == "text":
                text = getattr(b, "text", "") if hasattr(b, "text") else b.get("text", "")
                text_parts.append(text)
        text = "".join(text_parts)
        return _parse_extracted_facts(text)

    return []


def extract_memories_sync(
    messages: List[Any],
    response: str,
    client: Any,
    create_fn: Any = None,
) -> List[str]:
    """Extract memorable facts synchronously using the user's own LLM client.

    Uses a cheap/small model (gpt-4o-mini or claude-3-haiku) to minimize cost.
    Returns an empty list if extraction fails for any reason.

    Args:
        messages: The conversation messages (to find the last user message).
        response: The assistant's response text.
        client: An OpenAI or Anthropic client instance.
        create_fn: Optional original (unpatched) create function to avoid recursion.
    """
    if os.environ.get("HIPPOCORTEX_EXTRACT") == "false":
        return []

    try:
        user_message = _extract_last_user_message(messages)
        if not user_message or not response:
            return []

        if _is_openai_client(client):
            return _extract_via_openai_sync(client, user_message, response, create_fn)

        if _is_anthropic_client(client):
            return _extract_via_anthropic_sync(client, user_message, response, create_fn)

        return []
    except Exception as e:
        logger.warning(f"memory extraction failed (non-fatal): {e}")
        return []


async def extract_memories(
    messages: List[Any],
    response: str,
    client: Any,
    create_fn: Any = None,
) -> List[str]:
    """Extract memorable facts asynchronously using the user's own LLM client.

    Uses a cheap/small model (gpt-4o-mini or claude-3-haiku) to minimize cost.
    Returns an empty list if extraction fails for any reason.

    Args:
        messages: The conversation messages (to find the last user message).
        response: The assistant's response text.
        client: An OpenAI or Anthropic client instance.
        create_fn: Optional original (unpatched) create function to avoid recursion.
    """
    if os.environ.get("HIPPOCORTEX_EXTRACT") == "false":
        return []

    try:
        # Run synchronous extraction with a timeout
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                extract_memories_sync,
                messages,
                response,
                client,
                create_fn,
            ),
            timeout=EXTRACTION_TIMEOUT_S,
        )
        return result
    except asyncio.TimeoutError:
        logger.warning("memory extraction timed out (non-fatal)")
        return []
    except Exception as e:
        logger.warning(f"memory extraction failed (non-fatal): {e}")
        return []
