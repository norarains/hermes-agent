"""Transcript helpers for the OneBot/NapCat adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional


_SYSTEM_INSTRUCTION_MARKER = "\n\n[System instruction:"


def strip_system_instruction(content: str) -> str:
    before, marker, _ = content.partition(_SYSTEM_INSTRUCTION_MARKER)
    if marker:
        return before
    if content.startswith("[System instruction:"):
        return ""
    return content


def sanitize_transcript_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if entry.get("role") != "user":
        return entry
    content = entry.get("content")
    if not isinstance(content, str):
        return entry
    cleaned = strip_system_instruction(content)
    if cleaned == content:
        return entry
    sanitized = dict(entry)
    sanitized["content"] = cleaned
    return sanitized
