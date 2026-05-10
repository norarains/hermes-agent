"""NapCat voice helpers.

Keep QQ voice handling out of gateway core.  NapCat speech is transcribed
inside the adapter so the message reaches the agent as plain text — that
bypasses the gateway's auto-TTS-on-voice path (designed for voice-first
platforms like Discord channels), which on QQ DMs would otherwise produce
a redundant TTS bubble alongside the text reply.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Tuple

from .prompts import build_voice_system_instruction

logger = logging.getLogger(__name__)


VOICE_PLACEHOLDER = "[语音]"


async def transcribe_voice_files(
    voice_paths: List[str],
) -> List[Tuple[str, Optional[str]]]:
    """Run STT on each voice file in order.

    Returns one ``(transcript, error)`` tuple per input path.  A success
    has ``error=None`` (transcript may still be empty for silent audio);
    a failure has ``transcript=""`` and a non-empty ``error``.

    Sequential because local faster-whisper is CPU-bound and would just
    contend if run in parallel.
    """
    if not voice_paths:
        return []

    from tools.transcription_tools import transcribe_audio

    results: List[Tuple[str, Optional[str]]] = []
    for path in voice_paths:
        try:
            outcome = await asyncio.to_thread(transcribe_audio, path)
        except Exception as exc:
            logger.warning("NapCat voice transcription crashed (%s): %s", path, exc)
            results.append(("", str(exc)))
            continue
        if outcome.get("success"):
            results.append(((outcome.get("transcript") or "").strip(), None))
        else:
            results.append(("", outcome.get("error") or "unknown STT error"))
    return results


def compose_voice_message_text(
    *,
    parsed_text: str,
    transcripts: List[Tuple[str, Optional[str]]],
) -> str:
    """Strip voice placeholders from ``parsed_text`` and append transcripts.

    Each transcript becomes a paragraph in the order it was sent.  Failures
    and silent recordings get a brief bracketed marker so the agent can
    acknowledge them.  A trailing ``[System instruction: ...]`` line lets
    the agent know the preceding text came from STT (so it should tolerate
    transcription noise / ask for clarification).
    """
    if not transcripts:
        return (parsed_text or "").strip()

    stripped = parsed_text or ""
    for _ in transcripts:
        stripped = stripped.replace(VOICE_PLACEHOLDER, "", 1)
    stripped = stripped.strip()

    voice_lines: List[str] = []
    last_error: Optional[str] = None
    for transcript, err in transcripts:
        if err:
            last_error = err
            voice_lines.append("[Voice message — transcription failed]")
        elif not transcript:
            voice_lines.append("[Voice message — silent or unintelligible]")
        else:
            voice_lines.append(transcript)

    got_any_text = any(t for t, _ in transcripts)
    sys_instr = build_voice_system_instruction(
        transcription_failed=not got_any_text,
        error=last_error if not got_any_text else None,
    )

    parts: List[str] = []
    if stripped:
        parts.append(stripped)
    parts.extend(voice_lines)
    if sys_instr:
        parts.append(sys_instr)
    return "\n\n".join(p for p in parts if p)
