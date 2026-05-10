"""Tests for NapCat voice transcription handling.

Voice messages received from QQ are transcribed inside the NapCat
adapter and forwarded to the agent as plain text.  This guarantees the
gateway's auto-TTS-on-voice path (which is designed for voice-first
platforms like Discord channels) does not fire and produce a redundant
TTS bubble alongside the text reply.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from gateway.platforms.onebot_napcat.adapter import OneBotNapCatAdapter
from gateway.platforms.onebot_napcat.prompts import (
    build_voice_system_instruction,
)
from gateway.platforms.onebot_napcat.voice import (
    VOICE_PLACEHOLDER,
    compose_voice_message_text,
)


# ---------------------------------------------------------------------------
# build_voice_system_instruction
# ---------------------------------------------------------------------------


def test_voice_system_instruction_success_mentions_transcription():
    text = build_voice_system_instruction()
    assert text is not None
    assert "transcription" in text.lower()
    assert "text only" in text.lower()


def test_voice_system_instruction_failure_includes_reason():
    text = build_voice_system_instruction(
        transcription_failed=True, error="STT provider unreachable",
    )
    assert text is not None
    assert "failed" in text.lower()
    assert "STT provider unreachable" in text


def test_voice_system_instruction_failure_without_reason_still_actionable():
    text = build_voice_system_instruction(transcription_failed=True)
    assert text is not None
    assert "failed" in text.lower()
    # Must still tell the agent what to do
    assert "retry" in text.lower() or "type" in text.lower()


# ---------------------------------------------------------------------------
# compose_voice_message_text
# ---------------------------------------------------------------------------


def test_compose_voice_only_strips_placeholder_and_appends_transcript():
    text = compose_voice_message_text(
        parsed_text=VOICE_PLACEHOLDER,
        transcripts=[("hello there", None)],
    )
    assert "hello there" in text
    assert VOICE_PLACEHOLDER not in text
    assert "[System instruction:" in text


def test_compose_voice_with_text_caption_keeps_both():
    text = compose_voice_message_text(
        parsed_text=f"hi {VOICE_PLACEHOLDER}",
        transcripts=[("listening", None)],
    )
    assert text.startswith("hi")
    assert "listening" in text
    assert VOICE_PLACEHOLDER not in text


def test_compose_multiple_voice_messages_strips_each_placeholder():
    text = compose_voice_message_text(
        parsed_text=f"{VOICE_PLACEHOLDER}{VOICE_PLACEHOLDER}",
        transcripts=[("first", None), ("second", None)],
    )
    assert "first" in text
    assert "second" in text
    assert VOICE_PLACEHOLDER not in text
    # Only one trailing system instruction, not one per voice item
    assert text.count("[System instruction:") == 1


def test_compose_failure_uses_failure_flavored_instruction():
    text = compose_voice_message_text(
        parsed_text=VOICE_PLACEHOLDER,
        transcripts=[("", "STT timeout")],
    )
    assert "transcription failed" in text.lower()
    assert "STT timeout" in text
    assert "[Voice message" in text


def test_compose_silent_recording_marks_it_explicitly():
    text = compose_voice_message_text(
        parsed_text=VOICE_PLACEHOLDER,
        transcripts=[("", None)],
    )
    assert "silent or unintelligible" in text.lower()
    # No usable text recovered → use the failure-flavored instruction so
    # the agent prompts the user to retry, even though STT itself didn't
    # error.  The "Reason:" suffix is omitted because there's no error.
    sys_block = text.lower().split("[system instruction:", 1)[1]
    assert "failed" in sys_block
    assert "reason:" not in sys_block


def test_compose_partial_failure_uses_success_instruction_when_some_text_recovered():
    text = compose_voice_message_text(
        parsed_text=f"{VOICE_PLACEHOLDER}{VOICE_PLACEHOLDER}",
        transcripts=[("got this one", None), ("", "STT crash")],
    )
    assert "got this one" in text
    assert "[Voice message — transcription failed]" in text
    # Some content came through, so the instruction should not push
    # the agent into a "please retry" failure script.
    assert "transcription failed" not in text.lower().split("[system instruction:", 1)[1]


def test_compose_no_transcripts_is_a_noop_pass_through():
    """Defensive: caller should not invoke us with an empty list, but if
    they do we must not inject a fake failure instruction."""
    text = compose_voice_message_text(
        parsed_text="just text",
        transcripts=[],
    )
    assert text == "just text"


# ---------------------------------------------------------------------------
# Adapter integration: voice → STT → TEXT MessageEvent
# ---------------------------------------------------------------------------


def _make_adapter() -> OneBotNapCatAdapter:
    adapter = OneBotNapCatAdapter(PlatformConfig(enabled=True, extra={
        "group_sessions_per_user": False,
    }))
    adapter.self_id = 1903883693
    adapter.self_name = "小麻雀"
    adapter._allow_all_users = True
    return adapter


def _voice_event() -> Dict[str, Any]:
    """Synthetic NapCat voice DM payload."""
    return {
        "time": 1777073808,
        "post_type": "message",
        "message_type": "private",
        "user_id": 891088473,
        "message_id": "abc",
        "message": [
            {"type": "record", "data": {"file": "v.amr", "file_id": "fid-1"}},
        ],
        "sender": {"user_id": 891088473, "nickname": "tester"},
    }


def _photo_plus_voice_event() -> Dict[str, Any]:
    return {
        "time": 1777073808,
        "post_type": "message",
        "message_type": "private",
        "user_id": 891088473,
        "message_id": "abc",
        "message": [
            {"type": "image", "data": {"file": "p.jpg", "url": "https://x/p.jpg"}},
            {"type": "record", "data": {"file": "v.amr", "file_id": "fid-1"}},
        ],
        "sender": {"user_id": 891088473, "nickname": "tester"},
    }


@pytest.mark.asyncio
async def test_voice_dm_becomes_text_event_with_transcript_and_no_voice_media(
    monkeypatch,
):
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()

    async def fake_resolve(file_id: str, kind: str):
        return f"/cache/{file_id}.mp3"

    monkeypatch.setattr(adapter, "_resolve_via_get_file", fake_resolve)

    async def fake_transcribe(paths: List[str]) -> List[Tuple[str, Any]]:
        return [("听到啦", None) for _ in paths]

    monkeypatch.setattr(
        "gateway.platforms.onebot_napcat.adapter.transcribe_voice_files",
        fake_transcribe,
    )

    await adapter._handle_message_event(_voice_event())

    adapter.handle_message.assert_awaited_once()
    sent = adapter.handle_message.await_args.args[0]
    # The crux of the fix: agent sees TEXT (so auto-TTS gate `==
    # MessageType.VOICE` cannot fire) and the audio file is gone from
    # media (so downstream STT in run.py cannot re-transcribe).
    assert sent.message_type == MessageType.TEXT
    assert sent.media_urls == []
    assert sent.media_types == []
    assert "听到啦" in sent.text
    assert VOICE_PLACEHOLDER not in sent.text
    assert "[System instruction:" in sent.text


@pytest.mark.asyncio
async def test_voice_plus_photo_keeps_photo_strips_voice(monkeypatch):
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()

    async def fake_resolve(file_id: str, kind: str):
        return f"/cache/{file_id}.mp3"

    async def fake_resolve_ref(ref, kind, ext, filename):
        if kind == "photo":
            return "/cache/photo.jpg"
        return None

    monkeypatch.setattr(adapter, "_resolve_via_get_file", fake_resolve)
    monkeypatch.setattr(adapter, "_resolve_media_ref", fake_resolve_ref)

    async def fake_transcribe(paths):
        return [("hi", None)]

    monkeypatch.setattr(
        "gateway.platforms.onebot_napcat.adapter.transcribe_voice_files",
        fake_transcribe,
    )

    await adapter._handle_message_event(_photo_plus_voice_event())

    adapter.handle_message.assert_awaited_once()
    sent = adapter.handle_message.await_args.args[0]
    # Photo survives, voice is stripped — primary media is now PHOTO so
    # the vision pipeline still runs while auto-TTS still cannot fire.
    assert sent.message_type == MessageType.PHOTO
    assert sent.media_urls == ["/cache/photo.jpg"]
    assert sent.media_types == ["photo"]
    assert "hi" in sent.text


@pytest.mark.asyncio
async def test_voice_system_instruction_is_stripped_from_persisted_history(
    monkeypatch,
):
    """Like the poke flow: the trailing ``[System instruction: ...]`` is
    only attached for the live API call; once the adapter formats the
    event for the transcript and ``sanitize_transcript_entry`` runs on
    the way to disk, only the bare transcript should remain so the
    history isn't polluted by per-turn nudges to the model."""
    adapter = _make_adapter()

    async def fake_resolve(file_id: str, kind: str):
        return f"/cache/{file_id}.mp3"

    monkeypatch.setattr(adapter, "_resolve_via_get_file", fake_resolve)

    async def fake_transcribe(paths):
        return [("听到啦", None)]

    monkeypatch.setattr(
        "gateway.platforms.onebot_napcat.adapter.transcribe_voice_files",
        fake_transcribe,
    )

    captured: List[Any] = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture

    await adapter._handle_message_event(_voice_event())

    assert len(captured) == 1
    event = captured[0]
    # The live message that reaches the agent on this turn carries the
    # instruction so the model can frame the reply correctly.
    assert "[System instruction:" in event.text

    # The persisted user entry must NOT carry the instruction — the next
    # turn would otherwise replay a stale "this is voice" hint as if the
    # user typed it.  Mirror the gateway runner's user-entry format
    # (`<time prefix>: <event.text>`) inline; the gateway builds it in
    # ``GatewayRunner._prepare_inbound_message_text`` and we just need
    # something to feed the sanitizer to assert its strip behavior.
    persisted = adapter.sanitize_transcript_entry({
        "role": "user",
        "content": f"04-25 12:33: {event.text}",
    })
    assert persisted is not None
    persisted_content = persisted["content"]
    assert "听到啦" in persisted_content
    assert "[System instruction:" not in persisted_content


@pytest.mark.asyncio
async def test_voice_transcription_failure_still_forwards_text_event(monkeypatch):
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()

    async def fake_resolve(file_id: str, kind: str):
        return f"/cache/{file_id}.mp3"

    monkeypatch.setattr(adapter, "_resolve_via_get_file", fake_resolve)

    async def fake_transcribe(paths):
        return [("", "model unavailable")]

    monkeypatch.setattr(
        "gateway.platforms.onebot_napcat.adapter.transcribe_voice_files",
        fake_transcribe,
    )

    await adapter._handle_message_event(_voice_event())

    adapter.handle_message.assert_awaited_once()
    sent = adapter.handle_message.await_args.args[0]
    # Even on failure we still send TEXT (no audio in media), so the
    # agent can apologize and ask the user to retry instead of the
    # gateway silently dropping the message.
    assert sent.message_type == MessageType.TEXT
    assert sent.media_urls == []
    assert "transcription failed" in sent.text.lower()
    assert "model unavailable" in sent.text
