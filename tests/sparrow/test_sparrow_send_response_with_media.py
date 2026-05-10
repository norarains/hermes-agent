"""Sparrow tests for the multi-image follow-up delivery bug.

Bug it locks down
-----------------
When a user sends a new message while the bot is mid-turn, the gateway
queues the follow-up and ``_run_agent`` recurses to answer it.  Before
this fix, the *first* turn's response was delivered through a bare
``adapter.send(chat_id, first_response, ...)`` call at
``gateway/run.py:11163``, which bypassed
``base.py:_process_message_background``'s media-routing block.

For responses with one image URL inside markdown, the URL would still
land as a clickable link.  But for responses with multiple
``MEDIA:<path>`` tags (the TTS / image-tool output format), the user
would see *the raw text including ``MEDIA:/tmp/.../foo.jpg``* and zero
images — exactly the symptom in the user-reported sparrow.log timeline
(napcat sent only the text portion of a 5-image reply, no
``[图片]`` send records).

The fix: extract the inline media-routing block in
``_process_message_background`` into
``BasePlatformAdapter.send_response_with_media`` and route the
queued-follow-up shortcut through the same helper.  These tests pin
that contract so the helper can never silently drift back into a
text-only ``send`` call.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub the telegram package so importing gateway.platforms.base doesn't
# require the real PTB install (matches the pattern used by sibling
# tests in this directory).
_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import (  # noqa: E402
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SessionSource,
)


_FIXED_TS = datetime(2026, 4, 27, 21, 4, 16, tzinfo=timezone.utc)


def _make_adapter() -> BasePlatformAdapter:
    """Build a minimal concrete adapter with all I/O methods mocked."""

    class _StubAdapter(BasePlatformAdapter):
        name = "stub"

        async def connect(self) -> bool:
            return True

        async def disconnect(self) -> None:
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True, message_id="text_msg_id")

        async def get_chat_info(self, chat_id):
            return {"name": chat_id, "type": "group"}

    config = MagicMock()
    platform = MagicMock(value="stub")
    adapter = _StubAdapter(config, platform)

    # Mock attachment senders so we can assert routing without needing
    # real platform clients.  Each returns a successful SendResult.
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="text_msg_id"),
    )
    adapter.send_image = AsyncMock(
        return_value=SendResult(success=True, message_id="img_url_id"),
    )
    adapter.send_animation = AsyncMock(
        return_value=SendResult(success=True, message_id="anim_id"),
    )
    adapter.send_image_file = AsyncMock(
        return_value=SendResult(success=True, message_id="img_file_id"),
    )
    adapter.send_voice = AsyncMock(
        return_value=SendResult(success=True, message_id="voice_id"),
    )
    adapter.send_video = AsyncMock(
        return_value=SendResult(success=True, message_id="video_id"),
    )
    adapter.send_document = AsyncMock(
        return_value=SendResult(success=True, message_id="doc_id"),
    )
    adapter.play_tts = AsyncMock()

    return adapter


def _make_event(message_id: str = "msg_original") -> MessageEvent:
    source = SessionSource(
        platform=MagicMock(value="stub"),
        chat_id="group_940134062",
        chat_type="group",
        user_id="user_891088473",
    )
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
        timestamp=_FIXED_TS,
    )


# ---------------------------------------------------------------------------
# The actual bug repro: multi-image MEDIA: response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_media_tags_route_to_send_image_file(tmp_path):
    """Concrete reproduction of the user-reported sparrow.log incident.

    When the LLM emits 5 ``MEDIA:/tmp/.../*.jpg`` tags, every one of
    those paths must reach ``send_image_file`` AND none of them may
    survive in the text payload — otherwise the user reads
    ``MEDIA:/tmp/...`` literally and sees zero images.
    """
    adapter = _make_adapter()

    # Create real files so the helper's path expansion + suffix routing
    # match production semantics.  Five images, mirroring the bug report.
    paths = []
    for i in range(5):
        p = tmp_path / f"img_{i}.jpg"
        p.write_bytes(b"\xff\xd8\xff")  # any non-empty bytes
        paths.append(str(p))

    response_text = "5 张全到位——\n" + "\n".join(f"MEDIA:{p}" for p in paths)

    result = await adapter.send_response_with_media(
        chat_id="group_940134062",
        response=response_text,
        reply_to="orig_msg_id",
        metadata={"thread_id": "t1"},
    )

    # Text portion sent ONCE, with all MEDIA: tags scrubbed.
    assert adapter._send_with_retry.await_count == 1
    sent_text = adapter._send_with_retry.await_args.kwargs["content"]
    assert "MEDIA:" not in sent_text
    for p in paths:
        assert p not in sent_text
    assert "5 张全到位" in sent_text  # cleaned text still includes prose

    # Every MEDIA path routed to send_image_file in order.
    assert adapter.send_image_file.await_count == 5
    actual_paths = [
        call.kwargs["image_path"] for call in adapter.send_image_file.await_args_list
    ]
    assert actual_paths == paths

    # Returned SendResult is the TEXT one (not any image one) so the
    # caller can decorate the persisted assistant entry with the right
    # outbound message_id.
    assert result is not None
    assert result.message_id == "text_msg_id"


@pytest.mark.asyncio
async def test_media_routing_by_extension(tmp_path):
    """Audio → send_voice, video → send_video, image → send_image_file,
    other → send_document.  Future ``MEDIA:`` extensions added to the
    routing dispatch must come with explicit assertions here, otherwise
    a typo in one branch silently sends through the document fallback."""
    adapter = _make_adapter()

    img = tmp_path / "a.png"; img.write_bytes(b"\x89PNG")
    audio = tmp_path / "b.opus"; audio.write_bytes(b"OggS")
    video = tmp_path / "c.mp4"; video.write_bytes(b"\x00")
    other = tmp_path / "d.pdf"; other.write_bytes(b"%PDF")

    response_text = (
        f"MEDIA:{img}\n"
        f"MEDIA:{audio}\n"
        f"MEDIA:{video}\n"
        f"MEDIA:{other}\n"
    )

    await adapter.send_response_with_media(
        chat_id="cid",
        response=response_text,
    )

    adapter.send_image_file.assert_awaited_once()
    adapter.send_voice.assert_awaited_once()
    adapter.send_video.assert_awaited_once()
    adapter.send_document.assert_awaited_once()
    # send_image is for URL images, NOT for MEDIA: file paths
    adapter.send_image.assert_not_called()


@pytest.mark.asyncio
async def test_image_urls_route_to_send_image():
    """Markdown image URLs (``![alt](https://...)``) go to ``send_image``,
    not ``send_image_file``.  The two paths use different platform APIs
    on QQ/Telegram (URL vs file upload) so a swap would silently break
    one of the most common response shapes."""
    adapter = _make_adapter()

    response_text = (
        "Here you go: ![cat](https://example.com/cat.png) "
        "and ![dog](https://example.com/dog.jpg)"
    )

    await adapter.send_response_with_media(
        chat_id="cid",
        response=response_text,
    )

    assert adapter.send_image.await_count == 2
    urls = [call.kwargs["image_url"] for call in adapter.send_image.await_args_list]
    assert urls == ["https://example.com/cat.png", "https://example.com/dog.jpg"]
    adapter.send_image_file.assert_not_called()


@pytest.mark.asyncio
async def test_animation_urls_route_to_send_animation():
    """``.gif`` URLs must reach ``send_animation`` so platforms render
    them as auto-playing animations rather than static images."""
    adapter = _make_adapter()

    await adapter.send_response_with_media(
        chat_id="cid",
        response="![dance](https://example.com/dance.gif)",
    )

    adapter.send_animation.assert_awaited_once()
    adapter.send_image.assert_not_called()


@pytest.mark.asyncio
async def test_empty_response_is_noop():
    """Empty/None responses must not call any sender — otherwise a
    streamed-already turn with no text would post phantom blank
    messages."""
    adapter = _make_adapter()

    assert await adapter.send_response_with_media(chat_id="cid", response="") is None
    assert await adapter.send_response_with_media(chat_id="cid", response=None) is None  # type: ignore[arg-type]

    adapter._send_with_retry.assert_not_called()
    adapter.send_image_file.assert_not_called()
    adapter.send_image.assert_not_called()


@pytest.mark.asyncio
async def test_response_with_only_media_tags_skips_text_send(tmp_path):
    """When a response is JUST ``MEDIA:`` tags (no prose), the text
    payload after stripping is empty, so ``_send_with_retry`` MUST NOT
    be called.  Sending an empty string would create a blank quoted
    message on the user's side."""
    adapter = _make_adapter()
    img = tmp_path / "x.png"; img.write_bytes(b"\x89PNG")

    result = await adapter.send_response_with_media(
        chat_id="cid",
        response=f"MEDIA:{img}",
    )

    adapter._send_with_retry.assert_not_called()
    adapter.send_image_file.assert_awaited_once()
    # No text was sent → returned result is None (no SendResult to surface)
    assert result is None


@pytest.mark.asyncio
async def test_voice_event_triggers_auto_tts_when_no_media(monkeypatch):
    """Voice-channel users opted into spoken replies.  When ``event``
    is a voice message AND the response carries no MEDIA tags AND the
    chat hasn't disabled voice mode, auto-TTS plays the cleaned text
    *before* the text send — that's the voice-first UX contract."""
    import asyncio as _aio
    import json as _json
    import gateway.platforms.base as base_mod

    fake_tts_module = types.ModuleType("tools.tts_tool")
    fake_tts_module.check_tts_requirements = lambda: True
    tts_path_holder: list = []

    def _fake_tts(text: str) -> str:
        # Real TTS writes a file and returns a JSON blob with file_path.
        # Here we just return a sentinel path the test can intercept.
        tts_path_holder.append(text)
        return _json.dumps({"file_path": "/dev/null"})

    fake_tts_module.text_to_speech_tool = _fake_tts
    monkeypatch.setitem(sys.modules, "tools.tts_tool", fake_tts_module)

    # Make Path("/dev/null").exists() True so play_tts actually fires.
    # /dev/null exists on Linux, so the default check passes; os.remove
    # would raise on /dev/null but the helper swallows OSError.
    adapter = _make_adapter()
    # Opt the chat into auto-TTS via _should_auto_tts_for_chat's allowlist
    # path (matches upstream's /voice on|tts UX).
    adapter._auto_tts_enabled_chats.add(_make_event().source.chat_id)
    event = _make_event()
    event.message_type = MessageType.VOICE

    await adapter.send_response_with_media(
        chat_id=event.source.chat_id,
        response="hello there",
        event=event,
    )

    assert tts_path_holder == ["hello there"]
    adapter.play_tts.assert_awaited_once()


@pytest.mark.asyncio
async def test_voice_event_skips_auto_tts_when_media_present(tmp_path, monkeypatch):
    """Voice-channel + MEDIA tag means the agent already chose a media
    response (image, video, voice file).  Auto-TTS would double-up on
    the audio output, so it must be skipped."""
    fake_tts_module = types.ModuleType("tools.tts_tool")
    fake_tts_module.check_tts_requirements = lambda: True
    called: list = []
    fake_tts_module.text_to_speech_tool = lambda text: called.append(text) or "{}"
    monkeypatch.setitem(sys.modules, "tools.tts_tool", fake_tts_module)

    adapter = _make_adapter()
    event = _make_event()
    event.message_type = MessageType.VOICE

    img = tmp_path / "p.png"; img.write_bytes(b"\x89PNG")

    await adapter.send_response_with_media(
        chat_id="cid",
        response=f"see this MEDIA:{img}",
        event=event,
    )

    assert called == []
    adapter.play_tts.assert_not_called()
    adapter.send_image_file.assert_awaited_once()


# ---------------------------------------------------------------------------
# The follow-up shortcut at gateway/run.py must use the helper
# ---------------------------------------------------------------------------


def test_followup_shortcut_calls_send_response_with_media():
    """Source-level guard: ``gateway/run.py`` must route the queued
    follow-up first_response delivery through
    ``adapter.send_response_with_media``, NOT bare ``adapter.send``.
    A regression here is exactly the bug this whole change exists to
    fix — the future maintainer who "simplifies" this back to
    ``adapter.send`` would silently re-break multi-image delivery."""
    import pathlib

    run_py = pathlib.Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    src = run_py.read_text()

    # The fix block
    assert "adapter.send_response_with_media(" in src, (
        "gateway/run.py no longer calls adapter.send_response_with_media — "
        "the queued-follow-up first_response is back on the bare send path "
        "and multi-image deliveries will fail (sparrow.log 21:04:16 incident)."
    )

    # The buggy form must not reappear in the same neighborhood.  We
    # scan the entire file for a literal ``adapter.send(\n`` followed
    # within a few lines by ``first_response`` — that's the exact
    # shape the bug took before the fix.
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "adapter.send(" in line and "send_response_with_media" not in line:
            window = "\n".join(lines[i : i + 6])
            assert "first_response" not in window, (
                f"gateway/run.py:{i + 1} regressed to bare adapter.send for "
                "first_response — use adapter.send_response_with_media instead."
            )
