"""Tests for NapCat's atomic combined-delivery (text + media in one OneBot call).

Why this exists
---------------
Normal QQ presents ``text + N images`` as a single chat bubble with an
image-grid layout.  NapCat, being an OneBot v11 implementation, supports
this natively: a single ``send_group_msg`` / ``send_private_msg`` action
takes a ``message: [...]`` segment array combining ``reply`` + text +
image / audio / video / file segments.

Earlier the gateway split text and each attachment across N+1 separate
``send_*`` calls (one bubble per attachment).  That worked but produced
a chat experience that's distinctly worse than native QQ, especially
for multi-image responses (mushroom-photo bursts, document scans,
albums).

These tests pin the contract of the combined path:

  1. ``BasePlatformAdapter.send_response_with_media`` dispatches to
     ``adapter.send_combined`` when there are attachments AND the
     adapter implements the method (returns non-None).
  2. NapCat's ``send_combined`` produces ONE OneBot call carrying every
     attachment as a separate segment in a single message array.
  3. ``_send_segments`` emits ``napcat_send_starting`` and
     ``napcat_send_finished`` with the segment list, regardless of
     content (pure text, text+image, image-only, etc.).
  4. POKE / REPLY / [SILENT] directives in the text portion behave
     identically in the combined path and the text-only path.

If anyone later refactors send_combined to fall back to per-attachment
sends (re-creating the multi-bubble bug), the tests here fail loudly.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub the telegram package so importing gateway.platforms.base doesn't
# require the real PTB install.
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

from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms.base import SendResult  # noqa: E402
from gateway.platforms.onebot_napcat.adapter import OneBotNapCatAdapter  # noqa: E402


def _make_adapter() -> OneBotNapCatAdapter:
    adapter = OneBotNapCatAdapter(PlatformConfig(enabled=True, extra={
        "group_sessions_per_user": False,
    }))
    adapter.self_id = 792836799
    adapter.self_name = "stub"
    adapter._allow_all_users = True
    # _send_segments calls _call_action, which needs a non-closed http
    # client.  We mock _call_action directly so we never touch the
    # network and can inspect the params for assertions.
    adapter._call_action = AsyncMock(return_value={  # type: ignore[method-assign]
        "status": "ok",
        "data": {"message_id": 999},
    })
    return adapter


# ---------------------------------------------------------------------------
# send_combined builds ONE OneBot call with every attachment as a segment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_combined_text_plus_images_single_call(tmp_path):
    """The user-reported "send N mushroom photos" scenario.

    text + 5 images must hit the OneBot transport as a single
    ``send_group_msg`` action whose ``message`` array contains:
      - reply (since group chats default to quote-reply)
      - text segment
      - 5 image segments (in order)
    """
    adapter = _make_adapter()
    paths = [str(tmp_path / f"img_{i}.jpg") for i in range(5)]
    for p in paths:
        open(p, "wb").write(b"\xff\xd8\xff")

    result = await adapter.send_combined(
        chat_id="group_940134062",
        text="5 张全到位——！",
        images=(),
        media_files=[(p, False) for p in paths],
        reply_to="orig_msg_id",
    )

    assert result is not None
    assert result.success
    # ONE call, not 6
    assert adapter._call_action.await_count == 1

    action, params = adapter._call_action.await_args.args[:2]
    assert action == "send_group_msg"
    assert params["group_id"] == 940134062
    msgs = params["message"]
    types_ = [seg["type"] for seg in msgs]
    assert types_ == ["reply", "text", "image", "image", "image", "image", "image"]

    # Every image segment carries the right file:// URI in order
    image_files = [seg["data"]["file"] for seg in msgs if seg["type"] == "image"]
    assert image_files == [f"file://{p}" for p in paths]

    # Text segment carries the prose
    text_seg = next(seg for seg in msgs if seg["type"] == "text")
    assert "5 张全到位" in text_seg["data"]["text"]


@pytest.mark.asyncio
async def test_send_combined_routes_by_extension(tmp_path):
    """Mixed-type combined send — image / audio / video / unknown
    must each get the right OneBot segment type."""
    adapter = _make_adapter()
    img = str(tmp_path / "a.png");   open(img, "wb").write(b"\x89PNG")
    audio = str(tmp_path / "b.opus"); open(audio, "wb").write(b"OggS")
    video = str(tmp_path / "c.mp4");  open(video, "wb").write(b"\x00")
    other = str(tmp_path / "d.pdf");  open(other, "wb").write(b"%PDF")

    await adapter.send_combined(
        chat_id="private_891088473",
        text="here",
        media_files=[(img, False), (audio, False), (video, False), (other, False)],
    )

    msgs = adapter._call_action.await_args.args[1]["message"]
    types_ = [seg["type"] for seg in msgs]
    # Private chat → no auto reply segment.  text + image + record + video + file.
    assert types_ == ["text", "image", "record", "video", "file"]


@pytest.mark.asyncio
async def test_send_combined_image_url_uses_url_directly(tmp_path):
    """Markdown image URLs (``![alt](https://...)``) should NOT be
    rewritten to ``file://``.  They go through the OneBot image segment
    with the raw URL so NapCat fetches them itself."""
    adapter = _make_adapter()

    await adapter.send_combined(
        chat_id="group_940134062",
        text="see this",
        images=[("https://example.com/cat.png", "cat")],
    )

    msgs = adapter._call_action.await_args.args[1]["message"]
    img_seg = next(seg for seg in msgs if seg["type"] == "image")
    assert img_seg["data"]["file"] == "https://example.com/cat.png"


# ---------------------------------------------------------------------------
# Sparrow logging from _send_segments — fires for every send, content listed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_napcat_send_events_fire_with_segment_summary(monkeypatch, tmp_path):
    """Every OneBot RPC call must emit napcat_send_starting +
    napcat_send_finished into sparrow.log with a compact representation
    of the segment array.  This is the operator's at-a-glance view."""
    captured: List[tuple] = []

    def _fake_event(event: str, **fields: Any) -> None:
        captured.append((event, fields))

    # Patch the log_event symbol in the sparrow_log module.  The adapter
    # imports it lazily so monkey-patching the module attribute is what
    # the lazy import will pick up.
    fake_log = types.ModuleType("gateway.sparrow_log")
    fake_log.log_event = _fake_event
    monkeypatch.setitem(sys.modules, "gateway.sparrow_log", fake_log)

    adapter = _make_adapter()
    img = str(tmp_path / "x.jpg"); open(img, "wb").write(b"\xff\xd8\xff")

    await adapter.send_combined(
        chat_id="group_940134062",
        text="ok",
        media_files=[(img, False)],
        reply_to="msg_orig",
    )

    events = [e for e, _ in captured]
    assert events == ["napcat_send_starting", "napcat_send_finished"]

    starting = captured[0][1]
    finished = captured[1][1]
    # segment summary lists each segment as type[:target]
    assert "reply:msg_orig" in starting["segments"]
    assert "text" in starting["segments"]
    assert f"image:file://{img}" in starting["segments"]
    assert finished["status"] == "ok"
    assert finished["message_id"] == "999"


@pytest.mark.asyncio
async def test_napcat_send_events_fire_for_pure_text(monkeypatch):
    """The user explicitly wanted napcat_send_* to fire for text-only
    sends too — that path goes through ``send`` → ``_send_segments``
    natively, no special wiring needed.  This test pins that."""
    captured: List[tuple] = []
    fake_log = types.ModuleType("gateway.sparrow_log")
    fake_log.log_event = lambda e, **f: captured.append((e, f))
    monkeypatch.setitem(sys.modules, "gateway.sparrow_log", fake_log)

    adapter = _make_adapter()

    await adapter.send(
        chat_id="private_891088473",
        content="just words, no media",
    )

    events = [e for e, _ in captured]
    assert "napcat_send_starting" in events
    assert "napcat_send_finished" in events
    starting = next(f for e, f in captured if e == "napcat_send_starting")
    # No reply, no media — just a text segment in private chat
    assert starting["segments"] == "text"


@pytest.mark.asyncio
async def test_napcat_send_finished_reports_failure_status(monkeypatch):
    """When the OneBot action reports status=failed, finished event
    must carry status=failed + the error message so operators can
    triage from sparrow.log alone."""
    captured: List[tuple] = []
    fake_log = types.ModuleType("gateway.sparrow_log")
    fake_log.log_event = lambda e, **f: captured.append((e, f))
    monkeypatch.setitem(sys.modules, "gateway.sparrow_log", fake_log)

    adapter = _make_adapter()
    adapter._call_action = AsyncMock(return_value={  # type: ignore[method-assign]
        "status": "failed",
        "message": "ENOENT: no such file or directory",
    })

    result = await adapter.send_combined(
        chat_id="group_940134062",
        text="this will fail",
        media_files=[("/tmp/does_not_exist.jpg", False)],
    )
    assert result is not None
    assert not result.success

    finished = next(f for e, f in captured if e == "napcat_send_finished")
    assert finished["status"] == "failed"
    assert "ENOENT" in finished["error"]


# ---------------------------------------------------------------------------
# send_response_with_media dispatches to send_combined when applicable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_response_with_media_uses_combined_path(tmp_path):
    """End-to-end: agent emits ``text + MEDIA:<path>`` → base.py extracts
    media → calls send_combined → ONE OneBot RPC.  The per-attachment
    fallback loop must NOT run (it would produce N+1 bubbles)."""
    adapter = _make_adapter()
    img1 = str(tmp_path / "a.jpg"); open(img1, "wb").write(b"\xff\xd8\xff")
    img2 = str(tmp_path / "b.jpg"); open(img2, "wb").write(b"\xff\xd8\xff")

    response = (
        "好——试试连发！\n\n"
        f"MEDIA:{img1}\n"
        f"MEDIA:{img2}"
    )

    result = await adapter.send_response_with_media(
        chat_id="group_940134062",
        response=response,
        reply_to="trigger_msg_id",
    )

    assert result is not None
    assert result.success
    # ONE call total, NOT three (text + img1 + img2).  This is the bug
    # this whole change exists to fix.
    assert adapter._call_action.await_count == 1

    msgs = adapter._call_action.await_args.args[1]["message"]
    types_ = [seg["type"] for seg in msgs]
    assert types_ == ["reply", "text", "image", "image"]


@pytest.mark.asyncio
async def test_send_response_with_media_text_only_skips_combined(tmp_path):
    """No attachments → send_response_with_media must NOT call
    send_combined (the if-has_attachments guard).  Text-only goes
    through _send_with_retry → send → _send_segments, preserving the
    text-only retry behavior on transient errors."""
    adapter = _make_adapter()

    # Spy on send_combined to confirm it's never invoked
    adapter.send_combined = AsyncMock(return_value=None)  # type: ignore[method-assign]

    await adapter.send_response_with_media(
        chat_id="private_891088473",
        response="just words",
    )

    adapter.send_combined.assert_not_called()
    # But the OneBot send did happen via the normal path
    assert adapter._call_action.await_count == 1


# ---------------------------------------------------------------------------
# POKE / REPLY / [SILENT] directives keep working in combined path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_combined_silent_marker_short_circuits(tmp_path):
    """[SILENT] response with media still produces no chat bubble —
    the silent marker is honored regardless of attached media."""
    adapter = _make_adapter()
    img = str(tmp_path / "x.jpg"); open(img, "wb").write(b"\xff\xd8\xff")

    result = await adapter.send_combined(
        chat_id="group_940134062",
        text="[SILENT]",
        media_files=[(img, False)],
    )
    assert result is not None
    assert result.success
    # Silent → no actual OneBot call
    adapter._call_action.assert_not_called()


@pytest.mark.asyncio
async def test_combined_reply_override_takes_precedence(tmp_path):
    """``REPLY:<id>`` line in the text overrides whatever ``reply_to``
    the caller passed.  In the combined path this must still be honored
    so the same agent prompt that controls reply targeting works for
    both single-bubble and multi-bubble deliveries."""
    adapter = _make_adapter()
    img = str(tmp_path / "x.jpg"); open(img, "wb").write(b"\xff\xd8\xff")

    await adapter.send_combined(
        chat_id="group_940134062",
        text="REPLY:888\nactual content",
        media_files=[(img, False)],
        reply_to="123",  # should be ignored — REPLY:888 wins
    )

    msgs = adapter._call_action.await_args.args[1]["message"]
    reply = next(seg for seg in msgs if seg["type"] == "reply")
    assert reply["data"]["id"] == "888"
    # Text segment should NOT contain the REPLY: line itself
    text_seg = next(seg for seg in msgs if seg["type"] == "text")
    assert "REPLY:" not in text_seg["data"]["text"]
    assert "actual content" in text_seg["data"]["text"]


@pytest.mark.asyncio
async def test_combined_reply_none_disables_reply_segment(tmp_path):
    """``REPLY:none`` means "explicitly do not quote" — no reply
    segment should be added even though the platform default
    (group chat) would otherwise quote-reply."""
    adapter = _make_adapter()
    img = str(tmp_path / "x.jpg"); open(img, "wb").write(b"\xff\xd8\xff")

    await adapter.send_combined(
        chat_id="group_940134062",
        text="REPLY:none\ndon't quote me",
        media_files=[(img, False)],
        reply_to="orig",
    )

    msgs = adapter._call_action.await_args.args[1]["message"]
    types_ = [seg["type"] for seg in msgs]
    assert "reply" not in types_
