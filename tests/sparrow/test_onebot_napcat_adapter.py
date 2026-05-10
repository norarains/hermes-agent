from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.onebot_napcat.adapter import OneBotNapCatAdapter


def _adapter() -> OneBotNapCatAdapter:
    adapter = OneBotNapCatAdapter(PlatformConfig(enabled=True))
    adapter.self_id = 1903883693
    adapter.self_name = "小麻雀"
    return adapter


@pytest.mark.asyncio
async def test_group_at_message_dispatches_to_agent_with_normalized_source():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    await adapter._handle_message_event({
        "time": 1777073808,
        "message_type": "group",
        "group_id": 123,
        "group_name": "测试群",
        "user_id": 891088473,
        "message_id": "m1",
        "message": [
            {"type": "at", "data": {"qq": "1903883693"}},
            {"type": "text", "data": {"text": " hello"}},
        ],
        "sender": {"nickname": "近环"},
    })

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello"
    assert event.source.chat_id == "group_123"
    assert event.source.chat_name == "测试群"
    assert event.source.user_id == "891088473"
    assert event.source.user_name == "近环"
    assert event.message_id == "m1"
    assert "POKE:<QQ号>" in event.channel_prompt


@pytest.mark.asyncio
async def test_reply_segment_resolves_quoted_user_message_text(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    adapter._call_action = AsyncMock(return_value={
        "status": "ok",
        "data": {
            "time": 1777073808,
            "message_id": "quoted-1",
            "user_id": 222333444,
            "sender": {"card": "伊乐", "nickname": "yile"},
            "message": [{"type": "text", "data": {"text": "被引用的原文"}}],
        },
    })

    await adapter._handle_message_event({
        "time": 1777073810,
        "message_type": "group",
        "group_id": 123,
        "group_name": "测试群",
        "user_id": 891088473,
        "message_id": "current-1",
        "message": [
            {"type": "reply", "data": {"id": "quoted-1"}},
            {"type": "text", "data": {"text": " 我在回复这句"}},
        ],
        "sender": {"nickname": "近环"},
    })

    adapter._call_action.assert_awaited_once_with(
        "get_msg", {"message_id": "quoted-1"}, timeout=10,
    )
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.reply_to_message_id == "quoted-1"
    assert event.reply_to_text == (
        "04-24 16:36:48 [msg=quoted-1] 222333444 伊乐: 被引用的原文"
    )
    assert event.text == "我在回复这句"


@pytest.mark.asyncio
async def test_reply_segment_resolves_quoted_bot_message_with_bot_identity(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    adapter._call_action = AsyncMock(return_value={
        "status": "ok",
        "data": {
            "time": 1777073808,
            "message_id": "bot-msg",
            "user_id": 1903883693,
            "sender": {},
            "message": [{"type": "text", "data": {"text": "小麻雀自己的回复"}}],
        },
    })

    await adapter._handle_message_event({
        "message_type": "private",
        "user_id": 891088473,
        "message_id": "current-2",
        "message": [
            {"type": "reply", "data": {"id": "bot-msg"}},
            {"type": "text", "data": {"text": " 对这句补充"}},
        ],
        "sender": {"nickname": "近环"},
    })

    event = adapter.handle_message.await_args.args[0]
    assert event.reply_to_text == (
        "04-24 16:36:48 [msg=bot-msg] 1903883693 小麻雀: 小麻雀自己的回复"
    )


@pytest.mark.asyncio
async def test_reply_segment_marks_unresolved_when_get_msg_fails():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    adapter._call_action = AsyncMock(side_effect=RuntimeError("get_msg boom"))

    await adapter._handle_message_event({
        "message_type": "private",
        "user_id": 891088473,
        "message_id": "current-3",
        "message": [
            {"type": "reply", "data": {"id": "missing-msg"}},
            {"type": "text", "data": {"text": " 这条引用解析不到"}},
        ],
        "sender": {"nickname": "近环"},
    })

    adapter._call_action.assert_awaited_once_with(
        "get_msg", {"message_id": "missing-msg"}, timeout=10,
    )
    event = adapter.handle_message.await_args.args[0]
    assert event.reply_to_message_id == "missing-msg"
    assert event.reply_to_text == "unresolved QQ message [msg=missing-msg]"


@pytest.mark.asyncio
async def test_unmentioned_group_message_is_ignored_when_passive_group_disabled():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    await adapter._handle_message_event({
        "message_type": "group",
        "group_id": 123,
        "user_id": 891088473,
        "message_id": "m1",
        "message": [{"type": "text", "data": {"text": "hello"}}],
        "sender": {"nickname": "近环"},
    })

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_message_uses_napcat_get_record_mp3_before_local_amr_path(
    tmp_path, monkeypatch,
):
    """NapCat voice DM is fetched via get_record(out_format=mp3), then
    transcribed inside the adapter and forwarded as a TEXT event so the
    gateway's auto-TTS-on-voice path doesn't fire."""
    from gateway.platforms.base import MessageType

    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    mp3_path = tmp_path / "voice.amr.mp3"
    mp3_path.write_bytes(b"mp3")
    adapter._call_action = AsyncMock(
        return_value={"status": "ok", "data": {"file": str(mp3_path)}},
    )

    async def fake_transcribe(paths):
        return [("hello bot", None) for _ in paths]

    monkeypatch.setattr(
        "gateway.platforms.onebot_napcat.adapter.transcribe_voice_files",
        fake_transcribe,
    )

    await adapter._handle_message_event({
        "message_type": "private",
        "user_id": 891088473,
        "message_id": "m1",
        "message": [{
            "type": "record",
            "data": {
                "file": "napcat-record-id",
                "path": "/opt/data/.config/QQ/nt_data/Ptt/voice.amr",
                "file_size": "1234",
            },
        }],
        "sender": {"nickname": "近环"},
    })

    adapter._call_action.assert_awaited_once_with(
        "get_record",
        {"file_id": "napcat-record-id", "out_format": "mp3"},
        timeout=60,
    )
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.message_type == MessageType.TEXT
    assert event.media_urls == []
    assert event.media_types == []
    assert "hello bot" in event.text
    assert "[语音]" not in event.text
    assert "[System instruction:" in event.text


@pytest.mark.asyncio
async def test_send_expands_cq_at_code_to_onebot_segment():
    adapter = _adapter()
    adapter._call_action = AsyncMock(
        return_value={"status": "ok", "data": {"message_id": 7}},
    )

    result = await adapter.send("group_123", "[CQ:at,qq=891088473] hi")

    assert result.success is True
    adapter._call_action.assert_awaited_once_with(
        "send_group_msg",
        {
            "group_id": 123,
            "message": [
                {"type": "at", "data": {"qq": "891088473"}},
                {"type": "text", "data": {"text": " hi"}},
            ],
        },
        timeout=30,
    )


@pytest.mark.asyncio
async def test_send_silent_marker_does_not_call_onebot_action():
    adapter = _adapter()
    adapter._call_action = AsyncMock()

    result = await adapter.send("group_123", "[SILENT]")

    assert result.success is True
    adapter._call_action.assert_not_awaited()
