"""Tests for the NapCat quote-reply behavior.

Defaults:
  - Group chats: quote the triggering message (matches QQ etiquette
    where group context can be ambiguous about whom you're addressing).
  - Private DMs: do NOT quote (the conversation is already 1:1, a
    quote is pure noise).

Override directive (parsed inside ``adapter.send``):
  - ``REPLY:<message_id>`` quotes that specific message — IDs are
    surfaced to the LLM via the ``[msg=<id>]`` tag in transcript lines,
    so it can quote any past message, not only the triggering one.
  - ``REPLY:none`` explicitly skips the default group auto-quote.
  - When multiple REPLY: lines appear, the last wins.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.onebot_napcat.adapter import OneBotNapCatAdapter
from gateway.platforms.onebot_napcat.prompts import (
    BASE_CHANNEL_PROMPT,
    build_base_channel_prompt,
)
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adapter() -> OneBotNapCatAdapter:
    adapter = OneBotNapCatAdapter(PlatformConfig(enabled=True))
    adapter.self_id = 1903883693
    adapter.self_name = "小麻雀"
    return adapter


def _attach_call_action(adapter: OneBotNapCatAdapter) -> AsyncMock:
    mock = AsyncMock(return_value={"status": "ok", "data": {"message_id": 99}})
    adapter._call_action = mock
    return mock


def _sent_segments(call_action: AsyncMock) -> List[Dict[str, Any]]:
    """Pull the OneBot ``message`` array from the most recent send call."""
    args, _kwargs = call_action.await_args
    payload = args[1]
    return payload.get("message", [])


def _has_reply_segment(segments: List[Dict[str, Any]]) -> bool:
    return any(s.get("type") == "reply" for s in segments)


def _reply_id(segments: List[Dict[str, Any]]) -> str | None:
    for s in segments:
        if s.get("type") == "reply":
            return str(s.get("data", {}).get("id", ""))
    return None


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dm_default_does_not_quote_even_when_reply_to_is_passed():
    """DMs are 1:1 — the auto-quote is just noise.  Even when the gateway
    helpfully passes the triggering ``message_id`` (the existing
    behavior in ``base.py``), the napcat adapter should drop it."""
    adapter = _adapter()
    call = _attach_call_action(adapter)

    await adapter.send("private_891088473", "hi", reply_to="trigger-1")

    assert not _has_reply_segment(_sent_segments(call))


@pytest.mark.asyncio
async def test_group_default_quotes_triggering_message():
    """Groups default to quoting so the bot's reply is unambiguously
    attached to the message it's responding to."""
    adapter = _adapter()
    call = _attach_call_action(adapter)

    await adapter.send("group_123", "hi", reply_to="trigger-2")

    assert _reply_id(_sent_segments(call)) == "trigger-2"


# ---------------------------------------------------------------------------
# REPLY:<id> override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply_directive_in_dm_enables_quote_for_specific_message():
    adapter = _adapter()
    call = _attach_call_action(adapter)

    await adapter.send(
        "private_891088473",
        "REPLY:55555\nremember this from earlier?",
        reply_to=None,
    )

    segs = _sent_segments(call)
    assert _reply_id(segs) == "55555"
    # Directive is consumed — only the chat text reaches QQ.
    text_seg = next(s for s in segs if s.get("type") == "text")
    assert "REPLY" not in text_seg["data"]["text"]
    assert "remember this" in text_seg["data"]["text"]


@pytest.mark.asyncio
async def test_reply_directive_in_group_overrides_default_target():
    """Group default would quote the triggering message; an explicit
    REPLY:<other-id> redirects the quote at a different (e.g. historical)
    message."""
    adapter = _adapter()
    call = _attach_call_action(adapter)

    await adapter.send(
        "group_123",
        "REPLY:99999\ncalling back to that",
        reply_to="current-trigger",
    )

    assert _reply_id(_sent_segments(call)) == "99999"


@pytest.mark.asyncio
async def test_reply_none_in_group_suppresses_default_auto_quote():
    """A continuous back-and-forth where the auto-quote is just noise."""
    adapter = _adapter()
    call = _attach_call_action(adapter)

    await adapter.send(
        "group_123",
        "REPLY:none\njust a follow-up",
        reply_to="current-trigger",
    )

    assert not _has_reply_segment(_sent_segments(call))


@pytest.mark.asyncio
async def test_last_reply_directive_wins():
    """If the LLM hedges (e.g. emits both forms), the last directive is
    its final decision."""
    adapter = _adapter()
    call = _attach_call_action(adapter)

    await adapter.send(
        "group_123",
        "REPLY:111\nREPLY:none\nactually, no quote",
        reply_to="current-trigger",
    )

    assert not _has_reply_segment(_sent_segments(call))


@pytest.mark.asyncio
async def test_reply_directive_strips_from_chat_text_with_poke_directive():
    """Both POKE and REPLY directives should be consumed in the same pass
    — neither leaks into the chat text the user sees."""
    adapter = _adapter()
    call = _attach_call_action(adapter)
    adapter.send_poke = AsyncMock(return_value=None)

    await adapter.send(
        "group_123",
        "POKE:891088473\nREPLY:42\nhey there",
        reply_to="current-trigger",
    )

    segs = _sent_segments(call)
    assert _reply_id(segs) == "42"
    text_seg = next(s for s in segs if s.get("type") == "text")
    assert "POKE" not in text_seg["data"]["text"]
    assert "REPLY" not in text_seg["data"]["text"]
    assert "hey there" in text_seg["data"]["text"]
    adapter.send_poke.assert_awaited_once_with("group_123", "891088473")


@pytest.mark.asyncio
async def test_invalid_reply_directive_is_left_as_chat_text():
    """``REPLY:foo`` (non-numeric, non-`none`) doesn't match the regex —
    treat it as ordinary chat text rather than a silently-dropped line.
    Defaults still apply for the actual reply target."""
    adapter = _adapter()
    call = _attach_call_action(adapter)

    await adapter.send(
        "group_123",
        "REPLY:foo\nactual content",
        reply_to="current-trigger",
    )

    segs = _sent_segments(call)
    # Default group behavior kicks in.
    assert _reply_id(segs) == "current-trigger"
    text_seg = next(s for s in segs if s.get("type") == "text")
    assert "REPLY:foo" in text_seg["data"]["text"]


# ---------------------------------------------------------------------------
# Transcript surfaces message_id so LLM has material for REPLY:<id>
#
# These exercise the REAL persistence path — `GatewayRunner._prepare_inbound_
# message_text` — the same function that builds the user-message string sent
# to the LLM and persisted to SQLite.  Earlier we had tests against an
# adapter-level helper (`_format_event_for_transcript`) that turned out to
# be dead code, so the tests passed while production silently produced
# `<time>: <text>` with no `[msg=<id>]`.  These tests catch that for real.
# ---------------------------------------------------------------------------


def _make_event(*, message_id: Optional[str], chat_type: str) -> MessageEvent:
    chat_id = "private_891088473" if chat_type == "dm" else "group_123"
    source = SessionSource(
        platform=Platform.ONEBOT_NAPCAT,
        chat_id=chat_id,
        chat_name="测试群" if chat_type == "group" else "近环",
        chat_type=chat_type,
        user_id="891088473",
        user_name="近环",
    )
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
        timestamp=datetime(2026, 4, 25, 12, 33, tzinfo=timezone.utc),
    )


def _make_runner(*, group_sessions_per_user: bool = True) -> "GatewayRunner":
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    # Register a real napcat adapter so `extend_inbound_prefix` fires —
    # the runner looks up the adapter by `source.platform` to call the
    # hook.  Without registration the hook is silently skipped.
    runner.adapters = {Platform.ONEBOT_NAPCAT: _adapter()}
    runner.config = SimpleNamespace(
        group_sessions_per_user=group_sessions_per_user,
        thread_sessions_per_user=False,
    )
    return runner


@pytest.mark.asyncio
async def test_dm_inbound_text_for_agent_is_time_msg_qqid_name_text(monkeypatch):
    """DM line: ``<time> [msg=<id>] <qq_id> <name>: <text>``.

    Sender shows in DMs too (not just shared groups) so the LLM has a
    stable `<qq_id>` for ``[CQ:at,qq=<号>]`` / ``POKE:<号>`` directives
    even mid-DM (e.g. when discussing a third party in a group context).
    Asserting the entire line (not substring) locks the exact format —
    any drift fails loud.
    """
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    runner = _make_runner()
    event = _make_event(message_id="12345", chat_type="dm")

    text = await runner._prepare_inbound_message_text(
        event=event, source=event.source, history=[],
    )

    # 2026-04-25 12:33:00 UTC → 05:33:00 PDT (UTC-7).
    assert text == "04-25 05:33:00 [msg=12345] 891088473 近环: hello"


@pytest.mark.asyncio
async def test_group_default_inbound_text_includes_sender(monkeypatch):
    """Group default config (``group_sessions_per_user=True``) used to
    suppress the sender slot — the LLM had no idea who was speaking.
    Now sender always shows in groups too, with both qq_id and name."""
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    runner = _make_runner()
    event = _make_event(message_id="67890", chat_type="group")

    text = await runner._prepare_inbound_message_text(
        event=event, source=event.source, history=[],
    )

    assert text == "04-25 05:33:00 [msg=67890] 891088473 近环: hello"


@pytest.mark.asyncio
async def test_group_shared_session_uses_same_format_as_default(monkeypatch):
    """Shared group sessions previously got sender attribution while
    per-user sessions did not — that asymmetry is gone.  Both modes
    now produce identical output so the LLM sees one consistent
    transcript shape regardless of operator config."""
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    runner = _make_runner(group_sessions_per_user=False)
    event = _make_event(message_id="67890", chat_type="group")

    text = await runner._prepare_inbound_message_text(
        event=event, source=event.source, history=[],
    )

    assert text == "04-25 05:33:00 [msg=67890] 891088473 近环: hello"


@pytest.mark.asyncio
async def test_inbound_text_omits_msg_tag_when_message_id_missing(monkeypatch):
    """Defensive: synthetic events (e.g. background-process notifications,
    cron-triggered messages) may have no message_id.  Don't render
    `[msg=None]` or `[msg=]` garbage — sender slot still appears."""
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    runner = _make_runner()
    event = _make_event(message_id=None, chat_type="dm")

    text = await runner._prepare_inbound_message_text(
        event=event, source=event.source, history=[],
    )

    assert text == "04-25 05:33:00 891088473 近环: hello"


@pytest.mark.asyncio
async def test_sender_falls_back_to_qq_id_only_when_name_missing(monkeypatch):
    """When the user has no group card and no account nickname, the
    NapCat parser falls back to setting ``user_name = user_id``.  Don't
    render ``"891088473 891088473"`` — collapse to a single id."""
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    runner = _make_runner()
    event = _make_event(message_id="42", chat_type="group")
    # Mirror NapCat's `sender.get("card") or sender.get("nickname") or user_id`
    # fallback by setting user_name == user_id.
    event.source.user_name = "891088473"

    text = await runner._prepare_inbound_message_text(
        event=event, source=event.source, history=[],
    )

    assert text == "04-25 05:33:00 [msg=42] 891088473: hello"


@pytest.mark.asyncio
async def test_reply_context_to_other_user_message_is_inline_under_current_sender(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    runner = _make_runner()
    event = _make_event(message_id="current-1", chat_type="group")
    event.text = "我的消息"
    event.reply_to_message_id = "quoted-1"
    event.reply_to_text = (
        "04-24 16:36:48 [msg=quoted-1] 222333444 伊乐: 被引用的原文"
    )

    text = await runner._prepare_inbound_message_text(
        event=event, source=event.source, history=[],
    )

    assert text == (
        "04-25 05:33:00 [msg=current-1] 891088473 近环: "
        "[Replying to 04-24 16:36:48 [msg=quoted-1] "
        "222333444 伊乐: 被引用的原文]\n\n"
        "我的消息"
    )


@pytest.mark.asyncio
async def test_reply_context_to_assistant_message_is_inline_under_current_sender(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    runner = _make_runner()
    event = _make_event(message_id="current-2", chat_type="group")
    event.text = "对这句补充"
    event.reply_to_message_id = "bot-msg"
    event.reply_to_text = (
        "04-24 16:36:48 [msg=bot-msg] 1903883693 小麻雀: 小麻雀自己的回复"
    )

    text = await runner._prepare_inbound_message_text(
        event=event, source=event.source, history=[],
    )

    assert text == (
        "04-25 05:33:00 [msg=current-2] 891088473 近环: "
        "[Replying to 04-24 16:36:48 [msg=bot-msg] "
        "1903883693 小麻雀: 小麻雀自己的回复]\n\n"
        "对这句补充"
    )


# ---------------------------------------------------------------------------
# Assistant entry decoration (post-send finalization)
#
# The bot's own past messages must reach the LLM in EXACTLY the same
# `<time> [msg=<id>] <qq_id> <name>: <text>` shape as user messages, so
# the LLM can `REPLY:<own_id>` to quote its own messages on the next
# turn.  Without this, REPLY:<id> only works for inbound messages.
# ---------------------------------------------------------------------------


def _bot_adapter() -> OneBotNapCatAdapter:
    """Adapter with the bot's identity wired up so the assistant prefix
    can render `<bot_qq> <bot_name>` symmetrically with user inbound."""
    a = _adapter()
    a.self_id = 792836799
    a.self_name = "小麻雀"
    return a


def test_decorate_assistant_entry_prepends_full_inbound_style_prefix(monkeypatch):
    """When the assistant-header feature is opted in, send succeeded
    and message_id is known: the persisted assistant entry must start
    with the same prefix shape the LLM sees on inbound user lines."""
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    monkeypatch.setenv("NAPCAT_ASSISTANT_HEADER", "true")
    a = _bot_adapter()
    entry = {
        "role": "assistant",
        "content": "在的在的——！(◕‿◕)",
        # Wall-clock ts the runner stamped on the entry; format helper
        # will localize it.  2026-04-25 21:35 UTC → 14:35 PDT (UTC-7).
        "timestamp": 1777152900.0,
    }
    sr = SimpleNamespace(success=True, message_id=948824002, error=None)

    out = a.decorate_assistant_entry(entry, sr)

    assert out["content"] == (
        "04-25 14:35:00 [msg=948824002] 792836799 小麻雀: 在的在的——！(◕‿◕)"
    )
    assert out["role"] == "assistant"
    assert out["timestamp"] == 1777152900.0


def test_decorate_assistant_entry_omits_msg_tag_when_send_result_missing(
    monkeypatch,
):
    """With the feature opted in: send was skipped (interrupted / no
    text content) — `send_result` is None.  We still want to persist a
    sensible prefix so the entry is searchable in history; just drop
    the `[msg=<id>]` segment."""
    monkeypatch.setenv("NAPCAT_ASSISTANT_HEADER", "true")
    a = _bot_adapter()
    entry = {
        "role": "assistant",
        "content": "...",
        "timestamp": 1777152900.0,
    }
    out = a.decorate_assistant_entry(entry, None)
    assert out["content"].startswith("04-25 ")
    assert "[msg=" not in out["content"]
    assert "792836799 小麻雀:" in out["content"]


def test_decorate_assistant_entry_omits_msg_tag_when_send_failed(monkeypatch):
    """With the feature opted in: send returned a SendResult but with
    no message_id (e.g. failure or platform that doesn't return one).
    Same shape as the no-send case — prefix without `[msg=]`."""
    monkeypatch.setenv("NAPCAT_ASSISTANT_HEADER", "true")
    a = _bot_adapter()
    entry = {"role": "assistant", "content": "...", "timestamp": 1777152900.0}
    sr = SimpleNamespace(success=False, message_id=None, error="boom")
    out = a.decorate_assistant_entry(entry, sr)
    assert "[msg=" not in out["content"]


def test_decorate_assistant_entry_passes_through_when_content_empty(monkeypatch):
    """Defensive (feature opted in): don't manufacture a prefix for an
    entry with no real content — leave the entry alone for the runner
    / sanitizer."""
    monkeypatch.setenv("NAPCAT_ASSISTANT_HEADER", "true")
    a = _bot_adapter()
    entry = {"role": "assistant", "content": "", "timestamp": 1777152900.0}
    sr = SimpleNamespace(success=True, message_id=42, error=None)
    out = a.decorate_assistant_entry(entry, sr)
    assert out["content"] == ""
    # Same dict identity is fine; the contract is "don't mutate empty"
    assert out is entry


# ---------------------------------------------------------------------------
# Default-OFF behavior: with NAPCAT_ASSISTANT_HEADER unset, neither the
# adapter nor the prompt do anything around assistant headers.  This is
# the safe default that ships in production.
# ---------------------------------------------------------------------------


def test_decorate_assistant_entry_is_noop_by_default(monkeypatch):
    """Without ``NAPCAT_ASSISTANT_HEADER=true``, the hook must return
    the entry untouched — no `[msg=...]` prefix, no `<qq> <name>`,
    nothing.  Required because in-context-learning made the LLM mimic
    the prefix in its own output (doubled prefix bug, ENOENT for
    `MEDIA:<absolute path>`, `Get Uid Error` for `[CQ:at,qq=<号>]`)."""
    monkeypatch.delenv("NAPCAT_ASSISTANT_HEADER", raising=False)
    a = _bot_adapter()
    entry = {
        "role": "assistant",
        "content": "在的在的——！(◕‿◕)",
        "timestamp": 1777152900.0,
    }
    sr = SimpleNamespace(success=True, message_id=948824002, error=None)
    out = a.decorate_assistant_entry(entry, sr)
    assert out is entry  # identity — no copy, no rewrite
    assert out["content"] == "在的在的——！(◕‿◕)"


def test_base_channel_prompt_omits_assistant_header_warning_by_default(
    monkeypatch,
):
    """The anti-mimicry warning only ships when the decoration feature
    is enabled.  Otherwise the LLM doesn't need to worry about it (and
    the prompt stays a tad shorter)."""
    monkeypatch.delenv("NAPCAT_ASSISTANT_HEADER", raising=False)
    rendered = build_base_channel_prompt()
    flat = " ".join(rendered.split())
    assert "your OWN past assistant lines" not in flat
    assert "Do NOT include this prefix in your replies" not in flat


def test_base_channel_prompt_adds_assistant_header_warning_when_enabled(
    monkeypatch,
):
    """When ``NAPCAT_ASSISTANT_HEADER=true``, the prompt MUST gain the
    anti-mimicry warning so the LLM has any chance of not parroting the
    decoration in its own output."""
    monkeypatch.setenv("NAPCAT_ASSISTANT_HEADER", "true")
    rendered = build_base_channel_prompt()
    flat = " ".join(rendered.split())
    assert "your OWN past assistant lines" in flat
    assert "Do NOT include this prefix in your replies" in flat


def test_build_base_channel_prompt_raises_when_anchor_missing(monkeypatch):
    """If a future edit reshuffles BASE_CHANNEL_PROMPT and drops the
    SYSTEM-bullet anchor that the splice targets, the build function
    must raise — silently shipping a prompt that promises decoration
    without the anti-mimicry warning would re-open the production
    bugs (doubled prefix, fake msg_ids, ENOENT on placeholder paths)
    we added the warning to prevent."""
    monkeypatch.setenv("NAPCAT_ASSISTANT_HEADER", "true")
    monkeypatch.setattr(
        "gateway.platforms.onebot_napcat.prompts.BASE_CHANNEL_PROMPT",
        "QQ via NapCat.\n# Transcript format\n(anchor removed)\n",
    )
    with pytest.raises(RuntimeError, match="_TRANSCRIPT_FORMAT_ANCHOR"):
        build_base_channel_prompt()


def test_build_base_channel_prompt_does_not_raise_when_disabled_even_without_anchor(
    monkeypatch,
):
    """Symmetric: when the feature is off we don't splice anything, so
    a missing anchor must NOT raise — that would punish unrelated
    prompt edits that have nothing to do with assistant headers."""
    monkeypatch.delenv("NAPCAT_ASSISTANT_HEADER", raising=False)
    monkeypatch.setattr(
        "gateway.platforms.onebot_napcat.prompts.BASE_CHANNEL_PROMPT",
        "QQ via NapCat.\n# Transcript format\n(anchor removed)\n",
    )
    # Should produce a prompt without raising
    rendered = build_base_channel_prompt()
    assert "QQ via NapCat" in rendered
    assert "your OWN past assistant lines" not in rendered


def test_base_channel_prompt_contains_the_splice_anchor():
    """Lock the anchor invariant: ``_TRANSCRIPT_FORMAT_ANCHOR`` must
    appear verbatim in BASE_CHANNEL_PROMPT.  If a prompt edit drops or
    rewords the SYSTEM bullet, this test fires before the
    anchor-missing RuntimeError ever ships to production."""
    from gateway.platforms.onebot_napcat.prompts import _TRANSCRIPT_FORMAT_ANCHOR

    assert _TRANSCRIPT_FORMAT_ANCHOR in BASE_CHANNEL_PROMPT


def _make_session_store_with_db(tmp_path):
    """Real SessionStore + SessionDB bound to tmp_path so we exercise the
    actual SQLite + JSONL persistence the production load_transcript
    path reads from."""
    from gateway.session import SessionStore
    from gateway.config import GatewayConfig
    from hermes_state import SessionDB

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    store = SessionStore(sessions_dir=sessions_dir, config=GatewayConfig(platforms={}))
    # Replace the auto-created SessionDB with one bound to tmp_path so
    # the test never touches the user's real state.db.
    store._db = SessionDB(db_path=tmp_path / "state.db")
    return store


@pytest.mark.asyncio
async def test_reply_context_persists_to_sqlite_and_jsonl(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    monkeypatch.setenv("ONEBOT_NAPCAT_HOME_CHANNEL", "group_123")

    store = _make_session_store_with_db(tmp_path)
    runner = _make_runner()
    runner.config = GatewayConfig(
        platforms={Platform.ONEBOT_NAPCAT: PlatformConfig(enabled=True)},
        group_sessions_per_user=True,
    )
    runner.session_store = store
    runner._session_db = store._db
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._show_reasoning = False
    runner._set_session_env = lambda context: []
    runner._clear_session_env = lambda tokens: None
    runner._bind_adapter_run_generation = lambda *args, **kwargs: None
    runner._is_session_run_current = lambda *args, **kwargs: True
    runner._clear_restart_failure_count = lambda *args, **kwargs: None
    runner._should_send_voice_reply = lambda *args, **kwargs: False

    persisted_messages = {}

    async def fake_run_agent(**kwargs):
        persisted_messages["user"] = kwargs["message"]
        runner._session_db.ensure_session(
            kwargs["session_id"],
            source="onebot_napcat",
            model="test-model",
        )
        runner._session_db.append_message(
            session_id=kwargs["session_id"],
            role="user",
            content=kwargs["message"],
        )
        runner._session_db.append_message(
            session_id=kwargs["session_id"],
            role="assistant",
            content="收到",
        )
        return {
            "final_response": "收到",
            "messages": [
                {"role": "user", "content": kwargs["message"]},
                {"role": "assistant", "content": "收到"},
            ],
            "api_calls": 1,
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "tools": [],
        }

    runner._run_agent = fake_run_agent

    event = _make_event(message_id="current-1", chat_type="group")
    event.text = "我的消息"
    event.reply_to_message_id = "quoted-1"
    event.reply_to_text = (
        "04-24 16:36:48 [msg=quoted-1] 222333444 伊乐: 被引用的原文"
    )
    event.channel_prompt = ""
    event.metadata = None

    result = await runner._handle_message_with_agent(
        event,
        event.source,
        "onebot_napcat:group_123:891088473",
        1,
    )

    expected = (
        "04-25 05:33:00 [msg=current-1] 891088473 近环: "
        "[Replying to 04-24 16:36:48 [msg=quoted-1] "
        "222333444 伊乐: 被引用的原文]\n\n"
        "我的消息"
    )
    assert result == "收到"
    assert persisted_messages["user"] == expected

    session_id = store.get_or_create_session(event.source).session_id
    sqlite_users = [
        msg["content"]
        for msg in store._db.get_messages_as_conversation(session_id)
        if msg.get("role") == "user"
    ]
    assert sqlite_users == [expected]

    jsonl_users = []
    for line in store.get_transcript_path(session_id).read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("role") == "user":
            jsonl_users.append(entry.get("content"))
    assert jsonl_users == [expected]


def _seed_raw_turn(store, session_id: str, raw_assistant: str) -> str:
    """Mirror what the agent + runner persist for one turn pre-decoration.

    Both layers (SQLite via SessionDB.append_message, JSONL via
    SessionStore.append_to_transcript with skip_db=True) end up holding
    the bot's RAW assistant text — that's the starting state right
    before base.py's post-send decoration kicks in.
    """
    user_text = "04-25 14:34:55 [msg=948824002] 891088473 近环: 你好呀"
    store._db.ensure_session(session_id, source="onebot_napcat", model="m")
    store._db.append_message(session_id=session_id, role="user", content=user_text)
    store._db.append_message(session_id=session_id, role="assistant", content=raw_assistant)
    store.append_to_transcript(
        session_id, {"role": "user", "content": user_text}, skip_db=True,
    )
    store.append_to_transcript(
        session_id, {"role": "assistant", "content": raw_assistant}, skip_db=True,
    )
    return user_text


@pytest.mark.asyncio
async def test_post_send_decoration_reaches_load_transcript_end_to_end(
    monkeypatch, tmp_path,
):
    """End-to-end happy path: simulate agent's raw flush, then run
    base.py's `_decorate_persisted_assistant_entry` exactly the way it
    runs in production, then load history back the way the next turn
    does and assert the LLM sees the DECORATED entry.

    This is the contract test for plan B — if it fails, the LLM still
    cannot ``REPLY:<own_id>`` because the loader doesn't see the
    decoration.
    """
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    monkeypatch.setenv("NAPCAT_ASSISTANT_HEADER", "true")
    raw_assistant = "在的在的——！"
    store = _make_session_store_with_db(tmp_path)
    session_id = "sid-decorate-e2e"
    _seed_raw_turn(store, session_id, raw_assistant)

    a = _bot_adapter()
    a.set_session_store(store)

    # Stand-in MessageEvent that carries the session_id stash + a source
    # the adapter's decorate_assistant_entry expects.
    event = SimpleNamespace(_session_id=session_id)
    sr = SimpleNamespace(success=True, message_id=948824003, error=None)
    a._decorate_persisted_assistant_entry(event, raw_assistant, sr)

    history = store.load_transcript(session_id)
    last_assistant = [m for m in history if m.get("role") == "assistant"][-1]
    assert "[msg=948824003]" in last_assistant["content"], (
        f"Decoration invisible to LLM. Loaded: {last_assistant['content']!r}"
    )
    # Must NOT have appended a duplicate — exactly one assistant entry.
    assert sum(1 for m in history if m.get("role") == "assistant") == 1


def test_session_store_update_last_assistant_writes_both_sqlite_and_jsonl(
    tmp_path,
):
    """Update API must keep SQLite and JSONL in lockstep.  After a
    successful update both layers must contain the new content for the
    last assistant entry, and load_transcript (which can prefer either
    source) must return the new content from whichever it picks."""
    raw = "raw response"
    new = "decorated: raw response"
    store = _make_session_store_with_db(tmp_path)
    session_id = "sid-store-update"
    _seed_raw_turn(store, session_id, raw)

    ok = store.update_last_assistant_content(session_id, raw, new)
    assert ok is True

    # SQLite layer
    assert store._db.get_last_assistant_content(session_id) == new
    # JSONL layer
    transcript = store.get_transcript_path(session_id).read_text(encoding="utf-8")
    last_jsonl_assistant = None
    import json as _json
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        entry = _json.loads(line)
        if entry.get("role") == "assistant":
            last_jsonl_assistant = entry
    assert last_jsonl_assistant is not None
    assert last_jsonl_assistant["content"] == new
    # Loader (whichever source it picks) must see the new content
    history = store.load_transcript(session_id)
    last_assistant = [m for m in history if m.get("role") == "assistant"][-1]
    assert last_assistant["content"] == new


def test_session_store_update_refuses_when_sqlite_diverged(tmp_path):
    """Safety: if SQLite's last assistant content has drifted from
    ``expected_content`` (e.g. a concurrent turn already wrote, a
    /retry rewrote, decoration ran twice), refuse to update — better
    no decoration than corrupted history."""
    store = _make_session_store_with_db(tmp_path)
    session_id = "sid-sqlite-drift"
    _seed_raw_turn(store, session_id, "first response")

    # Something else mutates SQLite under us.
    store._db.update_last_assistant_content(session_id, "concurrent rewrite")

    # Now decoration tries with the OLD expected — must refuse.
    ok = store.update_last_assistant_content(
        session_id, "first response", "decorated: first response",
    )
    assert ok is False
    # SQLite stays at the concurrent value (no clobber)
    assert store._db.get_last_assistant_content(session_id) == "concurrent rewrite"
    # JSONL still has its original
    transcript = store.get_transcript_path(session_id).read_text(encoding="utf-8")
    assert "first response" in transcript
    assert "decorated:" not in transcript


def test_session_store_update_refuses_when_jsonl_diverged(tmp_path):
    """Symmetric safety: JSONL has been rewritten externally; refuse to
    update SQLite even though SQLite still matches expected."""
    store = _make_session_store_with_db(tmp_path)
    session_id = "sid-jsonl-drift"
    _seed_raw_turn(store, session_id, "first response")

    # Corrupt the JSONL last assistant line in place.
    transcript_path = store.get_transcript_path(session_id)
    lines = transcript_path.read_text(encoding="utf-8").splitlines(keepends=True)
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if not line:
            continue
        import json as _json
        entry = _json.loads(line)
        if entry.get("role") == "assistant":
            entry["content"] = "manually edited"
            lines[i] = _json.dumps(entry, ensure_ascii=False) + "\n"
            break
    transcript_path.write_text("".join(lines), encoding="utf-8")

    ok = store.update_last_assistant_content(
        session_id, "first response", "decorated: first response",
    )
    assert ok is False
    # SQLite untouched
    assert store._db.get_last_assistant_content(session_id) == "first response"


# ---------------------------------------------------------------------------
# Directive-placeholder safety: reject CQ:at and MEDIA: directives whose
# argument is obviously prompt-example boilerplate the LLM is quoting in
# prose (instead of using as a real action).
#
# Without these guards, an explanation like "use `[CQ:at,qq=<号>]` to
# mention someone" emitted by the bot would be sent to NapCat as a real
# at-segment with `qq=<号>`, triggering "Get Uid Error" and aborting
# the entire send (text body included), and `MEDIA:<absolute path>`
# would be treated as a real file and fail with ENOENT for
# `/opt/hermes_live/<absolute>`.  Both observed in production.
# ---------------------------------------------------------------------------


def test_at_segment_with_placeholder_qq_is_kept_as_text_not_real_at():
    """Bot quoted `[CQ:at,qq=<号>]` from the prompt — adapter must NOT
    emit a real at-segment (which would hit NapCat 'Get Uid Error' and
    fail the entire send).  Render the literal string as text so the
    user sees what the bot actually wrote."""
    from gateway.platforms.onebot_napcat.adapter import _expand_cq_codes_in_text

    segs = _expand_cq_codes_in_text("use [CQ:at,qq=<号>] to mention")
    # Must NOT contain an at-segment for the placeholder
    assert not any(s.get("type") == "at" for s in segs)
    # Whole input survives as text content (joined)
    joined = "".join(
        s["data"].get("text", "") for s in segs if s.get("type") == "text"
    )
    assert "[CQ:at,qq=<号>]" in joined


def test_at_segment_with_real_numeric_qq_still_works():
    """Don't break the actual mention path: a numeric qq must still
    produce a real at-segment so @-mentions in groups keep working."""
    from gateway.platforms.onebot_napcat.adapter import _expand_cq_codes_in_text

    segs = _expand_cq_codes_in_text("hi [CQ:at,qq=891088473] there")
    at = [s for s in segs if s.get("type") == "at"]
    assert len(at) == 1
    assert at[0]["data"]["qq"] == "891088473"


def test_at_segment_all_sentinel_still_works():
    """`[CQ:at,qq=all]` is the @everyone variant in OneBot — must keep
    being a real at-segment."""
    from gateway.platforms.onebot_napcat.adapter import _expand_cq_codes_in_text

    segs = _expand_cq_codes_in_text("[CQ:at,qq=all] hello group")
    at = [s for s in segs if s.get("type") == "at"]
    assert len(at) == 1
    assert at[0]["data"]["qq"] == "all"


def test_at_segment_with_empty_qq_is_kept_as_text():
    """`[CQ:at,qq=]` (empty value) is also a placeholder — never a real
    at-segment."""
    from gateway.platforms.onebot_napcat.adapter import _expand_cq_codes_in_text

    segs = _expand_cq_codes_in_text("[CQ:at,qq=] hi")
    assert not any(s.get("type") == "at" for s in segs)


def test_extract_media_skips_placeholder_path_with_angle_brackets():
    """Bot quoted `MEDIA:<absolute path>` from the prompt — extract_media
    must NOT return that as a real file path (which would later trigger
    ENOENT for `/opt/hermes_live/<absolute>` after path expansion)."""
    from gateway.platforms.base import BasePlatformAdapter

    media, cleaned = BasePlatformAdapter.extract_media(
        "you can attach files with `MEDIA:<absolute path>` like this"
    )
    assert media == []


def test_extract_media_skips_chinese_placeholder_path():
    """Same protection for the Chinese placeholder shape `MEDIA:<绝对路径>`
    that the bot may produce when explaining the directive in Chinese."""
    from gateway.platforms.base import BasePlatformAdapter

    media, cleaned = BasePlatformAdapter.extract_media("MEDIA:<绝对路径>")
    assert media == []


def test_extract_media_keeps_real_absolute_path():
    """Don't break the actual MEDIA: path — a real-looking absolute
    path with no angle brackets must still be picked up so TTS / file
    sends keep working."""
    from gateway.platforms.base import BasePlatformAdapter

    media, _ = BasePlatformAdapter.extract_media("MEDIA:/tmp/chart.png")
    assert len(media) == 1
    assert media[0][0] == "/tmp/chart.png"


def test_session_store_update_noop_when_no_assistant_entries(tmp_path):
    """Defensive: refuse silently if there's no assistant entry to
    target — avoids creating one out of thin air."""
    store = _make_session_store_with_db(tmp_path)
    session_id = "sid-empty"
    store._db.ensure_session(session_id, source="onebot_napcat", model="m")
    store._db.append_message(session_id=session_id, role="user", content="hi")

    ok = store.update_last_assistant_content(session_id, "anything", "new")
    assert ok is False
    # Nothing was written
    assert store._db.get_last_assistant_content(session_id) is None


@pytest.mark.asyncio
async def test_system_pseudo_sender_renders_bare_label(monkeypatch):
    """The poke handler builds a SessionSource with
    ``user_name='SYSTEM'`` for gateway-injected events.  That sentinel
    must render as a bare ``"SYSTEM"`` label — NOT ``"<actor_id> SYSTEM"``
    — so it stays visually distinct from real user messages and matches
    the prompt's ``SYSTEM:`` documentation."""
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    runner = _make_runner()
    # No message_id (poke notices don't carry one), user_name="SYSTEM".
    event = _make_event(message_id=None, chat_type="group")
    event.source.user_name = "SYSTEM"
    # Simulate a poke text payload so the assertion is against a
    # realistic line.
    event.text = "891088473 poked you"

    text = await runner._prepare_inbound_message_text(
        event=event, source=event.source, history=[],
    )

    assert text == "04-25 05:33:00 SYSTEM: 891088473 poked you"


# ---------------------------------------------------------------------------
# Channel prompt advertises the directive (so future-proofing fails loud
# if someone deletes the wording the model relies on).
# ---------------------------------------------------------------------------


def test_base_channel_prompt_documents_reply_directive():
    """The LLM only knows about REPLY: because the channel prompt tells
    it.  If this assertion fails, a code change has dropped the contract
    and any dependence the model has on ``REPLY:<id>`` will silently
    regress to default behavior."""
    assert "REPLY:<message_id>" in BASE_CHANNEL_PROMPT
    assert "REPLY:none" in BASE_CHANNEL_PROMPT
    assert "[msg=<id>]" in BASE_CHANNEL_PROMPT


def test_base_channel_prompt_warns_against_mimicking_assistant_prefix(
    monkeypatch,
):
    """When the assistant-header feature is opted in, the post-send
    adapter hook decorates persisted assistant entries with the same
    `<time> [msg=<id>] <qq> <name>:` shape user inbound lines have.
    Without an explicit warning, the LLM in-context-learns that this
    prefix IS its own output style and starts mimicking it (observed
    in production: bot fabricated a 10-digit fake msg_id and prefixed
    its reply, then our hook layered a SECOND real prefix on top —
    doubled prefix bug).

    This assertion locks in the anti-mimicry instruction: if a future
    prompt edit drops the warning, this test fires and forces the
    author to either restore it or migrate to a non-mimickable
    metadata format.
    """
    monkeypatch.setenv("NAPCAT_ASSISTANT_HEADER", "true")
    # Normalize whitespace so prompt line-wrapping doesn't break the
    # contract check.
    flat = " ".join(build_base_channel_prompt().split())
    # Must explicitly call out that the prefix on assistant lines is
    # adapter-injected, not bot-produced.
    assert "your OWN past assistant lines" in flat
    # Must explicitly tell the bot not to include the prefix.
    assert "Do NOT include this prefix in your replies" in flat
    # Must explain WHY (so the bot has a model, not just a rule) — the
    # adapter adds it post-send.
    assert "adapter" in flat.lower()
    assert "post-send" in flat.lower() or "automatically" in flat.lower()


# ---------------------------------------------------------------------------
# Response policy is mutually exclusive between passive / non-passive.
# Failing these tests means the gateway is shipping conflicting guidance
# (e.g. "always respond" alongside "[SILENT] when noise"), which lets
# the LLM rationalize either behavior on a coin flip.
# ---------------------------------------------------------------------------


def _render_prompt(passive: bool, monkeypatch) -> str:
    from gateway.platforms.onebot_napcat.prompts import build_base_channel_prompt

    if passive:
        monkeypatch.setenv("NAPCAT_GROUP_PASSIVE", "true")
    else:
        monkeypatch.delenv("NAPCAT_GROUP_PASSIVE", raising=False)
    return build_base_channel_prompt()


def test_non_passive_prompt_says_always_respond_and_omits_silent_directive(
    monkeypatch,
):
    rendered = _render_prompt(passive=False, monkeypatch=monkeypatch)
    assert "Always respond" in rendered
    # `[SILENT]` is a passive-only escape hatch.  Leaking it into the
    # default prompt would let the model skip arbitrary DMs.
    assert "[SILENT]" not in rendered
    assert "Group passive mode" not in rendered


def test_passive_prompt_uses_selective_policy_and_omits_always_respond(
    monkeypatch,
):
    rendered = _render_prompt(passive=True, monkeypatch=monkeypatch)
    assert "[SILENT]" in rendered
    assert "Skip when" in rendered
    assert "Respond when" in rendered
    # The "always respond" policy contradicts passive mode's selective
    # rules — they must not both ship in the same prompt.
    assert "Always respond" not in rendered
