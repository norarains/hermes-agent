"""Sparrow-specific busy-session queue tests kept out of upstream test files."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

import sys
import types

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

from gateway.platforms.base import MessageEvent, MessageType, SessionSource, build_session_key


_FIXED_TS = datetime(2026, 4, 24, 23, 36, 48, tzinfo=timezone.utc)


def _make_event(text="hello", chat_id="123", platform_val="telegram"):
    source = SessionSource(
        platform=MagicMock(value=platform_val),
        chat_id=chat_id,
        chat_type="private",
        user_id="user1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
        timestamp=_FIXED_TS,
    )


def _make_runner():
    from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._busy_ack_enabled = True
    runner._draining = False
    runner.adapters = {}
    runner.config = MagicMock()
    runner.config.group_sessions_per_user = True
    runner.config.thread_sessions_per_user = False
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    # Auth gate added by upstream #17775 — busy path now calls
    # ``_is_user_authorized`` before processing.  The test bypasses the
    # full ``__init__`` so we wire up the minimum it needs to authorise.
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved = MagicMock(return_value=True)
    runner._is_user_authorized = MagicMock(return_value=True)
    return runner, _AGENT_PENDING_SENTINEL


def _make_adapter(platform_val="telegram"):
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value=platform_val)
    return adapter


class TestBusySessionAck:
    @pytest.mark.asyncio
    async def test_busy_ack_can_be_disabled(self, monkeypatch):
        """Disabling busy_ack should still queue input without sending chat noise.

        Upstream's _handle_active_session_busy_message reads
        ``HERMES_GATEWAY_BUSY_ACK_ENABLED`` directly from os.environ
        rather than self._busy_ack_enabled, so set the env var.
        """
        monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        runner._busy_ack_enabled = False
        adapter = _make_adapter()

        event = _make_event(text="quiet queue")
        session_key = build_session_key(event.source)
        agent = MagicMock()
        runner._running_agents[session_key] = agent
        runner.adapters[event.source.platform] = adapter

        result = await runner._handle_active_session_busy_message(event, session_key)

        assert result is True
        assert adapter._pending_messages[session_key].text == "quiet queue"
        adapter._send_with_retry.assert_not_called()
        agent.interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_queue_mode_merges_text_followups_with_agent_hint(self):
        """Queued text follow-ups should be delivered as an ordered bundle."""
        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()

        first = _make_event(text="3")
        second = MessageEvent(
            text="4",
            message_type=MessageType.TEXT,
            source=first.source,
            message_id="msg2",
            timestamp=_FIXED_TS,
        )
        session_key = build_session_key(first.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 1,
            "max_iterations": 90,
            "current_tool": None,
        }
        runner._running_agents[session_key] = agent
        runner.adapters[first.source.platform] = adapter

        await runner._handle_active_session_busy_message(first, session_key)
        await runner._handle_active_session_busy_message(second, session_key)

        agent.interrupt.assert_not_called()
        queued = adapter._pending_messages[session_key].text
        assert "multiple queued messages" in queued
        assert "Process them in order" in queued
        assert "Timezone:" in queued
        assert "1. " in queued and ": 3" in queued
        assert "2. " in queued and ": 4" in queued

    @pytest.mark.asyncio
    async def test_queue_mode_group_bundle_preserves_sender_labels(self):
        """Shared group queued bundles should not collapse all text under one sender."""
        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()
        platform = MagicMock(value="telegram")
        source1 = SessionSource(
            platform=platform,
            chat_id="group_1",
            chat_type="group",
            user_id="u1",
            user_name="Alice",
        )
        source2 = SessionSource(
            platform=platform,
            chat_id="group_1",
            chat_type="group",
            user_id="u2",
            user_name="Bob",
        )
        first = MessageEvent(
            text="3",
            message_type=MessageType.TEXT,
            source=source1,
            message_id="msg1",
            timestamp=_FIXED_TS,
        )
        second = MessageEvent(
            text="4",
            message_type=MessageType.TEXT,
            source=source2,
            message_id="msg2",
            timestamp=_FIXED_TS,
        )
        session_key = build_session_key(source1, group_sessions_per_user=False)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 1,
            "max_iterations": 90,
            "current_tool": None,
        }
        runner._running_agents[session_key] = agent
        runner.adapters[platform] = adapter

        await runner._handle_active_session_busy_message(first, session_key)
        await runner._handle_active_session_busy_message(second, session_key)

        queued_event = adapter._pending_messages[session_key]
        assert getattr(queued_event, "_sender_attributed_bundle", False)
        assert "1. " in queued_event.text and "Alice: 3" in queued_event.text
        assert "2. " in queued_event.text and "Bob: 4" in queued_event.text

        runner.config.group_sessions_per_user = False
        prepared = await runner._prepare_inbound_message_text(
            event=queued_event,
            source=queued_event.source,
            history=[],
        )
        assert prepared == queued_event.text
        assert not prepared.startswith("[Alice] [The user sent multiple queued messages")
