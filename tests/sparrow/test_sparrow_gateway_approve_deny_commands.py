"""Sparrow-specific approval command tests kept out of upstream test files."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source(
    *,
    user_id: str = "u1",
    chat_id: str = "c1",
    chat_type: str = "dm",
    platform: Platform = Platform.TELEGRAM,
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="tester",
        chat_type=chat_type,
    )


def _make_event(text: str, source: SessionSource | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=source or _make_source(),
        message_id="m1",
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner.pairing_store = SimpleNamespace(
        is_approved=lambda _platform, user_id: user_id == "u1"
    )
    runner._set_session_env = lambda _context: None
    return runner


def _clear_approval_state():
    from tools import approval as mod

    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()


class TestApproveCommand:
    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_group_allowlist_sender_cannot_approve(self, monkeypatch):
        """Group authorization alone must not allow command approval."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        for name in (
            "GATEWAY_APPROVAL_ALLOWED_USERS",
            "TELEGRAM_APPROVAL_ALLOWED_USERS",
            "GATEWAY_ALLOWED_USERS",
            "TELEGRAM_ALLOWED_USERS",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "group1")

        runner = _make_runner()
        runner.pairing_store = SimpleNamespace(
            is_approved=lambda _platform, _user_id: False
        )
        source = _make_source(user_id="group-user", chat_id="group1", chat_type="group")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        result = await runner._handle_approve_command(_make_event("/approve", source))
        assert "主人账号" in result
        assert not entry.event.is_set()

    @pytest.mark.asyncio
    async def test_approval_allowlist_sender_can_approve_from_group(self, monkeypatch):
        """Explicit approval allowlist grants approval even in a shared group."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "group1")
        monkeypatch.setenv("TELEGRAM_APPROVAL_ALLOWED_USERS", "owner")

        runner = _make_runner()
        runner.pairing_store = SimpleNamespace(
            is_approved=lambda _platform, _user_id: False
        )
        source = _make_source(user_id="owner", chat_id="group1", chat_type="group")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        result = await runner._handle_approve_command(_make_event("/approve", source))
        assert "approved" in result.lower()
        assert entry.event.is_set()
