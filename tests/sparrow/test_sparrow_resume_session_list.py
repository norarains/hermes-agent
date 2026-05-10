"""Sparrow tests for the gateway /resume session list display."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_state import SessionDB


def _ts(year, month, day, hour, minute):
    return datetime(
        year, month, day, hour, minute, tzinfo=ZoneInfo("America/Los_Angeles")
    ).timestamp()


def _make_event(text="/resume"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.ONEBOT_NAPCAT,
            user_id="891088473",
            chat_id="group_940134062",
            chat_type="group",
            user_name="近环",
        ),
    )


def _make_runner(db, session_store=None):
    runner = object.__new__(GatewayRunner)
    runner._session_db = db
    runner.config = GatewayConfig()
    runner.session_store = session_store
    return runner


def _set_session_times(db, session_id, *, started_at, ended_at=None):
    def _do(conn):
        conn.execute(
            "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
            (started_at, ended_at, session_id),
        )

    db._execute_write(_do)


@pytest.mark.asyncio
async def test_resume_list_shows_time_range_and_message_count(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("old_group_session", "onebot_napcat", user_id="891088473")
        db.set_session_title("old_group_session", "Casual Greeting Check-in")
        db.append_message("old_group_session", "user", "在吗")
        db.append_message("old_group_session", "assistant", "在")
        _set_session_times(
            db,
            "old_group_session",
            started_at=_ts(2026, 4, 24, 4, 20),
            ended_at=_ts(2026, 4, 25, 22, 57),
        )

        db.create_session("current_group_session", "onebot_napcat", user_id="891088473")
        db.set_session_title("current_group_session", "Casual Chat Greeting Response")
        db.append_message("current_group_session", "user", "你好")
        _set_session_times(
            db,
            "current_group_session",
            started_at=_ts(2026, 4, 25, 22, 57),
        )

        result = await _make_runner(db)._handle_resume_command(_make_event())

        assert (
            "• **Casual Chat Greeting Response** — 04-25 22:57 → current · 1 msg"
            in result
        )
        assert (
            "• **Casual Greeting Check-in** — 04-24 04:20 → 04-25 22:57 · 2 msgs"
            in result
        )
        assert "你好" not in result
        assert "在吗" not in result
    finally:
        db.close()


@pytest.mark.asyncio
async def test_resume_success_uses_same_total_message_count_as_list(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("old_group_session", "onebot_napcat", user_id="891088473")
        db.set_session_title("old_group_session", "Casual Greeting Check-in")
        db.append_message("old_group_session", "user", "在吗")
        db.append_message("old_group_session", "assistant", "在")
        db.append_message("old_group_session", "tool", "tool result")
        db.append_message("old_group_session", "user", "继续")
        db.append_message("old_group_session", "assistant", "好")

        current_entry = SimpleNamespace(session_id="current_group_session")
        switched_entry = SimpleNamespace(session_id="old_group_session")
        session_store = MagicMock()
        session_store._generate_session_key.return_value = "onebot_napcat:group_940134062"
        session_store.get_or_create_session.return_value = current_entry
        session_store.switch_session.return_value = switched_entry
        session_store.load_transcript.return_value = [
            {"role": "user", "content": "在吗"},
            {"role": "user", "content": "继续"},
        ]

        runner = _make_runner(db, session_store=session_store)
        runner._release_running_agent_state = lambda session_key: None
        runner._clear_session_boundary_security_state = lambda session_key: None

        result = await runner._handle_resume_command(
            _make_event("/resume Casual Greeting Check-in")
        )

        assert (
            result
            == "↻ Resumed session **Casual Greeting Check-in** (5 msgs). Conversation restored."
        )
        session_store.load_transcript.assert_not_called()
    finally:
        db.close()
