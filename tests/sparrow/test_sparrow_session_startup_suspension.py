"""Sparrow policy tests for gateway startup session suspension."""

import json

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore


def _make_store(tmp_path, policy):
    config = GatewayConfig(default_reset_policy=policy)
    return SessionStore(sessions_dir=tmp_path / "sessions", config=config)


def _make_group_source():
    return SessionSource(
        platform=Platform.ONEBOT_NAPCAT,
        chat_id="group_940134062",
        user_id="891088473",
        chat_type="group",
    )


def test_mode_none_skips_recent_startup_suspension(tmp_path):
    store = _make_store(tmp_path, SessionResetPolicy(mode="none"))
    source = _make_group_source()
    entry = store.get_or_create_session(source)

    count = store.suspend_recently_active(max_age_seconds=120)

    assert count == 0
    assert entry.suspended is False

    same_entry = store.get_or_create_session(source)
    assert same_entry.session_id == entry.session_id
    assert same_entry.was_auto_reset is False


def test_non_none_policy_still_suspends_recent_sessions(tmp_path):
    store = _make_store(tmp_path, SessionResetPolicy(mode="idle", idle_minutes=1440))
    source = _make_group_source()
    entry = store.get_or_create_session(source)

    count = store.suspend_recently_active(max_age_seconds=120)

    assert count == 1
    assert entry.suspended is True


def test_mode_none_skips_stuck_loop_startup_suspension(tmp_path, monkeypatch):
    from gateway import run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    store = _make_store(tmp_path, SessionResetPolicy(mode="none"))
    entry = store.get_or_create_session(_make_group_source())
    failure_counts = {entry.session_key: GatewayRunner._STUCK_LOOP_THRESHOLD}
    (tmp_path / GatewayRunner._STUCK_LOOP_FILE).write_text(json.dumps(failure_counts))

    runner = object.__new__(GatewayRunner)
    runner.session_store = store

    suspended = runner._suspend_stuck_loop_sessions()

    assert suspended == 0
    assert entry.suspended is False


def test_stuck_loop_startup_suspends_when_policy_allows_it(tmp_path, monkeypatch):
    from gateway import run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    store = _make_store(tmp_path, SessionResetPolicy(mode="idle", idle_minutes=1440))
    entry = store.get_or_create_session(_make_group_source())
    failure_counts = {entry.session_key: GatewayRunner._STUCK_LOOP_THRESHOLD}
    (tmp_path / GatewayRunner._STUCK_LOOP_FILE).write_text(json.dumps(failure_counts))

    runner = object.__new__(GatewayRunner)
    runner.session_store = store

    suspended = runner._suspend_stuck_loop_sessions()

    assert suspended == 1
    assert entry.suspended is True
