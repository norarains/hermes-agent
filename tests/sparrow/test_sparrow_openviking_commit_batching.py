"""Sparrow customization: OpenViking commit batching.

Each ``/commit`` call triggers OpenViking's memory extractor — an LLM
round-trip over the session's transcript.  Per-turn commit (the
upstream default) is the dominant per-turn cost when ``vlm.provider =
openai`` and the underlying model is GPT-class.

Sparrow batches commits: messages POST every turn (so transcripts stay
fresh and queryable), but ``/commit`` only fires every
``OPENVIKING_COMMIT_EVERY_N_TURNS`` turns (default 10).  Gateway
shutdown / Ctrl+C / atexit force-flushes the trailing batch so no
data is lost.

These tests pin the contract by mocking the OpenViking HTTP client
and counting message vs commit POSTs across a sequence of turns +
shutdown.
"""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from plugins.memory.openviking import OpenVikingMemoryProvider


def _make_provider(monkeypatch, *, commit_every: int = 10):
    """Build a provider with HTTP fully mocked, threading made
    synchronous so test assertions can run right after each
    ``sync_turn``."""
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://test")
    monkeypatch.setenv("OPENVIKING_API_KEY", "")
    monkeypatch.setenv("OPENVIKING_COMMIT_EVERY_N_TURNS", str(commit_every))

    posts = []  # captures (path, payload) for every client.post call

    class FakeClient:
        def __init__(self, *_a, **_kw):
            pass

        def post(self, path, payload=None, **_kw):
            posts.append((path, payload))
            return {}

        def get(self, path, **_kw):
            return {}

        def health(self):
            # Treat fake server as healthy so initialize() keeps the
            # client wired up.
            return True

    # Patch threading.Thread to run synchronously so post counts are
    # observable immediately after sync_turn returns.  Daemon-friendly
    # and avoids racy joins in CI.
    class SyncThread:
        def __init__(self, target, daemon=None, name=None, **_kw):
            self._target = target
            self._done = False

        def start(self):
            self._target()
            self._done = True

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(
        "plugins.memory.openviking._VikingClient", FakeClient
    )
    monkeypatch.setattr(
        "plugins.memory.openviking.threading.Thread", SyncThread
    )

    p = OpenVikingMemoryProvider()
    # Bypass network init: stub out the client probe so initialize()
    # treats us as connected.
    p.initialize("test-session")
    p._client = FakeClient()
    return p, posts


def _commit_count(posts):
    return sum(1 for path, _ in posts if path.endswith("/commit"))


def _message_count(posts):
    return sum(1 for path, _ in posts if path.endswith("/messages"))


def test_default_commit_interval_is_10(monkeypatch):
    """Sparrow default: commit fires once per 10 turns, not per turn.

    Pinning the default guards against a regression to the upstream
    per-turn behavior, which doubled OpenAI bills overnight when the
    extractor is gpt-class."""
    monkeypatch.delenv("OPENVIKING_COMMIT_EVERY_N_TURNS", raising=False)
    p = OpenVikingMemoryProvider()
    p.initialize("s")
    assert p._commit_every_n_turns == 10


def test_commits_only_every_n_turns(monkeypatch):
    """N=3: turns 1,2 post messages with no commit; turn 3 commits;
    turns 4,5 again no commit; turn 6 commits; etc."""
    p, posts = _make_provider(monkeypatch, commit_every=3)

    for i in range(6):
        p.sync_turn(f"u{i}", f"a{i}")

    # 6 turns × 2 messages each = 12 message posts
    assert _message_count(posts) == 12
    # Commits at turn 3 and turn 6 = 2 commits
    assert _commit_count(posts) == 2
    # After turn 6 (a multiple of N), nothing pending
    assert p._uncommitted_count == 0


def test_uncommitted_state_between_intervals(monkeypatch):
    """Mid-batch: counter tracks how many turns are pending commit."""
    p, posts = _make_provider(monkeypatch, commit_every=5)

    p.sync_turn("u1", "a1")
    p.sync_turn("u2", "a2")
    assert p._uncommitted_count == 2
    assert _commit_count(posts) == 0

    p.sync_turn("u3", "a3")
    p.sync_turn("u4", "a4")
    p.sync_turn("u5", "a5")
    assert p._uncommitted_count == 0
    assert _commit_count(posts) == 1


def test_flush_forces_commit_when_uncommitted(monkeypatch):
    """Mid-batch flush() must fire a commit so the trailing turns
    are not lost on shutdown.

    DESIGN INVARIANT regression: the whole point of batching is that
    *no* turn data is silently dropped — flush is the safety net."""
    p, posts = _make_provider(monkeypatch, commit_every=10)

    for i in range(3):
        p.sync_turn(f"u{i}", f"a{i}")
    assert _commit_count(posts) == 0  # 3 < 10, no commit yet
    assert p._uncommitted_count == 3

    p.flush()

    assert _commit_count(posts) == 1, (
        "flush() must force a commit when uncommitted_count > 0"
    )
    assert p._uncommitted_count == 0


def test_flush_is_noop_when_nothing_pending(monkeypatch):
    """flush() with zero uncommitted turns must not fire a commit —
    that would re-extract over already-processed transcript and
    double-bill the LLM."""
    p, posts = _make_provider(monkeypatch, commit_every=2)

    p.sync_turn("u1", "a1")
    p.sync_turn("u2", "a2")  # commits here
    assert _commit_count(posts) == 1
    assert p._uncommitted_count == 0

    p.flush()
    assert _commit_count(posts) == 1, (
        "flush() must NOT re-fire commit when nothing is pending"
    )


def test_on_session_end_calls_flush(monkeypatch):
    """on_session_end is the atexit + gateway-shutdown entry point.
    Must trigger the same flush behavior (force-commit pending batch)
    so Ctrl+C never loses turn data."""
    p, posts = _make_provider(monkeypatch, commit_every=10)

    for i in range(4):
        p.sync_turn(f"u{i}", f"a{i}")
    assert _commit_count(posts) == 0  # 4 < 10, nothing committed

    p.on_session_end([])  # gateway shutdown / atexit path

    assert _commit_count(posts) == 1, (
        "on_session_end must flush the pending batch — losing turn "
        "data on Ctrl+C is the bug this whole feature exists to "
        "prevent."
    )
    assert p._uncommitted_count == 0


def test_shutdown_flushes_before_thread_cleanup(monkeypatch):
    """shutdown() is the explicit path used by code-driven cleanups.
    Must flush before joining threads so the trailing batch reaches
    the extractor."""
    p, posts = _make_provider(monkeypatch, commit_every=10)

    for i in range(7):
        p.sync_turn(f"u{i}", f"a{i}")
    assert _commit_count(posts) == 0

    p.shutdown()

    assert _commit_count(posts) == 1
    assert p._uncommitted_count == 0


def test_invalid_env_falls_back_to_default(monkeypatch):
    """Garbage in OPENVIKING_COMMIT_EVERY_N_TURNS must not crash
    initialization — fall back to default 10."""
    monkeypatch.setenv("OPENVIKING_COMMIT_EVERY_N_TURNS", "not-a-number")
    p = OpenVikingMemoryProvider()
    p.initialize("s")
    assert p._commit_every_n_turns == 10


def test_zero_or_negative_clamped_to_one(monkeypatch):
    """0 or negative N would mean 'never commit' — that contradicts
    the whole feature.  Clamp to 1 (commit every turn = upstream
    behavior) instead of silently disabling."""
    for raw in ("0", "-5"):
        monkeypatch.setenv("OPENVIKING_COMMIT_EVERY_N_TURNS", raw)
        p = OpenVikingMemoryProvider()
        p.initialize("s")
        assert p._commit_every_n_turns >= 1, (
            f"raw={raw!r} should clamp to >=1, got {p._commit_every_n_turns}"
        )
