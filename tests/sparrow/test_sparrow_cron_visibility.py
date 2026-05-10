"""Sparrow tests for cron-job observability and live-session mirroring.

Bugs these tests pin
--------------------
1. **sparrow.log silence on cron firings.** Before the
   ``cron_run_starting`` / ``cron_run_finished`` events, sparrow.log
   only emitted ``model_call_*`` / ``tool_call_*`` lines for cron
   activity — the operator could not tell from the timeline whether a
   given run was user-triggered or cron-triggered, what job it was, or
   whether delivery to the target chat actually succeeded.

2. **Cron-amnesia in the live session.** Cron jobs run in their own
   isolated session_id, deliver the agent's response to the target
   chat via ``adapter.send``, then exit — without writing the
   delivered message back into the target chat's live session
   transcript.  When the user later pings the bot in the same chat,
   the live session loads transcript history that's missing those
   cron messages, so the bot has amnesia about its own past output
   ("did I send the morning greeting?  I have no idea").

These tests lock in the fix for both: every cron firing produces a
``cron_run_*`` event pair, and every successful delivery mirrors the
agent response (annotated ``[cron @ <job_name>] ...``) into every
live session that matches the target ``(platform, chat_id)``.
"""

from __future__ import annotations

import json
import sys
import time as _time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_events(monkeypatch):
    """Replace ``gateway.sparrow_log.log_event`` with a spy that records
    every (event, fields) tuple. The lazy import inside cron.scheduler
    will pick this up because it imports the symbol from the module."""
    captured: List[Tuple[str, Dict[str, Any]]] = []
    fake = types.ModuleType("gateway.sparrow_log")
    fake.log_event = lambda event, **fields: captured.append((event, fields))
    monkeypatch.setitem(sys.modules, "gateway.sparrow_log", fake)
    return captured


@pytest.fixture
def cron_runtime(monkeypatch, tmp_path):
    """Build a synthetic cron environment.  Returns a SimpleNamespace
    holding the stubbed-out ``run_job``, ``save_job_output``,
    ``mark_job_run`` so each test can drive _process_job_with_events
    against any agent outcome."""
    import cron.scheduler as cs

    captured_marks: List[Tuple[str, bool, Optional[str], Optional[str]]] = []

    def fake_mark_job_run(job_id, success, error=None, delivery_error=None):
        captured_marks.append((job_id, success, error, delivery_error))

    def fake_save_job_output(job_id, output):
        return tmp_path / f"{job_id}.md"

    monkeypatch.setattr(cs, "mark_job_run", fake_mark_job_run)
    monkeypatch.setattr(cs, "save_job_output", fake_save_job_output)
    return types.SimpleNamespace(cs=cs, marks=captured_marks)


# ---------------------------------------------------------------------------
# Option E — cron sparrow events
# ---------------------------------------------------------------------------


def test_cron_event_pair_for_successful_run(monkeypatch, cron_runtime, captured_events):
    """Happy path: agent runs, returns text, delivery succeeds → exactly
    one starting + one finished event with status=ok."""
    cs = cron_runtime.cs

    monkeypatch.setattr(
        cs, "run_job",
        lambda job: (True, "# log\nresponse", "morning greeting", None),
    )
    monkeypatch.setattr(cs, "_deliver_result", lambda *_a, **_kw: None)

    job = {
        "id": "morning_job",
        "name": "Morning Greeting",
        "schedule_display": "0 8 * * *",
        "deliver": "onebot_napcat",
    }

    ok = cs._process_job_with_events(job, adapters={}, loop=None, verbose=False)
    assert ok is True

    events = [e for e, _ in captured_events]
    assert events == ["cron_run_starting", "cron_run_finished"]

    starting = captured_events[0][1]
    assert starting["job_id"] == "morning_job"
    assert starting["name"] == "Morning Greeting"
    assert starting["schedule"] == "0 8 * * *"
    assert starting["deliver"] == "onebot_napcat"

    finished = captured_events[1][1]
    assert finished["job_id"] == "morning_job"
    assert finished["status"] == "ok"
    assert finished["response_chars"] == len("morning greeting")
    assert "duration_ms" in finished


def test_cron_event_silent_status(monkeypatch, cron_runtime, captured_events):
    """``[SILENT]`` response → status=silent, response_chars=0, no
    delivery attempted.  Without distinct status flavors operators
    can't tell silent-by-design from silent-due-to-bug."""
    cs = cron_runtime.cs

    monkeypatch.setattr(
        cs, "run_job",
        lambda job: (True, "# log", "[SILENT]", None),
    )
    deliver_calls: List[Any] = []
    monkeypatch.setattr(
        cs, "_deliver_result",
        lambda *a, **kw: deliver_calls.append((a, kw)),
    )

    cs._process_job_with_events(
        {"id": "j1", "name": "j1", "deliver": "local"},
        adapters={}, loop=None, verbose=False,
    )

    finished = next(f for e, f in captured_events if e == "cron_run_finished")
    assert finished["status"] == "silent"
    assert finished["response_chars"] == 0
    # Silent must skip delivery
    assert deliver_calls == []


def test_cron_event_delivery_failed(monkeypatch, cron_runtime, captured_events):
    """Agent succeeded but the platform send returned an error → status
    must distinguish ``delivery_failed`` from ``failed``.  Operators
    triage these very differently: one is a network/auth issue on the
    target side, the other is the agent itself misbehaving."""
    cs = cron_runtime.cs

    monkeypatch.setattr(
        cs, "run_job",
        lambda job: (True, "# log", "all good", None),
    )
    monkeypatch.setattr(cs, "_deliver_result", lambda *_a, **_kw: "send_group_msg HTTP 500")

    cs._process_job_with_events(
        {"id": "j2", "name": "j2", "deliver": "onebot_napcat"},
        adapters={}, loop=None, verbose=False,
    )

    finished = next(f for e, f in captured_events if e == "cron_run_finished")
    assert finished["status"] == "delivery_failed"
    assert "HTTP 500" in finished["error"]
    assert finished["response_chars"] == len("all good")


def test_cron_event_failed_status_when_run_job_reports_failure(
    monkeypatch, cron_runtime, captured_events,
):
    """Agent itself errored → status=failed (not delivery_failed).
    The error is propagated so triage can find it without reading
    agent.log."""
    cs = cron_runtime.cs

    monkeypatch.setattr(
        cs, "run_job",
        lambda job: (False, "# log", "", "OpenAI 429 rate-limited"),
    )
    monkeypatch.setattr(cs, "_deliver_result", lambda *_a, **_kw: None)

    cs._process_job_with_events(
        {"id": "j3", "name": "j3", "deliver": "local"},
        adapters={}, loop=None, verbose=False,
    )

    finished = next(f for e, f in captured_events if e == "cron_run_finished")
    assert finished["status"] == "failed"
    assert "rate-limited" in finished["error"]


def test_cron_event_error_status_on_unexpected_exception(
    monkeypatch, cron_runtime, captured_events,
):
    """Even if run_job raises (not just returns success=False), the
    finished event must still fire so cron firings are NEVER silently
    invisible.  No event at all = the operator can't tell whether the
    job ran or was even due."""
    cs = cron_runtime.cs

    def _boom(job):
        raise RuntimeError("disk full")

    monkeypatch.setattr(cs, "run_job", _boom)
    monkeypatch.setattr(cs, "_deliver_result", lambda *_a, **_kw: None)

    ok = cs._process_job_with_events(
        {"id": "j4", "name": "j4", "deliver": "local"},
        adapters={}, loop=None, verbose=False,
    )
    assert ok is False

    events = [e for e, _ in captured_events]
    assert events == ["cron_run_starting", "cron_run_finished"]
    finished = captured_events[1][1]
    assert finished["status"] == "error"
    assert "disk full" in finished["error"]


def test_cron_finished_carries_duration(monkeypatch, cron_runtime, captured_events):
    """Operators want at-a-glance "how long did this cron take" for
    drift / slowness triage.  Pin the field so refactors can't drop
    it."""
    cs = cron_runtime.cs

    def _slow(job):
        _time.sleep(0.05)  # 50ms — measurable but quick
        return True, "# log", "ok", None

    monkeypatch.setattr(cs, "run_job", _slow)
    monkeypatch.setattr(cs, "_deliver_result", lambda *_a, **_kw: None)

    cs._process_job_with_events(
        {"id": "j5", "name": "j5", "deliver": "local"},
        adapters={}, loop=None, verbose=False,
    )

    finished = next(f for e, f in captured_events if e == "cron_run_finished")
    assert finished["duration_ms"] >= 40  # leave some slack for clock jitter


# ---------------------------------------------------------------------------
# Option A — mirror cron output to live session transcript
# ---------------------------------------------------------------------------


def test_cron_output_mirrored_to_live_session_transcript(monkeypatch):
    """End-to-end: cron job delivers a response to a chat that has a
    live gateway session.  After delivery the live session's
    transcript MUST contain the response (with cron annotation) so
    the next live turn knows what was sent.

    The bug this fixes (per sparrow.log timeline at 01:00:36→01:04):
    cron's morning greeting hit the user but never landed in the
    chat's live transcript, so the bot had amnesia about it.
    """
    import cron.scheduler as cs
    from gateway.platforms.base import SendResult

    captured_mirror_calls: List[Dict[str, Any]] = []

    def fake_mirror_to_session(
        platform, chat_id, message_text, *,
        source_label="cli", thread_id=None, user_id=None,
    ):
        captured_mirror_calls.append({
            "platform": platform,
            "chat_id": chat_id,
            "message_text": message_text,
            "source_label": source_label,
            "thread_id": thread_id,
        })
        return True

    fake_mirror_module = types.ModuleType("gateway.mirror")
    fake_mirror_module.mirror_to_session = fake_mirror_to_session
    monkeypatch.setitem(sys.modules, "gateway.mirror", fake_mirror_module)

    # Build a live adapter that succeeds.  cron uses asyncio bridges
    # to call adapter.send; we provide a stub adapter and a real loop
    # so that path is exercised end-to-end.
    import asyncio

    class _StubAdapter:
        async def send(self, chat_id, content, metadata=None, **_kw):
            return SendResult(success=True, message_id="sent_id")

    loop = asyncio.new_event_loop()
    loop_thread = None
    try:
        import threading
        def _run_loop():
            asyncio.set_event_loop(loop)
            loop.run_forever()
        loop_thread = threading.Thread(target=_run_loop, daemon=True)
        loop_thread.start()

        from gateway.config import Platform
        adapters = {Platform.ONEBOT_NAPCAT: _StubAdapter()}

        # Stub config so the standalone fallback path doesn't try to
        # read a real platform config in case the live adapter path
        # short-circuits.  Live adapter SHOULD win here, but keep the
        # standalone path safe.
        fake_config = MagicMock()
        fake_pconfig = MagicMock()
        fake_pconfig.enabled = True
        fake_config.platforms = {Platform.ONEBOT_NAPCAT: fake_pconfig}
        monkeypatch.setattr(cs, "load_config", lambda: {"cron": {"wrap_response": False}})
        with patch("gateway.config.load_gateway_config", return_value=fake_config):
            job = {
                "id": "morning_job",
                "name": "Morning Greeting",
                "deliver": "onebot_napcat:group_940134062",
            }
            err = cs._deliver_result(
                job,
                "早安——！(◕‿◕)♪ 今天天气真好★",
                adapters=adapters,
                loop=loop,
            )
            assert err is None, f"unexpected delivery failure: {err}"

    finally:
        loop.call_soon_threadsafe(loop.stop)
        if loop_thread is not None:
            loop_thread.join(timeout=2)
        loop.close()

    # The mirror call carries the agent's raw response (no cron-specific
    # prefix — the gateway transcript loader prepends ``[Delivered from
    # cron:<name>] ...`` on read via the existing mirror-record handling).
    # ``source_label`` uses the job NAME so the gateway-applied prefix is
    # human-readable in transcripts and search results.
    assert len(captured_mirror_calls) == 1
    call = captured_mirror_calls[0]
    assert call["platform"] == "onebot_napcat"
    assert call["chat_id"] == "group_940134062"
    assert "[cron @" not in call["message_text"]  # double-tagging guard
    assert call["message_text"].startswith("早安")
    assert call["source_label"] == "cron:Morning Greeting"


def test_cron_output_with_media_tags_strips_them_before_mirror(monkeypatch, tmp_path):
    """Mirrored content keeps the agent's prose but drops MEDIA: tags
    (the temp files those tags reference are gone by the time the
    live session would read them, and they'd render as raw text in
    the transcript anyway, polluting context with stale paths)."""
    import cron.scheduler as cs
    from gateway.platforms.base import SendResult

    captured: List[Dict[str, Any]] = []

    def fake_mirror(platform, chat_id, message_text, *, source_label="cli", thread_id=None, user_id=None):
        captured.append({"message_text": message_text})
        return True

    fake_mirror_module = types.ModuleType("gateway.mirror")
    fake_mirror_module.mirror_to_session = fake_mirror
    monkeypatch.setitem(sys.modules, "gateway.mirror", fake_mirror_module)

    img = tmp_path / "x.jpg"; img.write_bytes(b"\xff\xd8\xff")

    import asyncio
    import threading

    class _StubAdapter:
        async def send(self, chat_id, content, metadata=None, **_kw):
            return SendResult(success=True, message_id="sid")

    loop = asyncio.new_event_loop()
    def _run(): asyncio.set_event_loop(loop); loop.run_forever()
    th = threading.Thread(target=_run, daemon=True); th.start()

    try:
        from gateway.config import Platform
        adapters = {Platform.ONEBOT_NAPCAT: _StubAdapter()}
        monkeypatch.setattr(cs, "load_config", lambda: {"cron": {"wrap_response": False}})
        fake_config = MagicMock()
        fake_pconfig = MagicMock(); fake_pconfig.enabled = True
        fake_config.platforms = {Platform.ONEBOT_NAPCAT: fake_pconfig}
        with patch("gateway.config.load_gateway_config", return_value=fake_config):
            cs._deliver_result(
                {"id": "j", "name": "Photo Job", "deliver": "onebot_napcat:group_42"},
                f"check this out\nMEDIA:{img}",
                adapters=adapters, loop=loop,
            )
    finally:
        loop.call_soon_threadsafe(loop.stop); th.join(timeout=2); loop.close()

    assert len(captured) == 1
    msg = captured[0]["message_text"]
    assert "MEDIA:" not in msg
    assert "check this out" in msg
    # The gateway's mirror-record loader prepends ``[Delivered from
    # cron:<name>] ...`` on read; we MUST NOT prepend it again here or
    # the LLM would see a doubled prefix.
    assert "[cron @" not in msg
    assert "[Delivered from" not in msg


def test_cron_failed_delivery_does_not_mirror(monkeypatch):
    """If the live adapter returns failure AND the standalone fallback
    fails too, mirror MUST NOT run.  Mirroring on failure would tell
    the bot it sent something it didn't actually send — worse than
    the original bug."""
    import cron.scheduler as cs
    from gateway.platforms.base import SendResult

    mirror_calls: List[Any] = []

    fake_mirror_module = types.ModuleType("gateway.mirror")
    fake_mirror_module.mirror_to_session = lambda *a, **kw: mirror_calls.append((a, kw)) or True
    monkeypatch.setitem(sys.modules, "gateway.mirror", fake_mirror_module)

    import asyncio
    import threading

    class _BrokenAdapter:
        async def send(self, *a, **kw):
            return SendResult(success=False, error="auth error")

    loop = asyncio.new_event_loop()
    def _run(): asyncio.set_event_loop(loop); loop.run_forever()
    th = threading.Thread(target=_run, daemon=True); th.start()

    try:
        from gateway.config import Platform
        adapters = {Platform.ONEBOT_NAPCAT: _BrokenAdapter()}

        # Standalone fallback: pretend the platform isn't enabled so
        # _deliver_result records a delivery error and returns it.
        fake_config = MagicMock()
        fake_pconfig = MagicMock(); fake_pconfig.enabled = False
        fake_config.platforms = {Platform.ONEBOT_NAPCAT: fake_pconfig}
        monkeypatch.setattr(cs, "load_config", lambda: {"cron": {"wrap_response": False}})
        with patch("gateway.config.load_gateway_config", return_value=fake_config):
            err = cs._deliver_result(
                {"id": "j", "name": "Bad Job", "deliver": "onebot_napcat:group_42"},
                "should not be mirrored",
                adapters=adapters, loop=loop,
            )
            assert err is not None  # must report failure to caller

    finally:
        loop.call_soon_threadsafe(loop.stop); th.join(timeout=2); loop.close()

    assert mirror_calls == []
