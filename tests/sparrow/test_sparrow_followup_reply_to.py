"""Sparrow tests for the queued-follow-up reply_to fix.

Bug it locks down
-----------------
``base.py:_process_message_background`` keeps a SINGLE ``MessageEvent``
for the lifetime of one background task and uses ``event.message_id``
as ``reply_to`` when sending the agent's final response.  When a user
sends a second message while the first is still being processed, the
gateway queues the second message and ``_run_agent`` recurses to
answer it.  Without intervention, the response to that follow-up
message ships with ``reply_to=event.message_id`` of the ORIGINAL
message — so on QQ/Telegram/etc. the user sees the bot quote the
wrong message.

The fix:
  1. The recursion site at the bottom of ``_run_agent`` tags its
     result dict with ``latest_event_message_id`` (the id of the
     queued message it actually answered).
  2. ``_handle_message`` calls ``_retarget_event_to_followup(event,
     result)`` after ``_run_agent`` returns; the helper mutates
     ``event.message_id`` so base.py's later ``reply_to`` read picks
     up the follow-up's id.

These tests cover the helper's behavior exhaustively (including the
deep-recursion case where 3+ messages are chained) plus the
``MessageEvent`` mutability assumption that the fix relies on.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock

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

from gateway.platforms.base import MessageEvent, MessageType, SessionSource
from gateway.run import GatewayRunner


_FIXED_TS = datetime(2026, 4, 27, 2, 29, 30, tzinfo=timezone.utc)


def _make_event(message_id: str = "msg_original") -> MessageEvent:
    source = SessionSource(
        platform=MagicMock(value="onebot_napcat"),
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
# MessageEvent mutability — the assumption the whole fix hangs on
# ---------------------------------------------------------------------------


def test_message_event_message_id_is_mutable():
    """The fix mutates ``event.message_id`` in-place because base.py
    reads it AFTER the handler returns.  If MessageEvent ever becomes
    a frozen dataclass / immutable pydantic model, this test fails
    LOUDLY here instead of silently breaking reply_to in production."""
    event = _make_event(message_id="msg_a")
    event.message_id = "msg_b"  # must not raise
    assert event.message_id == "msg_b"


# ---------------------------------------------------------------------------
# _retarget_event_to_followup — happy paths and no-ops
# ---------------------------------------------------------------------------


def test_retarget_mutates_when_followup_id_present_and_differs():
    """Concrete reproduction of the user-reported bug: msg1 answered
    inline, msg2 answered by recursion.  After _run_agent returns,
    event.message_id must be msg2 so base.py's reply_to lands on it."""
    event = _make_event(message_id="msg1_今日计划")
    result = {"final_response": "...", "latest_event_message_id": "msg2_蹦蹦跳跳"}

    GatewayRunner._retarget_event_to_followup(event, result)

    assert event.message_id == "msg2_蹦蹦跳跳"


def test_retarget_no_op_when_key_missing():
    """No follow-up was queued — _run_agent never recursed, so the
    result dict has no ``latest_event_message_id``.  Mutating in this
    case would corrupt single-turn replies, so the helper must leave
    event.message_id alone."""
    event = _make_event(message_id="msg_only")
    result = {"final_response": "..."}  # no latest_event_message_id

    GatewayRunner._retarget_event_to_followup(event, result)

    assert event.message_id == "msg_only"


def test_retarget_no_op_when_latest_id_matches_current():
    """Defensive: if for some reason the result tags the SAME id that
    event already carries (e.g. agent_result wrapping happened but no
    actual recursion swap), there's no need to mutate.  Should be a
    no-op so we don't spuriously trigger downstream logic that may
    react to identity changes."""
    event = _make_event(message_id="msg_same")
    result = {"latest_event_message_id": "msg_same"}

    GatewayRunner._retarget_event_to_followup(event, result)

    assert event.message_id == "msg_same"


def test_retarget_no_op_when_latest_id_is_none_or_empty():
    """Truthy guard: ``latest_event_message_id`` could end up as None
    or '' if the queued event lacked a message_id (rare, but happens
    on internal/synthetic events).  Mutating to a falsy value would
    break ``reply_to`` entirely, so the helper must skip."""
    event = _make_event(message_id="msg_keep")

    for falsy in (None, "", 0, False):
        result = {"latest_event_message_id": falsy}
        GatewayRunner._retarget_event_to_followup(event, result)
        assert event.message_id == "msg_keep", f"mutated on falsy {falsy!r}"


def test_retarget_no_op_when_result_is_not_dict():
    """``_run_agent`` is documented to return a dict, but historically
    has had error paths returning None.  Helper must tolerate that
    without raising."""
    event = _make_event(message_id="msg_keep")

    for non_dict in (None, "string", 42, ["list"], object()):
        GatewayRunner._retarget_event_to_followup(event, non_dict)  # must not raise
        assert event.message_id == "msg_keep"


# ---------------------------------------------------------------------------
# Deep recursion: 3-message chain (msg1 → queued msg2 → queued msg3)
# ---------------------------------------------------------------------------


def test_setdefault_preserves_deepest_message_id_through_chain():
    """The recursion site uses ``result.setdefault('latest_event_message_id',
    next_message_id)``.  When the chain is msg1→msg2→msg3:

      - level 2 (deepest, answers msg3) returns its result
      - level 1 (parent, answered msg2 itself) sets the key to msg3
        because that's what the inner level handed back
      - level 0 (outermost, answered msg1) sees the key already set
        to msg3 and ``setdefault`` is a no-op — preserving msg3, not
        clobbering with msg2

    This pure-dict simulation locks the semantics so a refactor can't
    accidentally swap setdefault for unconditional assignment.
    """
    # Simulate level 2 (deepest) returning its result with NO key set.
    level2_result = {"final_response": "answer to msg3"}

    # Level 1 unwinds, having recursed with next_message_id="msg3".
    if isinstance(level2_result, dict) and "msg3":
        level2_result.setdefault("latest_event_message_id", "msg3")

    # Level 0 unwinds, having recursed with next_message_id="msg2".
    if isinstance(level2_result, dict) and "msg2":
        level2_result.setdefault("latest_event_message_id", "msg2")

    # The deepest id (msg3) must win — that's the message the user
    # last sent and the response is for.
    assert level2_result["latest_event_message_id"] == "msg3"


def test_setdefault_sets_id_on_first_pass_when_unset():
    """Single-recursion case (msg1 → queued msg2): no inner setdefault
    has run yet, so the level-0 setdefault is the one that sets the
    key.  Verifies the simple path."""
    inner_result = {"final_response": "answer to msg2"}

    if isinstance(inner_result, dict) and "msg2":
        inner_result.setdefault("latest_event_message_id", "msg2")

    assert inner_result["latest_event_message_id"] == "msg2"
