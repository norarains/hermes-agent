"""Tests for the memory-provider runtime-failure notification path.

When a memory provider call fails at runtime (per-turn sync, MCP tool,
session commit, ...), the gateway pushes an out-of-band ``memory_unhealthy``
system message to the user instead of either crashing the gateway or
swallowing the failure as a DEBUG log.

These tests pin:
  - The ``memory_unhealthy`` user-message kind exists with the contracted
    vars and a default template that uses all of them.
  - ``MemoryProvider._notify_user`` is a no-op without a callback (CLI/tests)
    and otherwise invokes the callback with (name, endpoint, error).  An
    exception inside the callback must NOT crash the provider call site.
  - ``MemoryManager.set_notify_user_callback`` fans out / clears the
    callback across all registered providers.
  - The OpenViking plugin actually calls ``self._notify_user(...)`` from
    each of its four runtime failure paths: sync_turn, handle_tool_call,
    on_session_end (commit), on_memory_write.

Background-thread paths (sync_turn, on_memory_write) join the spawned
worker before asserting so the test deterministically observes the
post-failure callback invocation.
"""

from __future__ import annotations

import re
import threading
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from gateway.user_messages import (
    MESSAGE_KINDS,
    _DEFAULT_TEMPLATES,
    render_user_message,
)
from plugins.memory.openviking import OpenVikingMemoryProvider


# ---------------------------------------------------------------------------
# user_messages: kind / template contract for memory_unhealthy
# ---------------------------------------------------------------------------


def test_memory_unhealthy_kind_is_registered_with_three_vars():
    """The kind contract is the one place callers and config validation
    look up which placeholders are legal — it must list provider/endpoint/error."""
    assert "memory_unhealthy" in MESSAGE_KINDS
    assert set(MESSAGE_KINDS["memory_unhealthy"]) == {"provider", "endpoint", "error"}


def test_memory_unhealthy_default_template_uses_all_three_vars():
    """Without all three vars the user can't tell WHICH provider broke,
    where it was reaching, or WHY — so the default template must surface
    every contracted variable."""
    template = _DEFAULT_TEMPLATES["memory_unhealthy"]
    used = set(re.findall(r"\{([a-zA-Z_]\w*)", template))
    assert used == {"provider", "endpoint", "error"}


def test_memory_unhealthy_renders_concrete_failure(monkeypatch):
    """End-to-end render with the no-config layer forced off — verifies
    the default template stringifies cleanly with realistic inputs."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: None,
    )
    out = render_user_message(
        "memory_unhealthy",
        provider="openviking",
        endpoint="http://127.0.0.1:1933",
        error="connection refused",
    )
    assert "openviking" in out
    assert "http://127.0.0.1:1933" in out
    assert "connection refused" in out


# ---------------------------------------------------------------------------
# MemoryProvider._notify_user behavior
# ---------------------------------------------------------------------------


class _StubProvider(MemoryProvider):
    """Minimal concrete provider for testing the base-class helper —
    every abstract method gets a trivial implementation."""

    @property
    def name(self) -> str:
        return "stub"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []


def test_notify_user_is_noop_when_callback_unset():
    """Default state (no gateway attached) must not raise — providers
    will call this unconditionally on every failure path."""
    p = _StubProvider()
    assert p.notify_user_callback is None
    # Must not raise.
    p._notify_user(endpoint="http://x", error="boom")


def test_notify_user_invokes_callback_with_name_endpoint_error():
    """Callback receives the provider's own name (so the gateway can
    template-substitute {provider}) plus endpoint + error verbatim."""
    p = _StubProvider()
    received: List[Tuple[str, str, str]] = []
    p.notify_user_callback = lambda name, endpoint, error: received.append(
        (name, endpoint, error)
    )

    p._notify_user(endpoint="http://x:1933", error="connection refused")

    assert received == [("stub", "http://x:1933", "connection refused")]


def test_notify_user_swallows_callback_exception():
    """Notification plumbing must NEVER take down the provider thread —
    if the gateway-installed callback raises (e.g. asyncio loop closed
    mid-shutdown), the provider's failure path keeps going."""
    p = _StubProvider()

    def _broken_cb(name, endpoint, error):
        raise RuntimeError("event loop closed")

    p.notify_user_callback = _broken_cb
    # Must not propagate.
    p._notify_user(endpoint="http://x", error="boom")


# ---------------------------------------------------------------------------
# MemoryManager.set_notify_user_callback fan-out
# ---------------------------------------------------------------------------


class _NamedStub(_StubProvider):
    """Variant that lets the test give each instance a unique name so the
    fan-out can be observed across multiple registered providers."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name


def test_set_notify_user_callback_fans_out_to_all_providers():
    mm = MemoryManager()
    p1 = _NamedStub("builtin")  # name=='builtin' bypasses the "one external" guard
    p2 = _NamedStub("ext")
    mm.add_provider(p1)
    mm.add_provider(p2)
    cb = lambda *a: None  # noqa: E731

    mm.set_notify_user_callback(cb)

    assert p1.notify_user_callback is cb
    assert p2.notify_user_callback is cb


def test_set_notify_user_callback_none_clears_callback():
    """Tests / shutdown / non-gateway agent re-use must be able to drop
    the callback so a stale closure can't fire on a dead loop."""
    mm = MemoryManager()
    p = _NamedStub("builtin")
    mm.add_provider(p)
    p.notify_user_callback = lambda *a: None

    mm.set_notify_user_callback(None)

    assert p.notify_user_callback is None


# ---------------------------------------------------------------------------
# OpenViking plugin — runtime failure paths invoke _notify_user
# ---------------------------------------------------------------------------


def _wired_provider(endpoint: str = "http://127.0.0.1:1933"):
    """Build an OpenViking provider with a recording callback installed
    and the bookkeeping that real ``initialize`` would have done.  Avoids
    starting a real server / httpx client."""
    p = OpenVikingMemoryProvider()
    p._endpoint = endpoint
    p._api_key = ""
    p._account = "default"
    p._user = "default"
    p._agent = "hermes"
    p._session_id = "sess-1"
    p._turn_count = 0
    p._client = MagicMock()
    received: List[Tuple[str, str, str]] = []
    p.notify_user_callback = lambda name, ep, err: received.append((name, ep, err))
    return p, received


def test_openviking_sync_turn_failure_notifies_user(monkeypatch):
    """sync_turn runs the API write inside a background thread — failure
    used to be a DEBUG log (silent to the user).  Must now route to the
    notify callback so the gateway can surface a system message."""
    p, received = _wired_provider()

    def _boom(*args, **kwargs):
        raise RuntimeError("http 500: backend down")

    # The thread spawns its OWN _VikingClient, not self._client, so patch
    # the class.  The post call inside _sync raises immediately.
    fake_client_cls = MagicMock()
    fake_client_cls.return_value.post.side_effect = _boom
    monkeypatch.setattr(
        "plugins.memory.openviking._VikingClient", fake_client_cls,
    )

    p.sync_turn("user msg", "assistant msg", session_id="sess-1")

    # Deterministic wait: join the worker thread the provider spawned.
    if p._sync_thread is not None:
        p._sync_thread.join(timeout=5.0)
        assert not p._sync_thread.is_alive(), "sync_turn worker did not finish"

    assert len(received) == 1
    name, endpoint, error = received[0]
    assert name == "openviking"
    assert endpoint == "http://127.0.0.1:1933"
    assert "sync_turn" in error
    assert "http 500: backend down" in error


def test_openviking_handle_tool_call_failure_notifies_user():
    """MCP tool failures used to bubble up as a tool_error string only —
    the user got no signal that the memory subsystem was broken.  Must
    now also fire the notify callback before returning the tool_error."""
    p, received = _wired_provider()
    p._client.post.side_effect = RuntimeError("read timed out")

    out = p.handle_tool_call("viking_search", {"query": "anything"})

    # Tool error is still returned (so the model gets a proper tool result),
    # but the user notification ALSO fires so they know something's wrong.
    assert "read timed out" in out
    assert len(received) == 1
    name, endpoint, error = received[0]
    assert name == "openviking"
    assert endpoint == "http://127.0.0.1:1933"
    assert "viking_search" in error
    assert "read timed out" in error


def test_openviking_on_session_end_does_not_commit():
    """OpenViking commits per turn in sync_turn; session end must not
    issue a second commit or resurrect the old end-of-session lifecycle."""
    p, received = _wired_provider()
    p._turn_count = 3
    p._client.post.side_effect = RuntimeError("commit refused: vector store full")

    p.on_session_end(messages=[])

    p._client.post.assert_not_called()
    assert received == []


def test_openviking_on_memory_write_failure_notifies_user(monkeypatch):
    """Background mirror thread: same pattern as sync_turn — used to
    swallow as DEBUG, must now surface."""
    p, received = _wired_provider()

    fake_client_cls = MagicMock()
    fake_client_cls.return_value.post.side_effect = RuntimeError("auth expired")
    monkeypatch.setattr(
        "plugins.memory.openviking._VikingClient", fake_client_cls,
    )

    # The on_memory_write thread isn't tracked on the provider; the only
    # way to deterministically wait is to walk current threads after the
    # call and join the named one.
    p.on_memory_write(action="add", target="memory", content="something")

    for t in threading.enumerate():
        if t.name == "openviking-memwrite":
            t.join(timeout=5.0)

    assert len(received) == 1
    name, endpoint, error = received[0]
    assert name == "openviking"
    assert "memory mirror" in error
    assert "auth expired" in error
