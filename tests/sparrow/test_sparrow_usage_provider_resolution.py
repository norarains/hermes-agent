"""Sparrow regression: ``/usage`` provider resolution must reflect the
*current* configured model/provider, not the historical ``billing_provider``
persisted to SessionDB.

Bug it locks down
-----------------
Before this fix, ``_handle_usage_command`` resolved the provider for
the account-balance fetch via:

  1. live agent's ``.provider``
  2. SessionDB ``billing_provider``  ← historical, last actual API call

When the user ran ``/model <new-provider>`` mid-session and then asked
``/usage`` *before* talking to the bot again, no API call had updated
the SessionDB row.  ``billing_provider`` still held the OLD provider —
the one that was billed for the previous turn.  ``/usage`` then queried
the wrong account-balance API and showed credentials/limits for a
provider the user is no longer using.

The fix inserts ``config.yaml model.provider + per-session override``
between the live-agent and SessionDB layers, so ``/usage`` reads the
same source-of-truth that ``/model`` reports as ``Current:`` —
exactly what the user *will* be billed by on their next turn.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub telegram package so importing gateway.platforms.base doesn't
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

from gateway.config import GatewayConfig, Platform, PlatformConfig  # noqa: E402
from gateway.platforms.base import MessageEvent  # noqa: E402
from gateway.session import SessionSource  # noqa: E402


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.ONEBOT_NAPCAT,
        user_id="891088473",
        chat_id="group_940134062",
        user_name="近环",
        chat_type="group",
    )


def _make_event() -> MessageEvent:
    return MessageEvent(text="/usage", source=_make_source(), message_id="m1")


def _make_runner_no_agent(persisted_billing_provider: Optional[str] = None):
    """Build a minimal GatewayRunner with no live agent in cache —
    forcing ``_handle_usage_command`` down its provider-resolution
    fallback path."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.ONEBOT_NAPCAT: PlatformConfig(enabled=True)},
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.ONEBOT_NAPCAT: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)

    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = MagicMock()
    runner._agent_cache_lock.__enter__ = lambda self_=None: None
    runner._agent_cache_lock.__exit__ = lambda *_: None
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._session_model_overrides = {}
    runner._is_user_authorized = lambda _source: True

    # Stub session_store so get_or_create_session returns a fake entry
    # and load_transcript returns an empty history (so /usage falls
    # into the "no agent at all + empty session" branch — focusing
    # the test on provider resolution, not message-count rendering).
    fake_entry = SimpleNamespace(session_id="20260428_fake_session")
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session = MagicMock(return_value=fake_entry)
    runner.session_store.load_transcript = MagicMock(return_value=[])

    # SessionDB returns the (historical) billing_provider.  Real
    # SessionDB also has billing_base_url etc. — only billing_provider
    # is relevant to the priority decision under test.
    if persisted_billing_provider is not None:
        fake_db = MagicMock()
        fake_db.get_session = MagicMock(return_value={
            "billing_provider": persisted_billing_provider,
            "billing_base_url": f"https://api.{persisted_billing_provider}.example/v1",
        })
        runner._session_db = fake_db
    else:
        runner._session_db = None

    return runner


# ---------------------------------------------------------------------------
# The actual bug repro
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_uses_current_config_provider_not_persisted_billing(monkeypatch):
    """The user reproduction: session was last billed against anthropic,
    user ran ``/model deepseek-v4-pro --provider deepseek`` to switch,
    then asked ``/usage`` before talking to the bot.  Expected:
    ``/usage`` queries the **deepseek** account-balance API, not
    anthropic's stale-credentials API."""
    runner = _make_runner_no_agent(persisted_billing_provider="anthropic")

    # config.yaml says deepseek is the current provider.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {
                "provider": "deepseek",
                "default": "deepseek-v4-pro",
                "base_url": "https://api.deepseek.com/v1",
            },
        },
    )

    captured: Dict[str, Any] = {}

    def _spy_fetch(provider, *, base_url=None, api_key=None):
        captured["provider"] = provider
        captured["base_url"] = base_url
        return None  # no snapshot — we only care which provider was queried

    monkeypatch.setattr("gateway.run.fetch_account_usage", _spy_fetch)

    await runner._handle_usage_command(_make_event())

    assert captured["provider"] == "deepseek", (
        f"Expected /usage to query the CURRENTLY configured provider "
        f"(deepseek), but it queried {captured['provider']!r} — likely "
        f"falling back to the historical SessionDB billing_provider, "
        f"which is the bug this test pins."
    )
    assert captured["base_url"] == "https://api.deepseek.com/v1"


@pytest.mark.asyncio
async def test_usage_session_override_beats_config(monkeypatch):
    """If the user ran ``/model ... --provider X`` only for *this
    session* (no ``--global``), the override is stored in
    ``self._session_model_overrides[session_key]`` and must take
    priority over the config.yaml default — same as ``/model`` does."""
    runner = _make_runner_no_agent(persisted_billing_provider=None)

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {"provider": "anthropic", "default": "claude-opus-4-7"},
        },
    )

    # Per-session override → openrouter.  /usage must follow.
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    runner._session_model_overrides[session_key] = {
        "provider": "openrouter",
        "model": "deepseek/deepseek-v4-pro",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "sk-or-v1-...",
    }

    captured: Dict[str, Any] = {}

    def _spy_fetch(provider, *, base_url=None, api_key=None):
        captured["provider"] = provider
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return None

    monkeypatch.setattr("gateway.run.fetch_account_usage", _spy_fetch)

    await runner._handle_usage_command(_make_event())

    assert captured["provider"] == "openrouter"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["api_key"] == "sk-or-v1-..."


@pytest.mark.asyncio
async def test_usage_falls_through_to_billing_provider_when_no_config(monkeypatch):
    """Defensive: if config.yaml load fails AND no per-session override
    AND no live agent, the SessionDB persisted billing_provider is the
    last-resort source.  This preserves the prior fallback behavior so
    rare config-load failures don't make ``/usage`` silently report
    nothing — better stale than empty."""
    runner = _make_runner_no_agent(persisted_billing_provider="anthropic")

    def _boom():
        raise RuntimeError("config.yaml unreadable")
    monkeypatch.setattr("hermes_cli.config.load_config", _boom)

    captured: Dict[str, Any] = {}

    def _spy_fetch(provider, *, base_url=None, api_key=None):
        captured["provider"] = provider
        captured["base_url"] = base_url
        return None

    monkeypatch.setattr("gateway.run.fetch_account_usage", _spy_fetch)

    await runner._handle_usage_command(_make_event())

    assert captured["provider"] == "anthropic"
    assert captured["base_url"] == "https://api.anthropic.example/v1"


@pytest.mark.asyncio
async def test_usage_with_no_provider_anywhere_skips_account_section(monkeypatch):
    """Symmetric with the existing behavior: if no provider can be
    resolved at all, the account-balance fetch must not run.  This
    used to be tested implicitly via the fallback chain; now that the
    chain has three steps instead of two, we re-pin it."""
    runner = _make_runner_no_agent(persisted_billing_provider=None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})

    fetch_calls: list = []
    monkeypatch.setattr(
        "gateway.run.fetch_account_usage",
        lambda *a, **kw: fetch_calls.append((a, kw)),
    )

    await runner._handle_usage_command(_make_event())

    assert fetch_calls == []
