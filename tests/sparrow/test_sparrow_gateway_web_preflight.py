"""Sparrow customization: gateway startup pre-flight check for web tools.

Sparrow policy: web tools are MANDATORY.  Cron jobs and interactive
turns alike depend on ``web_search`` / ``web_extract``; without them
the model falls back to ``execute_code + curl`` which trips tirith
approval prompts every time, every cron run, every interactive turn.
The gateway refuses to start without a reachable backend — there is
intentionally no "opt out by removing web from toolsets" escape hatch.

The hermes default behavior is silent degradation: web tools fail
their ``check_fn`` at registry-load time, get filtered out of the
agent's tool list, and the model never knows.  We refuse to ship that
state to production traffic.

These tests pin the decision logic in
``gateway.run._check_web_preflight``.  The actual ``start_gateway``
caller turns a non-None return into ``return False`` after firing the
``gateway_preflight_failed`` sparrow event and printing to stderr —
covered by the call site itself, not these tests.
"""

from __future__ import annotations

from unittest.mock import patch

from gateway.run import _check_web_preflight, _WEB_BACKEND_ENV_VARS


def test_returns_none_when_backend_available():
    """Backend probe says OK — gateway proceeds."""
    with patch(
        "tools.web_tools.check_web_api_key",
        return_value=True,
    ):
        result = _check_web_preflight()
    assert result is None


def test_fails_when_backend_unavailable(monkeypatch):
    """The actual bug being prevented: no backend is reachable —
    must produce a structured failure record with diagnostic fields."""
    for name in _WEB_BACKEND_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with patch(
        "hermes_cli.config.load_config",
        return_value={"toolsets": ["hermes-cli"]},
    ), patch(
        "tools.web_tools.check_web_api_key",
        return_value=False,
    ):
        result = _check_web_preflight()

    assert result is not None
    # Actionable one-liner naming BOTH the env var and the
    # ``web.backend`` config key — without the explicit backend
    # setting, the runtime falls back through firecrawl→parallel→
    # tavily→exa in fixed order and silently picks a backend the
    # operator may not have intended (web_tools.py:_get_backend).
    msg = result["user_message"]
    assert "TAVILY_API_KEY" in msg, (
        "Operator-facing message must name a concrete env var to set"
    )
    assert "web.backend" in msg, (
        "Operator-facing message must point at the config.yaml "
        "`web.backend` key — env-only setup relies on a fragile "
        "fallback order (web_tools.py:97-105)"
    )
    # Critical: do NOT advertise an "or remove `web` from toolsets"
    # opt-out.  Sparrow's policy is that web tools are mandatory; the
    # operator's only resolution is to configure a backend.
    msg_lower = msg.lower()
    assert "remove" not in msg_lower
    assert "toolset" not in msg_lower
    # Pin one-line brevity — refuse to drift back into a paragraph.
    assert len(msg) <= 110, (
        f"user_message must be ≤110 chars; got {len(msg)}: {msg!r}"
    )
    # Diagnostic fields landing in the sparrow event are populated.
    assert result["enabled_toolsets"] == "hermes-cli"
    assert result["reason"] == "check_web_api_key returned false"
    # All four backend env vars are reported missing in the event.
    missing = set(result["missing_env"].split(","))
    assert missing == set(_WEB_BACKEND_ENV_VARS)


def test_fails_regardless_of_toolset_config(monkeypatch):
    """Even when no toolset references web, a missing backend still
    blocks startup.  The check is unconditional — there's no opt-out
    via toolset config."""
    for name in _WEB_BACKEND_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    # Toolset has nothing web-related — this should NOT save startup.
    with patch(
        "hermes_cli.config.load_config",
        return_value={"toolsets": ["hermes-cron"]},
    ), patch(
        "tools.web_tools.check_web_api_key",
        return_value=False,
    ):
        result = _check_web_preflight()

    assert result is not None, (
        "Sparrow policy: web tools are mandatory.  Removing web from "
        "the toolset must NOT bypass the preflight."
    )
    assert result["reason"] == "check_web_api_key returned false"


def test_partial_env_set_still_fails_when_check_fn_returns_false(monkeypatch):
    """If env vars are set but ``check_web_api_key`` says no backend is
    reachable, fail.  The env-var report only drives the diagnostic
    field; the gating decision is the check_fn's call.

    Pinning this prevents a future regression where someone "smarts up"
    the preflight to skip the check when any env var is set, and ships
    over a wrong-backend / typo-key state."""
    for name in _WEB_BACKEND_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-set-but-invalid")

    with patch(
        "hermes_cli.config.load_config",
        return_value={"toolsets": ["hermes-cli"]},
    ), patch(
        "tools.web_tools.check_web_api_key",
        return_value=False,
    ):
        result = _check_web_preflight()

    assert result is not None
    # TAVILY_API_KEY is set, so it's NOT in missing_env; the others are.
    missing = set(result["missing_env"].split(","))
    assert "TAVILY_API_KEY" not in missing
    assert {"EXA_API_KEY", "PARALLEL_API_KEY", "FIRECRAWL_API_KEY"} <= missing


def test_skips_check_when_check_fn_raises():
    """If ``check_web_api_key`` itself raises, don't block startup.
    The preflight surfaces real backend gaps, not its own probe-side
    failures."""
    with patch(
        "tools.web_tools.check_web_api_key",
        side_effect=RuntimeError("backend probe network error"),
    ):
        result = _check_web_preflight()
    assert result is None


def test_failure_record_when_config_unavailable(monkeypatch):
    """A bad / missing config must NOT bypass the preflight.  Only the
    diagnostic ``enabled_toolsets`` field is best-effort — when config
    can't be loaded it's empty, but the failure record still fires
    because backend is the gate."""
    for name in _WEB_BACKEND_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    with patch(
        "tools.web_tools.check_web_api_key",
        return_value=False,
    ), patch(
        "hermes_cli.config.load_config",
        side_effect=RuntimeError("config corrupt"),
    ):
        result = _check_web_preflight()
    assert result is not None
    assert result["enabled_toolsets"] == ""
    assert result["reason"] == "check_web_api_key returned false"
