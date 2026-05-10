"""Sparrow customization: background_review summary delivery routing.

The review summary the agent emits after a background memory/skill pass
takes one of two paths:

* **Gateway mode** (``background_review_callback`` set): the user-facing
  message is delivered by the callback (e.g. as a QQ message via the
  napcat platform).  The local audit trail must go through
  ``logger.info`` so it lands in ``~/.hermes/logs/agent.log`` —
  *not* through ``_safe_print`` to stdout, which would pollute the
  gateway's terminal output (otherwise only structured log lines).

* **CLI mode** (no callback): the prompt_toolkit TUI renderer wired up
  via ``_print_fn`` is the only path the user can see, so the summary
  MUST go through ``_safe_print`` to reach them.

Regression guard: before this fix, gateway operators saw bare kawaii
lines like ``  💾 诶嘿——小麻雀新学了个本事「foo」啦~`` mixed into
WARNING/INFO log output, with no audit trail in agent.log.
"""

from __future__ import annotations

import logging

import run_agent as run_agent_module
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    """Minimal AIAgent suitable for driving ``_spawn_background_review``."""
    agent = object.__new__(AIAgent)
    agent.model = "fake-model"
    agent.platform = "telegram"
    agent.provider = "openai"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = ""
    agent.session_id = "test-session"
    agent._parent_session_id = ""
    agent._credential_pool = None
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.background_review_callback = None
    agent.status_callback = None
    agent._safe_print = lambda *_args, **_kwargs: None
    return agent


class _ImmediateThread:
    """threading.Thread stand-in that runs target inline so the test can
    observe side effects synchronously."""

    def __init__(self, *, target, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


def _setup_review_with_actions(monkeypatch, *, bg_callback):
    """Wire up a fake review run that produces a single canned action."""
    canned_actions = ["Skill 'foo' created."]

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        AIAgent,
        "_summarize_background_review_actions",
        staticmethod(lambda review_msgs, prior: list(canned_actions)),
    )

    agent = _bare_agent()
    agent.background_review_callback = bg_callback
    return agent, canned_actions


def test_gateway_mode_routes_review_summary_to_logger_not_stdout(monkeypatch, caplog):
    """Gateway mode: the user-facing summary goes via the callback (QQ
    message); the audit trail goes via logger.info — NOT ``_safe_print``.

    Pinning this prevents a regression where a bare ``_safe_print`` would
    pollute the gateway's terminal output with kawaii lines mixed in with
    WARNING/INFO logs.
    """
    delivered = []

    def bg_cb(summary):
        delivered.append(summary)

    agent, canned = _setup_review_with_actions(monkeypatch, bg_callback=bg_cb)

    print_calls = []
    agent._safe_print = lambda *a, **kw: print_calls.append(a)

    with caplog.at_level(logging.INFO, logger="run_agent"):
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[{"role": "user", "content": "hi"}],
            review_memory=True,
        )

    # Callback delivered the summary (raw English, untranslated; the
    # gateway adapter handles per-platform translation downstream).
    assert delivered == ["Skill 'foo' created."], (
        "Gateway delivery callback was not invoked with the raw summary."
    )
    # Logger captured the audit trail.
    assert any(
        "background_review" in r.getMessage() and canned[0] in r.getMessage()
        for r in caplog.records
    ), "Expected logger.info('background_review: ...') in gateway mode."
    # _safe_print MUST NOT fire — that's the bug being prevented.
    assert print_calls == [], (
        f"_safe_print should not run in gateway mode; got: {print_calls}"
    )


def test_cli_mode_prints_review_summary_when_no_callback(monkeypatch, caplog):
    """CLI mode (no bg callback): the summary must still reach the user,
    so ``_safe_print`` fires — that's the only path to the TUI renderer
    wired up via ``_print_fn``."""
    agent, _ = _setup_review_with_actions(monkeypatch, bg_callback=None)

    print_calls = []
    agent._safe_print = lambda *a, **kw: print_calls.append(a)

    with caplog.at_level(logging.INFO, logger="run_agent"):
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[{"role": "user", "content": "hi"}],
            review_memory=True,
        )

    assert len(print_calls) == 1, (
        f"Expected exactly one _safe_print in CLI mode; got {print_calls}"
    )
    assert "💾" in print_calls[0][0], (
        "CLI _safe_print should still carry the 💾 prefix the user sees."
    )
