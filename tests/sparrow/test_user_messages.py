"""Tests for the user-facing message renderer used by the gateway for
system events (approval prompts, background-review notifications).

The renderer has a 3-layer resolution: config.yaml override → caller
fallback → built-in default.  These tests pin each layer so future
changes can't silently break the contract operators rely on.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.onebot_napcat.adapter import OneBotNapCatAdapter
from gateway.user_messages import (
    MESSAGE_KINDS,
    known_kinds,
    render_user_message,
    translate_for_kind,
)


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_every_registered_kind_has_a_default_template():
    """``MESSAGE_KINDS`` is the contract: any kind listed there must
    have a built-in default so callers always get a renderable string
    even with no config override and no adapter fallback."""
    from gateway.user_messages import _DEFAULT_TEMPLATES
    missing = [k for k in MESSAGE_KINDS if k not in _DEFAULT_TEMPLATES]
    assert not missing, f"kinds without defaults: {missing}"


def test_every_default_template_uses_only_declared_variables():
    """Default templates may only reference variable names listed in
    ``MESSAGE_KINDS[kind]``.  Otherwise a perfectly valid call would
    blow up with KeyError because the call site followed the contract
    but the template referenced an undeclared var."""
    import re
    from gateway.user_messages import _DEFAULT_TEMPLATES

    for kind, declared_vars in MESSAGE_KINDS.items():
        template = _DEFAULT_TEMPLATES[kind]
        # str.format placeholders look like {name} or {name:format} or {name!conv}
        used = set(re.findall(r"\{([a-zA-Z_]\w*)", template))
        undeclared = used - set(declared_vars)
        assert not undeclared, (
            f"default template for {kind!r} references undeclared "
            f"vars {sorted(undeclared)}; declared vars are {declared_vars}"
        )


def test_known_kinds_returns_registered_keys():
    """Public helper used by callers / config validation must reflect
    ``MESSAGE_KINDS`` exactly."""
    assert set(known_kinds()) == set(MESSAGE_KINDS.keys())


# ---------------------------------------------------------------------------
# render_user_message: resolution order
# ---------------------------------------------------------------------------


def _no_config(monkeypatch):
    """Force config layer to be absent so default + fallback paths can
    be tested without contamination from the operator's actual
    ~/.hermes/config.yaml."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: None,
    )


def test_render_uses_default_when_no_config_no_fallback(monkeypatch):
    _no_config(monkeypatch)
    out = render_user_message("background_review", summary="Memory updated")
    assert out == "💾 Memory updated"


def test_render_uses_caller_fallback_when_no_config(monkeypatch):
    _no_config(monkeypatch)
    out = render_user_message(
        "background_review",
        summary="Memory updated",
        fallback="adapter says: {summary}",
    )
    assert out == "adapter says: Memory updated"


def test_config_override_beats_fallback_and_default(monkeypatch):
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: "config wins: {summary}" if kind == "background_review" else None,
    )
    out = render_user_message(
        "background_review",
        summary="Memory updated",
        fallback="adapter says: {summary}",
    )
    assert out == "config wins: Memory updated"


def test_unknown_kind_raises_keyerror(monkeypatch):
    """Defensive: typos in the kind name should fail loudly, not return
    an empty string or silently fall through to a different kind."""
    _no_config(monkeypatch)
    with pytest.raises(KeyError, match="unknown user-message kind"):
        render_user_message("approval_requst", command="x", description="x")  # noqa: typo


def test_missing_variable_raises_keyerror(monkeypatch):
    """Caller forgot to pass a required variable — the error message
    must name BOTH the missing var and the kind so the bug is
    debuggable from a single log line."""
    _no_config(monkeypatch)
    with pytest.raises(KeyError) as exc:
        render_user_message("approval_request", command="ls")
        # missing: description
    assert "description" in str(exc.value)
    assert "approval_request" in str(exc.value)


# ---------------------------------------------------------------------------
# Approval-request template content
# ---------------------------------------------------------------------------


def test_default_approval_request_explains_all_three_approval_modes(monkeypatch):
    """The user-facing text must document /approve, /approve session,
    /approve always, and /deny — those are the only ways the user can
    respond to an approval prompt and missing any one strands them.
    Lock the contract so a future prompt edit can't silently drop one."""
    _no_config(monkeypatch)
    out = render_user_message(
        "approval_request",
        command="rm -rf /tmp/foo",
        description="recursive delete",
    )
    flat = " ".join(out.split())
    for word in ("/approve", "session", "always", "/deny"):
        assert word in flat, f"missing {word!r} in default approval template"
    # Must show the actual command so the user knows what they're approving
    assert "rm -rf /tmp/foo" in out
    assert "recursive delete" in out


# ---------------------------------------------------------------------------
# Adapter integration
# ---------------------------------------------------------------------------


def test_napcat_adapter_uses_generic_english_when_config_unset(monkeypatch):
    """NapCat ships with NO persona override at the adapter level —
    persona text lives in operator config.yaml, never in adapter code.
    With config absent, NapCat must produce the same generic English
    text as any other platform (no surprise persona baked in)."""
    _no_config(monkeypatch)
    a = OneBotNapCatAdapter(PlatformConfig(enabled=True))
    out = a.render_user_message(
        "approval_request", command="ls", description="list files",
    )
    assert "Reply `/approve`" in out
    # Persona text MUST NOT appear unless config.yaml put it there.
    assert "小麻雀" not in out
    assert "主人" not in out


def test_napcat_adapter_uses_config_chinese_template_when_set(monkeypatch):
    """When operator puts the Chinese persona template in config.yaml,
    NapCat picks it up — without any adapter-level code change."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: (
            "小麻雀需要主人确认后再运行这条命令：\n"
            "```\n{command}\n```\n"
            "安全检查：{description}\n\n"
            "回复 `/approve` 只放行这一次，`/approve session` 本次会话信任这个模式，"
            "`/approve always` 永久信任这个模式；不运行就回复 `/deny`。\n"
            "只有主人账号可以 approve。"
        ) if kind == "approval_request" else None,
    )
    a = OneBotNapCatAdapter(PlatformConfig(enabled=True))
    out = a.render_user_message(
        "approval_request", command="rm -rf /", description="dangerous",
    )
    assert "小麻雀" in out
    assert "主人" in out
    assert "rm -rf /" in out
    assert "dangerous" in out
    # All four user actions still documented in the operator's template
    for word in ("/approve", "session", "always", "/deny"):
        assert word in out


def test_napcat_background_review_uses_default_emoji(monkeypatch):
    """No adapter-level customization for background_review either —
    the `💾 {summary}` shape comes straight from the generic default."""
    _no_config(monkeypatch)
    a = OneBotNapCatAdapter(PlatformConfig(enabled=True))
    out = a.render_user_message("background_review", summary="Memory updated")
    assert out == "💾 Memory updated"


def test_config_template_for_background_review_can_be_customized(monkeypatch):
    """Operator changes `messaging.background_review.template` — both
    NapCat and any other platform pick it up, no code edit needed."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: "[bg-review] {summary}" if kind == "background_review" else None,
    )
    a = OneBotNapCatAdapter(PlatformConfig(enabled=True))
    out = a.render_user_message("background_review", summary="Memory updated")
    assert out == "[bg-review] Memory updated"


# ---------------------------------------------------------------------------
# Config loader resilience
# ---------------------------------------------------------------------------


def test_config_loader_returns_none_on_missing_section(monkeypatch):
    """Operator hasn't added a `messaging:` section yet — loader must
    return None so adapter/default fallback chain runs."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"some_other_section": {}},
    )
    from gateway.user_messages import _load_config_template
    assert _load_config_template("approval_request") is None


def test_config_loader_returns_none_on_malformed_entry(monkeypatch):
    """`messaging.approval_request` exists but isn't a dict (e.g.
    operator wrote a bare string by mistake) — must NOT raise on
    every render call; treat as no override."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"messaging": {"approval_request": "wrong shape"}},
    )
    from gateway.user_messages import _load_config_template
    assert _load_config_template("approval_request") is None


def test_config_loader_returns_none_when_template_field_missing(monkeypatch):
    """Operator wrote `messaging: {approval_request: {}}` (empty entry
    or wrong key) — must NOT crash; treat as no override."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"messaging": {"approval_request": {"note": "TODO"}}},
    )
    from gateway.user_messages import _load_config_template
    assert _load_config_template("approval_request") is None


def test_config_loader_swallows_load_config_exceptions(monkeypatch):
    """Broken YAML / IO error in config must NOT prevent message
    delivery.  The renderer has to keep working with defaults — a bad
    config file should never silence the gateway."""
    def _boom():
        raise RuntimeError("YAML parse error at line 4")
    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    from gateway.user_messages import _load_config_template
    assert _load_config_template("approval_request") is None


# ---------------------------------------------------------------------------
# Variable-translation layer (config-driven label localization)
#
# The agent synthesizes hard-coded English labels like "Memory updated"
# inside run_agent.py.  Operators can't easily patch upstream code, so
# this layer translates per-variable values at render time via the
# `messaging.<kind>.translations` config dict.  Tests pin the
# semantics: matched values get swapped, unmatched pass through, and
# malformed config never crashes delivery.
# ---------------------------------------------------------------------------


def test_summary_variable_gets_translated_when_match_in_config(monkeypatch):
    """The headline use case: agent emits 'Memory updated' as the
    background-review summary; operator's translations dict swaps it
    to localized text BEFORE the template renders."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: "💾 {summary}" if kind == "background_review" else None,
    )
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translations",
        lambda kind: {"Memory updated": "小麻雀的记忆已经更新"}
        if kind == "background_review" else None,
    )
    out = render_user_message("background_review", summary="Memory updated")
    assert out == "💾 小麻雀的记忆已经更新"


def test_unmatched_summary_passes_through_translation_layer(monkeypatch):
    """An agent-synthesized message that's NOT in the operator's
    translations dict (e.g. 'Skill created at /tmp/foo' — exact path
    isn't worth a static dict entry) must pass through unchanged so
    the user still sees something meaningful."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: "💾 {summary}" if kind == "background_review" else None,
    )
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translations",
        lambda kind: {"Memory updated": "中文版"}
        if kind == "background_review" else None,
    )
    out = render_user_message(
        "background_review", summary="Skill created at /tmp/foo",
    )
    assert out == "💾 Skill created at /tmp/foo"


def test_translation_only_swaps_string_values_not_other_types(monkeypatch):
    """Defensive: a future kind may pass non-string variables (ints,
    paths, etc.).  The translation lookup must skip those — only
    strings can match dict keys, and trying to lookup a Path object
    in a string-keyed dict would silently miss anyway, but be explicit."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: "{summary}" if kind == "background_review" else None,
    )
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translations",
        lambda kind: {"42": "should not match because it's a string lookup"}
        if kind == "background_review" else None,
    )
    # An int 42 is NOT a string, so it doesn't enter the lookup path —
    # render still works and produces "42".
    out = render_user_message("background_review", summary=42)
    assert out == "42"


def test_translation_layer_is_noop_when_no_config(monkeypatch):
    """When operator hasn't configured translations at all, summaries
    pass through verbatim — no surprise label rewriting from a stale
    test fixture or env-leakage."""
    _no_config(monkeypatch)
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translations",
        lambda kind: None,
    )
    out = render_user_message("background_review", summary="Memory updated")
    assert out == "💾 Memory updated"


def test_translations_loader_returns_none_on_missing_field(monkeypatch):
    """Operator added `messaging.background_review:` block but no
    `translations:` key — loader must return None so render falls
    through to the no-translation path."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"messaging": {"background_review": {"template": "X {summary}"}}},
    )
    from gateway.user_messages import _load_config_translations
    assert _load_config_translations("background_review") is None


def test_translations_loader_filters_malformed_entries(monkeypatch):
    """A malformed translations entry (list value, dict value) must NOT
    crash render — coerce to a clean str→str dict and silently drop
    bad rows.  Better one missing translation than zero deliveries."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "messaging": {
                "background_review": {
                    "translations": {
                        "Memory updated": "小麻雀的记忆已经更新",  # good
                        "Bad value": ["list", "value"],            # ignored
                        42: "non-string key",                       # ignored
                    },
                },
            },
        },
    )
    from gateway.user_messages import _load_config_translations
    out = _load_config_translations("background_review")
    assert out == {"Memory updated": "小麻雀的记忆已经更新"}


def test_translations_loader_swallows_load_config_exceptions(monkeypatch):
    """Same resilience as the template loader — broken config never
    silences delivery."""
    def _boom():
        raise RuntimeError("YAML parse error")
    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    from gateway.user_messages import _load_config_translations
    assert _load_config_translations("background_review") is None


# ---------------------------------------------------------------------------
# Regex-pattern translation layer (for dynamic labels like "Skill 'X' created")
# ---------------------------------------------------------------------------


def test_regex_pattern_translation_substitutes_named_group(monkeypatch):
    """The headline use case for patterns: agent emits dynamic labels
    like ``Skill 'foo' created.`` where ``foo`` is the skill name.
    Operator pattern with a named group lets the translation reuse
    that name in the rendered output."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: "💾 {summary}" if kind == "background_review" else None,
    )
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translations",
        lambda kind: None,
    )
    import re as _re
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translation_patterns",
        lambda kind: [
            (_re.compile(r"^Skill '(?P<name>.+)' created\.$"),
             "新本事「{name}」get！"),
        ] if kind == "background_review" else [],
    )
    out = render_user_message(
        "background_review", summary="Skill 'qq-helper' created.",
    )
    assert out == "💾 新本事「qq-helper」get！"


def test_regex_pattern_passthrough_when_no_match(monkeypatch):
    """A summary that doesn't match any pattern AND isn't in the
    exact-match dict must pass through verbatim — the operator
    didn't translate it, so the user sees raw English."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: "💾 {summary}" if kind == "background_review" else None,
    )
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translations",
        lambda kind: None,
    )
    import re as _re
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translation_patterns",
        lambda kind: [
            (_re.compile(r"^Skill '(?P<name>.+)' created\.$"), "x {name}"),
        ] if kind == "background_review" else [],
    )
    out = render_user_message("background_review", summary="Random other label")
    assert out == "💾 Random other label"


def test_exact_match_takes_precedence_over_regex(monkeypatch):
    """Two-layer resolution: exact-match dict beats regex patterns when
    both could match (operator can override a pattern for a specific
    value).  Tested by setting both up so that a match-anything regex
    would normally fire — the dict entry must win."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: "{summary}" if kind == "background_review" else None,
    )
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translations",
        lambda kind: {"Memory updated": "EXACT WIN"}
        if kind == "background_review" else None,
    )
    import re as _re
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translation_patterns",
        lambda kind: [
            (_re.compile(r"^.*$"), "REGEX WOULD HAVE FIRED"),
        ] if kind == "background_review" else [],
    )
    out = render_user_message("background_review", summary="Memory updated")
    assert out == "EXACT WIN"


def test_first_matching_pattern_wins(monkeypatch):
    """Order matters in the patterns list — first hit decides.  Lets
    operators put more specific patterns above broader ones."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: "{summary}" if kind == "background_review" else None,
    )
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translations",
        lambda kind: None,
    )
    import re as _re
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translation_patterns",
        lambda kind: [
            (_re.compile(r"^Skill 'foo' created\.$"), "SPECIFIC"),
            (_re.compile(r"^Skill '(?P<name>.+)' created\.$"), "GENERAL {name}"),
        ] if kind == "background_review" else [],
    )
    assert render_user_message(
        "background_review", summary="Skill 'foo' created.",
    ) == "SPECIFIC"
    assert render_user_message(
        "background_review", summary="Skill 'bar' created.",
    ) == "GENERAL bar"


def test_pattern_template_with_undefined_group_passes_value_through(monkeypatch):
    """If operator's template references a group name the regex doesn't
    capture (typo), don't crash delivery — fall back to raw value
    with a warning."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: "{summary}" if kind == "background_review" else None,
    )
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translations",
        lambda kind: None,
    )
    import re as _re
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translation_patterns",
        lambda kind: [
            (_re.compile(r"^Skill '(?P<actual>.+)' created\.$"),
             "Bad ref {wrong_name}"),
        ] if kind == "background_review" else [],
    )
    out = render_user_message(
        "background_review", summary="Skill 'foo' created.",
    )
    assert out == "Skill 'foo' created."  # raw passthrough


# ---------------------------------------------------------------------------
# translation_patterns loader resilience
# ---------------------------------------------------------------------------


def test_pattern_loader_returns_empty_when_field_missing(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"messaging": {"background_review": {"template": "X"}}},
    )
    from gateway.user_messages import _load_config_translation_patterns
    assert _load_config_translation_patterns("background_review") == []


def test_pattern_loader_filters_malformed_items(monkeypatch):
    """Bad pattern entries (non-dict, missing keys, uncompilable
    regex) are dropped silently with a warning so one bad entry
    doesn't break delivery for the rest of the patterns."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "messaging": {
                "background_review": {
                    "translation_patterns": [
                        "not a dict",                                          # ignored
                        {"pattern": "^good\\.$", "template": "ok"},            # kept
                        {"pattern": "[invalid", "template": "x"},              # uncompilable, ignored
                        {"template": "missing pattern key"},                    # ignored
                        {"pattern": "no-template", "template": None},           # template not str, ignored
                    ],
                },
            },
        },
    )
    from gateway.user_messages import _load_config_translation_patterns
    out = _load_config_translation_patterns("background_review")
    assert len(out) == 1
    assert out[0][0].pattern == r"^good\.$"
    assert out[0][1] == "ok"


def test_pattern_loader_swallows_load_config_exceptions(monkeypatch):
    def _boom():
        raise RuntimeError("YAML parse error")
    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    from gateway.user_messages import _load_config_translation_patterns
    assert _load_config_translation_patterns("background_review") == []


# ---------------------------------------------------------------------------
# translate_for_kind: per-segment translation for joined multi-action
# summaries (the original user-reported bug — "User profile updated · Memory
# updated" passed through untranslated because dict lookup is whole-string).
# ---------------------------------------------------------------------------


def test_translate_for_kind_applies_dict_match_for_single_segment(monkeypatch):
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translations",
        lambda kind: {"Memory updated": "记忆已更新"} if kind == "background_review" else {},
    )
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translation_patterns",
        lambda kind: [],
    )
    assert translate_for_kind("background_review", "Memory updated") == "记忆已更新"


def test_translate_for_kind_lets_caller_pre_translate_segments_for_join(monkeypatch):
    """Reproduction of the bug.  When ``_summarize_background_review_actions``
    returns multiple actions, run_agent.py joins them with ``" · "`` BEFORE
    rendering.  Without per-segment translation the joined string ``"User
    profile updated · Memory updated"`` matches no dict key and ships in
    English regardless of how complete the operator's translation map is.

    The fix moves the translation step to per-action: each segment hits the
    dict on its own and the caller joins translated segments.  This test
    pins that contract end-to-end."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translations",
        lambda kind: {
            "User profile updated": "主人画像已更新",
            "Memory updated": "记忆已更新",
        } if kind == "background_review" else {},
    )
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translation_patterns",
        lambda kind: [],
    )

    actions = ["User profile updated", "Memory updated"]
    translated = [translate_for_kind("background_review", a) for a in actions]
    summary = " · ".join(dict.fromkeys(translated))
    assert summary == "主人画像已更新 · 记忆已更新"

    # Sanity: rendering the joined translated summary as a background_review
    # should NOT double-translate or otherwise mangle it; the joined form
    # has no dict entry so it falls through, then the template wraps it.
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: None,
    )
    assert render_user_message("background_review", summary=summary) == f"💾 {summary}"


def test_translate_for_kind_unknown_kind_is_passthrough():
    """Defensive: callers in non-renderer code paths (run_agent.py) must
    not break when a hypothetical future kind hasn't been registered yet
    or someone passes a typo."""
    assert translate_for_kind("not_a_real_kind", "anything") == "anything"


def test_translate_for_kind_no_translations_configured_is_passthrough(monkeypatch):
    """When the operator hasn't set ``messaging.<kind>.translations`` /
    ``translation_patterns``, the helper returns the value unchanged
    rather than raising — this is the default-config experience."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translations",
        lambda kind: {},
    )
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translation_patterns",
        lambda kind: [],
    )
    assert translate_for_kind("background_review", "Memory updated") == "Memory updated"


def test_translate_for_kind_falls_through_to_pattern_when_no_dict_match(monkeypatch):
    """Pattern fallback works the same as inside render_user_message — the
    helper is just exposing the existing ``_translate_value`` mechanism for
    pre-join use."""
    import re
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translations",
        lambda kind: {},
    )
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translation_patterns",
        lambda kind: [(re.compile(r"^Skill '(?P<name>.+)' created\.$"), "新本事「{name}」")],
    )
    assert (
        translate_for_kind("background_review", "Skill 'foo' created.")
        == "新本事「foo」"
    )


# ---------------------------------------------------------------------------
# Approval / denial outcome kinds — added so the /approve and /deny status
# replies can be localized via config.yaml instead of being hard-coded
# English strings in gateway/run.py.
# ---------------------------------------------------------------------------


def test_approval_resolved_default_scope_once(monkeypatch):
    _no_config(monkeypatch)
    out = render_user_message(
        "approval_resolved",
        count_label="Command", scope_suffix="", count=1,
    )
    assert out == "✅ Command approved. The agent is resuming..."


def test_approval_resolved_default_scope_always_plural(monkeypatch):
    """Plural label is computed by the caller (gateway/run.py picks
    ``Command`` vs ``Commands`` based on count) — the template just
    interpolates.  This is the contract that lets operators write a
    single Chinese template that handles both cases via the
    ``Command``/``Commands`` translation entries."""
    _no_config(monkeypatch)
    out = render_user_message(
        "approval_resolved",
        count_label="Commands",
        scope_suffix=" (pattern approved permanently)",
        count=3,
    )
    assert out == "✅ Commands approved (pattern approved permanently). The agent is resuming..."


def test_approval_resolved_chinese_template_translates_count_and_scope(monkeypatch):
    """Operator-config path: a Chinese template consumes the same
    ``{count_label}`` and ``{scope_suffix}`` placeholders, with each
    English value swapped to Chinese via per-kind translations BEFORE
    being interpolated.  This is exactly the data/config.yaml shipped
    setup — pin it so refactors can't silently break Chinese rendering."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: (
            "✅ {count_label}通过啦{scope_suffix}——小麻雀继续干活了~ (≧▽≦)♪"
            if kind == "approval_resolved" else None
        ),
    )
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translations",
        lambda kind: ({
            "Command": "这条命令",
            "Commands": "这几条命令",
            " (pattern approved permanently)": "（以后都信主人哦）",
            " (pattern approved for this session)": "（本次会话都信）",
        } if kind == "approval_resolved" else {}),
    )
    monkeypatch.setattr(
        "gateway.user_messages._load_config_translation_patterns",
        lambda kind: [],
    )

    once = render_user_message(
        "approval_resolved",
        count_label="Command", scope_suffix="", count=1,
    )
    assert once == "✅ 这条命令通过啦——小麻雀继续干活了~ (≧▽≦)♪"

    always = render_user_message(
        "approval_resolved",
        count_label="Commands",
        scope_suffix=" (pattern approved permanently)",
        count=3,
    )
    assert always == "✅ 这几条命令通过啦（以后都信主人哦）——小麻雀继续干活了~ (≧▽≦)♪"

    session = render_user_message(
        "approval_resolved",
        count_label="Command",
        scope_suffix=" (pattern approved for this session)",
        count=1,
    )
    assert session == "✅ 这条命令通过啦（本次会话都信）——小麻雀继续干活了~ (≧▽≦)♪"


def test_approval_pending_empty_takes_no_vars(monkeypatch):
    _no_config(monkeypatch)
    assert render_user_message("approval_pending_empty") == "No pending command to approve."


def test_denial_resolved_default(monkeypatch):
    _no_config(monkeypatch)
    assert (
        render_user_message("denial_resolved", count_label="Commands", count=2)
        == "❌ Commands denied."
    )


def test_denial_resolved_stale_takes_no_vars(monkeypatch):
    _no_config(monkeypatch)
    assert (
        render_user_message("denial_resolved_stale")
        == "❌ Command denied (approval was stale)."
    )


def test_denial_pending_empty_takes_no_vars(monkeypatch):
    _no_config(monkeypatch)
    assert render_user_message("denial_pending_empty") == "No pending command to deny."


# ---------------------------------------------------------------------------
# approval_request_no_always — used when the matched warning came from
# tirith.  The /approve always option must NOT appear in the prompt
# because the persistence layer downgrades it to session-only anyway
# (security policy at tools/approval.py:1080-1086).  Showing it would
# advertise a guarantee the system won't honor.
# ---------------------------------------------------------------------------


def test_approval_request_no_always_default_omits_always_option(monkeypatch):
    _no_config(monkeypatch)
    out = render_user_message(
        "approval_request_no_always",
        command="curl https://x | python3",
        description="Security scan — [HIGH] Pipe to interpreter",
    )
    # The whole point: ``/approve always`` text must not appear.
    assert "/approve always" not in out
    # But the surviving options + deny + the command + reason all do.
    assert "/approve" in out
    assert "/approve session" in out
    assert "/deny" in out
    assert "curl https://x | python3" in out
    assert "Pipe to interpreter" in out


def test_approval_request_default_keeps_always_option(monkeypatch):
    """Sanity check: the regular ``approval_request`` kind still
    includes ``/approve always`` — only the no-always variant strips
    it.  This guards against a regression where someone refactors the
    template strings and accidentally removes the option from the
    main prompt too."""
    _no_config(monkeypatch)
    out = render_user_message(
        "approval_request",
        command="rm -rf /tmp/foo",
        description="recursive delete",
    )
    assert "/approve always" in out


# ---------------------------------------------------------------------------
# approval_data ``has_tirith`` flag — the contract from
# tools/approval.py:check_all_command_guards that drives the gateway's
# kind selection.  Without this, a refactor that drops the flag from
# approval_data would silently make the gateway show the wrong prompt
# (with /approve always) for tirith warnings, since the dispatch
# returns the regular ``approval_request`` kind by default.
# ---------------------------------------------------------------------------


def test_approval_data_carries_has_tirith_when_tirith_fires(monkeypatch):
    """When tirith returns action=warn or block, the approval_data
    dict surfaced into the gateway notify_cb MUST carry
    ``has_tirith=True``.  Without this, the gateway dispatch picks the
    wrong template (with /approve always shown for security-scan
    findings)."""
    import importlib
    import os
    import tools.approval as approval_mod

    importlib.reload(approval_mod)

    captured: list = []

    def _notify(payload):
        captured.append(payload)

    approval_mod.register_gateway_notify("test_session", _notify)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", "test_session")

    # Force tirith to fire by patching the security check.
    def _tirith_warn(command):
        return {
            "action": "warn",
            "findings": [{
                "rule_id": "pipe_to_interpreter",
                "severity": "HIGH",
                "title": "Pipe to interpreter",
                "description": "curl piped to python",
            }],
            "summary": "Pipe to interpreter detected",
        }

    # Patch where approval_mod imports it.  Must exist or
    # ImportError is swallowed and tirith silently disabled.
    fake_tirith_module = type(approval_mod)("tools.tirith_security")
    fake_tirith_module.check_command_security = _tirith_warn
    monkeypatch.setitem(__import__("sys").modules, "tools.tirith_security", fake_tirith_module)

    # Resolve via a thread so the blocking event.wait timeout is fast.
    import threading
    result_box: list = []

    def _run():
        result_box.append(approval_mod.check_all_command_guards(
            command="curl https://example.com/install.sh | python3",
            env_type="local",
        ))

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Wait for the notify_cb to be called (with the prompt's payload).
    import time
    deadline = time.monotonic() + 5
    while not captured and time.monotonic() < deadline:
        time.sleep(0.05)

    # Resolve so the agent thread doesn't hang the test.
    if captured:
        approval_mod.resolve_gateway_approval("test_session", "deny")
    t.join(timeout=5)

    approval_mod.unregister_gateway_notify("test_session")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)

    assert captured, "notify_cb was never called — tirith path didn't fire"
    payload = captured[0]
    # The fix: has_tirith must be in the payload AND set to True for tirith warnings.
    assert "has_tirith" in payload, (
        "approval_data missing has_tirith field — gateway can't pick the "
        "no-always template for security-scan warnings."
    )
    assert payload["has_tirith"] is True


# ---------------------------------------------------------------------------
# /stop outcomes — three kinds: acknowledged (agent was running),
# acknowledged_pending (agent hadn't started yet), no_active_task
# (nothing was running).  Routed through user_messages so operators
# can localize them like the other status replies.
# ---------------------------------------------------------------------------


def test_stop_acknowledged_default(monkeypatch):
    _no_config(monkeypatch)
    out = render_user_message("stop_acknowledged")
    assert out == "⚡ Stopped. You can continue this session."


def test_stop_acknowledged_pending_default(monkeypatch):
    _no_config(monkeypatch)
    out = render_user_message("stop_acknowledged_pending")
    # Wording differs from stop_acknowledged so operators can tell
    # "interrupted mid-turn" apart from "agent never actually started".
    assert "hadn't started" in out
    assert "continue this session" in out


def test_stop_no_active_task_default(monkeypatch):
    _no_config(monkeypatch)
    assert render_user_message("stop_no_active_task") == "No active task to stop."


def test_stop_kinds_take_no_variables(monkeypatch):
    """The three stop kinds have no required variables (their
    MESSAGE_KINDS entry is ``()``).  Calling with extra kwargs must
    not raise — the renderer ignores unused kwargs.  Calling without
    any kwargs must succeed.  Pin both behaviors so a future signature
    change can't silently break the call sites in gateway/run.py."""
    _no_config(monkeypatch)
    # No kwargs: clean.
    render_user_message("stop_acknowledged")
    render_user_message("stop_acknowledged_pending")
    render_user_message("stop_no_active_task")
    # Extra unused kwargs: tolerated.
    render_user_message("stop_acknowledged", count=3, ignored="x")


def test_stop_acknowledged_chinese_override(monkeypatch):
    """Operator sets ``messaging.stop_acknowledged.template`` in
    config.yaml → renderer uses the Chinese template instead of the
    English default.  Same path as the other approval/denial kinds."""
    monkeypatch.setattr(
        "gateway.user_messages._load_config_template",
        lambda kind: (
            "⚡ 停下来啦——！(◕‿◕)✨ 主人随时继续聊就好~"
            if kind == "stop_acknowledged" else None
        ),
    )
    out = render_user_message("stop_acknowledged")
    assert out == "⚡ 停下来啦——！(◕‿◕)✨ 主人随时继续聊就好~"


def test_approval_data_has_tirith_false_for_regular_dangerous_pattern(monkeypatch):
    """For dangerous patterns that DON'T involve tirith (e.g. plain
    ``rm -rf /tmp/x``), ``has_tirith`` must be False so the gateway
    shows the regular prompt with /approve always available."""
    import importlib
    import tools.approval as approval_mod

    importlib.reload(approval_mod)

    captured: list = []

    def _notify(payload):
        captured.append(payload)

    approval_mod.register_gateway_notify("test_session_2", _notify)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", "test_session_2")

    # Tirith stays "allow" by default (or no module): patch to be sure.
    def _tirith_allow(command):
        return {"action": "allow", "findings": [], "summary": ""}

    fake_tirith_module = type(approval_mod)("tools.tirith_security")
    fake_tirith_module.check_command_security = _tirith_allow
    monkeypatch.setitem(__import__("sys").modules, "tools.tirith_security", fake_tirith_module)

    import threading
    result_box: list = []

    def _run():
        result_box.append(approval_mod.check_all_command_guards(
            command="rm -rf /tmp/some_dir/",
            env_type="local",
        ))

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    import time
    deadline = time.monotonic() + 5
    while not captured and time.monotonic() < deadline:
        time.sleep(0.05)

    if captured:
        approval_mod.resolve_gateway_approval("test_session_2", "deny")
    t.join(timeout=5)

    approval_mod.unregister_gateway_notify("test_session_2")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)

    assert captured, "notify_cb was never called"
    payload = captured[0]
    assert payload.get("has_tirith") is False, (
        "Non-tirith dangerous patterns must report has_tirith=False so the "
        "gateway picks the regular approval_request template (with /approve always)."
    )
