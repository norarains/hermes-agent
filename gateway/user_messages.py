"""User-facing gateway message templates.

Centralizes the strings the gateway sends to users for system events
(approval prompts, background-review notifications, etc.) so adapters
and operators can customize without touching call-site code.

Resolution order for any (kind, vars) call:
  1. ``messaging.<kind>.template`` from ~/.hermes/config.yaml (operator override)
  2. ``fallback`` argument the caller provided (adapter-specific default)
  3. ``_DEFAULT_TEMPLATES[kind]`` (generic English baseline)

Templates use ``str.format`` with named placeholders.  Missing variables
raise ``KeyError`` early so config typos surface as a startup error
rather than silently shipping a half-rendered message.

DESIGN INVARIANT: be careful not to break this — every kind name listed
in ``MESSAGE_KINDS`` MUST have a default template, and every default
template MUST consume exactly the variables documented next to it
below.  Tests enforce both invariants.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kind → required-vars contract
# ---------------------------------------------------------------------------

# Each entry maps a stable string key (used by callers + config) to the
# tuple of placeholder names that template MUST reference.  Keep this
# small and stable; downstream code (test_user_messages, this module's
# render path, adapter overrides) all key off it.
MESSAGE_KINDS: Dict[str, tuple] = {
    "approval_request": ("command", "description"),
    # Same shape as ``approval_request`` but rendered when the matched
    # warning came from tirith.  The persistence layer downgrades
    # ``/approve always`` to session-only for tirith findings (security
    # policy at ``approval.py:1080-1086``), so the prompt MUST hide that
    # option — otherwise the UI advertises a guarantee the system won't
    # honor.  Operators can localize this kind separately from
    # ``approval_request`` because the wording differs (only 3 options
    # to mention, not 4).
    "approval_request_no_always": ("command", "description"),
    "background_review": ("summary",),
    # Surfaced when a memory provider's runtime call (sync_turn, MCP tool,
    # session commit, ...) fails.  Sent as an out-of-band gateway message
    # so the user knows the memory subsystem is degraded without crashing
    # the gateway or polluting chat history.
    "memory_unhealthy": ("provider", "endpoint", "error"),
    # /approve outcomes — sent back to the chat after the user resolves
    # a pending dangerous-command approval.  Routed through this layer
    # (instead of hard-coded strings at the call site) so operators can
    # localize without touching gateway source.
    "approval_resolved": ("count_label", "scope_suffix", "count"),
    "approval_pending_empty": (),
    # /deny outcomes — same rationale.
    "denial_resolved": ("count_label", "count"),
    "denial_resolved_stale": (),
    "denial_pending_empty": (),
    # /stop outcomes — sent back to the chat after the user runs /stop.
    # Three flavors so operators can localize each precisely:
    "stop_acknowledged": (),         # agent was running and was interrupted
    "stop_acknowledged_pending": (),  # agent hadn't actually started yet
    "stop_no_active_task": (),        # nothing was running
    # Gateway lifecycle notifications, sent to active chats when the
    # gateway is about to stop or restart.  Two kinds so operators can
    # localize each precisely — restart adds resume guidance, shutdown
    # does not.
    "gateway_shutdown": (),
    "gateway_restart": (),
}


# ---------------------------------------------------------------------------
# Generic English defaults
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATES: Dict[str, str] = {
    "approval_request": (
        "I need approval before running this command:\n"
        "```\n{command}\n```\n"
        "Reason: {description}\n\n"
        "Reply `/approve` to allow once, `/approve session` to trust this "
        "pattern for the rest of this session, or `/approve always` to "
        "trust it permanently.  Reply `/deny` to refuse."
    ),
    "approval_request_no_always": (
        "I need approval before running this command:\n"
        "```\n{command}\n```\n"
        "Reason: {description}\n\n"
        "Reply `/approve` to allow once or `/approve session` to trust "
        "this pattern for the rest of this session.  Reply `/deny` to "
        "refuse.  (`always` is unavailable for security-scan warnings.)"
    ),
    "background_review": "💾 {summary}",
    "memory_unhealthy": "⚠️ Memory provider {provider} unhealthy ({endpoint}): {error}",
    "approval_resolved": (
        "✅ {count_label} approved{scope_suffix}. The agent is resuming..."
    ),
    "approval_pending_empty": "No pending command to approve.",
    "denial_resolved": "❌ {count_label} denied.",
    "denial_resolved_stale": "❌ Command denied (approval was stale).",
    "denial_pending_empty": "No pending command to deny.",
    "stop_acknowledged": "⚡ Stopped. You can continue this session.",
    "stop_acknowledged_pending": (
        "⚡ Stopped. The agent hadn't started yet — you can continue this session."
    ),
    "stop_no_active_task": "No active task to stop.",
    "gateway_shutdown": "⚠️ Gateway shutting down — Your current task will be interrupted.",
    "gateway_restart": (
        "⚠️ Gateway restarting — Your current task will be interrupted. "
        "Send any message after restart and I'll try to resume where you left off."
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_user_message(
    kind: str,
    *,
    fallback: Optional[str] = None,
    **variables: Any,
) -> str:
    """Render the user-facing message for ``kind`` using ``variables``.

    See module docstring for the resolution order.  Raises ``KeyError``
    for unknown kinds or missing template variables, so configuration
    typos and call-site bugs surface loudly instead of shipping a
    half-rendered string to end users.

    Variable values are also passed through an optional per-kind
    translation dict from ``messaging.<kind>.translations`` in
    config.yaml — that's how operators localize hard-coded English
    labels the agent code synthesizes (e.g. ``"Memory updated"``)
    without having to touch upstream source.
    """
    if kind not in MESSAGE_KINDS:
        raise KeyError(f"unknown user-message kind: {kind!r}")

    translations = _load_config_translations(kind) or {}
    patterns = _load_config_translation_patterns(kind) or []
    if translations or patterns:
        variables = {
            name: _translate_value(value, translations, patterns)
            for name, value in variables.items()
        }

    template = (
        _load_config_template(kind)
        or fallback
        or _DEFAULT_TEMPLATES[kind]
    )

    try:
        return template.format(**variables)
    except KeyError as exc:
        raise KeyError(
            f"missing variable {exc} for user-message kind {kind!r}; "
            f"required vars are {MESSAGE_KINDS[kind]!r}"
        ) from exc


def _translate_value(
    value: Any,
    translations: Dict[str, str],
    patterns: List[Tuple["re.Pattern[str]", str]],
) -> Any:
    """Apply per-value translation: exact-match dict first, then regex
    patterns (first hit wins).  Non-string values pass through.

    Pattern templates use named groups: ``(?P<name>...)`` in the
    regex feeds ``{name}`` in the template.
    """
    if not isinstance(value, str):
        return value
    if value in translations:
        return translations[value]
    for regex, template in patterns:
        m = regex.fullmatch(value)
        if not m:
            continue
        try:
            return template.format(**m.groupdict())
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning(
                "user_messages: translation pattern %r matched but "
                "template %r failed (%s) — passing value through",
                regex.pattern, template, exc,
            )
            return value
    return value


def translate_for_kind(kind: str, value: str) -> str:
    """Translate a single string against ``messaging.<kind>``'s
    ``translations`` map (and ``translation_patterns``).

    Useful when a caller assembles a *multi-segment* user message by
    joining several pre-translated parts (e.g. background_review's
    ``"User profile updated · Memory updated"``).  Doing the join after
    per-segment translation gives every segment a chance to hit the
    dict; doing the lookup on the joined string only matches when the
    operator wrote a translation entry for the exact joined form,
    which is brittle and almost never what they want.

    Returns ``value`` unchanged when the kind is unknown, no
    translations are configured, or no pattern matches.
    """
    if kind not in MESSAGE_KINDS or not isinstance(value, str):
        return value
    translations = _load_config_translations(kind) or {}
    patterns = _load_config_translation_patterns(kind) or []
    if not translations and not patterns:
        return value
    return _translate_value(value, translations, patterns)


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


def _load_config_template(kind: str) -> Optional[str]:
    """Look up ``messaging.<kind>.template`` in config.yaml.

    Returns None when the operator hasn't customized this kind, so the
    caller falls through to the adapter fallback then the default.
    Failures (missing file, bad YAML) are logged at debug and treated
    as "no override" — a broken config must never break delivery.
    """
    try:
        from hermes_cli.config import load_config
    except ImportError:
        return None
    try:
        cfg = load_config()
    except Exception as exc:
        logger.debug("user_messages: load_config failed (%s)", exc)
        return None
    if not isinstance(cfg, dict):
        return None
    section = cfg.get("messaging")
    if not isinstance(section, dict):
        return None
    entry = section.get(kind)
    if not isinstance(entry, dict):
        return None
    template = entry.get("template")
    if isinstance(template, str) and template:
        return template
    return None


def _load_config_translations(kind: str) -> Optional[Dict[str, str]]:
    """Look up ``messaging.<kind>.translations`` in config.yaml.

    Returns a string→string mapping or None when no translations are
    configured / the entry is malformed.  Used to localize hard-coded
    English labels the agent synthesizes (``"Memory updated"`` etc.)
    without patching upstream code.
    """
    try:
        from hermes_cli.config import load_config
    except ImportError:
        return None
    try:
        cfg = load_config()
    except Exception as exc:
        logger.debug("user_messages: load_config failed (%s)", exc)
        return None
    if not isinstance(cfg, dict):
        return None
    section = cfg.get("messaging")
    if not isinstance(section, dict):
        return None
    entry = section.get(kind)
    if not isinstance(entry, dict):
        return None
    raw = entry.get("translations")
    if not isinstance(raw, dict):
        return None
    # Filter to string→string pairs so a malformed entry (e.g. a list
    # value) doesn't crash render at delivery time.
    return {
        str(k): str(v)
        for k, v in raw.items()
        if isinstance(k, str) and isinstance(v, (str, int, float))
    }


def _load_config_translation_patterns(
    kind: str,
) -> List[Tuple["re.Pattern[str]", str]]:
    """Load regex translation patterns for ``messaging.<kind>.translation_patterns``.

    Schema (list of dicts):
        translation_patterns:
          - pattern: "^Skill '(?P<name>.+)' created\\.$"
            template: "新本事「{name}」get！"

    Bad entries (non-dict, missing keys, uncompilable regex) are
    silently dropped with a warning so a typo in one pattern doesn't
    break delivery for the rest.
    """
    try:
        from hermes_cli.config import load_config
    except ImportError:
        return []
    try:
        cfg = load_config()
    except Exception as exc:
        logger.debug("user_messages: load_config failed (%s)", exc)
        return []
    if not isinstance(cfg, dict):
        return []
    section = cfg.get("messaging")
    if not isinstance(section, dict):
        return []
    entry = section.get(kind)
    if not isinstance(entry, dict):
        return []
    raw = entry.get("translation_patterns")
    if not isinstance(raw, list):
        return []

    out: List[Tuple["re.Pattern[str]", str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            logger.warning(
                "user_messages: translation_patterns[%d] for kind %s is "
                "not a dict — skipping", idx, kind,
            )
            continue
        pattern_str = item.get("pattern")
        template = item.get("template")
        if not (isinstance(pattern_str, str) and isinstance(template, str)):
            logger.warning(
                "user_messages: translation_patterns[%d] for kind %s "
                "missing pattern/template — skipping", idx, kind,
            )
            continue
        try:
            regex = re.compile(pattern_str)
        except re.error as exc:
            logger.warning(
                "user_messages: bad regex %r in translation_patterns[%d] "
                "for kind %s: %s — skipping",
                pattern_str, idx, kind, exc,
            )
            continue
        out.append((regex, template))
    return out


def known_kinds() -> Iterable[str]:
    """Return the registered message kinds (for tests / config validation)."""
    return MESSAGE_KINDS.keys()
