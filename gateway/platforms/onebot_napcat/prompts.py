"""Channel prompts for the OneBot/NapCat adapter."""

from __future__ import annotations

import os
from typing import Optional


def is_assistant_header_enabled() -> bool:
    """Return True iff NapCat should decorate persisted assistant
    entries with a ``<time> [msg=<id>] <QQ号> <name>:`` header (and the
    channel prompt should warn the LLM about it).

    Disabled by default.  Even with the warning in the prompt, the LLM
    in-context-learns from past decorated entries and starts mimicking
    the format in its own output (observed in production: bot
    fabricated 10-digit fake msg_ids, double-prefix bug, and triggered
    `Get Uid Error` / ENOENT by quoting prompt placeholder text as if
    it were a real directive).  Opt-in with
    ``NAPCAT_ASSISTANT_HEADER=true`` if you have a model robust enough
    to honor the anti-mimicry instruction.
    """
    return os.getenv("NAPCAT_ASSISTANT_HEADER", "").lower() in ("1", "true", "yes")


_ASSISTANT_HEADER_NOTE = (
    "- IMPORTANT: your OWN past assistant lines in this transcript also\n"
    "  carry this `<time> [msg=<id>] <QQ号> <name>:` prefix — the adapter\n"
    "  injects it post-send, it is NOT something you produced.  Do NOT\n"
    "  include this prefix in your replies; write the message body only\n"
    "  (the adapter will add the prefix automatically when persisting).\n"
)


# Anchor used by ``build_base_channel_prompt`` to splice
# ``_ASSISTANT_HEADER_NOTE`` in when the feature is enabled.  Keep the
# marker exactly equal to the line in BASE_CHANNEL_PROMPT below.
_TRANSCRIPT_FORMAT_ANCHOR = (
    "  <QQ号>`), not user-typed content; never echo or invent them.\n"
)


BASE_CHANNEL_PROMPT = (
    "QQ via NapCat.\n"
    "\n"
    "# Transcript format\n"
    "Each line you read is `<time> [msg=<id>] <QQ号> <name>: <content>`.\n"
    "- `[msg=<id>]` is the QQ message_id; quote any past message with\n"
    "  the `REPLY:<id>` directive below.\n"
    "- `<QQ号>` is the sender's numeric QQ id — use it directly in\n"
    "  `[CQ:at,qq=<号>]` and `POKE:<号>` directives without asking the\n"
    "  user for it.\n"
    "- `<name>` is the display name (group card / account nickname).\n"
    "  When the user has neither, only the bare QQ id appears.\n"
    "- `SYSTEM:` lines are gateway-injected events (e.g. `you poked\n"
    "  <QQ号>`), not user-typed content; never echo or invent them.\n"
    "\n"
    "# Output directives\n"
    "Standalone lines matching the formats below are consumed by the\n"
    "adapter: it performs the QQ action and strips the directive from\n"
    "the chat text the user sees. History keeps your raw output\n"
    "(directives included) so you can see your own past actions.\n"
    "\n"
    "  `[CQ:at,qq=<QQ号>]`       mention a user (inline, anywhere in text)\n"
    "  `MEDIA:<absolute path>`    send a file (e.g. `MEDIA:/tmp/chart.png`)\n"
    "  `POKE:<QQ号>`              poke a user\n"
    "  `REPLY:<message_id>`       quote-reply that message (any past one)\n"
    "  `REPLY:none`               skip the auto-quote in groups\n"
    "\n"
    "Default quote behavior: groups quote the triggering message (QQ\n"
    "etiquette / disambiguates whom you're addressing); DMs do not\n"
    "auto-quote. Use the `REPLY:` directives above to override.\n"
    "\n"
    "Never write history/event text like `you poked <QQ号>` yourself —\n"
    "the adapter injects the `SYSTEM:` line after your directive runs.\n"
    "\n"
    "# Style\n"
    "Be concise; no code blocks/tables."
)


# When this channel is NOT in passive group mode (default), the gateway
# only delivers messages directed at you (DMs or @-mentions in groups),
# so the response policy is trivial — always answer.  Passive mode
# delivers every group message, so it overrides this with a
# selective skip/respond policy below.
_NON_PASSIVE_POLICY = (
    "# Response policy\n"
    "Always respond — you only see messages directed at you (DMs or\n"
    "@-mentions in groups)."
)


_PASSIVE_GROUP_POLICY = (
    "# Group passive mode (this channel)\n"
    "You see every group message. Output `[SILENT]` alone to skip\n"
    "a turn (the adapter drops it without sending anything).\n"
    "\n"
    "Skip when:\n"
    "  - the topic is unrelated to you\n"
    "  - you've been told to shut up / be quiet / pretend not to see\n"
    "  - your reply would just add noise\n"
    "\n"
    "Respond when:\n"
    "  - you're @'d or replied to\n"
    "  - one of your known names / nicknames is mentioned\n"
    "  - you're being asked about\n"
    "  - you have unique value to add\n"
    "\n"
    "Judge each message on its own merits."
)


def build_base_channel_prompt() -> str:
    base = BASE_CHANNEL_PROMPT
    if is_assistant_header_enabled():
        # Splice the anti-mimicry note into the Transcript format
        # section right after the SYSTEM: bullet — same place a
        # human reader would expect a related clarification.  Fail
        # loudly if the anchor was edited away: silent fallthrough
        # would ship a prompt that promises decoration without warning
        # the LLM about it, re-opening the mimicry bugs the warning
        # was added to prevent.
        if _TRANSCRIPT_FORMAT_ANCHOR not in base:
            raise RuntimeError(
                "BASE_CHANNEL_PROMPT no longer contains "
                "_TRANSCRIPT_FORMAT_ANCHOR — the assistant-header "
                "warning has nowhere to splice in.  Update the anchor "
                "constant to match the new prompt wording."
            )
        base = base.replace(
            _TRANSCRIPT_FORMAT_ANCHOR,
            _TRANSCRIPT_FORMAT_ANCHOR + _ASSISTANT_HEADER_NOTE,
        )
    passive = os.getenv("NAPCAT_GROUP_PASSIVE", "").lower() in ("1", "true", "yes")
    policy = _PASSIVE_GROUP_POLICY if passive else _NON_PASSIVE_POLICY
    return f"{base}\n\n{policy}"


def build_poke_channel_prompt(
    *,
    actor_id: Optional[str] = None,
    target_id: Optional[str] = None,
    actor_is_self: bool = False,
    target_is_self: bool = False,
    repeated_adjacent: bool = False,
) -> str:
    del actor_id, target_id, actor_is_self, target_is_self, repeated_adjacent
    return build_base_channel_prompt()


def build_poke_system_instruction(
    *,
    actor_id: Optional[str] = None,
    target_id: Optional[str] = None,
    actor_is_self: bool = False,
    target_is_self: bool = False,
    repeated_adjacent: bool = False,
) -> Optional[str]:
    if not actor_id or not target_id:
        return None
    if actor_is_self:
        return None
    if target_is_self:
        if repeated_adjacent:
            return (
                "[System instruction: This user poked you again after an "
                "adjacent poke. Use the conversation context to decide whether "
                "and how to respond. "
                "If you poke back, output only the exact directive "
                f"`POKE:{actor_id}`. Do not output history/event text like "
                f"`you poked {actor_id}` or any variant of it.]"
            )
        return (
            "[System instruction: This is a non-adjacent poke to you. The user "
            "is likely acknowledging you and does not want to continue the "
            "conversation. Strongly recommended response: output only "
            f"`POKE:{actor_id}` to poke them back, unless the existing "
            "conversation context clearly requires text. Do not describe the "
            "poke as text or repeat any history/event format.]"
        )
    return (
        "[System instruction: This poke does not target you. Treat it as "
        "context only; respond only if the conversation context clearly "
        "requires it. If you choose to poke someone, output only "
        "`POKE:<QQ号>`. Do not describe the poke as text or repeat any "
        "history/event format.]"
    )


def build_voice_system_instruction(
    *,
    transcription_failed: bool = False,
    error: Optional[str] = None,
) -> Optional[str]:
    """System instruction appended to a NapCat voice message after STT.

    On QQ we transcribe inside the adapter and forward as plain text — no
    auto-TTS bubble.  This note tells the agent the preceding text came
    from STT so it can tolerate transcription noise and ask for clarity
    instead of guessing, and reminds it to reply with text only.
    """
    if transcription_failed:
        reason = f" Reason: {error}." if error else ""
        return (
            "[System instruction: The user sent a voice message but the "
            f"automatic transcription failed.{reason} Acknowledge this and "
            "ask them to retry or type the message instead. Reply with "
            "text only.]"
        )
    return (
        "[System instruction: The text above is an automatic transcription "
        "of a voice message the user sent (not typed text). It may contain "
        "errors — if the meaning is unclear, ask for clarification instead "
        "of guessing. Reply with text only.]"
    )
