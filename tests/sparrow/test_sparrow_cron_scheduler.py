"""Sparrow-specific cron tests kept out of upstream test files."""

from cron.scheduler import _build_job_prompt, _resolve_delivery_target, _resolve_delivery_targets


def test_origin_delivery_without_origin_falls_back_to_supported_home_channels(monkeypatch):
    for fallback_env in (
        "MATRIX_HOME_ROOM",
        "MATRIX_HOME_CHANNEL",
        "TELEGRAM_HOME_CHANNEL",
        "DISCORD_HOME_CHANNEL",
        "SLACK_HOME_CHANNEL",
        "SIGNAL_HOME_CHANNEL",
        "MATTERMOST_HOME_CHANNEL",
        "SMS_HOME_CHANNEL",
        "EMAIL_HOME_ADDRESS",
        "DINGTALK_HOME_CHANNEL",
        "BLUEBUBBLES_HOME_CHANNEL",
        "FEISHU_HOME_CHANNEL",
        "WECOM_HOME_CHANNEL",
        "WEIXIN_HOME_CHANNEL",
        "QQ_HOME_CHANNEL",
        "QQBOT_HOME_CHANNEL",
        "ONEBOT_NAPCAT_HOME_CHANNEL",
    ):
        monkeypatch.delenv(fallback_env, raising=False)
    monkeypatch.setenv("ONEBOT_NAPCAT_HOME_CHANNEL", "group_123")

    assert _resolve_delivery_target({"deliver": "origin"}) == {
        "platform": "onebot_napcat",
        "chat_id": "group_123",
        "thread_id": None,
    }


def test_bare_onebot_napcat_delivery_uses_home_channel(monkeypatch):
    monkeypatch.setenv("ONEBOT_NAPCAT_HOME_CHANNEL", "private_891088473")

    assert _resolve_delivery_target({"deliver": "onebot_napcat"}) == {
        "platform": "onebot_napcat",
        "chat_id": "private_891088473",
        "thread_id": None,
    }


def test_output_only_guidance_present():
    job = {"prompt": "Generate a report"}
    result = _build_job_prompt(job)

    assert "Return only the final content to deliver" in result
    assert "do not include planning, reasoning" in result
    assert job["prompt"] == "Generate a report"


# ---------------------------------------------------------------------------
# Multi-target delivery — `deliver: "foo,bar"` pin
# ---------------------------------------------------------------------------


def test_multi_target_delivery_parses_comma_separated_into_distinct_targets(monkeypatch):
    """``deliver: "telegram,onebot_napcat"`` must resolve to TWO
    independent target dicts (one per platform), each with its own
    chat_id from the configured home channel.

    Without this, an operator setting up cron-broadcast to two
    platforms would silently see the comma treated as part of a single
    platform name and drop the whole job at validation."""
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "tg_home_1234")
    monkeypatch.setenv("ONEBOT_NAPCAT_HOME_CHANNEL", "group_5678")

    targets = _resolve_delivery_targets({"deliver": "telegram,onebot_napcat"})

    assert len(targets) == 2
    platforms = {t["platform"] for t in targets}
    assert platforms == {"telegram", "onebot_napcat"}
    by_platform = {t["platform"]: t["chat_id"] for t in targets}
    assert by_platform["telegram"] == "tg_home_1234"
    assert by_platform["onebot_napcat"] == "group_5678"


def test_multi_target_delivery_with_explicit_chat_ids(monkeypatch):
    """``deliver: "telegram:specific_id,discord:other_id"`` overrides
    the home-channel default per-target.  Pins that the colon-separated
    suffix is parsed independently for each comma-segment."""
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "should_be_overridden")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "also_overridden")

    targets = _resolve_delivery_targets({
        "deliver": "telegram:custom_chat,discord:other_chat",
    })

    assert len(targets) == 2
    by_platform = {t["platform"]: t["chat_id"] for t in targets}
    assert by_platform["telegram"] == "custom_chat"
    assert by_platform["discord"] == "other_chat"


def test_multi_target_delivery_skips_unresolvable_segment(monkeypatch):
    """If one segment of a comma-list resolves to a missing home
    channel, the OTHER segments must still produce targets — partial
    delivery beats total failure for cron broadcast.  Without this, a
    misconfigured TELEGRAM_HOME_CHANNEL on a 3-platform cron would
    block delivery to the two correctly-configured platforms too."""
    # Only configure ONE of the two requested platforms.
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL_NAME", raising=False)
    monkeypatch.setenv("ONEBOT_NAPCAT_HOME_CHANNEL", "group_99")

    targets = _resolve_delivery_targets({"deliver": "telegram,onebot_napcat"})

    # The unresolvable telegram segment is dropped; napcat still resolves.
    assert len(targets) == 1
    assert targets[0]["platform"] == "onebot_napcat"
    assert targets[0]["chat_id"] == "group_99"


def test_multi_target_delivery_empty_string_segments_are_ignored(monkeypatch):
    """``"telegram,,onebot_napcat"`` (typo with double comma) — the
    empty segment from the doubled separator must NOT produce a
    "platform=''" target; should just be skipped.  Without this guard,
    cron tick would crash on the empty-platform lookup."""
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "tg_home")
    monkeypatch.setenv("ONEBOT_NAPCAT_HOME_CHANNEL", "group_1")

    targets = _resolve_delivery_targets({"deliver": "telegram,,onebot_napcat"})

    assert len(targets) == 2
    assert {t["platform"] for t in targets} == {"telegram", "onebot_napcat"}
