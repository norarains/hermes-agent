"""Sparrow gateway /cron command tests kept out of upstream test files."""

from datetime import datetime, timedelta, timezone

import pytest

from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner

_PACIFIC = timezone(timedelta(hours=-7))


def _runner() -> GatewayRunner:
    return object.__new__(GatewayRunner)


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text)


def _freeze_now(monkeypatch):
    monkeypatch.setattr(
        "gateway.cron_command._now_for_tz",
        lambda tzinfo: datetime(2026, 4, 25, 19, 48, tzinfo=tzinfo or _PACIFIC),
    )


@pytest.mark.asyncio
async def test_gateway_cron_command_lists_enabled_jobs(monkeypatch):
    _freeze_now(monkeypatch)
    calls = []

    def fake_list_jobs(*, include_disabled=False):
        calls.append(include_disabled)
        return [
            {
                "id": "weekly",
                "name": "每周维护提醒",
                "prompt": "提醒主人更新 Hermes Claude NapCat",
                "schedule_display": "0 10 * * 6",
                "next_run_at": "2026-05-02T10:00:00-07:00",
                "deliver": "origin",
                "enabled": True,
                "state": "scheduled",
                "last_status": "ok",
            },
            {
                "id": "morning",
                "name": "每日早报",
                "prompt": "给主人发一份简短的每日早报",
                "schedule_display": "0 7 * * *",
                "next_run_at": "2026-04-26T07:00:00-07:00",
                "deliver": "onebot_napcat:group_940134062",
                "enabled": True,
                "state": "scheduled",
            },
            {
                "id": "breakfast",
                "name": "每日早安提醒伊乐姐姐吃早饭",
                "prompt": "提醒伊乐姐姐吃早饭",
                "schedule_display": "0 18 * * *",
                "next_run_at": "2026-04-26T18:00:00-07:00",
                "deliver": "origin",
                "enabled": True,
                "state": "scheduled",
            }
        ]

    monkeypatch.setattr("cron.jobs.list_jobs", fake_list_jobs)

    result = await _runner()._handle_cron_command(_event("/cron"))

    assert calls == [False]
    assert result.startswith("定时任务（按下次触发时间）:")
    assert "明天 04-26 周日" in result
    assert "下周六 05-02" in result
    assert "1. 07:00 每日早报" in result
    assert "2. 18:00 每日早安提醒伊乐姐姐吃早饭" in result
    assert "3. 10:00 每周维护提醒" in result
    assert "id: morning | 状态: 已启用 | 投递: onebot_napcat:group_940134062" in result
    assert "规则: 每天 07:00" in result
    assert "规则: 每周六 10:00" in result
    assert "任务: 给主人发一份简短的每日早报" in result
    assert "2026-04-26T07:00:00-07:00" not in result
    assert result.index("明天 04-26 周日") < result.index("下周六 05-02")
    assert "Use `/cron all`" in result


@pytest.mark.asyncio
async def test_gateway_cron_all_includes_disabled_jobs(monkeypatch):
    _freeze_now(monkeypatch)
    calls = []

    def fake_list_jobs(*, include_disabled=False):
        calls.append(include_disabled)
        return [
            {
                "id": "paused1",
                "name": "paused reminder",
                "prompt": "old task",
                "schedule": {"display": "every 1h"},
                "next_run_at": None,
                "deliver": ["telegram:dm", "onebot_napcat:group_123"],
                "enabled": False,
                "state": "paused",
            }
        ]

    monkeypatch.setattr("cron.jobs.list_jobs", fake_list_jobs)

    result = await _runner()._handle_cron_command(_event("/cron all"))

    assert calls == [True]
    assert "未安排下次运行" in result
    assert "1. --:-- paused reminder" in result
    assert "id: paused1 | 状态: 已禁用 | 投递: telegram:dm, onebot_napcat:group_123" in result
    assert "规则: every 1h" in result
    assert "Use `/cron all`" not in result


@pytest.mark.asyncio
async def test_gateway_cron_rejects_mutating_subcommands(monkeypatch):
    def fake_list_jobs(*, include_disabled=False):
        raise AssertionError("/cron pause must not read or mutate cron jobs")

    monkeypatch.setattr("cron.jobs.list_jobs", fake_list_jobs)

    result = await _runner()._handle_cron_command(_event("/cron pause job123"))

    assert "Usage: `/cron [list|all]`" in result
    assert "only shows scheduled jobs" in result


@pytest.mark.asyncio
async def test_gateway_cron_empty_list(monkeypatch):
    monkeypatch.setattr("cron.jobs.list_jobs", lambda *, include_disabled=False: [])

    result = await _runner()._handle_cron_command(_event("/cron list"))

    assert result == "No scheduled cron jobs."


def test_gateway_cron_command_is_registered_for_chat_and_busy_sessions():
    from hermes_cli.commands import (
        ACTIVE_SESSION_BYPASS_COMMANDS,
        GATEWAY_KNOWN_COMMANDS,
        gateway_help_lines,
        resolve_command,
    )

    command = resolve_command("cron")

    assert command is not None
    assert command.cli_only is False
    assert "cron" in GATEWAY_KNOWN_COMMANDS
    assert "cron" in ACTIVE_SESSION_BYPASS_COMMANDS
    assert any(line.startswith("`/cron [subcommand]`") for line in gateway_help_lines())
