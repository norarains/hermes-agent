"""Read-only gateway rendering for chat /cron."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_WEEKDAYS_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def render_cron_command(raw_args: str) -> str:
    """Render `/cron` output for chat without mutating cron state."""
    normalized_args = " ".join(raw_args.strip().lower().split())
    if normalized_args in ("", "list"):
        include_disabled = False
    elif normalized_args in (
        "all",
        "--all",
        "-a",
        "list all",
        "list --all",
        "list -a",
    ):
        include_disabled = True
    else:
        return "Usage: `/cron [list|all]`\nThis chat command only shows scheduled jobs."

    try:
        from cron.jobs import list_jobs

        jobs = list_jobs(include_disabled=include_disabled)
    except Exception as exc:
        logger.warning("Failed to list cron jobs for gateway command: %s", exc)
        return f"Failed to list cron jobs: {exc}"

    if not jobs:
        return "No scheduled cron jobs."

    lines = ["定时任务（按下次触发时间）:"]
    current_group = object()
    for index, job in enumerate(_sort_jobs_by_next_run(jobs), start=1):
        next_run_dt = _parse_next_run(job.get("next_run_at"))
        group = next_run_dt.date() if next_run_dt else None
        if group != current_group:
            lines.append("")
            lines.append(_format_group_heading(next_run_dt))
            current_group = group

        job_id = _field(job.get("id"), "?")
        name = _preview(job.get("name") or job.get("prompt") or "cron job", limit=80)
        enabled = bool(job.get("enabled", True))
        state = _state_label(job.get("state"), enabled)
        schedule = _format_schedule(job)
        deliver = _field(job.get("deliver"))
        trigger_time = next_run_dt.strftime("%H:%M") if next_run_dt else "--:--"

        lines.append(f"{index}. {trigger_time} {name}")
        lines.append(f"   id: {job_id} | 状态: {state} | 投递: {deliver}")
        lines.append(f"   规则: {schedule}")
        prompt_preview = _preview(job.get("prompt"), limit=90)
        if prompt_preview and prompt_preview != name:
            lines.append(f"   任务: {prompt_preview}")
        last_status = _field(job.get("last_status"), "")
        if last_status and last_status != "ok":
            last_run = _field(job.get("last_run_at"))
            lines.append(f"   上次: {last_status} at {last_run}")

    if not include_disabled:
        lines.append("Use `/cron all` to include disabled jobs.")
    return "\n".join(lines)


def _sort_jobs_by_next_run(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(job: dict[str, Any]) -> tuple[float, str]:
        next_run = _parse_next_run(job.get("next_run_at"))
        timestamp = next_run.timestamp() if next_run else float("inf")
        name = str(job.get("name") or job.get("prompt") or "")
        return timestamp, name

    return sorted(jobs, key=sort_key)


def _parse_next_run(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        next_run = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if next_run.tzinfo is None:
        next_run = next_run.astimezone()
    return next_run


def _format_group_heading(next_run: datetime | None) -> str:
    if next_run is None:
        return "未安排下次运行"

    today = _now_for_tz(next_run.tzinfo).date()
    target = next_run.date()
    day_diff = (target - today).days
    weekday = _WEEKDAYS_ZH[target.weekday()]
    date_text = target.strftime("%m-%d")
    if day_diff == 0:
        return f"今天 {date_text} {weekday}"
    if day_diff == 1:
        return f"明天 {date_text} {weekday}"
    if day_diff == 2:
        return f"后天 {date_text} {weekday}"

    next_week_start = today - _date_delta(today.weekday()) + _date_delta(7)
    if target < next_week_start:
        return f"本{weekday} {date_text}"
    if target < next_week_start + _date_delta(7):
        return f"下{weekday} {date_text}"
    return f"{target:%Y-%m-%d} {weekday}"


def _date_delta(days: int):
    from datetime import timedelta

    return timedelta(days=days)


def _now_for_tz(tzinfo) -> datetime:
    return datetime.now(tzinfo).astimezone(tzinfo) if tzinfo else datetime.now().astimezone()


def _state_label(state: Any, enabled: bool) -> str:
    if not enabled:
        return "已禁用"
    normalized = str(state or "scheduled").lower()
    if normalized == "paused":
        return "已暂停"
    if normalized == "running":
        return "运行中"
    if normalized == "error":
        return "错误"
    if normalized == "scheduled":
        return "已启用"
    return str(state)


def _format_schedule(job: dict[str, Any]) -> str:
    schedule_display = _field(job.get("schedule_display"))
    if schedule_display == "-" and isinstance(job.get("schedule"), dict):
        schedule_display = _field(job["schedule"].get("display") or job["schedule"].get("value"))
    translated = _translate_cron_expression(schedule_display)
    return translated or schedule_display


def _translate_cron_expression(value: str) -> str | None:
    parts = value.split()
    if len(parts) != 5:
        return None

    minute, hour, day_of_month, month, day_of_week = parts
    if day_of_month != "*" or month != "*":
        return None
    if not minute.isdigit() or not hour.isdigit():
        return None

    hhmm = f"{int(hour):02d}:{int(minute):02d}"
    if day_of_week == "*":
        return f"每天 {hhmm}"
    weekday = _cron_weekday_label(day_of_week)
    if weekday:
        return f"每{weekday} {hhmm}"
    return None


def _cron_weekday_label(value: str) -> str | None:
    labels = {
        "0": "周日",
        "7": "周日",
        "1": "周一",
        "2": "周二",
        "3": "周三",
        "4": "周四",
        "5": "周五",
        "6": "周六",
        "sun": "周日",
        "mon": "周一",
        "tue": "周二",
        "wed": "周三",
        "thu": "周四",
        "fri": "周五",
        "sat": "周六",
    }
    return labels.get(value.lower())


def _preview(value: Any, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _field(value: Any, fallback: str = "-") -> str:
    if value in (None, ""):
        return fallback
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value) or fallback
    return str(value)
