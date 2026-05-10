from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from agent.anthropic_adapter import (
    _is_oauth_token,
    refresh_anthropic_usage_token,
    resolve_anthropic_usage_token,
)
from hermes_cli.auth import _read_codex_tokens, resolve_codex_runtime_credentials
from hermes_cli.runtime_provider import resolve_runtime_provider


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class AccountUsageWindow:
    label: str
    used_percent: Optional[float] = None
    reset_at: Optional[datetime] = None
    detail: Optional[str] = None


@dataclass(frozen=True)
class AccountUsageSnapshot:
    provider: str
    source: str
    fetched_at: datetime
    title: str = "Account limits"
    plan: Optional[str] = None
    windows: tuple[AccountUsageWindow, ...] = ()
    details: tuple[str, ...] = ()
    unavailable_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.windows or self.details) and not self.unavailable_reason


def _title_case_slug(value: Optional[str]) -> Optional[str]:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    return cleaned.replace("_", " ").replace("-", " ").title()


def _parse_dt(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _format_reset(dt: Optional[datetime]) -> str:
    if not dt:
        return "unknown"
    local_dt = dt.astimezone()
    delta = dt - _utc_now()
    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        return f"now ({local_dt.strftime('%Y-%m-%d %H:%M %Z')})"
    hours, rem = divmod(total_seconds, 3600)
    minutes = rem // 60
    if hours >= 24:
        days, hours = divmod(hours, 24)
        rel = f"in {days}d {hours}h"
    elif hours > 0:
        rel = f"in {hours}h {minutes}m"
    else:
        rel = f"in {minutes}m"
    return f"{rel} ({local_dt.strftime('%Y-%m-%d %H:%M %Z')})"


def render_account_usage_lines(snapshot: Optional[AccountUsageSnapshot], *, markdown: bool = False) -> list[str]:
    if not snapshot:
        return []
    header = f"📈 {'**' if markdown else ''}{snapshot.title}{'**' if markdown else ''}"
    lines = [header]
    if snapshot.plan:
        lines.append(f"Provider: {snapshot.provider} ({snapshot.plan})")
    else:
        lines.append(f"Provider: {snapshot.provider}")
    for window in snapshot.windows:
        if window.used_percent is None:
            base = f"{window.label}: unavailable"
        else:
            remaining = max(0, round(100 - float(window.used_percent)))
            used = max(0, round(float(window.used_percent)))
            base = f"{window.label}: {remaining}% remaining ({used}% used)"
        if window.reset_at:
            base += f" • resets {_format_reset(window.reset_at)}"
        elif window.detail:
            base += f" • {window.detail}"
        lines.append(base)
    for detail in snapshot.details:
        lines.append(detail)
    if snapshot.unavailable_reason:
        lines.append(f"Unavailable: {snapshot.unavailable_reason}")
    return lines


def _usage_source_for_provider(provider: str) -> str:
    if provider == "anthropic":
        return "oauth_usage_api"
    if provider == "openai-codex":
        return "usage_api"
    if provider == "openrouter":
        return "credits_api"
    if provider == "deepseek":
        return "balance_api"
    return "usage_api"


def _usage_unavailable_reason(provider: str, exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        body = exc.response.text or ""
        if provider == "anthropic":
            if status == 401:
                return (
                    "Anthropic usage token is invalid or expired. Refresh "
                    "ANTHROPIC_USAGE_TOKEN from Claude Code login credentials."
                )
            if status == 403 and "user:profile" in body:
                return (
                    "Anthropic usage token lacks required user:profile scope. "
                    "Use a Claude Code login accessToken, not a claude setup-token."
                )
            return f"Anthropic usage query failed with HTTP {status}."
        return f"Usage query failed with HTTP {status}."

    message = str(exc).strip() or exc.__class__.__name__
    return f"Usage query failed: {message}"


def _usage_unavailable_snapshot(provider: str, exc: Exception) -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider=provider,
        source=_usage_source_for_provider(provider),
        fetched_at=_utc_now(),
        unavailable_reason=_usage_unavailable_reason(provider, exc),
    )


def _resolve_codex_usage_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        normalized = "https://chatgpt.com/backend-api/codex"
    if normalized.endswith("/codex"):
        normalized = normalized[: -len("/codex")]
    if "/backend-api" in normalized:
        return normalized + "/wham/usage"
    return normalized + "/api/codex/usage"


def _fetch_codex_account_usage() -> Optional[AccountUsageSnapshot]:
    creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
    token_data = _read_codex_tokens()
    tokens = token_data.get("tokens") or {}
    account_id = str(tokens.get("account_id", "") or "").strip() or None
    headers = {
        "Authorization": f"Bearer {creds['api_key']}",
        "Accept": "application/json",
        "User-Agent": "codex-cli",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    with httpx.Client(timeout=15.0) as client:
        response = client.get(_resolve_codex_usage_url(creds.get("base_url", "")), headers=headers)
        response.raise_for_status()
    payload = response.json() or {}
    rate_limit = payload.get("rate_limit") or {}
    windows: list[AccountUsageWindow] = []
    for key, label in (("primary_window", "Session"), ("secondary_window", "Weekly")):
        window = rate_limit.get(key) or {}
        used = window.get("used_percent")
        if used is None:
            continue
        windows.append(
            AccountUsageWindow(
                label=label,
                used_percent=float(used),
                reset_at=_parse_dt(window.get("reset_at")),
            )
        )
    details: list[str] = []
    credits = payload.get("credits") or {}
    if credits.get("has_credits"):
        balance = credits.get("balance")
        if isinstance(balance, (int, float)):
            details.append(f"Credits balance: ${float(balance):.2f}")
        elif credits.get("unlimited"):
            details.append("Credits balance: unlimited")
    return AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=_utc_now(),
        plan=_title_case_slug(payload.get("plan_type")),
        windows=tuple(windows),
        details=tuple(details),
    )


def _fetch_anthropic_account_usage() -> Optional[AccountUsageSnapshot]:
    token = (resolve_anthropic_usage_token() or "").strip()
    if not token:
        return AccountUsageSnapshot(
            provider="anthropic",
            source="oauth_usage_api",
            fetched_at=_utc_now(),
            unavailable_reason=(
                "No Anthropic usage token configured. Set ANTHROPIC_USAGE_TOKEN "
                "to a Claude Code OAuth accessToken with user:profile scope."
            ),
        )
    if not _is_oauth_token(token):
        return AccountUsageSnapshot(
            provider="anthropic",
            source="oauth_usage_api",
            fetched_at=_utc_now(),
            unavailable_reason="Anthropic account limits are only available for OAuth-backed Claude accounts.",
        )

    def _headers(access_token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "anthropic-beta": "oauth-2025-04-20",
            "User-Agent": "claude-code/2.1.0",
        }

    with httpx.Client(timeout=15.0) as client:
        response = client.get(
            "https://api.anthropic.com/api/oauth/usage",
            headers=_headers(token),
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 401:
                raise
            refreshed = (refresh_anthropic_usage_token() or "").strip()
            if not refreshed or refreshed == token:
                raise
            response = client.get(
                "https://api.anthropic.com/api/oauth/usage",
                headers=_headers(refreshed),
            )
            response.raise_for_status()
    payload = response.json() or {}
    windows: list[AccountUsageWindow] = []
    mapping = (
        ("five_hour", "Current session"),
        ("seven_day", "Current week"),
        ("seven_day_opus", "Opus week"),
        ("seven_day_sonnet", "Sonnet week"),
    )
    for key, label in mapping:
        window = payload.get(key) or {}
        util = window.get("utilization")
        if util is None:
            continue
        used = float(util) * 100 if float(util) <= 1 else float(util)
        windows.append(
            AccountUsageWindow(
                label=label,
                used_percent=used,
                reset_at=_parse_dt(window.get("resets_at")),
            )
        )
    details: list[str] = []
    extra = payload.get("extra_usage") or {}
    if extra.get("is_enabled"):
        used_credits = extra.get("used_credits")
        monthly_limit = extra.get("monthly_limit")
        currency = extra.get("currency") or "USD"
        if isinstance(used_credits, (int, float)) and isinstance(monthly_limit, (int, float)):
            details.append(
                f"Extra usage: {used_credits:.2f} / {monthly_limit:.2f} {currency}"
            )
    return AccountUsageSnapshot(
        provider="anthropic",
        source="oauth_usage_api",
        fetched_at=_utc_now(),
        windows=tuple(windows),
        details=tuple(details),
    )


def _fetch_openrouter_account_usage(base_url: Optional[str], api_key: Optional[str]) -> Optional[AccountUsageSnapshot]:
    runtime = resolve_runtime_provider(
        requested="openrouter",
        explicit_base_url=base_url,
        explicit_api_key=api_key,
    )
    token = str(runtime.get("api_key", "") or "").strip()
    if not token:
        return None
    normalized = str(runtime.get("base_url", "") or "").rstrip("/")
    credits_url = f"{normalized}/credits"
    key_url = f"{normalized}/key"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    with httpx.Client(timeout=10.0) as client:
        credits_resp = client.get(credits_url, headers=headers)
        credits_resp.raise_for_status()
        credits = (credits_resp.json() or {}).get("data") or {}
        try:
            key_resp = client.get(key_url, headers=headers)
            key_resp.raise_for_status()
            key_data = (key_resp.json() or {}).get("data") or {}
        except Exception:
            key_data = {}
    total_credits = float(credits.get("total_credits") or 0.0)
    total_usage = float(credits.get("total_usage") or 0.0)
    details = [f"Credits balance: ${max(0.0, total_credits - total_usage):.2f}"]
    windows: list[AccountUsageWindow] = []
    limit = key_data.get("limit")
    limit_remaining = key_data.get("limit_remaining")
    limit_reset = str(key_data.get("limit_reset") or "").strip()
    usage = key_data.get("usage")
    if (
        isinstance(limit, (int, float))
        and float(limit) > 0
        and isinstance(limit_remaining, (int, float))
        and 0 <= float(limit_remaining) <= float(limit)
    ):
        limit_value = float(limit)
        remaining_value = float(limit_remaining)
        used_percent = ((limit_value - remaining_value) / limit_value) * 100
        detail_parts = [f"${remaining_value:.2f} of ${limit_value:.2f} remaining"]
        if limit_reset:
            detail_parts.append(f"resets {limit_reset}")
        windows.append(
            AccountUsageWindow(
                label="API key quota",
                used_percent=used_percent,
                detail=" • ".join(detail_parts),
            )
        )
    if isinstance(usage, (int, float)):
        usage_parts = [f"API key usage: ${float(usage):.2f} total"]
        for value, label in (
            (key_data.get("usage_daily"), "today"),
            (key_data.get("usage_weekly"), "this week"),
            (key_data.get("usage_monthly"), "this month"),
        ):
            if isinstance(value, (int, float)) and float(value) > 0:
                usage_parts.append(f"${float(value):.2f} {label}")
        details.append(" • ".join(usage_parts))
    return AccountUsageSnapshot(
        provider="openrouter",
        source="credits_api",
        fetched_at=_utc_now(),
        windows=tuple(windows),
        details=tuple(details),
    )


def _fetch_deepseek_account_usage(
    base_url: Optional[str], api_key: Optional[str],
) -> Optional[AccountUsageSnapshot]:
    """Pull DeepSeek account balance via ``GET /user/balance``.

    Endpoint shape (verified against api.deepseek.com on 2026-04-28):
        Request:  GET /user/balance  (or /v1/user/balance — both route)
                  Authorization: Bearer <api_key>
        Response: {
          "is_available": bool,
          "balance_infos": [
            {"currency": "CNY"|"USD",
             "total_balance":   "7.34",
             "granted_balance": "0.00",
             "topped_up_balance": "7.34"}, ...
          ]
        }

    We surface every field the API returns (no client-side filtering)
    because the operator wanted "whatever they return, show that".
    Each currency bucket becomes one detail line; the granted/topped-up
    breakdown is appended in parens when either is non-zero so users can
    tell free credits from paid balance at a glance.
    """
    runtime = resolve_runtime_provider(
        requested="deepseek",
        explicit_base_url=base_url,
        explicit_api_key=api_key,
    )
    token = str(runtime.get("api_key", "") or "").strip()
    if not token:
        return None

    base = str(runtime.get("base_url", "") or "https://api.deepseek.com").rstrip("/")
    # ``base_url`` from config is typically ``.../v1`` (matches the OpenAI-
    # style chat-completions URL), but the balance endpoint lives one
    # level up.  DeepSeek currently routes BOTH ``/user/balance`` and
    # ``/v1/user/balance`` to the same handler, so we just trim a trailing
    # ``/v1`` defensively rather than picking sides.
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    url = f"{base}/user/balance"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json() or {}

    is_available = bool(data.get("is_available", True))
    balance_infos = data.get("balance_infos") or []

    details: list[str] = []
    if not is_available:
        details.append("Account marked unavailable by DeepSeek")

    for info in balance_infos:
        if not isinstance(info, dict):
            continue
        currency = str(info.get("currency", "") or "").strip() or "?"
        total = info.get("total_balance")
        granted = info.get("granted_balance")
        topped = info.get("topped_up_balance")

        try:
            total_f = float(total) if total is not None else None
        except (TypeError, ValueError):
            total_f = None
        try:
            granted_f = float(granted) if granted is not None else 0.0
        except (TypeError, ValueError):
            granted_f = 0.0
        try:
            topped_f = float(topped) if topped is not None else 0.0
        except (TypeError, ValueError):
            topped_f = 0.0

        if total_f is None:
            line = f"Balance ({currency}): unknown"
        else:
            line = f"Balance: {total_f:.2f} {currency}"
        # Show granted/topped breakdown only when either is non-zero so a
        # zero-zero bucket (the empty USD bucket on a CNY-only account)
        # isn't padded with redundant "(granted 0.00 + top-up 0.00)".
        if granted_f or topped_f:
            line += f" (granted {granted_f:.2f} + top-up {topped_f:.2f})"
        details.append(line)

    if not details:
        details.append("No balance buckets returned")

    return AccountUsageSnapshot(
        provider="deepseek",
        source="balance_api",
        fetched_at=_utc_now(),
        details=tuple(details),
    )


def fetch_account_usage(
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    normalized = str(provider or "").strip().lower()
    if normalized in {"", "auto", "custom"}:
        return None
    try:
        if normalized == "openai-codex":
            return _fetch_codex_account_usage()
        if normalized == "anthropic":
            return _fetch_anthropic_account_usage()
        if normalized == "openrouter":
            return _fetch_openrouter_account_usage(base_url, api_key)
        if normalized == "deepseek":
            return _fetch_deepseek_account_usage(base_url, api_key)
    except Exception as exc:
        return _usage_unavailable_snapshot(normalized, exc)
    return None
