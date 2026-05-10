import httpx

from agent.account_usage import fetch_account_usage, render_account_usage_lines
from agent.anthropic_adapter import resolve_anthropic_usage_token


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _CapturingClient:
    def __init__(self, payload, captured):
        self._payload = payload
        self._captured = captured

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        self._captured["url"] = url
        self._captured["headers"] = dict(headers or {})
        return _Response(self._payload)


class _HttpErrorClient:
    def __init__(self, status_code, payload):
        self._status_code = status_code
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        return httpx.Response(
            self._status_code,
            json=self._payload,
            request=httpx.Request("GET", url),
        )


class _SequenceClient:
    def __init__(self, responses, captured):
        self._responses = list(responses)
        self._captured = captured

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        self._captured.append({"url": url, "headers": dict(headers or {})})
        response = self._responses.pop(0)
        response.request = httpx.Request("GET", url)
        return response


def test_fetch_account_usage_anthropic_uses_usage_token_without_changing_runtime_token(monkeypatch):
    captured = {}
    monkeypatch.setenv("ANTHROPIC_USAGE_TOKEN", "sk-ant-oat01-usage")
    monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-runtime")
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=15.0: _CapturingClient(
            {
                "five_hour": {
                    "utilization": 27.0,
                    "resets_at": "2026-04-26T05:10:00+00:00",
                },
                "seven_day": {
                    "utilization": 42.0,
                    "resets_at": "2026-04-28T16:00:00+00:00",
                },
            },
            captured,
        ),
    )

    snapshot = fetch_account_usage("anthropic")

    assert snapshot is not None
    assert snapshot.source == "oauth_usage_api"
    assert captured["url"] == "https://api.anthropic.com/api/oauth/usage"
    assert captured["headers"]["Authorization"] == "Bearer sk-ant-oat01-usage"
    assert [window.label for window in snapshot.windows] == ["Current session", "Current week"]
    assert snapshot.windows[0].used_percent == 27.0


def test_anthropic_usage_token_never_falls_back_to_runtime_token(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_USAGE_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_USAGE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_USAGE_EXPIRES_AT_MS", raising=False)
    monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-runtime-only")

    assert resolve_anthropic_usage_token() is None


def test_anthropic_usage_token_refreshes_and_persists_dotenv(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ANTHROPIC_USAGE_TOKEN=sk-ant-oat01-old-usage\n"
        "ANTHROPIC_USAGE_REFRESH_TOKEN=sk-ant-oat01-old-refresh\n"
        "ANTHROPIC_USAGE_EXPIRES_AT_MS=1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_USAGE_TOKEN", "sk-ant-oat01-old-usage")
    monkeypatch.setenv("ANTHROPIC_USAGE_REFRESH_TOKEN", "sk-ant-oat01-old-refresh")
    monkeypatch.setenv("ANTHROPIC_USAGE_EXPIRES_AT_MS", "1")
    monkeypatch.setattr(
        "agent.anthropic_adapter.refresh_anthropic_oauth_pure",
        lambda refresh_token, use_json=False: {
            "access_token": "sk-ant-oat01-new-usage",
            "refresh_token": "sk-ant-oat01-new-refresh",
            "expires_at_ms": 1777272191243,
        },
    )

    assert resolve_anthropic_usage_token() == "sk-ant-oat01-new-usage"

    assert "ANTHROPIC_USAGE_TOKEN=sk-ant-oat01-new-usage" in env_path.read_text()
    assert "ANTHROPIC_USAGE_REFRESH_TOKEN=sk-ant-oat01-new-refresh" in env_path.read_text()
    assert "ANTHROPIC_USAGE_EXPIRES_AT_MS=1777272191243" in env_path.read_text()


def test_fetch_account_usage_refreshes_usage_token_once_on_401(monkeypatch):
    captured = []
    monkeypatch.setenv("ANTHROPIC_USAGE_TOKEN", "sk-ant-oat01-old-usage")
    monkeypatch.setenv("ANTHROPIC_USAGE_REFRESH_TOKEN", "sk-ant-oat01-refresh")
    monkeypatch.delenv("ANTHROPIC_USAGE_EXPIRES_AT_MS", raising=False)
    monkeypatch.setattr(
        "agent.account_usage.refresh_anthropic_usage_token",
        lambda: "sk-ant-oat01-new-usage",
    )
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=15.0: _SequenceClient(
            [
                httpx.Response(401, json={"error": {"message": "expired"}}),
                httpx.Response(
                    200,
                    json={
                        "five_hour": {
                            "utilization": 0.3,
                            "resets_at": "2026-04-26T05:10:00+00:00",
                        },
                    },
                ),
            ],
            captured,
        ),
    )

    snapshot = fetch_account_usage("anthropic")

    assert snapshot is not None
    assert snapshot.available
    assert [call["headers"]["Authorization"] for call in captured] == [
        "Bearer sk-ant-oat01-old-usage",
        "Bearer sk-ant-oat01-new-usage",
    ]


def test_fetch_account_usage_surfaces_anthropic_scope_failure(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_anthropic_usage_token",
        lambda: "sk-ant-oat01-inference-only",
    )
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=15.0: _HttpErrorClient(
            403,
            {
                "type": "error",
                "error": {
                    "type": "permission_error",
                    "message": "OAuth token does not meet scope requirement user:profile",
                },
            },
        ),
    )

    snapshot = fetch_account_usage("anthropic")

    assert snapshot is not None
    assert snapshot.provider == "anthropic"
    assert snapshot.source == "oauth_usage_api"
    assert snapshot.windows == ()
    assert snapshot.unavailable_reason == (
        "Anthropic usage token lacks required user:profile scope. "
        "Use a Claude Code login accessToken, not a claude setup-token."
    )
    assert "Unavailable: Anthropic usage token lacks required user:profile scope." in "\n".join(
        render_account_usage_lines(snapshot)
    )


# ---------------------------------------------------------------------------
# DeepSeek balance — pulled via GET /user/balance and rendered as one
# detail line per currency bucket.
# ---------------------------------------------------------------------------


def _stub_deepseek_runtime(monkeypatch, base_url="https://api.deepseek.com/v1", api_key="sk-test"):
    """Patch the runtime-provider resolver so the deepseek fetcher gets
    the configured base_url + api_key without touching real env vars or
    config files."""
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda **kwargs: {"base_url": base_url, "api_key": api_key},
    )


def test_fetch_account_usage_deepseek_returns_balance_per_currency(monkeypatch):
    """The exact response shape DeepSeek returns in production (verified
    against api.deepseek.com on 2026-04-28).  Pin every detail line so a
    silent change to the rendering — losing the granted/topped-up
    breakdown, dropping a currency, or wedging USD ahead of CNY — fails
    loudly instead of confusing the operator at /usage time."""
    captured = {}
    _stub_deepseek_runtime(monkeypatch)
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=10.0: _CapturingClient(
            {
                "is_available": True,
                "balance_infos": [
                    {"currency": "CNY", "total_balance": "7.34",
                     "granted_balance": "0.00", "topped_up_balance": "7.34"},
                    {"currency": "USD", "total_balance": "0.00",
                     "granted_balance": "0.00", "topped_up_balance": "0.00"},
                ],
            },
            captured,
        ),
    )

    snapshot = fetch_account_usage("deepseek")

    assert snapshot is not None
    assert snapshot.provider == "deepseek"
    assert snapshot.source == "balance_api"
    # base_url ``.../v1`` must be trimmed to the bare host before
    # appending /user/balance — DeepSeek routes both, but trimming gives
    # us a single canonical URL the operator can curl with.
    assert captured["url"] == "https://api.deepseek.com/user/balance"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"

    rendered = "\n".join(render_account_usage_lines(snapshot))
    # CNY bucket: non-zero topped-up → granted/top-up breakdown shown.
    assert "Balance: 7.34 CNY (granted 0.00 + top-up 7.34)" in rendered
    # USD bucket: all zeros → breakdown SUPPRESSED to avoid noise.
    assert "Balance: 0.00 USD" in rendered
    assert "(granted 0.00 + top-up 0.00)" not in rendered


def test_fetch_account_usage_deepseek_handles_unavailable_account(monkeypatch):
    """When DeepSeek flips ``is_available`` to false (account suspended,
    payment expired) the snapshot must surface that status loudly so the
    user understands why their next request is about to fail."""
    captured = {}
    _stub_deepseek_runtime(monkeypatch)
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=10.0: _CapturingClient(
            {
                "is_available": False,
                "balance_infos": [
                    {"currency": "CNY", "total_balance": "0.00",
                     "granted_balance": "0.00", "topped_up_balance": "0.00"},
                ],
            },
            captured,
        ),
    )

    snapshot = fetch_account_usage("deepseek")
    assert snapshot is not None
    rendered = "\n".join(render_account_usage_lines(snapshot))
    assert "Account marked unavailable by DeepSeek" in rendered


def test_fetch_account_usage_deepseek_returns_none_when_no_api_key(monkeypatch):
    """No api_key → fetcher returns None (rather than raising or hitting
    the network with an empty token).  This is the standard
    "provider configured but credential missing" behavior — /usage just
    shows session-token info without an account-balance section."""
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda **kwargs: {"base_url": "https://api.deepseek.com/v1", "api_key": ""},
    )
    assert fetch_account_usage("deepseek") is None


def test_fetch_account_usage_deepseek_keeps_base_when_no_v1_suffix(monkeypatch):
    """If the operator's base_url lacks the ``/v1`` suffix (some custom
    deployments, or a future API version bump), don't accidentally trim
    something we shouldn't.  Only the exact ``/v1`` tail gets stripped."""
    captured = {}
    _stub_deepseek_runtime(monkeypatch, base_url="https://api.deepseek.com")
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=10.0: _CapturingClient(
            {"is_available": True, "balance_infos": []},
            captured,
        ),
    )
    fetch_account_usage("deepseek")
    assert captured["url"] == "https://api.deepseek.com/user/balance"


def test_fetch_account_usage_deepseek_passes_explicit_base_and_key(monkeypatch):
    """The dispatcher's base_url/api_key kwargs must be forwarded to the
    runtime resolver — gateway calls /usage with those pulled off the
    live agent's runtime, not from env vars.  Without this, /usage would
    silently use the wrong account when the agent's runtime was rebound
    via fallback / per-session override."""
    captured_resolver = {}
    def _resolve(*, requested, explicit_base_url, explicit_api_key):
        captured_resolver["requested"] = requested
        captured_resolver["base_url"] = explicit_base_url
        captured_resolver["api_key"] = explicit_api_key
        return {"base_url": "https://api.deepseek.com/v1", "api_key": "sk-via-resolver"}
    monkeypatch.setattr("agent.account_usage.resolve_runtime_provider", _resolve)
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=10.0: _CapturingClient(
            {"is_available": True, "balance_infos": []}, {},
        ),
    )

    fetch_account_usage(
        "deepseek",
        base_url="https://override.example/v1",
        api_key="sk-override",
    )

    assert captured_resolver["requested"] == "deepseek"
    assert captured_resolver["base_url"] == "https://override.example/v1"
    assert captured_resolver["api_key"] == "sk-override"
