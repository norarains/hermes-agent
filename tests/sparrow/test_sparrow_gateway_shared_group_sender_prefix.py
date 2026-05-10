import pytest
from datetime import datetime, timezone

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(config: GatewayConfig) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


@pytest.mark.asyncio
async def test_preprocess_prefixes_time_and_sender_for_shared_non_thread_group_session(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
            group_sessions_per_user=False,
        )
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1002285219667",
        chat_name="Test Group",
        chat_type="group",
        user_name="Alice",
    )
    event = MessageEvent(
        text="hello",
        source=source,
        timestamp=datetime(2026, 4, 24, 23, 36, 48, tzinfo=timezone.utc),
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "04-24 16:36:48 Alice: hello"


@pytest.mark.asyncio
async def test_preprocess_prefixes_time_for_default_group_sessions(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
        )
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1002285219667",
        chat_name="Test Group",
        chat_type="group",
        user_name="Alice",
    )
    event = MessageEvent(
        text="hello",
        source=source,
        timestamp=datetime(2026, 4, 24, 23, 36, 48, tzinfo=timezone.utc),
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "04-24 16:36:48: hello"
