"""Sparrow-specific run_agent tests kept out of upstream test files."""

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        instance = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        instance.client = MagicMock()
        return instance


def test_includes_configured_timezone_name(agent, monkeypatch):
    monkeypatch.setattr(
        "hermes_time.now",
        lambda: datetime(2026, 4, 25, 9, 30, tzinfo=ZoneInfo("Asia/Tokyo")),
    )

    prompt = agent._build_system_prompt()

    assert "Conversation started: Saturday, April 25, 2026 09:30 AM JST" in prompt
    assert "Timezone: Asia/Tokyo" in prompt
