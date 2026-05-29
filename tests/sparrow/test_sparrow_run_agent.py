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


def test_external_memory_sync_summarizes_multimodal_user_message():
    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.session_id = "sess-image"
    user_message = [
        {
            "type": "text",
            "text": "User sent an image\n\n[Image attached at: /tmp/image.png]",
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,RAW_IMAGE_BYTES"},
        },
    ]

    agent._sync_external_memory_for_turn(
        original_user_message=user_message,
        final_response="Looks like release week.",
        interrupted=False,
    )

    agent._memory_manager.sync_all.assert_called_once()
    synced_user, synced_assistant = agent._memory_manager.sync_all.call_args.args[:2]
    assert isinstance(synced_user, str)
    assert synced_assistant == "Looks like release week."
    assert "User sent an image" in synced_user
    assert "[Image attached at: /tmp/image.png]" in synced_user
    assert "[image attachment]" in synced_user
    assert "data:image" not in synced_user
    assert "RAW_IMAGE_BYTES" not in synced_user
    assert agent._memory_manager.sync_all.call_args.kwargs == {"session_id": "sess-image"}
