"""Sparrow-specific send_message_tool tests kept out of upstream test files."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

from tools.send_message_tool import _send_onebot_napcat


class TestSendOneBotNapCat:
    def test_sends_group_message_over_onebot_ws(self, monkeypatch):
        import aiohttp

        sent_payloads = []

        class FakeWS:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def send_json(self, payload):
                sent_payloads.append(payload)

            async def receive(self):
                return SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data=json.dumps({
                        "echo": "send_message_tool",
                        "status": "ok",
                        "data": {"message_id": 42},
                    }),
                )

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            def ws_connect(self, *args, **kwargs):
                self.ws_args = args
                self.ws_kwargs = kwargs
                return FakeWS()

        session = FakeSession()
        monkeypatch.setenv("NAPCAT_WS_URL", "ws://napcat.example/ws")
        monkeypatch.setenv("NAPCAT_ACCESS_TOKEN", "secret")

        with patch("aiohttp.ClientSession", return_value=session):
            result = asyncio.run(
                _send_onebot_napcat(
                    SimpleNamespace(enabled=True, token=None, extra={}),
                    "group_123",
                    "hello",
                )
            )

        assert result == {
            "success": True,
            "platform": "onebot_napcat",
            "chat_id": "group_123",
            "message_id": "42",
        }
        assert session.ws_args == ("ws://napcat.example/ws",)
        assert session.ws_kwargs["headers"] == {"Authorization": "Bearer secret"}
        assert sent_payloads == [
            {
                "action": "send_group_msg",
                "params": {
                    "group_id": 123,
                    "message": [{"type": "text", "data": {"text": "hello"}}],
                },
                "echo": "send_message_tool",
            }
        ]

    def test_rejects_bad_chat_id_without_network(self):
        result = asyncio.run(
            _send_onebot_napcat(
                SimpleNamespace(enabled=True, token=None, extra={}),
                "room_123",
                "hello",
            )
        )

        assert "error" in result
        assert "group_" in result["error"]
