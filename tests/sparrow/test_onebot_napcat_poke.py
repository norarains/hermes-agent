from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.onebot_napcat.adapter import OneBotNapCatAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _adapter() -> OneBotNapCatAdapter:
    adapter = OneBotNapCatAdapter(PlatformConfig(enabled=True, extra={
        "group_sessions_per_user": False,
    }))
    adapter.self_id = 1903883693
    adapter.self_name = "小麻雀"
    return adapter


FIRST_POKE_SYSTEM_INSTRUCTION = (
    "[System instruction: This is a non-adjacent poke to you. The user "
    "is likely acknowledging you and does not want to continue the "
    "conversation. Strongly recommended response: output only "
    "`POKE:891088473` to poke them back, unless the existing "
    "conversation context clearly requires text. Do not describe the "
    "poke as text or repeat any history/event format.]"
)


def test_napcat_disables_streaming_edits_so_poke_directives_reach_send_hook():
    assert OneBotNapCatAdapter.SUPPORTS_MESSAGE_EDITING is False


@pytest.mark.asyncio
async def test_poke_notice_first_adjacent_self_poke_goes_through_normal_message_path():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    await adapter._handle_poke_notice({
        "time": 1777073808,
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 891088473,
        "target_id": 1903883693,
    })

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == f"891088473 poked you\n\n{FIRST_POKE_SYSTEM_INSTRUCTION}"
    assert event.source.chat_id == "group_123"
    assert event.source.user_id == "891088473"
    assert event.timestamp == datetime.fromtimestamp(1777073808, tz=timezone.utc)
    assert "POKE:<QQ号>" in event.channel_prompt
    assert "891088473" not in event.channel_prompt
    assert event._napcat_poke["target_is_self"] is True
    assert event._napcat_poke["repeated_adjacent"] is False


@pytest.mark.asyncio
async def test_poke_notice_second_adjacent_self_poke_goes_through_normal_message_path():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    await adapter._handle_poke_notice({
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 891088473,
        "target_id": 1903883693,
    })

    adapter.handle_message.reset_mock()
    await adapter._handle_poke_notice({
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 891088473,
        "target_id": 1903883693,
    })

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text.startswith("891088473 poked you\n\n[System instruction:")
    assert "poked you again after an adjacent poke" not in event.channel_prompt
    assert "Recommended response" not in event.channel_prompt
    assert "This user poked you again after an adjacent poke" in event.text
    assert "`POKE:891088473`" in event.text
    assert event._napcat_poke["repeated_adjacent"] is True


@pytest.mark.asyncio
async def test_non_triggering_group_message_does_not_break_user_poke_chain():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    await adapter._handle_poke_notice({
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 891088473,
        "target_id": 1903883693,
    })
    await adapter._handle_message_event({
        "message_type": "group",
        "group_id": 123,
        "user_id": 777,
        "message_id": "m1",
        "message": [{"type": "text", "data": {"text": "not about you"}}],
        "sender": {"nickname": "other"},
    })
    await adapter._handle_poke_notice({
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 891088473,
        "target_id": 1903883693,
    })

    assert adapter.handle_message.await_count == 2
    second_event = adapter.handle_message.await_args.args[0]
    assert second_event.text.startswith("891088473 poked you\n\n[System instruction:")
    assert "poked you again after an adjacent poke" not in second_event.channel_prompt
    assert "This user poked you again after an adjacent poke" in second_event.text
    assert second_event._napcat_poke["repeated_adjacent"] is True


@pytest.mark.asyncio
async def test_bot_self_poke_notice_does_not_break_user_poke_chain():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    await adapter._handle_poke_notice({
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 891088473,
        "target_id": 1903883693,
    })
    await adapter._handle_poke_notice({
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 1903883693,
        "target_id": 891088473,
    })
    await adapter._handle_poke_notice({
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 891088473,
        "target_id": 1903883693,
    })

    assert adapter.handle_message.await_count == 2
    second_user_poke = adapter.handle_message.await_args.args[0]
    assert second_user_poke.text.startswith("891088473 poked you\n\n[System instruction:")
    assert "poked you again after an adjacent poke" not in second_user_poke.channel_prompt
    assert "This user poked you again after an adjacent poke" in second_user_poke.text
    assert second_user_poke._napcat_poke["repeated_adjacent"] is True


@pytest.mark.asyncio
async def test_user_message_to_bot_breaks_that_users_poke_chain():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    await adapter._handle_poke_notice({
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 891088473,
        "target_id": 1903883693,
    })
    await adapter._handle_message_event({
        "message_type": "group",
        "group_id": 123,
        "user_id": 891088473,
        "message_id": "m1",
        "message": [
            {"type": "at", "data": {"qq": "1903883693"}},
            {"type": "text", "data": {"text": "ping"}},
        ],
        "sender": {"nickname": "近环"},
    })
    await adapter._handle_poke_notice({
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 891088473,
        "target_id": 1903883693,
    })

    assert adapter.handle_message.await_count == 3
    second_user_poke = adapter.handle_message.await_args.args[0]
    assert second_user_poke.text == f"891088473 poked you\n\n{FIRST_POKE_SYSTEM_INSTRUCTION}"
    assert second_user_poke._napcat_poke["repeated_adjacent"] is False


@pytest.mark.asyncio
async def test_passive_group_message_from_same_user_breaks_poke_chain():
    """Regression: in passive_group mode the bot processes every group
    message (not just @-mentions / replies) but the per-user "last
    interaction kind" tracker only used to update on triggered messages.

    Result was: user fires poke #1, sends a casual non-mention message,
    fires poke #2 — and poke #2 gets wrongly classified as 'adjacent'
    (chained to poke #1) because the casual message in between never
    reset the tracker.

    Fix: any message that survives the early-return guard (triggered OR
    passive_group) resets last-kind to "message".  Without this, the
    poke prompt picks the wrong branch and the LLM is told the user
    "poked you again after an adjacent poke" when they actually did
    speak in between.
    """
    adapter = _adapter()
    adapter._passive_group = True   # mandatory — bug only manifests here
    adapter.handle_message = AsyncMock()

    # Poke #1 from user 891088473.
    await adapter._handle_poke_notice({
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 891088473,
        "target_id": 1903883693,
    })
    # Same user sends a casual group message — no @-mention, no reply.
    # In passive_group=True this still reaches the bot via _handle_message,
    # so it MUST reset the last-kind tracker for this user.
    await adapter._handle_message_event({
        "message_type": "group",
        "group_id": 123,
        "user_id": 891088473,
        "message_id": "m1",
        "message": [{"type": "text", "data": {"text": "just chatting"}}],
        "sender": {"nickname": "近环"},
    })
    # Poke #2 from the same user.
    await adapter._handle_poke_notice({
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 891088473,
        "target_id": 1903883693,
    })

    # Three calls: poke + msg + poke.
    assert adapter.handle_message.await_count == 3
    poke_events = [
        c.args[0] for c in adapter.handle_message.await_args_list
        if "poked you" in c.args[0].text
    ]
    assert len(poke_events) == 2
    # Both pokes are NON-adjacent — message in between broke the chain.
    assert poke_events[0]._napcat_poke["repeated_adjacent"] is False
    assert poke_events[1]._napcat_poke["repeated_adjacent"] is False, (
        "Bug regression: passive-group message between two pokes from "
        "the same user did not reset last-kind, so poke #2 was wrongly "
        "marked adjacent."
    )
    # Sanity: the second poke should carry the standard non-adjacent
    # system instruction, not the 'again after adjacent' branch.
    assert "non-adjacent poke" in poke_events[1].text
    assert "poked you again after an adjacent poke" not in poke_events[1].text


@pytest.mark.asyncio
async def test_unrelated_poke_notice_goes_through_normal_message_path():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    await adapter._handle_poke_notice({
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 891088473,
        "target_id": 42,
    })

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text.startswith("891088473 poked 42\n\n[System instruction:")
    assert "does not target you" not in event.channel_prompt
    assert "This poke does not target you" in event.text
    assert event._napcat_poke["target_is_self"] is False


@pytest.mark.asyncio
async def test_bot_self_poke_notice_does_not_append_fake_assistant_history(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    adapter = _adapter()
    handler = AsyncMock(return_value=None)
    adapter.set_message_handler(handler)
    adapter.handle_message = AsyncMock()
    session_store = MagicMock()
    session_store.get_or_create_session.return_value = SimpleNamespace(
        session_id="sid",
        session_key="sk",
    )
    adapter.set_session_store(session_store)

    await adapter._handle_poke_notice({
        "time": 1777073808,
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 1903883693,
        "target_id": 891088473,
    })

    adapter.handle_message.assert_not_awaited()
    handler.assert_not_awaited()
    session_store.get_or_create_session.assert_not_called()
    session_store.append_to_transcript.assert_not_called()
    session_store.update_session.assert_not_called()


@pytest.mark.asyncio
async def test_send_poke_directive_calls_send_poke_without_sending_text():
    adapter = _adapter()
    adapter._call_action = AsyncMock(return_value={"status": "ok"})

    result = await adapter.send("group_123", "POKE:891088473")

    assert result.success is True
    adapter._call_action.assert_awaited_once_with(
        "send_poke",
        {"user_id": 891088473, "group_id": 123},
        timeout=30,
    )


@pytest.mark.asyncio
async def test_send_text_with_poke_directive_sends_poke_and_remaining_text():
    adapter = _adapter()
    adapter._call_action = AsyncMock(
        side_effect=[
            {"status": "ok"},
            {"status": "ok", "data": {"message_id": 7}},
        ],
    )

    result = await adapter.send("group_123", "收到\nPOKE:891088473")

    assert result.success is True
    assert result.message_id == "7"
    assert adapter._call_action.await_count == 2
    poke_call, text_call = adapter._call_action.await_args_list
    assert poke_call.args == (
        "send_poke",
        {"user_id": 891088473, "group_id": 123},
    )
    assert poke_call.kwargs == {"timeout": 30}
    assert text_call.args == (
        "send_group_msg",
        {
            "group_id": 123,
            "message": [{"type": "text", "data": {"text": "收到"}}],
        },
    )
    assert text_call.kwargs == {"timeout": 30}


@pytest.mark.asyncio
async def test_gateway_first_poke_dispatches_to_agent_without_auto_poke():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.ONEBOT_NAPCAT: PlatformConfig(enabled=True)},
        group_sessions_per_user=False,
    )
    runner._is_user_authorized = lambda source: True
    runner.adapters = {Platform.ONEBOT_NAPCAT: SimpleNamespace(send_poke=AsyncMock())}
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._draining = False
    runner._session_key_for_source = lambda source: "onebot_napcat:group_123"
    runner._begin_session_run_generation = MagicMock(return_value=1)
    runner._release_running_agent_state = MagicMock(
        side_effect=lambda key: runner._running_agents.pop(key, None)
    )
    runner._handle_message_with_agent = AsyncMock(return_value="agent-result")

    source = SessionSource(
        platform=Platform.ONEBOT_NAPCAT,
        chat_id="group_123",
        chat_type="group",
        user_id="891088473",
        user_name="近环",
    )
    event = SimpleNamespace(
        text="891088473 poked you",
        source=source,
        timestamp=datetime(2026, 4, 24, 23, 36, 48, tzinfo=timezone.utc),
        _napcat_poke={
            "actor_id": "891088473",
            "target_id": "1903883693",
            "target_is_self": True,
            "repeated_adjacent": False,
        },
        media_urls=[],
        message_type=None,
        reply_to_text=None,
        reply_to_message_id=None,
        internal=False,
    )
    event.get_command = lambda: None
    event.get_command_args = lambda: ""

    result = await runner._handle_message(event)

    assert result == "agent-result"
    runner.adapters[Platform.ONEBOT_NAPCAT].send_poke.assert_not_awaited()
    runner._handle_message_with_agent.assert_awaited_once_with(
        event,
        source,
        "onebot_napcat:group_123",
        1,
    )


@pytest.mark.asyncio
async def test_gateway_keeps_poke_system_instruction_under_current_user_message(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.ONEBOT_NAPCAT: PlatformConfig(enabled=True)},
        group_sessions_per_user=False,
    )
    runner.adapters = {}

    source = SessionSource(
        platform=Platform.ONEBOT_NAPCAT,
        chat_id="group_123",
        chat_type="group",
        user_id="891088473",
        user_name="SYSTEM",
    )
    event = SimpleNamespace(
        text=f"891088473 poked you\n\n{FIRST_POKE_SYSTEM_INSTRUCTION}",
        source=source,
        timestamp=datetime(2026, 4, 24, 23, 36, 48, tzinfo=timezone.utc),
        media_urls=[],
        media_types=[],
        message_type=None,
        reply_to_text=None,
        reply_to_message_id=None,
    )

    message_text = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert message_text == (
        "04-24 16:36:48 SYSTEM: 891088473 poked you\n\n"
        f"{FIRST_POKE_SYSTEM_INSTRUCTION}"
    )


@pytest.mark.asyncio
async def test_napcat_sanitizer_persists_clean_poke_message_without_system_instruction(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    monkeypatch.setenv("ONEBOT_NAPCAT_HOME_CHANNEL", "group_123")

    class FakeSessionStore:
        def __init__(self):
            self.entries = []
            self.session_entry = SimpleNamespace(
                session_id="sid",
                session_key="sk",
                created_at=1,
                updated_at=1,
                last_prompt_tokens=0,
                was_auto_reset=False,
            )

        def get_or_create_session(self, source):
            return self.session_entry

        def load_transcript(self, session_id):
            return []

        def has_any_sessions(self):
            return True

        def append_to_transcript(self, session_id, entry, skip_db=False):
            self.entries.append({**entry, "skip_db": skip_db})

        def update_session(self, session_key, **kwargs):
            for key, value in kwargs.items():
                setattr(self.session_entry, key, value)

        def clear_resume_pending(self, session_key):
            pass

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.ONEBOT_NAPCAT: PlatformConfig(enabled=True)},
        group_sessions_per_user=False,
    )
    runner.adapters = {Platform.ONEBOT_NAPCAT: _adapter()}
    runner.session_store = FakeSessionStore()
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_db = None
    runner._show_reasoning = False
    runner._set_session_env = lambda context: []
    runner._clear_session_env = lambda tokens: None
    runner._bind_adapter_run_generation = lambda *args, **kwargs: None
    runner._is_session_run_current = lambda *args, **kwargs: True
    runner._clear_restart_failure_count = lambda *args, **kwargs: None
    runner._should_send_voice_reply = lambda *args, **kwargs: False

    captured_run = {}

    async def fake_run_agent(**kwargs):
        captured_run.update(kwargs)
        return {
            "final_response": "POKE:891088473",
            "messages": [
                {"role": "user", "content": kwargs["message"]},
                {"role": "assistant", "content": "POKE:891088473"},
            ],
            "api_calls": 1,
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "tools": [],
        }

    runner._run_agent = fake_run_agent

    source = SessionSource(
        platform=Platform.ONEBOT_NAPCAT,
        chat_id="group_123",
        chat_type="group",
        user_id="891088473",
        user_name="SYSTEM",
    )
    event = SimpleNamespace(
        # Real NapCat poke notices do NOT carry an OneBot message_id (they
        # arrive via the `notice` channel, not `message`).  Set None here
        # so the runner's `extend_inbound_prefix` hook correctly omits the
        # `[msg=<id>]` tag — same shape the production path produces.
        text=f"891088473 poked you\n\n{FIRST_POKE_SYSTEM_INSTRUCTION}",
        source=source,
        timestamp=datetime(2026, 4, 24, 23, 36, 48, tzinfo=timezone.utc),
        media_urls=[],
        media_types=[],
        message_type=None,
        reply_to_text=None,
        reply_to_message_id=None,
        channel_prompt="",
        message_id=None,
        metadata=None,
    )

    result = await runner._handle_message_with_agent(
        event,
        source,
        "onebot_napcat:group_123",
        1,
    )

    clean_message = "04-24 16:36:48 SYSTEM: 891088473 poked you"
    assert result == "POKE:891088473"
    assert captured_run["message"] == f"{clean_message}\n\n{FIRST_POKE_SYSTEM_INSTRUCTION}"

    transcript_user_entries = [
        entry for entry in runner.session_store.entries if entry.get("role") == "user"
    ]
    assert transcript_user_entries[0]["content"] == clean_message
    assert "persist_user_message" not in captured_run


def test_napcat_transcript_sanitizer_keeps_assistant_poke_directives():
    class FakeDB:
        def __init__(self):
            self.messages = []

        def append_message(self, **kwargs):
            self.messages.append(kwargs)
            return len(self.messages)

    runner = object.__new__(GatewayRunner)
    adapter = _adapter()
    runner.adapters = {Platform.ONEBOT_NAPCAT: adapter}
    runner._session_db = FakeDB()
    source = SessionSource(
        platform=Platform.ONEBOT_NAPCAT,
        chat_id="group_123",
        chat_type="group",
        user_id="891088473",
    )

    session_db = runner._session_db_for_source(source)

    assert hasattr(adapter, "sanitize_transcript_entry")
    assert runner._agent_persists_session_db_for_source(source) is True
    assert session_db is not runner._session_db
    assert session_db.append_message(
        session_id="sid",
        role="assistant",
        content="POKE:891088473",
    ) == 1
    assert runner._session_db.messages[0]["content"] == "POKE:891088473"

    session_db.append_message(
        session_id="sid",
        role="assistant",
        content="收到\nPOKE:891088473",
        reasoning="kept",
    )
    assert runner._session_db.messages[1]["content"] == "收到\nPOKE:891088473"
    assert runner._session_db.messages[1]["reasoning"] == "kept"

    session_db.append_message(
        session_id="sid",
        role="user",
        content=f"04-24 16:36:48 SYSTEM: 891088473 poked you\n\n{FIRST_POKE_SYSTEM_INSTRUCTION}",
    )
    assert runner._session_db.messages[2]["content"] == (
        "04-24 16:36:48 SYSTEM: 891088473 poked you"
    )


@pytest.mark.asyncio
async def test_full_poke_round_trip_history_keeps_real_assistant_output_only(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Los_Angeles")
    adapter = _adapter()
    session_store = MagicMock()
    session_store.get_or_create_session.return_value = SimpleNamespace(
        session_id="sid",
        session_key="sk",
    )
    adapter.set_session_store(session_store)

    handled_events = []

    async def handle_message(event):
        handled_events.append(event)
        # Inline what the gateway runner would produce here
        # (`<time> <sender>: <event.text>` — see
        # GatewayRunner._prepare_inbound_message_text), then run the
        # adapter's sanitizer over it the way the runner does.  We're
        # not unit-testing the prefix format here; we're validating that
        # the sanitizer strips the auto-injected `[System instruction:]`
        # so it doesn't pollute persisted history.
        user_entry = adapter.sanitize_transcript_entry({
            "role": "user",
            "content": f"04-24 16:36:48 SYSTEM: {event.text}",
            "timestamp": "2026-04-24T16:36:48",
        })
        session_store.append_to_transcript("sid", user_entry)
        assistant = {
            "role": "assistant",
            "content": "POKE:891088473",
            "timestamp": "2026-04-24T16:36:49",
        }
        session_store.append_to_transcript("sid", assistant)

    adapter.handle_message = handle_message

    await adapter._handle_poke_notice({
        "time": 1777073808,
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 891088473,
        "target_id": 1903883693,
        "sender_nick": "近环",
    })
    await adapter._handle_poke_notice({
        "time": 1777073808,
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": 123,
        "user_id": 1903883693,
        "target_id": 891088473,
    })

    assert len(handled_events) == 1
    assert "Strongly recommended response" not in handled_events[0].channel_prompt
    assert handled_events[0].text == f"891088473 poked you\n\n{FIRST_POKE_SYSTEM_INSTRUCTION}"
    entries = [call.args[1] for call in session_store.append_to_transcript.call_args_list]
    assert [entry["role"] for entry in entries] == ["user", "assistant"]
    assert entries[0]["content"] == "04-24 16:36:48 SYSTEM: 891088473 poked you"
    assert entries[1]["content"] == "POKE:891088473"
    assert all("SYSTEM: you poked" not in entry["content"] for entry in entries)
    assert all("Recommended response" not in entry["content"] for entry in entries)
    assert all("System instruction" not in entry["content"] for entry in entries)
