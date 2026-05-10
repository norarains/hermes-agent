"""Sparrow tests pinning the **emission contract** of high-level events
that show up in ``sparrow.log``.

Why this file exists
--------------------
Sparrow events (``model_call_starting``, ``media_send_finished``,
``memory_prefetch_send``, ...) are how the operator triages what's
happening at runtime — they're the high-level timeline that lets us
debug without scrolling through ``agent.log``.  But their emission is
*incidental* in the source: a one-line ``_sparrow_event(...)`` tucked
inside a larger function.  A refactor that drops or renames any of
those calls is silent — the event just stops appearing in the log,
and we only notice when we *next try to debug* something and find the
trail goes cold.

This file pins each event family by exercising the smallest unit that
emits it AND asserting both the event name and the expected fields.

Coverage scope (events tested here):
  * ``media_send_starting`` / ``media_send_finished``  (status=ok|failed|error, with error= field on failure)
  * ``memory_prefetch_send`` / ``memory_prefetch_receive``  (query/content fields)
  * ``last_prefetch_query`` getter (the helper that backs the receive event's query field)

Other event families pinned in their own dedicated files (kept here as
a reference so a future operator looking for "where do I add a test
for X event" finds the right file):
  * ``napcat_send_*``, ``silent``           → ``test_onebot_napcat_inline_send.py``
  * ``cron_run_*``                           → ``test_sparrow_cron_visibility.py``
  * ``memory_unhealthy``                     → ``test_sparrow_memory_unhealthy_notification.py``
  * ``openviking_commit_*``                  → emitted by openviking-server, tested in upstream openviking repo
  * ``user_msg`` / ``assistant_msg`` /
    ``model_call_*`` / ``tool_call_*`` /
    ``gateway_*`` / ``napcat_starting`` /
    ``openviking_starting`` / ``interrupted`` /
    ``user_sent_new_message``               → integration-tested only (require driving the
                                              full agent loop or process lifecycle)
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub telegram package so importing gateway.platforms.base doesn't
# require the real PTB install (matches the pattern used by sibling
# tests in this directory).
_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import BasePlatformAdapter, SendResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: capture sparrow events by monkeypatching the log_event symbol
# ---------------------------------------------------------------------------


def _install_event_spy(monkeypatch) -> List[Tuple[str, Dict[str, Any]]]:
    """Replace ``gateway.sparrow_log.log_event`` with a list-recording
    spy.  Returns the list (each entry is ``(event_name, fields)``).
    Lazy imports inside source code pick this up because they read the
    module attribute at call time."""
    captured: List[Tuple[str, Dict[str, Any]]] = []
    fake_module = types.ModuleType("gateway.sparrow_log")
    fake_module.log_event = lambda event, **fields: captured.append((event, dict(fields)))
    monkeypatch.setitem(sys.modules, "gateway.sparrow_log", fake_module)
    return captured


def _make_minimal_adapter() -> BasePlatformAdapter:
    """A concrete BasePlatformAdapter with all attachment-send methods
    mocked to return success.  Used to drive ``send_response_with_media``
    without touching any real platform API."""

    class _StubAdapter(BasePlatformAdapter):
        name = "stub"

        async def connect(self) -> bool:
            return True

        async def disconnect(self) -> None:
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True, message_id="text_id")

        async def get_chat_info(self, chat_id):
            return {"name": chat_id, "type": "group"}

    config = MagicMock()
    platform = MagicMock(value="stub")
    adapter = _StubAdapter(config, platform)
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True, message_id="text_id"))
    adapter.send_image = AsyncMock(return_value=SendResult(success=True))
    adapter.send_animation = AsyncMock(return_value=SendResult(success=True))
    adapter.send_image_file = AsyncMock(return_value=SendResult(success=True))
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True))
    adapter.send_video = AsyncMock(return_value=SendResult(success=True))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True))
    adapter.play_tts = AsyncMock()
    return adapter


# ---------------------------------------------------------------------------
# media_send_* events — emitted from BasePlatformAdapter.send_response_with_media
# for every URL image / MEDIA: tag / bare-local-file path detected in the
# response.  Each loop emits a `_starting` then a `_finished` (success or
# failure variant).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_send_events_pair_for_image_url(monkeypatch):
    """Markdown image URL ``![alt](https://...)`` → starting + finished
    pair, with ``type=image`` + the URL in the starting event."""
    captured = _install_event_spy(monkeypatch)
    adapter = _make_minimal_adapter()

    await adapter.send_response_with_media(
        chat_id="c1",
        response="here ![cat](https://example.com/cat.png)",
    )

    media_events = [(e, f) for e, f in captured if e.startswith("media_send_")]
    assert [e for e, _ in media_events] == ["media_send_starting", "media_send_finished"]
    starting_fields = media_events[0][1]
    assert starting_fields["chat"] == "c1"
    assert starting_fields["type"] == "image"
    assert starting_fields["target"] == "https://example.com/cat.png"
    finished_fields = media_events[1][1]
    assert finished_fields["status"] == "ok"


@pytest.mark.asyncio
async def test_media_send_events_pair_for_animation_url(monkeypatch):
    """``.gif`` URL routes through ``send_animation`` and emits with
    ``type=animation`` so an operator searching "what kind of media did
    this turn ship" can answer from sparrow.log alone."""
    captured = _install_event_spy(monkeypatch)
    adapter = _make_minimal_adapter()

    await adapter.send_response_with_media(
        chat_id="c1",
        response="![dance](https://example.com/dance.gif)",
    )

    starting = next(f for e, f in captured if e == "media_send_starting")
    assert starting["type"] == "animation"


@pytest.mark.asyncio
async def test_media_send_events_route_by_media_tag_extension(monkeypatch, tmp_path):
    """``MEDIA:<path>`` tags route by file extension.  This pins the
    ``type=`` field for each variant (image/voice/video/document) so a
    refactor of the dispatch table can't silently relabel them."""
    captured = _install_event_spy(monkeypatch)
    adapter = _make_minimal_adapter()

    img = tmp_path / "a.png"; img.write_bytes(b"\x89PNG")
    audio = tmp_path / "b.opus"; audio.write_bytes(b"OggS")
    video = tmp_path / "c.mp4"; video.write_bytes(b"\x00")
    other = tmp_path / "d.pdf"; other.write_bytes(b"%PDF")

    await adapter.send_response_with_media(
        chat_id="c1",
        response=f"MEDIA:{img}\nMEDIA:{audio}\nMEDIA:{video}\nMEDIA:{other}",
    )

    starting_types = [
        f["type"] for e, f in captured if e == "media_send_starting"
    ]
    assert starting_types == ["image", "voice", "video", "document"]


@pytest.mark.asyncio
async def test_media_send_finished_status_failed_on_send_error(monkeypatch, tmp_path):
    """When the underlying send_image_file returns ``success=False``,
    the finished event MUST carry ``status=failed`` and the error
    message — this is the operator's only signal that a media send was
    attempted but rejected by the platform (e.g. the ENOENT case where
    the agent hallucinated a file path)."""
    captured = _install_event_spy(monkeypatch)
    adapter = _make_minimal_adapter()
    adapter.send_image_file = AsyncMock(
        return_value=SendResult(success=False, error="ENOENT: no such file"),
    )

    img = tmp_path / "ghost.jpg"; img.write_bytes(b"\xff\xd8\xff")

    await adapter.send_response_with_media(
        chat_id="c1",
        response=f"MEDIA:{img}",
    )

    finished = next(f for e, f in captured if e == "media_send_finished")
    assert finished["status"] == "failed"
    assert "ENOENT" in finished["error"]


@pytest.mark.asyncio
async def test_media_send_finished_status_error_on_exception(monkeypatch, tmp_path):
    """When the send method raises, the finished event must carry
    ``status=error`` (distinct from ``status=failed`` for known
    rejection) so operators can tell network/auth errors apart from
    "platform said no"."""
    captured = _install_event_spy(monkeypatch)
    adapter = _make_minimal_adapter()
    adapter.send_image_file = AsyncMock(side_effect=RuntimeError("connection refused"))

    img = tmp_path / "bad.jpg"; img.write_bytes(b"\xff\xd8\xff")

    await adapter.send_response_with_media(
        chat_id="c1",
        response=f"MEDIA:{img}",
    )

    finished = next(f for e, f in captured if e == "media_send_finished")
    assert finished["status"] == "error"
    assert "connection refused" in finished["error"]


@pytest.mark.asyncio
async def test_media_send_starting_omits_content_text(monkeypatch, tmp_path):
    """Starting event must NOT include the response text in any field —
    only the resolved attachment target.  Otherwise a long agent
    response or user message would inflate every sparrow.log line."""
    captured = _install_event_spy(monkeypatch)
    adapter = _make_minimal_adapter()
    img = tmp_path / "x.jpg"; img.write_bytes(b"\xff\xd8\xff")

    long_text = "lorem ipsum " * 200
    await adapter.send_response_with_media(
        chat_id="c1",
        response=f"{long_text}\nMEDIA:{img}",
    )

    starting = next(f for e, f in captured if e == "media_send_starting")
    # No field that contains the prose body
    for value in starting.values():
        if isinstance(value, str):
            assert "lorem ipsum lorem ipsum lorem ipsum" not in value


# ---------------------------------------------------------------------------
# memory_prefetch_* events — the cross-turn observability fix.  Tests live
# here (rather than in test_sparrow_send_response_with_media or similar)
# because the events fire from the agent's run_conversation prefetch
# block, not from any adapter method.
# ---------------------------------------------------------------------------


def _install_memory_manager_with_query_capture(monkeypatch):
    """Build a MemoryManager that captures (query, calls) for both
    ``queue_prefetch_all`` and ``prefetch_all`` so we can assert what
    the agent passed to each side of the handshake."""
    from agent.memory_manager import MemoryManager

    class _SpyProvider:
        name = "spy"
        def __init__(self):
            self.queued: List[str] = []
            self.read_count = 0
            self._cached_query = ""
            self._cached_content = ""

        def is_available(self):
            return True

        def initialize(self, **_):
            return None

        def get_config_schema(self):
            return []

        def system_prompt_block(self):
            return ""

        def prefetch(self, query, *, session_id=""):
            self.read_count += 1
            return self._cached_content

        def queue_prefetch(self, query, *, session_id=""):
            self.queued.append(query)
            self._cached_query = query
            self._cached_content = f"## Stub\n[1.0] match for {query} (viking://stub.md)"

        def last_prefetch_query(self) -> str:
            return self._cached_query

        def sync_turn(self, *_a, **_k):
            return None

        def get_tool_schemas(self):
            return []

        def handle_tool_call(self, *_a, **_k):
            return ""

    provider = _SpyProvider()
    manager = MemoryManager.__new__(MemoryManager)
    manager._providers = [provider]
    manager._tool_to_provider = {}
    return manager, provider


def test_memory_manager_queue_prefetch_all_passes_query_to_provider(monkeypatch):
    """``queue_prefetch_all(query)`` must forward ``query`` verbatim to
    every provider — that's the contract the
    ``memory_prefetch_send`` event relies on (its query field is
    ``original_user_message``, which IS this query)."""
    manager, provider = _install_memory_manager_with_query_capture(monkeypatch)
    manager.queue_prefetch_all("turn N user msg")
    assert provider.queued == ["turn N user msg"]


def test_memory_manager_last_prefetch_query_returns_provider_query(monkeypatch):
    """The ``memory_prefetch_receive`` event's ``query`` field is
    populated by ``manager.last_prefetch_query()``.  Pin that this
    returns the most recent query passed to ``queue_prefetch_all`` —
    NOT whatever the current turn happens to be asking about — so the
    receive event's query+content actually correspond."""
    manager, provider = _install_memory_manager_with_query_capture(monkeypatch)

    # Before any queue → empty.
    assert manager.last_prefetch_query() == ""

    manager.queue_prefetch_all("first")
    assert manager.last_prefetch_query() == "first"

    manager.queue_prefetch_all("second")
    assert manager.last_prefetch_query() == "second"


def test_memory_manager_last_prefetch_query_skips_providers_without_method(monkeypatch):
    """When a provider doesn't expose ``last_prefetch_query`` (older
    plugin, or one that doesn't queue), it must be skipped silently —
    not raise — so a future provider plugin without the method can be
    added without breaking the receive-event field."""
    from agent.memory_manager import MemoryManager

    class _BareProvider:
        name = "bare"
        def is_available(self): return True
        def initialize(self, **_): return None
        def get_config_schema(self): return []
        def system_prompt_block(self): return ""
        def prefetch(self, *_a, **_k): return ""
        def queue_prefetch(self, *_a, **_k): return None
        def sync_turn(self, *_a, **_k): return None
        def get_tool_schemas(self): return []
        def handle_tool_call(self, *_a, **_k): return ""

    manager = MemoryManager.__new__(MemoryManager)
    manager._providers = [_BareProvider()]
    manager._tool_to_provider = {}

    assert manager.last_prefetch_query() == ""


def test_memory_manager_last_prefetch_query_returns_first_nonempty(monkeypatch):
    """With multiple providers, return the FIRST one that has a query —
    skip empties.  This documents the resolution order so a future
    plugin that always returns something doesn't accidentally shadow a
    more recently queued one."""
    from agent.memory_manager import MemoryManager

    class _ProviderEmpty:
        name = "empty"
        def is_available(self): return True
        def initialize(self, **_): return None
        def get_config_schema(self): return []
        def system_prompt_block(self): return ""
        def prefetch(self, *_a, **_k): return ""
        def queue_prefetch(self, *_a, **_k): return None
        def sync_turn(self, *_a, **_k): return None
        def get_tool_schemas(self): return []
        def handle_tool_call(self, *_a, **_k): return ""
        def last_prefetch_query(self): return ""

    class _ProviderHasQuery(_ProviderEmpty):
        name = "has"
        def last_prefetch_query(self): return "real query"

    manager = MemoryManager.__new__(MemoryManager)
    manager._providers = [_ProviderEmpty(), _ProviderHasQuery()]
    manager._tool_to_provider = {}

    assert manager.last_prefetch_query() == "real query"


def test_openviking_provider_records_query_alongside_result(monkeypatch):
    """Live test of the openviking provider's cache invariant: every
    successful ``queue_prefetch`` call must update both
    ``_prefetch_result`` AND ``_prefetch_query`` under the lock so a
    subsequent ``last_prefetch_query()`` call returns the query that
    actually produced the cached content."""
    from plugins.memory.openviking import OpenVikingMemoryProvider

    captured_payloads: List[Dict[str, Any]] = []

    class _StubClient:
        def post(self, path, payload, **_):
            captured_payloads.append(payload)
            return {
                "result": {
                    "memories": [
                        {"uri": "viking://stub.md", "abstract": "abstract X", "score": 0.9},
                    ],
                    "resources": [],
                }
            }

    provider = OpenVikingMemoryProvider.__new__(OpenVikingMemoryProvider)
    provider._endpoint = "http://localhost:1933"
    provider._api_key = ""
    provider._account = "default"
    provider._user = "default"
    provider._agent = "hermes"
    provider._client = MagicMock()
    provider._notify_user = lambda **_: None
    provider._session_id = "test_session"
    provider._turn_count = 0
    provider._sync_thread = None
    provider._prefetch_result = ""
    provider._prefetch_query = ""
    import threading
    provider._prefetch_lock = threading.Lock()
    provider._prefetch_thread = None

    # Patch the client constructor inside queue_prefetch to return the stub.
    import plugins.memory.openviking as ov
    monkeypatch.setattr(ov, "_VikingClient", lambda *a, **kw: _StubClient())

    provider.queue_prefetch("query A", session_id="s1")
    if provider._prefetch_thread is not None:
        provider._prefetch_thread.join(timeout=5.0)

    assert provider.last_prefetch_query() == "query A"
    assert "abstract X" in provider._prefetch_result


# ---------------------------------------------------------------------------
# model_call_finished `text` field — preview of what the model emitted.
# Pins the helper used to populate the field.  Without this the operator
# scanning sparrow.log would see only token counts and have to scroll to
# the next [assistant_msg] OR [tool_call_starting] event to find out what
# the model decided.
# ---------------------------------------------------------------------------


def test_summarize_model_response_extracts_openai_text_content():
    """Plain chat reply on the OpenAI shape:
    ``response.choices[0].message.content`` is a string."""
    from run_agent import _summarize_model_response_for_event

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="hello, this is a reply", tool_calls=None))]
    out = _summarize_model_response_for_event(response)
    assert out == "hello, this is a reply"


def test_summarize_model_response_renders_openai_tool_call_only():
    """Tool-call-only iteration (no prose) on the OpenAI shape.
    ``message.content`` is None / empty; ``message.tool_calls`` carries
    the call.  Expected output: ``→ tool_name(args)``."""
    from run_agent import _summarize_model_response_for_event

    tool_call = MagicMock()
    tool_call.function = MagicMock(
        name="read_file",  # MagicMock's name attr is special; set explicitly below
        arguments='{"path": "/tmp/foo.py"}',
    )
    tool_call.function.name = "read_file"

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=None, tool_calls=[tool_call]))]
    out = _summarize_model_response_for_event(response)
    assert out.startswith("→ read_file(")
    assert "path=" in out
    assert "/tmp/foo.py" in out


def test_summarize_model_response_renders_openai_text_plus_tool_calls():
    """Mixed iteration: prose then tool calls.  Prose first, then one
    line per tool call.  Operators reading the log see both halves."""
    from run_agent import _summarize_model_response_for_event

    tc1 = MagicMock()
    tc1.function = MagicMock(arguments='{"pattern": "foo"}')
    tc1.function.name = "search_files"

    tc2 = MagicMock()
    tc2.function = MagicMock(arguments='{"path": "bar.py"}')
    tc2.function.name = "read_file"

    response = MagicMock()
    response.choices = [
        MagicMock(message=MagicMock(
            content="Let me search for foo first then read bar.py",
            tool_calls=[tc1, tc2],
        )),
    ]
    out = _summarize_model_response_for_event(response)
    lines = out.split("\n")
    assert lines[0].startswith("Let me search for foo first")
    assert lines[1].startswith("→ search_files(")
    assert lines[2].startswith("→ read_file(")


def test_summarize_model_response_renders_anthropic_text_blocks():
    """Anthropic shape: ``response.content`` is a list of blocks."""
    from run_agent import _summarize_model_response_for_event

    response = MagicMock()
    response.choices = None  # not OpenAI shape
    response.content = [
        {"type": "text", "text": "thinking out loud..."},
        {"type": "tool_use", "name": "viking_search", "input": {"query": "蛋包饭"}},
    ]
    out = _summarize_model_response_for_event(response)
    assert "thinking out loud..." in out
    assert "→ viking_search(" in out
    assert "蛋包饭" in out


def test_summarize_model_response_falls_back_to_reasoning_when_no_text():
    """Pure reasoning response (no tool call, no chat content) — the
    operator should still see WHAT the model thought instead of an
    empty line.  Real example: a deepseek reasoner mid-chain emitting
    only ``reasoning_content``."""
    from run_agent import _summarize_model_response_for_event

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(
        content=None,
        tool_calls=None,
        reasoning_content="The user asked about X. I should look at Y.",
        reasoning=None,
    ))]
    out = _summarize_model_response_for_event(response)
    assert out.startswith("[reasoning]")
    assert "look at Y" in out


def test_summarize_model_response_truncates_long_string_args():
    """Tool-call arguments with a long string field render with the
    string truncated and a ``…`` marker so one giant arg doesn't blow
    past the 512-char preview cap."""
    from run_agent import _summarize_model_response_for_event

    long_path = "/very/long/path/that/keeps/going/" * 5
    tc = MagicMock()
    tc.function = MagicMock(arguments=f'{{"path": "{long_path}"}}')
    tc.function.name = "read_file"

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=None, tool_calls=[tc]))]
    out = _summarize_model_response_for_event(response)
    assert "→ read_file(" in out
    assert "…" in out  # truncation marker on the path


# ---------------------------------------------------------------------------
# Streaming-call interrupt abandonment — when the agent's
# ``_interrupt_requested`` flag is set, the streaming call must NOT
# return a partial mock response built from accumulated chunks.  The
# operator's mental model is: ``stop`` means "drop this call entirely;
# don't pretend it produced anything".
# ---------------------------------------------------------------------------


def test_call_chat_completions_returns_none_when_interrupted_mid_stream(monkeypatch):
    """Pin the new contract of ``_call_chat_completions``: when the
    inner for-loop breaks because ``_interrupt_requested`` flipped,
    the function returns ``None`` instead of constructing a
    SimpleNamespace from the partial chunks.  Without this, the caller
    logs ``model_call_finished`` with whatever bytes happened to land
    before the break — misleading every operator reading sparrow.log
    into thinking the model produced output that actually went
    nowhere."""
    import inspect
    import re
    from run_agent import AIAgent

    src = inspect.getsource(AIAgent._interruptible_streaming_api_call)
    # Whitespace-tolerant search for the guard:
    #     if self._interrupt_requested:
    #         return None
    # The chat_completions branch must short-circuit *before* it
    # constructs a SimpleNamespace from accumulated chunks.
    pattern = re.compile(
        r"if\s+self\._interrupt_requested\s*:\s*\n\s+return\s+None",
    )
    matches = pattern.findall(src)
    assert len(matches) >= 2, (
        "Expected at least 2 ``if self._interrupt_requested: return None``"
        " guards inside _interruptible_streaming_api_call (one for"
        " chat_completions, one for anthropic), found"
        f" {len(matches)}.  Without these, partial mock_responses leak"
        " past the break and show up as model_call_finished with"
        " phantom content."
    )


def test_outer_streaming_wrapper_raises_on_inner_returned_none(monkeypatch):
    """Pin the second half of the contract: when the inner thread
    returns ``None`` because it broke on interrupt, the OUTER polling
    wrapper detects this and raises ``InterruptedError`` instead of
    returning ``None`` to the caller.  The caller expects to either
    receive a real response object OR see InterruptedError — never
    ``None``, which would crash downstream code that does
    ``response.choices[0]``."""
    import inspect
    from run_agent import AIAgent

    src = inspect.getsource(AIAgent._interruptible_streaming_api_call)
    # The guard checks both `_interrupt_requested` and `result["response"] is None`
    # to cover the race: inner thread broke + completed before the outer
    # poll could observe `_interrupt_requested`.
    assert 'result["response"] is None' in src, (
        "Outer polling wrapper missing the inner-returned-None guard."
    )
    assert "Agent interrupted; streaming call abandoned" in src, (
        "Outer wrapper must raise InterruptedError with a clear "
        "message so the trace points the operator straight at the "
        "abandonment path (not at some downstream None-deref)."
    )


def test_model_call_finished_logs_status_interrupted_on_abandoned_call(monkeypatch):
    """Pin the third half: when the streaming call raises
    ``InterruptedError``, ``model_call_finished`` MUST fire with
    ``status=interrupted`` and zeroed-out token / text fields.

    Otherwise a future operator would see no ``model_call_finished``
    event for that iteration at all (because the exception bypassed
    the normal log path) AND no marker that the call was abandoned —
    they'd just see a missing entry and have to guess.  Explicit
    ``status=interrupted`` is the contract that makes the timeline
    self-explanatory."""
    import inspect
    from run_agent import AIAgent

    src = inspect.getsource(AIAgent.run_conversation)
    # The except-InterruptedError block in run_conversation that wraps
    # the streaming/non-streaming call must emit a finished event with
    # status=interrupted before re-raising.
    assert "except InterruptedError:" in src, (
        "Missing InterruptedError catch around the streaming/non-"
        "streaming API call in run_conversation."
    )
    assert 'status="interrupted"' in src, (
        "InterruptedError handler must log model_call_finished with "
        "status='interrupted' so the timeline reflects abandonment."
    )


def test_summarize_model_response_returns_empty_on_unrecognized_shape():
    """Defensive: if the response object has no choices and no content,
    return ``""`` cleanly — never raise.  A future provider with a
    different response shape must not break the model_call_finished
    event emission."""
    from run_agent import _summarize_model_response_for_event

    response = MagicMock(spec=[])  # no attributes
    out = _summarize_model_response_for_event(response)
    assert out == ""


def test_openviking_provider_records_empty_query_on_no_results(monkeypatch):
    """When the search returns zero hits, ``last_prefetch_query`` must
    STILL report the query that ran — empty content is a legitimate
    outcome and the operator needs to see *what* was searched in the
    log, not a blank field that looks like the prefetch never fired."""
    from plugins.memory.openviking import OpenVikingMemoryProvider

    class _EmptyClient:
        def post(self, path, payload, **_):
            return {"result": {"memories": [], "resources": []}}

    provider = OpenVikingMemoryProvider.__new__(OpenVikingMemoryProvider)
    provider._endpoint = "http://localhost:1933"
    provider._api_key = ""
    provider._account = "default"
    provider._user = "default"
    provider._agent = "hermes"
    provider._client = MagicMock()
    provider._notify_user = lambda **_: None
    provider._session_id = "test_session"
    provider._turn_count = 0
    provider._sync_thread = None
    provider._prefetch_result = ""
    provider._prefetch_query = ""
    import threading
    provider._prefetch_lock = threading.Lock()
    provider._prefetch_thread = None

    import plugins.memory.openviking as ov
    monkeypatch.setattr(ov, "_VikingClient", lambda *a, **kw: _EmptyClient())

    provider.queue_prefetch("no-hit query", session_id="s1")
    if provider._prefetch_thread is not None:
        provider._prefetch_thread.join(timeout=5.0)

    assert provider.last_prefetch_query() == "no-hit query"
    assert provider._prefetch_result == ""
