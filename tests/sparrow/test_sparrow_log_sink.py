import importlib.util
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from gateway import sparrow_log


def _reset_sparrow_log_state() -> None:
    sparrow_log.stop_event_sink()
    logger = logging.getLogger("sparrow")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    sparrow_log._initialized = False
    sparrow_log._log_path = None


def _post(endpoint: str, token: str, payload: dict) -> int:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.status


def _load_openviking_sparrow_events_module():
    helper_path = (
        Path(__file__).resolve().parents[3]
        / "openviking"
        / "openviking"
        / "sparrow_events.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_openviking_sparrow_events", helper_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def clean_sparrow_log_state():
    _reset_sparrow_log_state()
    yield
    _reset_sparrow_log_state()


def test_event_sink_writes_authenticated_event(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sparrow_log.init_sparrow_log()

    endpoint = sparrow_log.start_event_sink()
    assert endpoint is not None
    token = os.environ["SPARROW_LOG_TOKEN"]

    assert _post(
        endpoint,
        token,
        {
            "event": "openviking_commit_starting",
            "fields": {"session_id": "session-1", "messages": 2},
        },
    ) == 204

    sparrow_log.stop_event_sink()
    text = (tmp_path / "logs" / "sparrow.log").read_text(encoding="utf-8")
    assert "[openviking_commit_starting]" in text
    assert "session_id=session-1" in text
    assert "messages=2" in text
    assert "SPARROW_LOG_ENDPOINT" not in os.environ
    assert "SPARROW_LOG_TOKEN" not in os.environ


def test_event_sink_rejects_missing_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sparrow_log.init_sparrow_log()
    endpoint = sparrow_log.start_event_sink()
    assert endpoint is not None

    request = urllib.request.Request(
        endpoint,
        data=json.dumps({"event": "openviking_commit_starting"}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(request, timeout=2)
    assert exc_info.value.code == 403

    sparrow_log.stop_event_sink()
    text = (tmp_path / "logs" / "sparrow.log").read_text(encoding="utf-8")
    assert "[openviking_commit_starting]" not in text


def test_openviking_helper_posts_to_sparrow_sink(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sparrow_log.init_sparrow_log()
    assert sparrow_log.start_event_sink() is not None

    openviking_events = _load_openviking_sparrow_events_module()
    openviking_events.emit_sparrow_event(
        "openviking_commit_finished",
        session_id="session-2",
        status="ok",
        memories=2,
        files=7,
    )

    sparrow_log.stop_event_sink()
    text = (tmp_path / "logs" / "sparrow.log").read_text(encoding="utf-8")
    assert "[openviking_commit_finished]" in text
    assert "session_id=session-2" in text
    assert "status=ok" in text
    assert "memories=2" in text
    assert "files=7" in text
    assert "archived=" not in text
    assert "archive_uri=" not in text
    assert "task_id=" not in text
    assert "messages=" not in text
    assert "trace_id=" not in text
