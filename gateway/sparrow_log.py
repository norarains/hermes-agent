"""Sparrow operator log — high-level events, one line per event.

This is an ADDITIONAL log alongside agent.log / errors.log; it does NOT
replace, intercept, or reconfigure any existing logger.  Its purpose
is to give the operator (and any agent inspecting the system) a clean
high-level timeline of what the bot is doing — gateway lifecycle,
backend services starting up, and every user→assistant message pair —
without the noise of every internal API call.

The file rewrites on each ``hermes gateway run`` (operator can rely on
"the top of this file = this run's startup, the bottom = right now").
A documentation preamble is written first so anyone (human or LLM)
opening sparrow.log can see what the sibling logs are for.
"""

from __future__ import annotations

import logging
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional


_LOGGER_NAME = "sparrow"
_FILE_NAME = "sparrow.log"
_SINK_ENDPOINT_ENV = "SPARROW_LOG_ENDPOINT"
_SINK_TOKEN_ENV = "SPARROW_LOG_TOKEN"
_SINK_PATH = "/events"

_internal_logger = logging.getLogger(__name__)

_lock = threading.Lock()
_initialized = False
_log_path: Optional[Path] = None
_sink_server: Optional[ThreadingHTTPServer] = None
_sink_thread: Optional[threading.Thread] = None
_sink_endpoint: Optional[str] = None
_sink_token: Optional[str] = None


_PREAMBLE = """\
# Sparrow runtime log — high-level events for operator monitoring.
#
# Relevant sibling files (paths inside the container):
#
#   /opt/data/logs/sparrow.log
#       This file.  High-level event timeline.  Rewritten per run.
#
#   /opt/data/logs/agent.log
#       Full hermes-agent runtime log: every adapter call, plugin
#       method, model API request.  Rotates on size; ``agent.log.1``
#       is the previous rotation.  Use for deep debugging.
#
#   /opt/data/logs/errors.log
#       Errors only (subset of agent.log) for fast triage.  Also
#       rotates; ``errors.log.1`` is the previous rotation.
#
#   /opt/data/.openviking/data/log/openviking.log
#       openviking-server (memory backend) runtime log: HTTP requests,
#       memory extraction phases, vector index ops, embedding calls.
#       Enabled via ov.conf ``log.output="file"``.
#
# ─────────────────────────────────────────────────────────────────────
"""


def _resolve_log_path() -> Path:
    """``$HERMES_HOME/logs/sparrow.log``, sitting next to agent.log."""
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    return Path(home) / "logs" / _FILE_NAME


def init_sparrow_log() -> None:
    """Truncate (or create) ``sparrow.log``, write the preamble, and
    install a file handler on the ``sparrow`` logger.

    Idempotent — calling twice in one process is a no-op after the
    first init.  Never raises: file IO failure logs to ``agent.log``
    via the module logger and proceeds without the sparrow log.
    """
    global _initialized, _log_path
    with _lock:
        if _initialized:
            return
        try:
            path = _resolve_log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            # mode='w' truncates on open — the contract this file
            # advertises in its preamble ("rewritten on each run").
            with open(path, "w", encoding="utf-8") as f:
                f.write(_PREAMBLE)

            sparrow_logger = logging.getLogger(_LOGGER_NAME)
            sparrow_logger.setLevel(logging.INFO)
            sparrow_logger.propagate = False  # don't pollute agent.log
            # Drop any prior handlers from a previous in-process
            # init (would otherwise hold a stale FD on the truncated file).
            for h in list(sparrow_logger.handlers):
                sparrow_logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
            handler = logging.FileHandler(path, mode="a", encoding="utf-8")
            _fmt = logging.Formatter(
                fmt="%(asctime)s.%(msecs)03dZ  %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
            # Force UTC so the "Z" suffix in the format string is honest
            # — operators reading sparrow.log across containers/hosts
            # don't have to think about timezone.
            import time as _time
            _fmt.converter = _time.gmtime
            handler.setFormatter(_fmt)
            handler.setLevel(logging.INFO)
            sparrow_logger.addHandler(handler)

            _log_path = path
            _initialized = True
        except Exception as exc:
            _internal_logger.warning(
                "sparrow_log init failed (%s); high-level events won't be recorded",
                exc,
            )


class _SparrowEventServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], token: str):
        super().__init__(server_address, _SparrowEventHandler)
        self.token = token


class _SparrowEventHandler(BaseHTTPRequestHandler):
    """Tiny localhost POST sink used by managed child processes.

    DESIGN INVARIANT: be careful not to break this.  The sink is
    intentionally private to the local machine and authenticated with a
    per-run random token.  It must never become a general remote logging
    endpoint.
    """

    server: _SparrowEventServer

    def do_POST(self) -> None:
        if self.path != _SINK_PATH:
            self._send_status(404)
            return
        if not self._authorized():
            self._send_status(403)
            return
        try:
            payload = self._read_json_body()
            event = str(payload.get("event") or "").strip()
            if not event:
                self._send_status(400)
                return
            raw_fields = payload.get("fields")
            if isinstance(raw_fields, dict):
                fields = {str(k): v for k, v in raw_fields.items()}
            else:
                fields = {
                    str(k): v
                    for k, v in payload.items()
                    if k not in {"event", "fields"}
                }
            log_event(event, **fields)
            self._send_status(204)
        except Exception:
            self._send_status(400)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _authorized(self) -> bool:
        token = self.server.token
        auth = self.headers.get("Authorization", "")
        header_token = self.headers.get("X-Sparrow-Log-Token", "")
        return auth == f"Bearer {token}" or header_token == token

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length") or "0"
        length = int(raw_length)
        if length <= 0 or length > 1024 * 1024:
            raise ValueError("invalid request body length")
        data = self.rfile.read(length)
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_status(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()


def start_event_sink(host: str = "127.0.0.1") -> Optional[str]:
    """Start the local HTTP sink and expose endpoint/token via env vars.

    Gateway-managed subprocesses inherit these env vars and can report
    high-level Sparrow events without importing Hermes code.  Never raises.
    """
    global _sink_server, _sink_thread, _sink_endpoint, _sink_token
    with _lock:
        if _sink_server is not None:
            return _sink_endpoint
        try:
            token = secrets.token_urlsafe(24)
            server = _SparrowEventServer((host, 0), token)
            bound_host, bound_port = server.server_address[:2]
            endpoint = f"http://{bound_host}:{bound_port}{_SINK_PATH}"
            thread = threading.Thread(
                target=server.serve_forever,
                name="sparrow-log-sink",
                daemon=True,
            )
            thread.start()

            _sink_server = server
            _sink_thread = thread
            _sink_endpoint = endpoint
            _sink_token = token
            os.environ[_SINK_ENDPOINT_ENV] = endpoint
            os.environ[_SINK_TOKEN_ENV] = token
            return endpoint
        except Exception as exc:
            _internal_logger.warning("sparrow log event sink failed to start: %s", exc)
            return None


def stop_event_sink(timeout: float = 2.0) -> None:
    """Stop the local HTTP sink and remove its env vars.  Never raises."""
    global _sink_server, _sink_thread, _sink_endpoint, _sink_token
    with _lock:
        server = _sink_server
        thread = _sink_thread
        endpoint = _sink_endpoint
        token = _sink_token
        _sink_server = None
        _sink_thread = None
        _sink_endpoint = None
        _sink_token = None

    if os.environ.get(_SINK_ENDPOINT_ENV) == endpoint:
        os.environ.pop(_SINK_ENDPOINT_ENV, None)
    if os.environ.get(_SINK_TOKEN_ENV) == token:
        os.environ.pop(_SINK_TOKEN_ENV, None)
    if server is None:
        return
    try:
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=timeout)
    except Exception as exc:
        _internal_logger.debug("sparrow log event sink shutdown failed: %s", exc)


def log_event(event: str, **fields: Any) -> None:
    """Emit one event line.  ``fields`` are appended as ``k=v`` pairs.

    Values containing whitespace are wrapped in double quotes; multiline
    values have their newlines escaped so each event stays on one line.
    Never raises.
    """
    if not _initialized:
        # Early callers (before init_sparrow_log) silently no-op rather
        # than buffer; a bot that starts a turn before the file exists
        # is not a normal state and we don't want to mask it.
        return
    try:
        sparrow_logger = logging.getLogger(_LOGGER_NAME)
        tag = f"[{event}]"
        if fields:
            parts = [tag]
            for k, v in fields.items():
                parts.append(f"{k}={_format_value(v)}")
            sparrow_logger.info("  ".join(parts))
        else:
            sparrow_logger.info(tag)
    except Exception as exc:
        _internal_logger.debug("sparrow_log.log_event(%s) failed: %s", event, exc)


def _format_value(v: Any) -> str:
    """Escape a value so it stays on a single sparrow-log line."""
    if v is None:
        return "<none>"
    s = str(v)
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if any(ch.isspace() for ch in s) or any(ch in s for ch in '"='):
        # Newline escaping above keeps the event on a single line; we
        # do NOT truncate.  Long values (memory dumps, big assistant
        # replies) are deliberately surfaced in full so the operator
        # can see every byte without round-tripping to agent.log.
        s = '"' + s.replace('"', '\\"') + '"'
    return s


def log_path() -> Optional[Path]:
    """Return the active sparrow.log path, or None if init hasn't run."""
    return _log_path
