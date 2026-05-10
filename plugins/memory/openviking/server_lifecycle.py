"""Spawn / wait / stop the openviking-server subprocess from the gateway.

Why this lives in the plugin (not in gateway core):
  - The decision about WHETHER to spawn (local endpoint? binary on PATH?
    already serving?) is openviking-specific.
  - Tearing down on Ctrl+C with SIGTERM→SIGKILL fallback is plumbing
    that doesn't belong in gateway/run.py.

How the gateway uses it:
  1. Very early in ``start_gateway`` (after PID-lock, before runner.start)
     call :func:`maybe_spawn_server` to kick off the subprocess so it has
     the longest possible time to come up while adapters connect.
  2. Right before announcing "Gateway ready" (after cron starts) call
     :func:`wait_for_ready` to confirm health.
  3. After ``wait_for_shutdown`` returns (Ctrl+C path) call
     :func:`stop_server` to gracefully shut it down.

If the user already started ``openviking-server`` manually OR points
``OPENVIKING_ENDPOINT`` at a remote host, every function here no-ops
silently — we never kill or interfere with a server we didn't spawn.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)


_DEFAULT_ENDPOINT = "http://127.0.0.1:1933"
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "0.0.0.0", "[::1]")


def _endpoint() -> str:
    return os.environ.get("OPENVIKING_ENDPOINT", _DEFAULT_ENDPOINT)


def _is_local(endpoint: str) -> bool:
    return any(h in endpoint for h in _LOCAL_HOSTS)


def _ping_health(endpoint: str, timeout: float = 2.0) -> bool:
    """Best-effort GET ``/health``.  Returns True only on HTTP 200."""
    try:
        import httpx
    except ImportError:
        return False
    try:
        resp = httpx.get(f"{endpoint}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _is_openviking_active_provider() -> bool:
    """Read ``memory.provider`` from config — only auto-spawn when
    openviking is the configured provider.  If the user picked
    hindsight / mem0 / honcho / built-in only, we shouldn't drag the
    openviking server up just because the binary happens to be there."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
    except Exception:
        return False
    provider = (cfg.get("memory") or {}).get("provider") or ""
    return provider.strip().lower() == "openviking"


def maybe_spawn_server() -> Optional[subprocess.Popen]:
    """Spawn ``openviking-server`` as a child process iff:
      * ``memory.provider`` in config is ``"openviking"``, AND
      * the configured endpoint is local (127.0.0.1 / localhost / ::1), AND
      * nothing is already serving at that endpoint, AND
      * the ``openviking-server`` binary is on PATH.

    Returns the ``Popen`` handle so the caller can later wait for it to
    become healthy and stop it on shutdown.  Returns ``None`` when we
    deliberately did NOT spawn (different provider, remote endpoint,
    already running, binary missing, or spawn failure) — caller should
    treat that as "no managed subprocess; assume external lifecycle".
    """
    if not _is_openviking_active_provider():
        logger.info(
            "[openviking] memory.provider is not 'openviking' — skipping auto-spawn",
        )
        return None
    endpoint = _endpoint()
    if not _is_local(endpoint):
        logger.info(
            "[openviking] endpoint %s is remote — not spawning a local server",
            endpoint,
        )
        return None
    if _ping_health(endpoint):
        logger.info(
            "[openviking] server already responding at %s — reusing instead of spawning",
            endpoint,
        )
        return None
    binary = shutil.which("openviking-server")
    if not binary:
        logger.info(
            "[openviking] 'openviking-server' not found on PATH — skipping auto-spawn",
        )
        return None
    logger.info("[openviking] spawning local server (endpoint=%s) ...", endpoint)
    try:
        proc = subprocess.Popen(
            [binary],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # New session/pgroup so terminating the gateway via Ctrl+C
            # doesn't deliver SIGINT to the server before we get a
            # chance to wait_for_shutdown — we want our own SIGTERM
            # path so the server can finish its own lifespan cleanup.
            start_new_session=True,
        )
    except Exception as exc:
        logger.warning("[openviking] failed to spawn server: %s", exc)
        return None
    logger.info(
        "[openviking] server spawned (PID %d); health check deferred to end of startup",
        proc.pid,
    )
    try:
        from gateway.sparrow_log import log_event as _sparrow_event
        _sparrow_event("openviking_starting", endpoint=endpoint, pid=proc.pid)
    except Exception:
        pass
    return proc


def wait_for_ready(
    proc: Optional[subprocess.Popen],
    timeout: float = 30.0,
    poll_interval: float = 0.5,
) -> bool:
    """Poll ``/health`` up to ``timeout`` seconds.

    Returns True when the server reports healthy.  Returns False when
    the timeout elapses OR the spawned process has exited (in which
    case the gateway proceeds with the memory plugin in degraded mode).
    """
    if proc is None:
        return False
    endpoint = _endpoint()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            logger.warning(
                "[openviking] server (PID %d) exited prematurely with code %s",
                proc.pid, proc.returncode,
            )
            return False
        if _ping_health(endpoint):
            logger.info("[openviking] server is healthy at %s", endpoint)
            try:
                from gateway.sparrow_log import log_event as _sparrow_event
                _sparrow_event("openviking_started", endpoint=endpoint, pid=proc.pid)
            except Exception:
                pass
            return True
        time.sleep(poll_interval)
    logger.warning(
        "[openviking] server did not become healthy within %.0fs — memory plugin may be degraded",
        timeout,
    )
    return False


def stop_server(
    proc: Optional[subprocess.Popen], timeout: float = 10.0
) -> None:
    """SIGTERM the spawned server, wait up to ``timeout`` seconds, then
    escalate to SIGKILL.  Idempotent — safe to call when ``proc`` is
    None or already exited."""
    if proc is None:
        return
    if proc.poll() is not None:
        logger.info(
            "[openviking] server (PID %d) already exited (code %s)",
            proc.pid, proc.returncode,
        )
        return
    logger.info("[openviking] sending SIGTERM to server (PID %d) ...", proc.pid)
    try:
        proc.terminate()
    except Exception as exc:
        logger.warning("[openviking] SIGTERM failed: %s", exc)
        return
    try:
        proc.wait(timeout=timeout)
        logger.info(
            "[openviking] server stopped cleanly (exit code %s)",
            proc.returncode,
        )
        return
    except subprocess.TimeoutExpired:
        pass
    logger.warning(
        "[openviking] server did not exit within %.0fs; sending SIGKILL",
        timeout,
    )
    try:
        proc.kill()
        proc.wait(timeout=5.0)
        logger.info(
            "[openviking] server killed (exit code %s)", proc.returncode,
        )
    except Exception as exc:
        logger.warning("[openviking] SIGKILL/wait failed: %s", exc)
