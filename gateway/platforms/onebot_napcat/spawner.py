"""
Manage the NapCat subprocess that runs alongside Hermes inside the container.

NapCat is the QQ NT client (Electron) wrapped to speak OneBot v11.  Unlike
Hermes's other platforms, it isn't just a library — it's a binary that
listens on ``ws://127.0.0.1:3001`` (OneBot WS) and ``http://0.0.0.0:6099``
(WebUI).  The adapter boots it before connecting, and tears it down on
gateway shutdown so we don't leak a zombie QQ session.

Runs as the ``hermes`` user (not root).  Requires the image built from
``control/hermes-agent/Dockerfile.patched`` which copies NapCat + QQ binaries and installs Xvfb.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp

from .constants import (
    NAPCAT_PATH,
    NAPCAT_SHUTDOWN_TIMEOUT,
    NAPCAT_SPAWN_TIMEOUT,
    NAPCAT_WS_URL,
    QQ_BINARY,
)

logger = logging.getLogger(__name__)


class NapCatSpawnError(RuntimeError):
    """Raised when NapCat cannot be started or doesn't come up in time."""


# -----------------------------------------------------------------------------
# Live state observed from NapCat's stdout
# -----------------------------------------------------------------------------

# Login state machine — the log-line patterns below drive transitions.
LOGIN_STATE_STARTING = "starting"      # NapCat just spawned, nothing observed yet
LOGIN_STATE_QR_NEEDED = "qr_needed"    # NapCat printed QR / session missing or expired
LOGIN_STATE_CAPTCHA_NEEDED = "captcha_needed"  # QQ requires SMS/slider verification (one-time)
LOGIN_STATE_LOGIN_FAILED = "login_failed"  # Explicit error from QQ server
LOGIN_STATE_LOGIN_OK = "login_ok"      # Successfully logged in
LOGIN_STATE_WS_READY = "ws_ready"      # OneBot WS server actually started


@dataclass
class NapCatState:
    """Mutable shared state populated by ``_pipe_output`` as NapCat logs.

    The spawner watches NapCat's stdout and surfaces observed events via
    this object so the adapter / wizard can react (e.g. fail fast when
    login can't complete instead of waiting 90s for a WS port that won't
    open).
    """

    login_state: str = LOGIN_STATE_STARTING
    last_error: str = ""
    qr_url: Optional[str] = None
    captcha_url: Optional[str] = None
    self_id: Optional[int] = None
    # Recent stdout lines kept in a small ring buffer for diagnostics
    recent_lines: List[str] = field(default_factory=list)

    def tail(self, n: int = 20) -> str:
        return "\n".join(self.recent_lines[-n:])


# Patterns driving the login state machine.  Order matters — more specific
# patterns go first (e.g. ``Login success`` before the generic QR prompt).
_PAT_LOGIN_ERROR = re.compile(
    r"\[Login\]\s+Login\s+Error.*?ErrType[:\s]+(\d+).*?ErrCode[:\s]+(\d+)",
    re.IGNORECASE,
)
_PAT_QR_NEEDED = re.compile(r"请扫描下面的二维码|Please scan", re.IGNORECASE)
_PAT_QR_URL = re.compile(r"(https?://txz\.qq\.com/\S+)")
# Password-fallback on a "new" device triggers SMS / slider captcha.
_PAT_CAPTCHA_NEEDED = re.compile(r"需要验证码|密码回退需要验证码|proofWaterUrl", re.IGNORECASE)
_PAT_CAPTCHA_URL = re.compile(r"(https?://ti\.qq\.com/\S+)")
# Self ID — only the ``1. <qq> <nickname>`` form is verified against real
# NapCat output; ``Uin=...`` is a guess based on common conventions.  Matching
# self_id is a nice-to-have for diagnostics, not required for correctness.
_PAT_SELF_ID = re.compile(r"Uin[=:\s]*(\d{5,12})|^\s*\d+\.\s+(\d{5,12})\s")

# Note: we intentionally do NOT match "login success" or "OneBot WS started"
# from NapCat's log output.  Those messages vary across NapCat versions and
# guessing risks false positives (my earlier ``[SelfInfo]``-based pattern
# matched unrelated diagnostic lines and masked real QR_NEEDED events).  The
# real success signal is ``ws_connect(ws://localhost:3001)`` returning OK —
# if the port is bound and handshakes, we're good regardless of what NapCat
# chose to print.  State transitions we DO track are only the failure ones:
# QR_NEEDED and LOGIN_FAILED, both verified against observed log lines.


def check_bundled() -> bool:
    """Return True if the image has NapCat + QQ binaries bundled in."""
    return (
        os.path.isdir(NAPCAT_PATH)
        and os.path.isfile(os.path.join(NAPCAT_PATH, "napcat", "napcat.mjs"))
        and os.path.isfile(QQ_BINARY)
        and shutil.which("xvfb-run") is not None
    )


# The OneBot v11 WS server config we need NapCat to expose.  Baked into the
# image at /app/napcat/config/onebot11.json (default template) but NapCat
# also writes per-account files like ``onebot11_<qq>.json`` that override
# the template — those are created on login and may come up empty if they
# were generated before this template existed.  ``ensure_ws_server_config``
# rewrites any per-account file that's missing the ws server entry.
_DESIRED_WS_SERVER = {
    "enable": True,
    "name": "ws",
    "host": "0.0.0.0",
    "port": 3001,
    "reportSelfMessage": False,
    "enableForcePushEvent": True,
    "messagePostFormat": "array",
    "token": "",
    "debug": False,
    "heartInterval": 30000,
}

# Companion HTTP server — the adapter uses this for ALL outbound actions
# (send_msg / get_file / …).  HTTP requests are independent so a wedge
# in one doesn't block others, unlike WS where actions share a socket.
_DESIRED_HTTP_SERVER = {
    "enable": True,
    "name": "http",
    "host": "0.0.0.0",
    "port": 3000,
    "enableCors": True,
    "enableWebsocket": False,
    "messagePostFormat": "array",
    "token": "",
    "debug": False,
}


def ensure_ws_server_config() -> None:
    """Make sure every onebot11_*.json config file exposes the ws server,
    and that napcat.json has the upstream-recommended o3HookMode=0
    (works around NapNeko/NapCatQQ#869-style gradual wedge).

    Called on each spawn so existing installs (where login happened before
    the Dockerfile's template update) self-heal without the user having to
    manually delete configs.
    """
    import glob
    import json

    config_dir = os.path.join(NAPCAT_PATH, "napcat", "config")
    if not os.path.isdir(config_dir):
        return

    # 1) onebot11 per-account configs must expose the ws server
    patterns = [
        os.path.join(config_dir, "onebot11.json"),
        os.path.join(config_dir, "onebot11_*.json"),
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                with open(path, encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception as exc:
                logger.warning("ensure_ws: can't read %s: %s", path, exc)
                continue

            network = cfg.setdefault("network", {})
            ws_servers = network.setdefault("websocketServers", [])
            http_servers = network.setdefault("httpServers", [])

            changed = False
            ws_present = any(
                isinstance(s, dict) and s.get("name") == "ws" and s.get("port") == 3001
                for s in ws_servers
            )
            if not ws_present:
                ws_servers.append(dict(_DESIRED_WS_SERVER))
                changed = True

            http_present = any(
                isinstance(s, dict) and s.get("name") == "http" and s.get("port") == 3000
                for s in http_servers
            )
            if not http_present:
                http_servers.append(dict(_DESIRED_HTTP_SERVER))
                changed = True

            if not changed:
                continue

            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                logger.info(
                    "ensure_ws: patched OneBot config %s (ws=%s, http=%s)",
                    path, ws_present or "added", http_present or "added",
                )
            except Exception as exc:
                logger.warning("ensure_ws: can't write %s: %s", path, exc)

    # 2) napcat.json — force o3HookMode=0 to mitigate send-wedging bug
    napcat_json = os.path.join(config_dir, "napcat.json")
    if os.path.isfile(napcat_json):
        try:
            with open(napcat_json, encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("o3HookMode") != 0:
                cfg["o3HookMode"] = 0
                with open(napcat_json, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                logger.info("ensure_ws: set o3HookMode=0 in %s (mitigates send-wedging)", napcat_json)
        except Exception as exc:
            logger.warning("ensure_ws: can't patch napcat.json: %s", exc)


def _parse_line_into_state(line: str, state: NapCatState) -> None:
    """Update ``state`` based on a single NapCat stdout line."""
    prev_state = state.login_state

    # Self ID — two alternation groups in the regex, take whichever matched
    m = _PAT_SELF_ID.search(line)
    if m:
        try:
            uin = m.group(1) or m.group(2)
            if uin:
                state.self_id = int(uin)
        except ValueError:
            pass

    # QR URL (useful even when login is just starting)
    m = _PAT_QR_URL.search(line)
    if m:
        state.qr_url = m.group(1)

    # Captcha URL (for password-fallback verification)
    m = _PAT_CAPTCHA_URL.search(line)
    if m:
        state.captcha_url = m.group(1)

    # Login error (highest priority — clobbers other states)
    m = _PAT_LOGIN_ERROR.search(line)
    if m:
        state.login_state = LOGIN_STATE_LOGIN_FAILED
        state.last_error = (
            f"QQ login rejected by server (ErrType={m.group(1)} ErrCode={m.group(2)}). "
            "Session is invalid or expired — re-run `hermes gateway setup` → "
            "NapCat to scan a fresh QR."
        )
    # QR needed — transition only from starting
    elif _PAT_QR_NEEDED.search(line) and state.login_state == LOGIN_STATE_STARTING:
        state.login_state = LOGIN_STATE_QR_NEEDED
        state.last_error = (
            "NapCat needs a fresh QR-code login. Either this is the first launch, "
            "the previous QQ session expired, or NAPCAT_SELF_ID isn't set so "
            "quick-login via -q didn't fire. Re-run `hermes gateway setup` → "
            "NapCat and scan the code with mobile QQ."
        )
    elif _PAT_CAPTCHA_NEEDED.search(line) and state.login_state in (
        LOGIN_STATE_STARTING, LOGIN_STATE_QR_NEEDED
    ):
        state.login_state = LOGIN_STATE_CAPTCHA_NEEDED
        state.last_error = (
            "QQ requires SMS/slider verification (password fallback to a new "
            "device). One-time: open the captcha URL below, complete SMS + "
            "slider, then NapCat will retry automatically."
        )

    # Surface failure transitions to the terminal (WARNING → stderr via the
    # default Python logging config; INFO-level logs are buried in the
    # agent.log file).  We only track failure states — the success path is
    # signalled by ws_connect() returning OK in wait_for_ws_ready.
    if state.login_state != prev_state:
        if state.login_state == LOGIN_STATE_QR_NEEDED:
            logger.warning("NapCat: QR-code login required — gateway cannot proceed. %s", state.last_error)
            if state.qr_url:
                logger.warning("NapCat: QR URL: %s", state.qr_url)
        elif state.login_state == LOGIN_STATE_CAPTCHA_NEEDED:
            logger.warning("NapCat: 🔐 SMS/captcha verification required — %s", state.last_error)
            if state.captcha_url:
                logger.warning("NapCat: 👉 Open this URL to complete verification: %s", state.captcha_url)
        elif state.login_state == LOGIN_STATE_LOGIN_FAILED:
            logger.warning("NapCat: %s", state.last_error)


async def _pipe_output(
    proc: asyncio.subprocess.Process,
    state: NapCatState,
    tee_stdout: bool = False,
) -> None:
    """Stream the child process's combined stdout/stderr.

    Always goes to our Python logger (for post-mortem inspection via
    ``journalctl`` / gateway logs).  When ``tee_stdout=True`` the lines
    are ALSO printed to the parent process's stdout — used by the setup
    wizard so the user can see the WebUI login token and QR code that
    NapCat prints to its own console on startup.

    Additionally parses lines into ``state`` so callers can react to
    login progress (e.g. fail fast when QQ rejects the session).
    """
    import sys as _sys

    assert proc.stdout is not None
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        if not line:
            continue
        logger.info("[napcat] %s", line)
        if tee_stdout:
            print(f"[napcat] {line}", file=_sys.stderr, flush=True)
        # Keep a small ring buffer for diagnostics
        state.recent_lines.append(line)
        if len(state.recent_lines) > 200:
            del state.recent_lines[: -200]
        _parse_line_into_state(line, state)


def ensure_persistent_config() -> None:
    """Make ``/app/napcat/config`` a symlink to ``/opt/data/napcat/config``.

    Why: NapCat writes per-account state (device_guid, protocol, OneBot
    config) into ``<cwd>/config/``.  Our container's cwd is /app/napcat,
    so that's /app/napcat/config — which lives INSIDE the image and is
    recreated fresh every ``./control/run.sh``.  Result: per-account
    files get wiped each run, NapCat regenerates a new device_guid, and
    QQ revokes the session on every restart.

    This function:
      1. Ensures /opt/data/napcat/config/ exists (volume-backed).
      2. Copies the image's default configs (napcat.json / webui.json /
         onebot11.json) into the volume on first run.
      3. Replaces /app/napcat/config with a symlink to the volume dir.
    """
    import shutil

    image_dir = os.path.join(NAPCAT_PATH, "napcat", "config")
    vol_dir = "/opt/data/napcat/config"

    # Already set up?
    if os.path.islink(image_dir):
        target = os.readlink(image_dir)
        if target == vol_dir:
            return
        # Weird — pointing somewhere else.  Rebuild the link.
        os.remove(image_dir)

    os.makedirs(vol_dir, exist_ok=True)

    # First-run: seed the volume with the image's defaults (but never
    # overwrite anything already in the volume — that would wipe user
    # login state).
    if os.path.isdir(image_dir):
        for name in os.listdir(image_dir):
            src = os.path.join(image_dir, name)
            dst = os.path.join(vol_dir, name)
            if os.path.isfile(src) and not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                    logger.info("NapCat: seeded %s into persistent volume", name)
                except Exception as exc:
                    logger.warning("NapCat: failed to seed %s: %s", name, exc)

    # Now atomically replace the in-image dir with a symlink to the volume.
    if os.path.isdir(image_dir):
        try:
            shutil.rmtree(image_dir)
        except Exception as exc:
            logger.warning(
                "NapCat: couldn't remove %s (%s) — per-account config may not persist",
                image_dir, exc,
            )
            return

    try:
        os.symlink(vol_dir, image_dir)
        logger.info(
            "NapCat: ✓ config persistence: %s → %s",
            image_dir, vol_dir,
        )
    except Exception as exc:
        logger.warning(
            "NapCat: couldn't symlink %s → %s (%s) — device_guid will regenerate each run",
            image_dir, vol_dir, exc,
        )


def log_session_diagnostics() -> None:
    """Log QQ session + NapCat config locations and contents.

    Emitted at WARNING level so it shows in the terminal (INFO is buried
    in agent.log).  Purpose: when ``-q`` fails and NapCat falls back to
    QR, we want to see *why* at a glance — wrong path, empty dir, bad
    permissions, stale session — without having to ssh in and poke.
    """
    import getpass
    import grp
    import pwd
    import stat

    # Where QQ will look for its cached session.
    # Our spawner forces XDG_CONFIG_HOME=/opt/data/.config, so QQ reads
    # /opt/data/.config/QQ/.  The per-account subdir name is nt_qq_<hash>
    # where <hash> is derived from machine-id; we list all of them.
    qq_root = "/opt/data/.config/QQ"
    napcat_config = os.path.join(NAPCAT_PATH, "napcat", "config")

    # Current effective identity
    try:
        euid = os.geteuid()
        egid = os.getegid()
        user = pwd.getpwuid(euid).pw_name
        group = grp.getgrgid(egid).gr_name
        logger.info("NapCat diag: running as %s(%d):%s(%d)", user, euid, group, egid)
    except Exception:
        logger.info("NapCat diag: running as uid=%d gid=%d", os.geteuid(), os.getegid())

    expected_quick_login = os.environ.get("NAPCAT_SELF_ID", "").strip()
    logger.info(
        "NapCat diag: quick-login target QQ=%s (from NAPCAT_SELF_ID)",
        expected_quick_login or "<unset>",
    )

    # Session dir inspection
    if not os.path.isdir(qq_root):
        logger.warning(
            "NapCat diag: ❌ QQ session root %s does NOT exist — first login expected",
            qq_root,
        )
    else:
        try:
            st = os.stat(qq_root)
            logger.info(
                "NapCat diag: QQ session root %s  (mode=%s, owner uid=%d gid=%d)",
                qq_root, stat.filemode(st.st_mode), st.st_uid, st.st_gid,
            )
            entries = sorted(os.listdir(qq_root))
            nt_qq_dirs = [e for e in entries if e.startswith("nt_qq_")]
            if not nt_qq_dirs:
                logger.warning(
                    "NapCat diag: ❌ no nt_qq_<hash>/ dirs found in %s — QQ has no cached session",
                    qq_root,
                )
            import time as _time
            now = _time.time()
            # Also look at the generic non-hashed nt_qq/ and NapCat/ siblings
            extra_dirs = [d for d in ("nt_qq", "NapCat") if os.path.isdir(os.path.join(qq_root, d))]
            for name in nt_qq_dirs + extra_dirs:
                full = os.path.join(qq_root, name)
                try:
                    sub = sorted(os.listdir(full))
                    readable = os.access(full, os.R_OK | os.X_OK)
                except Exception:
                    sub = []
                    readable = False
                logger.info(
                    "NapCat diag:   %s/ (%d children, readable=%s): %s",
                    name, len(sub), readable, sub[:12],
                )
                for leaf in sub:
                    leaf_path = os.path.join(full, leaf)
                    if not os.path.isdir(leaf_path):
                        continue
                    try:
                        leaf_files = sorted(os.listdir(leaf_path))
                    except PermissionError:
                        logger.warning("NapCat diag:     ❌ %s/%s/ not readable (perm)", name, leaf)
                        continue
                    # Highlight files likely to hold login credentials
                    login_hints = [f for f in leaf_files if any(
                        kw in f.lower() for kw in (
                            "login", "auth", "token", "session", "profile", "device", "account", "uin", "ticket",
                        )
                    )]
                    # Recency check: find newest file's mtime
                    newest_mtime = 0.0
                    newest_name = ""
                    for f in leaf_files:
                        try:
                            m = os.path.getmtime(os.path.join(leaf_path, f))
                            if m > newest_mtime:
                                newest_mtime = m
                                newest_name = f
                        except Exception:
                            pass
                    age_hours = (now - newest_mtime) / 3600 if newest_mtime else -1
                    logger.info(
                        "NapCat diag:     %s/ (%d files%s) newest=%s (%.1fh ago)",
                        leaf, len(leaf_files),
                        f", login-hints={login_hints[:5]}" if login_hints else "",
                        newest_name or "?", age_hours,
                    )
        except Exception as exc:
            logger.warning("NapCat diag: failed to stat %s: %s", qq_root, exc)

    # NapCat config inspection (onebot11 / webui / napcat.json)
    if os.path.isdir(napcat_config):
        try:
            cfgs = sorted(os.listdir(napcat_config))
            logger.info("NapCat diag: NapCat config dir %s: %s", napcat_config, cfgs)
            # Dump device/protocol configs verbatim — these drive the
            # ``device name`` QQ server sees, and if they regenerate per
            # run QQ treats every start as a new device.
            import hashlib as _hl
            for fname in cfgs:
                if not fname.endswith(".json"):
                    continue
                if not any(kw in fname.lower() for kw in ("device", "protocol", "napcat_")):
                    continue
                full = os.path.join(napcat_config, fname)
                try:
                    with open(full, encoding="utf-8") as _f:
                        content = _f.read()
                    body = content[:600] + ("…" if len(content) > 600 else "")
                    digest = _hl.md5(content.encode()).hexdigest()[:12]
                    logger.info(
                        "NapCat diag:   %s  (size=%d, md5[:12]=%s)\n%s",
                        fname, len(content), digest, body,
                    )
                except Exception as exc:
                    logger.warning("NapCat diag:   can't read %s: %s", fname, exc)

            # Also save a signature of all config files so the user can
            # compare across runs: if these change between starts, the
            # device identity is unstable.
            import json as _json
            sig = {}
            for fname in cfgs:
                full = os.path.join(napcat_config, fname)
                if os.path.isfile(full):
                    try:
                        with open(full, "rb") as _f:
                            sig[fname] = _hl.md5(_f.read()).hexdigest()[:12]
                    except Exception:
                        sig[fname] = "err"
            logger.info("NapCat diag:   config signature (md5[:12]): %s", _json.dumps(sig, ensure_ascii=False))
        except Exception as exc:
            logger.warning("NapCat diag: can't list %s: %s", napcat_config, exc)
    else:
        logger.warning("NapCat diag: ❌ NapCat config dir %s missing", napcat_config)


async def spawn(tee_stdout: bool = False) -> Tuple[asyncio.subprocess.Process, NapCatState]:
    """Start NapCat as a background subprocess.

    Uses ``xvfb-run`` to provide a virtual X display for QQ's Electron
    client.  QQ stores its session under ``~/.config/QQ`` which is the
    hermes user's home — set ``XDG_CONFIG_HOME`` to point inside the
    mounted data volume so login survives container recreation.

    When ``tee_stdout=True``, NapCat's console output is forwarded to the
    caller's stderr as well as our Python logger.  The setup wizard uses
    this so the user can read NapCat's dynamically-generated WebUI login
    token (printed on startup) and the QR code.
    """
    if not check_bundled():
        raise NapCatSpawnError(
            f"NapCat binaries not found under {NAPCAT_PATH}. "
            "Update the Hermes image via `./control/build.sh` "
            "(control/hermes-agent/Dockerfile.patched must be current)."
        )

    # Make /app/napcat/config a symlink into /opt/data so the per-account
    # files (napcat_<QQ>.json, napcat_protocol_<QQ>.json, onebot11_<QQ>.json)
    # — which include the device_guid — survive container restarts.  This
    # is THE fix for "QQ makes me scan QR on every start": without it, each
    # new container has a fresh config dir, NapCat regenerates the
    # device_guid, QQ's server sees a new device, and revokes the session.
    ensure_persistent_config()

    # Self-heal OneBot WS server config (runs pre-spawn so NapCat reads the
    # fixed config on startup).
    ensure_ws_server_config()

    # "Dummy-proof" pre-flight log: show where QQ will look for its cached
    # session, whether anything is actually there, and who owns it.  The
    # single biggest cause of "re-scan every time" is a config path /
    # permission mismatch we can't see otherwise.
    log_session_diagnostics()

    # Pin QQ's session directory to a known location on the mounted volume.
    #
    # Two reasons we can't just let QQ follow $HOME/.config:
    #   1. ``gosu hermes`` doesn't reset $HOME — inside the container it
    #      remains ``/root`` inherited from the entrypoint's root phase,
    #      even after dropping privileges.  QQ would look in
    #      ``/root/.config/QQ/`` (ephemeral, wiped every ``./control/run.sh``).
    #   2. Even with ``HOME=/opt/data``, QQ's fallback via XDG spec prefers
    #      $XDG_CONFIG_HOME when set — so explicit is safer.
    #
    # This path matches where the setup wizard initially stored the QQ
    # session, so auto-login via ``-q <qq>`` finds the cached credentials.
    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = "/opt/data/.config"
    env["HOME"] = "/opt/data"  # Defensive: some QQ code paths use $HOME directly.
    env["FFMPEG_PATH"] = "/usr/bin/ffmpeg"
    os.makedirs("/opt/data/.config/QQ", exist_ok=True)

    # ``xvfb-run -a`` picks an unused display number automatically.  ``--``
    # separates xvfb-run flags from the child command.
    #
    # ``-q <QQ号>`` asks QQ to silent-login with the given account's cached
    # session rather than showing a QR prompt.  Without it, NapCat defaults
    # to QR mode even when a valid session is on disk — which is useless
    # in a headless gateway context (no one can scan).
    cmd = [
        "xvfb-run",
        "-a",
        "--server-args=-screen 0 1080x760x16 +extension GLX +render",
        "--",
        QQ_BINARY,
        "--no-sandbox",
    ]
    quick_login_qq = os.environ.get("NAPCAT_SELF_ID", "").strip()
    if quick_login_qq.isdigit():
        cmd.extend(["-q", quick_login_qq])
        logger.info("NapCat: spawning with quick-login QQ %s (expect 10-20s to come up)", quick_login_qq)
        try:
            from gateway.sparrow_log import log_event as _sparrow_event
            _sparrow_event("napcat_starting", qq=quick_login_qq)
        except Exception:
            pass
    else:
        logger.warning("NapCat: NAPCAT_SELF_ID not set — will prompt for QR scan (likely fails in gateway mode)")

    logger.info("Starting NapCat: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        cwd=os.path.join(NAPCAT_PATH, "napcat"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # New session/pgroup so SIGINT to the gateway (Ctrl+C in the
        # terminal where ``hermes gateway run`` was launched) doesn't
        # propagate to NapCat.  Without this, the user-typed Ctrl+C
        # races: NapCat starts shutting down on its own (closes its
        # OneBot WS server), the gateway's WS listener exits with
        # ``ws-iter-ended`` BEFORE ``adapter.disconnect()`` has had a
        # chance to set ``_stopping=True``, and the noisy
        # "watchdog will attempt recovery" warning fires on what is
        # actually a clean shutdown.  Putting NapCat in its own session
        # makes ``adapter.disconnect()`` the single, ordered path that
        # terminates it.
        start_new_session=True,
    )

    # Drain output to the logger (and optionally the caller's terminal)
    state = NapCatState()
    asyncio.create_task(_pipe_output(proc, state, tee_stdout=tee_stdout))
    return proc, state


async def wait_for_ws_ready(
    ws_url: str = NAPCAT_WS_URL,
    timeout: float = NAPCAT_SPAWN_TIMEOUT,
    state: Optional[NapCatState] = None,
    qr_timeout: float = 600.0,
) -> None:
    """Poll the WS endpoint until it accepts connections or we time out.

    When a *state* object is provided, we adapt:
      * ``LOGIN_STATE_LOGIN_FAILED`` → raise immediately (session is
        rejected by QQ; no amount of waiting will fix it).
      * ``LOGIN_STATE_QR_NEEDED`` → extend the deadline to ``qr_timeout``
        (default 10 min) and keep polling so the user has time to scan
        the QR code that NapCat just printed.  We log the QR URL once so
        it's visible even though NapCat's own stdout is buried in the
        gateway log.
      * Otherwise → wait up to the normal *timeout*.

    Note: the OneBot WS only starts listening **after** QQ login completes.
    Use ``wait_for_webui_ready`` first if you need a readiness signal that
    doesn't require a logged-in session.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last_error: Optional[BaseException] = None
    qr_mode_entered = False

    while loop.time() < deadline:
        if state is not None:
            # Hard fail — the session was rejected by QQ's server.
            if state.login_state == LOGIN_STATE_LOGIN_FAILED:
                raise NapCatSpawnError(
                    f"NapCat cannot connect to QQ. {state.last_error}\n"
                    f"Recent NapCat log:\n{state.tail(15)}"
                )
            # Captcha branch — similar to QR, one-time interactive fix.
            if state.login_state == LOGIN_STATE_CAPTCHA_NEEDED and not qr_mode_entered:
                qr_mode_entered = True
                deadline = loop.time() + qr_timeout
                logger.warning(
                    "NapCat: extending WS wait to %.0f min so you can complete "
                    "the SMS/captcha verification.", qr_timeout / 60,
                )
                if state.captcha_url:
                    logger.warning("NapCat: 👉 open in browser: %s", state.captcha_url)
                logger.warning(
                    "NapCat: or use the WebUI (http://localhost:6099/webui). "
                    "After verification NapCat retries automatically."
                )

            # Soft branch — user needs to scan a QR.  Extend the wait
            # instead of killing NapCat so the user can complete login
            # interactively while the gateway is still running.  Dump the
            # ASCII QR (and a few surrounding context lines) directly to
            # stderr so the user can scan it from the terminal without
            # hunting through agent.log.
            if state.login_state == LOGIN_STATE_QR_NEEDED and not qr_mode_entered:
                qr_mode_entered = True
                deadline = loop.time() + qr_timeout
                import sys as _sys
                logger.warning(
                    "NapCat: extending WS wait to %.0f min — scan the QR below:",
                    qr_timeout / 60,
                )
                # Find the ASCII QR block in the captured NapCat stdout.
                # QR lines are composed of unicode box-drawing chars like
                # ▄ ▀ █.  Grab a contiguous block where >50% of chars are
                # these markers, plus a few context lines around it.
                _qr_chars = set("▄▀█ ")
                qr_block_lines: list[str] = []
                for line in state.recent_lines:
                    ratio = (
                        sum(c in _qr_chars for c in line) / len(line)
                        if line else 0
                    )
                    if ratio > 0.6 and len(line) > 20:
                        qr_block_lines.append(line)
                if qr_block_lines:
                    print("", file=_sys.stderr)  # blank line before QR
                    for line in qr_block_lines:
                        print(line, file=_sys.stderr)
                    print("", file=_sys.stderr)  # blank line after QR
                if state.qr_url:
                    logger.warning(
                        "NapCat: or paste this URL into any QR generator:",
                    )
                    logger.warning("NapCat:    %s", state.qr_url)
                logger.warning(
                    "NapCat: (alternatively: http://localhost:6099/webui — "
                    "the WebUI token is in the NapCat startup lines above)"
                )
                logger.warning(
                    "NapCat: once you scan, the gateway will auto-continue."
                )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url, timeout=aiohttp.ClientWSTimeout(ws_close=5)) as ws:
                    await ws.close()
                    logger.info("NapCat WS ready at %s", ws_url)
                    return
        except Exception as exc:  # noqa: BLE001 — we want broad retry
            last_error = exc
            await asyncio.sleep(2.0)

    # Timeout — include the latest state context if we have it
    context = ""
    if state is not None:
        context = (
            f"\n  Login state at timeout: {state.login_state}"
            f"\n  Last error from NapCat: {state.last_error or '(none)'}"
            f"\n  Recent NapCat log:\n{state.tail(15)}"
        )
    raise NapCatSpawnError(
        f"NapCat WS {ws_url} not ready after {timeout:.0f}s; "
        f"last network error: {last_error!r}{context}"
    )


async def wait_for_webui_ready(port: int, timeout: float = 60.0) -> None:
    """Poll the WebUI HTTP endpoint until it responds.

    The WebUI listens on ``0.0.0.0:<port>`` (default 6099) as soon as NapCat
    finishes booting — well before any QQ login.  This is the first
    observable signal that NapCat is alive and ready for the user to scan
    a QR.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    url = f"http://127.0.0.1:{port}/webui/"
    last_error: Optional[BaseException] = None
    async with aiohttp.ClientSession() as session:
        while loop.time() < deadline:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status < 500:
                        logger.info("NapCat WebUI ready at %s", url)
                        return
                    last_error = RuntimeError(f"HTTP {resp.status}")
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(1.5)
    raise NapCatSpawnError(
        f"NapCat WebUI on port {port} not ready after {timeout:.0f}s; last error: {last_error!r}"
    )


async def terminate(proc: asyncio.subprocess.Process) -> None:
    """Gracefully stop NapCat, falling back to SIGKILL on timeout."""
    if proc.returncode is not None:
        return
    logger.info("Stopping NapCat (pid=%d)", proc.pid)
    try:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=NAPCAT_SHUTDOWN_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("NapCat did not stop in %.1fs; sending SIGKILL", NAPCAT_SHUTDOWN_TIMEOUT)
        proc.kill()
        await proc.wait()
    except ProcessLookupError:
        pass  # already gone
