"""
Interactive setup wizard for the onebot_napcat platform.

Invoked from ``hermes gateway setup`` when the user picks the
"QQ (NapCat, real account)" option.  The wizard:

  1. Warns the user about ToS / ban risk.
  2. Starts NapCat as a subprocess (NTQQ + OneBot WS server).
  3. Points the user at http://localhost:6099/webui to scan the QR.
  4. Polls NapCat's ``get_login_info`` until it reports a logged-in user.
  5. Captures the QQ number (self_id) and nickname.
  6. Prompts for group allowlist / DM policy.
  7. Persists NAPCAT_* env vars to ~/.hermes/.env.
  8. Stops the NapCat subprocess (the gateway will respawn it on ``run``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

from .constants import NAPCAT_WEBUI_PORT, NAPCAT_WS_URL
from . import spawner

logger = logging.getLogger(__name__)


# =============================================================================
# Async core — spawn + poll login + teardown
# =============================================================================

async def _poll_login(ws_url: str, timeout: float = 600.0) -> Optional[dict]:
    """Try to WS-connect and poll ``get_login_info`` until login succeeds.

    Robust against the WS endpoint not being up yet — the OneBot WS only
    starts *after* QQ login.  Pre-login we get ConnectionRefused; we
    retry periodically until either login happens or *timeout* elapses.

    Returns the login_info data dict (with ``user_id`` + ``nickname``) on
    success, or None on timeout.
    """
    try:
        import aiohttp
    except ImportError:
        return None

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    echo_seq = 0

    while loop.time() < deadline:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    ws_url, timeout=aiohttp.ClientWSTimeout(ws_close=5), heartbeat=30,
                ) as ws:
                    # Connected — poll for a logged-in user.  Once the
                    # connection is up, it usually means login already
                    # happened, but double-check via get_login_info.
                    echo_seq += 1
                    echo_id = str(echo_seq)
                    await ws.send_json(
                        {"action": "get_login_info", "params": {}, "echo": echo_id}
                    )
                    try:
                        while True:
                            msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            frame = json.loads(msg.data)
                            if str(frame.get("echo", "")) == echo_id:
                                if frame.get("status") == "ok":
                                    data = frame.get("data") or {}
                                    if int(data.get("user_id", 0)) > 0:
                                        return data
                                break  # not logged in yet via this frame
                    except asyncio.TimeoutError:
                        pass
        except Exception:
            # WS not up yet (user hasn't scanned), or transient — retry.
            pass
        await asyncio.sleep(3.0)

    return None


async def _spawn_and_wait_for_login(ws_url: str) -> Optional[dict]:
    """Spawn NapCat, show QR instructions ASAP, then poll for login."""
    proc = None
    try:
        # tee_stdout=True so the user sees NapCat's dynamically-generated
        # WebUI login token and the QR in their terminal.
        proc, _state = await spawner.spawn(tee_stdout=True)
        # WebUI comes up within a few seconds of spawn — before login.
        # Wait for it so we know NapCat is alive before telling the user
        # to open the browser.
        try:
            await spawner.wait_for_webui_ready(NAPCAT_WEBUI_PORT, timeout=60)
        except spawner.NapCatSpawnError as exc:
            # Even if the probe fails, keep going — WebUI listening is a
            # strong hint but not strictly required (user may have
            # configured an external UI).  Warn and proceed.
            print(f"  ⚠  WebUI readiness probe: {exc}")

        print()
        print(f"  NapCat is up. Open http://localhost:{NAPCAT_WEBUI_PORT}/webui in your browser.")
        print()
        print("  The WebUI login token is printed by NapCat to the lines above")
        print("  (look for '[WebUi] 启动成功' or similar — it's a random token).")
        print("  Paste that token in the WebUI, then click 'QRCode Login' and scan with QQ.")
        print()
        print("  Waiting for login to complete (up to 10 min)… Ctrl+C to cancel.")
        return await _poll_login(ws_url, timeout=600)
    finally:
        if proc is not None:
            await spawner.terminate(proc)


# =============================================================================
# Sync entry point — called from hermes_cli/gateway.py dispatch
# =============================================================================

def cmd_setup_onebot_napcat() -> None:
    """Interactive wizard entry point.  Safe to call from non-async context."""
    # Import lazily to avoid a hard dep from hermes_cli on optional platforms.
    from hermes_cli.gateway import (  # type: ignore  — internal helpers
        color, Colors, prompt, prompt_yes_no, prompt_choice,
        print_info, print_success, print_warning, print_error,
        save_env_value, get_env_value,
    )

    print()
    print(color("  ─── 🐱 QQ (NapCat — real account) Setup ───", Colors.CYAN))
    print()
    print_warning("  ⚠  This uses an NTQQ-based bot framework (NapCat) that logs in")
    print_warning("     as a *real* QQ account — not the official bot API.")
    print_warning("     Third-party QQ frameworks violate Tencent's ToS in spirit;")
    print_warning("     accounts can be restricted or permanently banned.")
    print()
    print_info("  Recommendations:")
    print_info("    • Use a secondary/dedicated QQ account, not your primary.")
    print_info("    • Prefer residential IPs.  Datacenter IPs are heavily flagged.")
    print_info("    • Don't spam groups or send mass messages.")
    print()

    if not prompt_yes_no("  Continue with NapCat setup?", False):
        print_info("  Cancelled.")
        return

    if not spawner.check_bundled():
        print_error("  NapCat binaries not found in this image.")
        print_info("  Update with: ./control/build.sh  (ensure control/hermes-agent/Dockerfile.patched is current)")
        return

    existing_self_id = get_env_value("NAPCAT_SELF_ID")
    if existing_self_id:
        print()
        print_success(f"  NapCat is already configured for QQ {existing_self_id}.")
        if not prompt_yes_no("  Re-login / reconfigure?", False):
            return

    print()
    print_info(f"  Starting NapCat… (first launch takes ~20s)")
    print_info(f"  WebUI will be at http://localhost:{NAPCAT_WEBUI_PORT}/webui")
    print()

    ws_url = os.getenv("NAPCAT_WS_URL", NAPCAT_WS_URL)

    try:
        login_info = asyncio.run(_spawn_and_wait_for_login(ws_url))
    except KeyboardInterrupt:
        print()
        print_warning("  Setup cancelled.")
        return
    except spawner.NapCatSpawnError as exc:
        print_error(f"  NapCat failed to start: {exc}")
        return
    except Exception as exc:
        print_error(f"  Unexpected error: {exc}")
        return

    if not login_info:
        print_warning("  Login did not complete in time. Try again with `hermes gateway setup`.")
        return

    self_id = int(login_info.get("user_id", 0))
    nickname = login_info.get("nickname", "")
    print()
    print_success(f"  ✓ Logged in as {nickname} (QQ {self_id})")

    save_env_value("NAPCAT_ENABLED", "true")
    save_env_value("NAPCAT_SELF_ID", str(self_id))

    # --- Group policy ---
    # Derive current state from env so re-setup defaults to the user's
    # existing choice rather than overwriting blindly.
    current_groups = (get_env_value("NAPCAT_ALLOWED_GROUPS") or "").strip()
    if current_groups == "disabled":
        default_group_idx = 2
    elif current_groups:
        default_group_idx = 1
    else:
        default_group_idx = 0

    print()
    print_info("  Which groups should the bot engage in?")
    group_choices = [
        "Any group I'm in (bot responds when @-mentioned or replied to)",
        "Specific groups only (allowlist by group ID)",
        "No groups — DMs only",
    ]
    idx = prompt_choice("  Group policy", group_choices, default_group_idx)
    if idx == 0:
        save_env_value("NAPCAT_ALLOWED_GROUPS", "")  # empty = all
    elif idx == 1:
        current_list = current_groups if current_groups and current_groups != "disabled" else ""
        if current_list:
            print_info(f"  Current: {current_list}")
        raw = prompt("  Group IDs (comma-separated, empty to keep current)", password=False)
        if raw:
            save_env_value("NAPCAT_ALLOWED_GROUPS", raw.replace(" ", ""))
        elif not current_list:
            # No current value AND empty input — fall back to "any group"
            save_env_value("NAPCAT_ALLOWED_GROUPS", "")
        # else: keep existing value
    else:
        # Use a sentinel that matches no real group — effectively disables groups
        save_env_value("NAPCAT_ALLOWED_GROUPS", "disabled")

    # --- DM policy ---
    current_users = (get_env_value("NAPCAT_ALLOWED_USERS") or "").strip()
    current_allow_all = (
        (get_env_value("NAPCAT_ALLOW_ALL_USERS") or "").lower() in ("1", "true", "yes")
    )
    if current_users == "disabled":
        default_dm_idx = 2
    elif current_allow_all:
        default_dm_idx = 1
    else:
        default_dm_idx = 0

    print()
    dm_choices = [
        "Allow DMs from listed QQ user IDs (allowlist)",
        "Allow DMs from anyone (open access)",
        "Disable DMs",
    ]
    idx = prompt_choice("  DM policy", dm_choices, default_dm_idx)
    if idx == 0:
        current_list = current_users if current_users and current_users != "disabled" else ""
        if current_list:
            print_info(f"  Current: {current_list}")
        raw = prompt("  User QQ IDs (comma-separated, empty to keep current)", password=False)
        if raw:
            save_env_value("NAPCAT_ALLOWED_USERS", raw.replace(" ", ""))
        elif not current_list:
            print_warning("  No allowlist set — DMs will be denied by default.")
        save_env_value("NAPCAT_ALLOW_ALL_USERS", "false")
    elif idx == 1:
        save_env_value("NAPCAT_ALLOWED_USERS", "")
        save_env_value("NAPCAT_ALLOW_ALL_USERS", "true")
    else:
        save_env_value("NAPCAT_ALLOWED_USERS", "disabled")

    # --- Home channel (for cron + notifications) ---
    current_home = (get_env_value("ONEBOT_NAPCAT_HOME_CHANNEL") or "").strip()
    print()
    print_info("  Optional: set a 'home channel' for cron jobs & system notifications.")
    print_info("  Format: 'group_<group_id>' or 'private_<user_id>'")
    if current_home:
        print_info(f"  Current: {current_home}")
    home = prompt("  Home channel (empty to keep current, '-' to clear)", password=False)
    if home.strip() == "-":
        save_env_value("ONEBOT_NAPCAT_HOME_CHANNEL", "")
    elif home.strip():
        # Hermes looks up home channel by platform_value.upper() + "_HOME_CHANNEL"
        # (see gateway/run.py line ~4461), so the env var name must be this
        # specific string — not our shorter "NAPCAT_" prefix.
        save_env_value("ONEBOT_NAPCAT_HOME_CHANNEL", home.strip())

    # --- Shared-group session mode ---
    # QQ group UX expects bot to remember context across members (whoever @'s
    # it can build on prior exchanges).  Hermes's default is per-user isolation
    # which means each member has their own memory — not what you usually
    # want in QQ groups.
    #
    # This flag is GLOBAL, not per-platform, so it also affects Telegram /
    # Discord / Slack groups.  If you don't use those, turning it on is safe.
    try:
        from hermes_cli.config import load_config, set_config_value
        current = load_config().get("group_sessions_per_user", True)
        default_current = bool(current)  # True when currently per-user isolated
    except Exception:
        default_current = True

    if default_current:
        print()
        print_info("  Shared group mode: when ON, all @-invocations in a group share one")
        print_info("  conversation context.  Bot remembers what others asked it in the")
        print_info("  same group.  RECOMMENDED for QQ groups.")
        print_warning("  ⚠  This setting is GLOBAL — it also affects Telegram/Discord/Slack")
        print_warning("     groups if you use them there.  Per-user isolation is the default.")
        if prompt_yes_no("  Enable shared group mode?", True):
            try:
                set_config_value("group_sessions_per_user", "false")
                print_success("  Set group_sessions_per_user=false in config.yaml")
            except Exception as exc:
                print_warning(f"  Couldn't update config.yaml: {exc}")
                print_info("  Set it manually: `hermes config set group_sessions_per_user false`")

    print()
    print_success("  🐱 NapCat configured!")
    print_info("  Run `hermes gateway run` to start the gateway with QQ enabled.")
    print_info(f"  Session is persisted in /opt/data/napcat/ — you won't need to re-login.")
