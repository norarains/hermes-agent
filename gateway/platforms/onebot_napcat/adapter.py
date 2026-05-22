"""
OneBot v11 platform adapter, driven by an in-container NapCat subprocess.

Flow:
    ┌────────┐   scan QR   ┌────────┐   WS 3001   ┌──────────┐
    │ QQ app │ ──────────▶ │ NapCat │ ◀─────────▶ │  this    │
    │ (user) │             │ (NTQQ) │             │ adapter  │
    └────────┘             └────────┘             └──────────┘
                                                        │
                                                        ▼
                                              Hermes gateway
                                              (handle_message)

The adapter speaks OneBot v11 (https://github.com/botuniverse/onebot-11):
  - inbound events: message / notice / request / meta_event
  - outbound actions: send_group_msg, send_private_msg, get_login_info, etc.

All actions use the ``echo`` field to correlate responses with a Future so
callers can await a concrete SendResult.

Config (env vars, set via ``hermes gateway setup`` → NapCat wizard):

  NAPCAT_ENABLED=true                       # gate the platform on
  NAPCAT_WS_URL=ws://127.0.0.1:3001         # override default
  NAPCAT_ACCESS_TOKEN=<secret>              # if NapCat requires auth
  NAPCAT_AUTOSPAWN=true                     # default: spawn NapCat ourselves
  NAPCAT_SELF_ID=<your qq number>           # set by wizard after login
  NAPCAT_ALLOWED_USERS=12345,67890          # allowlist (comma-sep QQ UIDs)
  NAPCAT_ALLOW_ALL_USERS=false              # open access
  NAPCAT_ALLOWED_GROUPS=111,222             # only respond in these groups
  NAPCAT_HOME_CHANNEL=group_123             # for cron / notifications
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    cache_audio_from_url,
    cache_document_from_bytes,
    cache_image_from_bytes,
    cache_image_from_url,
    cache_video_from_bytes,
    format_message_event_time,
)
from gateway.session import SessionSource, build_session_key

from .constants import (
    CHAT_ID_GROUP_PREFIX,
    CHAT_ID_PRIVATE_PREFIX,
    MAX_MESSAGE_LENGTH,
    NAPCAT_ACCESS_TOKEN,
    NAPCAT_HTTP_URL,
    NAPCAT_WS_URL,
)
from . import spawner
from .prompts import build_poke_channel_prompt, build_poke_system_instruction
from .poke import format_poke_event_text
from .transcript import sanitize_transcript_entry as sanitize_napcat_transcript_entry
from .voice import compose_voice_message_text, transcribe_voice_files

logger = logging.getLogger(__name__)

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]


def check_onebot_napcat_requirements() -> bool:
    """Check if the OneBot/NapCat adapter can start."""
    if not AIOHTTP_AVAILABLE:
        logger.warning("onebot_napcat: aiohttp not installed")
        return False
    if os.getenv("NAPCAT_ENABLED", "").lower() not in ("1", "true", "yes"):
        return False
    autospawn = os.getenv("NAPCAT_AUTOSPAWN", "true").lower() in ("1", "true", "yes")
    if autospawn and not spawner.check_bundled():
        logger.warning(
            "onebot_napcat: NAPCAT_AUTOSPAWN=true but NapCat binaries are not "
            "bundled in the image. Update with ./control/build.sh "
            "(using control/hermes-agent/Dockerfile.patched), or set "
            "NAPCAT_AUTOSPAWN=false and run NapCat yourself.",
        )
        return False
    return True


# =============================================================================
# OneBot v11 helpers
# =============================================================================

def _infer_ext(filename: str, default: str) -> str:
    """Return the (lower-case) extension in ``filename`` or *default*."""
    if filename and "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower().strip()
        if 2 <= len(ext) <= 6 and ext[1:].isalnum():
            return ext
    return default


def _parse_segments(segments: List[Dict[str, Any]], self_id: int) -> Dict[str, Any]:
    """Flatten an OneBot message array into our adapter's intermediate form.

    Covers every segment type NapCat emits on inbound (per
    https://napneko.github.io/develop/msg ): text / at / reply / face /
    image / record / video / file / mface / forward / music / share /
    location / json / xml / poke / contact / dice / rps / shake / redbag.

    Returns a dict:
      - ``text``          concatenated text incl. annotations for non-text content
      - ``mentioned_self`` True iff any ``at`` targets our QQ or "all"
      - ``reply_to``      message_id of the first reply segment, or None
      - ``media``         list of dicts {kind, ref, ext, filename}
    """
    text_parts: List[str] = []
    mentioned_self = False
    reply_to: Optional[str] = None
    media: List[Dict[str, str]] = []

    self_id_str = str(self_id)

    def add_media(kind: str, data: Dict[str, Any], default_ext: str) -> None:
        """Helper: extract media info from a segment and record it.

        For segments where NapCat doesn't auto-download (file/record/
        video), ``data.path`` will usually be empty — we store the
        ``file_id`` so the caller can call ``get_file`` on it.
        """
        file_value = str(data.get("file", "") or "")
        ref = data.get("path") or data.get("url") or file_value or ""
        file_id = str(data.get("file_id", "") or file_value or "")
        # NapCat uses file_unique OR file_id — prefer file_id for get_file.
        filename = str(data.get("file", "") or data.get("file_unique", "") or "")
        ext = _infer_ext(filename, default_ext)
        media.append({
            "kind": kind,
            "ref": str(ref) if ref else "",
            "file_id": file_id,
            "ext": ext,
            "filename": filename,
        })

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        stype = seg.get("type")
        data = seg.get("data") or {}

        # --- Identity / threading ------------------------------------
        if stype == "text":
            text_parts.append(str(data.get("text", "")))
        elif stype == "at":
            target_qq = str(data.get("qq", ""))
            if target_qq == self_id_str or target_qq == "all":
                mentioned_self = True
            else:
                text_parts.append(f"@{data.get('name') or target_qq} ")
        elif stype == "reply":
            reply_to = str(data.get("id", "")) or None

        # --- Emoji / stickers ----------------------------------------
        elif stype == "face":
            text_parts.append(f"[face:{data.get('id', '')}]")
        elif stype == "mface":
            # Market-face / sticker-pack emoji.  Per NapCat spec the data
            # fields are emoji_id / emoji_package_id / key / summary — NOT
            # a downloadable file.  If a ``url`` or ``path`` happens to be
            # present (some builds include it), grab it as a sticker for
            # vision; otherwise just render the summary text.
            summary = data.get("summary") or "[表情]"
            if data.get("url") or data.get("path"):
                add_media("sticker", data, ".png")
            text_parts.append(summary)

        # --- Media files ---------------------------------------------
        elif stype == "image":
            add_media("photo", data, ".jpg")
        elif stype == "record":
            # QQ voice messages are usually AMR; NapCat may transcode to OGG.
            add_media("voice", data, ".amr")
            text_parts.append("[语音]")
        elif stype == "video":
            add_media("video", data, ".mp4")
            text_parts.append("[视频]")
        elif stype == "file":
            add_media("document", data, "")
            fname = data.get("file", "file")
            text_parts.append(f"[文件: {fname}]")

        # --- Card / structured content -------------------------------
        elif stype == "forward":
            # Merged-forward message — content is a nested array.  We
            # render a compact placeholder; deep unpacking is out of
            # scope for now (can reach hundreds of nested messages).
            count = len(data.get("content") or [])
            text_parts.append(f"[合并转发({count}条)]")
        elif stype == "music":
            title = str(data.get("title") or data.get("url") or "")
            text_parts.append(f"[音乐分享: {title}]" if title else "[音乐分享]")
        elif stype == "share":
            title = data.get("title") or "(无标题)"
            url = data.get("url") or ""
            text_parts.append(f"[分享: {title} {url}".rstrip() + "]")
        elif stype == "location":
            lat = data.get("lat", "")
            lon = data.get("lon", "")
            title = data.get("title") or "位置"
            text_parts.append(f"[{title} @ {lat},{lon}]")
        elif stype == "json":
            text_parts.append("[卡片消息(json)]")
        elif stype == "xml":
            text_parts.append("[卡片消息(xml)]")

        # --- Interactions --------------------------------------------
        elif stype == "poke":
            target = data.get("qq", "")
            text_parts.append(f"[戳一戳 @{target}]")
        elif stype == "contact":
            c_type = data.get("type", "")
            c_id = data.get("id", "")
            text_parts.append(f"[推荐{'群' if c_type == 'group' else '好友'}: {c_id}]")
        elif stype == "dice":
            text_parts.append(f"[骰子: {data.get('result', '?')}]")
        elif stype == "rps":
            # 1=rock 2=scissors 3=paper per OneBot convention
            result_map = {1: "石头", 2: "剪刀", 3: "布"}
            r = data.get("result")
            text_parts.append(f"[猜拳: {result_map.get(r, r)}]")
        elif stype == "shake":
            text_parts.append("[窗口抖动]")
        elif stype == "redbag":
            title = data.get("title", "")
            text_parts.append(f"[红包: {title}]")
        # Unknown types: silently ignored (we log at gateway diag level if needed).

    return {
        "text": "".join(text_parts).strip(),
        "mentioned_self": mentioned_self,
        "reply_to": reply_to,
        "media": media,
    }


def _build_text_segments(text: str) -> List[Dict[str, Any]]:
    return [{"type": "text", "data": {"text": text}}]


# CQ-code parser.  When sending in ``array`` message-post format, NapCat does
# NOT auto-convert embedded ``[CQ:type,k=v]`` codes in a text segment into
# proper segments — they stay as literal text and the user sees
# ``[CQ:at,qq=12345]`` in chat.  The LLM does use these codes when it wants
# to @-mention someone (because that's the OneBot string-format idiom), so
# we translate them to real segments here.
#
# Escape convention (OneBot v11):
#   &amp;  → &
#   &#91;  → [
#   &#93;  → ]
#   &#44;  → ,           (only inside CQ params)
_CQ_RE = re.compile(r"\[CQ:([a-zA-Z_]+)((?:,[^,\]]+=[^,\]]*)*)\]")


def _cq_unescape(value: str) -> str:
    return (
        value.replace("&#44;", ",")
        .replace("&#91;", "[")
        .replace("&#93;", "]")
        .replace("&amp;", "&")
    )


def _text_unescape(value: str) -> str:
    # Outside CQ codes, commas don't need escaping.
    return value.replace("&#91;", "[").replace("&#93;", "]").replace("&amp;", "&")


def _expand_cq_codes_in_text(text: str) -> List[Dict[str, Any]]:
    """Split *text* into OneBot segments, converting embedded CQ codes.

    Validates each parsed segment so that placeholder text the LLM may
    quote from the prompt (e.g. ``[CQ:at,qq=<号>]`` when explaining the
    directive instead of using it) does NOT get sent to NapCat as a
    real action.  An at-segment with a non-numeric ``qq`` would trigger
    NapCat's "Get Uid Error" and abort the entire send (including the
    plain-text body), so we keep such segments as literal text and let
    them through as-is.
    """
    segments: List[Dict[str, Any]] = []
    last_end = 0
    for m in _CQ_RE.finditer(text):
        if m.start() > last_end:
            chunk = _text_unescape(text[last_end : m.start()])
            if chunk:
                segments.append({"type": "text", "data": {"text": chunk}})
        seg_type = m.group(1)
        params_str = m.group(2) or ""
        data: Dict[str, str] = {}
        for part in params_str.split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, _, v = part.partition("=")
            data[k.strip()] = _cq_unescape(v.strip())

        # Validation: at-segments need a real numeric QQ (or the special
        # "all" sentinel for @everyone).  Anything else — placeholder
        # text like `<号>`, an empty qq, a string with brackets — is
        # almost certainly the bot quoting the prompt rather than
        # mentioning a user, so emit the original CQ code text instead.
        if seg_type == "at":
            qq_value = data.get("qq", "").strip()
            if qq_value != "all" and not qq_value.isdigit():
                segments.append({
                    "type": "text",
                    "data": {"text": text[m.start():m.end()]},
                })
                last_end = m.end()
                continue

        segments.append({"type": seg_type, "data": data})
        last_end = m.end()
    tail = _text_unescape(text[last_end:])
    if tail:
        segments.append({"type": "text", "data": {"text": tail}})
    if not segments:
        segments = [{"type": "text", "data": {"text": text}}]
    return segments


def _parse_csv_ids(value: str) -> List[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


# =============================================================================
# Adapter
# =============================================================================

class OneBotNapCatAdapter(BasePlatformAdapter):
    """Platform adapter that drives a real QQ account via NapCat (OneBot v11)."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.ONEBOT_NAPCAT)

        self.ws_url = os.getenv("NAPCAT_WS_URL", NAPCAT_WS_URL)
        self.http_url = os.getenv("NAPCAT_HTTP_URL", NAPCAT_HTTP_URL).rstrip("/")
        self.access_token = os.getenv("NAPCAT_ACCESS_TOKEN", NAPCAT_ACCESS_TOKEN)
        self.autospawn = os.getenv("NAPCAT_AUTOSPAWN", "true").lower() in ("1", "true", "yes")

        # Populated after login_info query; may also come from env for early use
        self.self_id: Optional[int] = None
        self.self_name: Optional[str] = None
        env_self_id = os.getenv("NAPCAT_SELF_ID", "").strip()
        if env_self_id.isdigit():
            self.self_id = int(env_self_id)

        self._napcat_proc: Optional[asyncio.subprocess.Process] = None
        self._napcat_state: Optional[spawner.NapCatState] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        self._stopping = False  # True after disconnect() — watchdog exits
        # Separate HTTP client session for actions.  Each HTTP request is
        # independent — a single wedged action doesn't drag others down,
        # unlike WS where actions share a socket.
        self._http_client: Optional["aiohttp.ClientSession"] = None
        self._session: Optional["aiohttp.ClientSession"] = None
        self._ws: Optional["aiohttp.ClientWebSocketResponse"] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._connected_since: Optional[float] = None

        # Allow-list / policy
        self._allowed_groups = set(_parse_csv_ids(os.getenv("NAPCAT_ALLOWED_GROUPS", "")))

        # Passive mode: forward every group message (not just @-mentions /
        # replies) to the LLM and let it decide whether to respond.  LLM
        # outputs the literal string ``[SILENT]`` to stay silent on any
        # given message.  Costs a full LLM call per group message — expect
        # 100+ messages/day per active group ≈ $5-10/day.
        self._passive_group = os.getenv("NAPCAT_GROUP_PASSIVE", "false").lower() in (
            "1", "true", "yes",
        )
        if self._passive_group:
            logger.info(
                "NapCat: passive group mode ENABLED — every group message in "
                "allowed groups will invoke the LLM.  LLM should output "
                "[SILENT] to stay silent.",
            )

        # Per-user last inbound event directed at the bot, keyed by
        # (Hermes session, QQ user).  Bot self-pokes and third-party pokes do
        # not affect this state; it answers "what did this user last do to me?"
        self._last_to_bot_by_session_user: Dict[tuple[str, str], str] = {}

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            self._set_fatal_error("deps_missing", "aiohttp not installed", retryable=False)
            return False

        # 1. Start NapCat if we own it
        if self.autospawn:
            try:
                self._napcat_proc, self._napcat_state = await spawner.spawn()
                await spawner.wait_for_ws_ready(self.ws_url, state=self._napcat_state)
            except spawner.NapCatSpawnError as exc:
                # ``exc`` already carries actionable guidance — log it as
                # multiple lines so the user can actually read it in the
                # gateway console.
                for line in str(exc).splitlines():
                    logger.error("NapCat: %s", line)
                # DESIGN INVARIANT: be careful not to break this.
                # Every successful autospawn must be paired with termination
                # if startup fails before the adapter reaches connected
                # state; otherwise reconnect attempts accumulate live QQ
                # clients for the same account.
                if self._napcat_proc is not None:
                    try:
                        await spawner.terminate(self._napcat_proc)
                    except Exception:
                        logger.exception("NapCat: failed to terminate process after startup error")
                    finally:
                        self._napcat_proc = None
                        self._napcat_state = None
                self._set_fatal_error("napcat_spawn", str(exc), retryable=False)
                return False

        # 2. Connect to the OneBot WS (for RECEIVING events only — all
        #    outbound actions go via HTTP below).
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        try:
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(self.ws_url, headers=headers, heartbeat=30)
        except Exception as exc:
            logger.error("OneBot WS connect to %s failed: %s", self.ws_url, exc)
            await self._cleanup_connection()
            return False

        # 2b. Separate HTTP session for outbound actions.  We don't block
        # on http readiness here — NapCat starts HTTP server alongside WS,
        # so if WS is up HTTP should be too.  First action call will
        # surface any misconfiguration.
        self._http_client = aiohttp.ClientSession(
            headers=({"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}),
        )

        # 3. Learn our own QQ number (needed for @mention detection)
        try:
            login_info = await self._call_action("get_login_info", {}, timeout=10)
            if login_info and login_info.get("status") == "ok":
                self.self_id = int(login_info["data"].get("user_id", 0)) or self.self_id
                nickname = login_info["data"].get("nickname", "")
                self.self_name = str(nickname or "") or self.self_name
                logger.info("NapCat: ✓ connected as QQ %s (%s)", self.self_id, nickname or "?")
                try:
                    from gateway.sparrow_log import log_event as _sparrow_event
                    _sparrow_event("napcat_started", qq=self.self_id, nickname=nickname or "?")
                except Exception:
                    pass
            else:
                logger.warning("NapCat: ✓ WS connected (but get_login_info returned no data)")
        except Exception as exc:
            logger.warning("NapCat: ✓ WS connected (get_login_info failed: %s)", exc)

        if not self.self_id:
            logger.warning(
                "onebot_napcat: no self_id yet — @-mention detection disabled "
                "until login completes. Complete QR login at "
                "http://localhost:%s/webui and restart the gateway.",
                os.getenv("NAPCAT_WEBUI_PORT", "6099"),
            )

        # 4. Start event listener + watchdog
        self._listener_task = asyncio.create_task(self._listen_loop(), name="onebot_napcat_listener")
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(
                self._watchdog_loop(), name="onebot_napcat_watchdog"
            )
        self._connected_since = time.time()
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._stopping = True
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except (asyncio.CancelledError, Exception):
                pass
            self._watchdog_task = None

        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, Exception):
                pass
            self._listener_task = None

        await self._cleanup_connection()

        if self._napcat_proc is not None:
            await spawner.terminate(self._napcat_proc)
            self._napcat_proc = None

        self._mark_disconnected()

    async def _cleanup_connection(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
        # HTTP client is a separate resource — close it too.
        if self._http_client is not None:
            try:
                await self._http_client.close()
            except Exception:
                pass
            self._http_client = None

    # -------------------------------------------------------------------
    # Inbound event loop
    # -------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        assert self._ws is not None
        exit_reason = "normal"
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        event = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.warning("non-JSON WS frame from NapCat: %r", msg.data[:200])
                        continue
                    try:
                        await self._dispatch_event(event)
                    except Exception:
                        logger.exception("error handling OneBot event")
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    exit_reason = f"ws-{msg.type.name.lower()}"
                    break
            else:
                exit_reason = "ws-iter-ended"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            exit_reason = f"exception: {exc!r}"
            logger.exception("OneBot listener crashed")
        finally:
            if not self._stopping:
                logger.warning("NapCat: OneBot WS listener exited (%s) — watchdog will attempt recovery", exit_reason)
            self._mark_disconnected()

    async def _watchdog_loop(self) -> None:
        """Monitor the listener and recover NapCat when the WS dies.

        Recovery strategy:
          1. Try WS reconnect (fast — works if NapCat's OneBot server is
             still listening but the connection dropped).
          2. If WS reconnect fails, kill NapCat and respawn it from
             scratch.  This is the heavy hammer that recovers from
             cases where NapCat's OneBot module crashed internally
             (port 3001 stops listening entirely).
          3. Exponential backoff between attempts, capped at 60s.
        """
        backoff = 5.0
        while not self._stopping:
            try:
                # Wait for the listener to exit (WS died) or us to be stopped.
                if self._listener_task is None:
                    await asyncio.sleep(2)
                    continue
                try:
                    await self._listener_task
                except (asyncio.CancelledError, Exception):
                    pass

                if self._stopping:
                    return

                logger.warning("NapCat: watchdog starting recovery (backoff=%.0fs)", backoff)
                await asyncio.sleep(backoff)
                if self._stopping:
                    return

                recovered = await self._try_recover()
                if recovered:
                    logger.warning("NapCat: ✓ recovered — listener restarted")
                    backoff = 5.0
                else:
                    backoff = min(backoff * 2, 60.0)
                    logger.warning("NapCat: recovery failed, will retry in %.0fs", backoff)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("NapCat watchdog error")
                await asyncio.sleep(backoff)

    async def _try_recover(self) -> bool:
        """One recovery attempt: reconnect WS; fall back to respawning NapCat."""
        # Step 1: try a plain WS reconnect.
        await self._cleanup_connection()
        try:
            self._session = aiohttp.ClientSession()
            headers = {"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}
            self._ws = await self._session.ws_connect(self.ws_url, headers=headers, heartbeat=30)
            logger.warning("NapCat: watchdog reconnected WS (kept same NapCat process)")
        except Exception as exc:
            logger.warning("NapCat: WS reconnect failed (%s) — respawning NapCat", exc)
            # Step 2: respawn.
            if self._napcat_proc is not None:
                await spawner.terminate(self._napcat_proc)
                self._napcat_proc = None
            try:
                self._napcat_proc, self._napcat_state = await spawner.spawn()
                await spawner.wait_for_ws_ready(self.ws_url, state=self._napcat_state)
                self._session = aiohttp.ClientSession()
                headers = {"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}
                self._ws = await self._session.ws_connect(self.ws_url, headers=headers, heartbeat=30)
                logger.warning("NapCat: watchdog respawned NapCat and reconnected WS")
            except Exception:
                logger.exception("NapCat: respawn attempt failed")
                return False

        # Re-create HTTP client (cleanup_connection closed the previous one).
        self._http_client = aiohttp.ClientSession(
            headers=({"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}),
        )

        # Restart listener.
        self._listener_task = asyncio.create_task(self._listen_loop(), name="onebot_napcat_listener")
        self._mark_connected()
        return True

    async def _dispatch_event(self, event: Dict[str, Any]) -> None:
        # Ignore action-response frames that might still arrive on WS
        # (legacy — we send actions via HTTP now, but an old NapCat
        # version or misconfig could still deliver one on WS).
        if event.get("echo") is not None:
            return

        # Every OneBot event carries self_id — piggyback off messages to
        # set self_id if the initial get_login_info race lost.
        if self.self_id is None:
            sid = event.get("self_id")
            if isinstance(sid, int) and sid > 0:
                self.self_id = sid
                logger.info("NapCat self_id learned from event: %s", sid)

        post_type = event.get("post_type")
        if post_type == "message":
            await self._handle_message_event(event)
        elif post_type == "meta_event":
            self._handle_meta_event(event)
        elif post_type == "notice":
            await self._handle_notice_event(event)
        elif post_type == "request":
            # TODO: auto-accept friend requests / group invites — not MVP.
            pass

    def _handle_meta_event(self, event: Dict[str, Any]) -> None:
        meta_type = event.get("meta_event_type")
        if meta_type == "lifecycle" and event.get("sub_type") == "connect":
            sid = event.get("self_id")
            if sid:
                self.self_id = int(sid)
                logger.info("NapCat lifecycle: connected as %s", self.self_id)
        # heartbeat → ignore

    def _session_key_for_source(self, source: SessionSource) -> str:
        return build_session_key(
            source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )

    def _last_to_bot_key(self, source: SessionSource, user_id: str) -> tuple[str, str]:
        return (self._session_key_for_source(source), str(user_id))

    def _get_last_to_bot_kind(self, source: SessionSource, user_id: str) -> Optional[str]:
        return self._last_to_bot_by_session_user.get(self._last_to_bot_key(source, user_id))

    def _set_last_to_bot_kind(self, source: SessionSource, user_id: str, kind: str) -> None:
        self._last_to_bot_by_session_user[self._last_to_bot_key(source, user_id)] = kind

    def sanitize_transcript_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return sanitize_napcat_transcript_entry(entry)

    def extend_inbound_prefix(self, event: MessageEvent) -> Optional[str]:
        # Surface OneBot ``message_id`` as ``[msg=<id>]`` so the LLM has
        # the material the channel prompt promises (it documents
        # ``REPLY:<message_id>`` and points the LLM at this tag).  Without
        # this hook the directive is unreachable — the LLM either skips
        # quoting or improvises an invalid ID and the directive parser
        # silently drops the line into chat text.
        return f"[msg={event.message_id}]" if event.message_id else None

    def decorate_assistant_entry(
        self,
        entry: Dict[str, Any],
        send_result: Optional[SendResult],
    ) -> Dict[str, Any]:
        # Symmetry with inbound: rewrite the bot's own assistant entries
        # with the same `<time> [msg=<id>] <qq_id> <name>: <text>` shape
        # the LLM sees for user messages, so it can quote-reply its own
        # past messages with `REPLY:<own_id>` (it had no way to do this
        # before — the assistant rows in SQLite were bare raw text).
        #
        # Off by default: in-context learning makes the LLM mimic the
        # decorated format in its own output (see
        # ``is_assistant_header_enabled`` for the failure modes).  Opt
        # in with ``NAPCAT_ASSISTANT_HEADER=true`` once the model is
        # known to honor the anti-mimicry warning in the channel prompt.
        from .prompts import is_assistant_header_enabled
        if not is_assistant_header_enabled():
            return entry
        content = entry.get("content")
        if not isinstance(content, str) or not content:
            return entry

        # Use the send timestamp when we have one, else now — matches the
        # "when did the bot speak" semantics the LLM expects.
        ts_value = entry.get("timestamp")
        try:
            if isinstance(ts_value, (int, float)):
                ts = datetime.fromtimestamp(float(ts_value), tz=timezone.utc)
            elif isinstance(ts_value, str):
                ts = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = datetime.now(tz=timezone.utc)
        except Exception:
            ts = datetime.now(tz=timezone.utc)

        _stub = MessageEvent(
            text="",
            source=SessionSource(
                platform=Platform.ONEBOT_NAPCAT,
                chat_id="",
                chat_name="",
                chat_type="dm",
                user_id=str(self.self_id or ""),
                user_name=self.self_name or "",
            ),
            timestamp=ts,
        )
        time_label = format_message_event_time(_stub)

        parts: List[str] = [time_label]
        msg_id = getattr(send_result, "message_id", None) if send_result else None
        if msg_id:
            parts.append(f"[msg={msg_id}]")
        sender = self.inbound_sender_label(_stub)
        if sender:
            parts.append(sender)

        rewritten = dict(entry)
        rewritten["content"] = f"{' '.join(parts)}: {content}"
        return rewritten

    def inbound_sender_label(self, event: MessageEvent) -> Optional[str]:
        # Always render sender as ``<qq_id> <display_name>`` so the LLM
        # has BOTH the numeric id (needed for ``[CQ:at,qq=<号>]`` and
        # ``POKE:<号>`` directives) AND the human name it can use in
        # natural prose, even in DMs and per-user group sessions where
        # the runner default would suppress the sender slot.
        #
        # ``"SYSTEM"`` is a sentinel set by the poke handler for
        # gateway-injected events; render bare so it stays visually
        # distinct from real user messages.
        user_name = (event.source.user_name or "").strip() if event.source else ""
        user_id = (event.source.user_id or "").strip() if event.source else ""
        if user_name == "SYSTEM":
            return "SYSTEM"
        if user_id and user_name and user_name != user_id:
            return f"{user_id} {user_name}"
        # When ``user_name`` was empty or already fell back to the qq id
        # (NapCat does this when neither group card nor account nickname
        # exists), don't duplicate it.
        return user_id or user_name or None

    def _unresolved_reply_to_text(self, message_id: str) -> str:
        return f"unresolved QQ message [msg={message_id}]"

    def _format_quoted_message_line(
        self,
        data: Dict[str, Any],
        *,
        fallback_message_id: str,
    ) -> Optional[str]:
        """Render a get_msg response in the same shape as NapCat inbound lines."""
        message_id = str(data.get("message_id") or fallback_message_id or "").strip()

        raw_time = data.get("time")
        timestamp = datetime.now(tz=timezone.utc)
        if raw_time is not None:
            try:
                timestamp = datetime.fromtimestamp(float(raw_time), tz=timezone.utc)
            except Exception:
                pass

        sender = data.get("sender") or {}
        user_id = str(
            data.get("user_id")
            or sender.get("user_id")
            or sender.get("uin")
            or ""
        ).strip()
        if user_id == str(self.self_id or "") and self.self_name:
            user_name = self.self_name
        else:
            user_name = str(
                sender.get("card") or sender.get("nickname") or user_id
            ).strip()

        raw_message = data.get("message")
        text = ""
        if isinstance(raw_message, list):
            text = _parse_segments(raw_message, self.self_id or 0)["text"]
        elif isinstance(raw_message, str):
            text = raw_message.strip()
        if not text:
            fallback_text = data.get("raw_message")
            if isinstance(fallback_text, str):
                text = fallback_text.strip()
        if not text:
            text = "[non-text QQ message]"

        stub = MessageEvent(
            text=text,
            source=SessionSource(
                platform=Platform.ONEBOT_NAPCAT,
                chat_id="",
                chat_name="",
                chat_type=str(data.get("message_type") or ""),
                user_id=user_id,
                user_name=user_name,
            ),
            message_id=message_id or None,
            timestamp=timestamp,
        )

        parts: List[str] = [format_message_event_time(stub)]
        if message_id:
            parts.append(f"[msg={message_id}]")
        sender_label = self.inbound_sender_label(stub)
        if sender_label:
            parts.append(sender_label)
        return f"{' '.join(parts)}: {text}"

    async def _resolve_reply_to_text(self, message_id: Optional[str]) -> Optional[str]:
        if not message_id:
            return None

        # DESIGN INVARIANT: the LLM must see that this is a reply even when
        # NapCat cannot resolve the quoted message body.  Otherwise the current
        # message looks like a standalone utterance and loses conversational
        # intent.
        unresolved = self._unresolved_reply_to_text(message_id)
        try:
            resp = await self._call_action(
                "get_msg", {"message_id": message_id}, timeout=10,
            )
        except Exception as exc:
            logger.warning("NapCat: get_msg(%s) failed: %s", message_id, exc)
            return unresolved

        if not resp or resp.get("status") != "ok":
            logger.warning(
                "NapCat: get_msg(%s) returned %s",
                message_id, (resp or {}).get("message") or resp,
            )
            return unresolved

        data = resp.get("data") or {}
        if not isinstance(data, dict):
            return unresolved
        return self._format_quoted_message_line(
            data,
            fallback_message_id=message_id,
        ) or unresolved

    async def _handle_notice_event(self, event: Dict[str, Any]) -> None:
        # TEMP DIAG: log every raw notice (esp. notify/poke) so we can
        # correlate phantom events with our own outbound send_poke actions.
        if event.get("notice_type") == "notify" and event.get("sub_type") == "poke":
            try:
                logger.info(
                    "POKE-DIAG inbound notice ts=%.3f raw=%s",
                    time.monotonic(),
                    json.dumps(event, ensure_ascii=False, default=str),
                )
            except Exception:
                logger.info("POKE-DIAG inbound notice (json-dump-failed): %r", event)
            await self._handle_poke_notice(event)

    async def _handle_poke_notice(self, event: Dict[str, Any]) -> None:
        actor_id = str(event.get("user_id") or event.get("sender_id") or "")
        target_id = str(event.get("target_id") or "")
        if not actor_id or not target_id:
            return
        # In DMs, NapCat fills user_id/target_id with the DM peer's QQ
        # regardless of who actually performed the poke; the real sender
        # of the action is in `sender_id`.  So if sender_id matches the
        # bot, this notice is the echo of our own send_poke and should be
        # treated as actor_is_self (ignore echo, never reply to it).
        sender_id_str = str(event.get("sender_id") or "")
        actor_is_self = bool(self.self_id and (
            actor_id == str(self.self_id)
            or (sender_id_str and sender_id_str == str(self.self_id))
        ))

        raw_group_id = event.get("group_id")
        if raw_group_id is not None:
            group_id = str(raw_group_id)
            if self._allowed_groups and group_id not in self._allowed_groups:
                return
            chat_id = f"{CHAT_ID_GROUP_PREFIX}{group_id}"
            chat_type = "group"
            chat_name = event.get("group_name") or f"QQ group {group_id}"
        else:
            chat_id = f"{CHAT_ID_PRIVATE_PREFIX}{target_id if actor_is_self else actor_id}"
            chat_type = "dm"
            chat_name = event.get("sender_nick") or (target_id if actor_is_self else actor_id)

        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=target_id if actor_is_self else actor_id,
            user_name="SYSTEM",
        )
        target_is_self = bool(self.self_id and target_id == str(self.self_id))
        repeated_adjacent_self_poke = (
            not actor_is_self
            and target_is_self
            and self._get_last_to_bot_kind(source, actor_id) == "poke"
        )

        text = format_poke_event_text(
            actor_id=actor_id,
            target_id=target_id,
            actor_is_self=actor_is_self,
            target_is_self=target_is_self,
            repeated_adjacent=repeated_adjacent_self_poke,
        )
        system_instruction = build_poke_system_instruction(
            actor_id=actor_id,
            target_id=target_id,
            actor_is_self=actor_is_self,
            target_is_self=target_is_self,
            repeated_adjacent=repeated_adjacent_self_poke,
        )
        if system_instruction:
            text = f"{text}\n\n{system_instruction}"
        event_timestamp = datetime.now().astimezone()
        raw_time = event.get("time")
        if raw_time is not None:
            try:
                event_timestamp = datetime.fromtimestamp(float(raw_time), tz=timezone.utc)
            except Exception:
                pass

        msg_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=event,
            timestamp=event_timestamp,
            channel_prompt=build_poke_channel_prompt(
                actor_id=actor_id,
                target_id=target_id,
                actor_is_self=actor_is_self,
                target_is_self=target_is_self,
                repeated_adjacent=repeated_adjacent_self_poke,
            ),
        )
        msg_event._napcat_poke = {
            "actor_id": actor_id,
            "target_id": target_id,
            "actor_is_self": actor_is_self,
            "target_is_self": target_is_self,
            "repeated_adjacent": repeated_adjacent_self_poke,
        }

        if actor_is_self:
            logger.info(
                "POKE-DIAG decision=ACTOR_IS_SELF (ignore echo, no agent) actor=%s target=%s chat=%s",
                actor_id, target_id, chat_id,
            )
            # This is NapCat's echo of a poke we already requested via the
            # assistant's real output (`POKE:<qq>`). Do not append a second
            # assistant-looking "SYSTEM: you poked ..." line to history.
            msg_event.internal = True
            logger.debug(
                "NapCat: self-poke echo observed, not appending transcript event: %s",
                msg_event.text,
            )
            return

        if target_is_self and not repeated_adjacent_self_poke:
            logger.info(
                "POKE-DIAG decision=TARGET_IS_SELF_FRESH (forward to agent) actor=%s target=%s chat=%s",
                actor_id, target_id, chat_id,
            )
            self._set_last_to_bot_kind(source, actor_id, "poke")
            await self.handle_message(msg_event)
            return

        # Catch-all: either a repeated re-poke at the bot, or a third-party
        # poke (group A→B, or — rare — a real user-self-poke in a DM).  Forward
        # to the agent so it can witness the event; in passive mode the LLM
        # gates with `[SILENT]`.  DM phantoms from our own send_poke are no
        # longer reachable here because actor_is_self now also matches against
        # `sender_id`, which NapCat fills with the bot's QQ for those echoes.
        if target_is_self:
            self._set_last_to_bot_kind(source, actor_id, "poke")
        logger.info(
            "POKE-DIAG decision=CATCHALL_FORWARD (forward to agent) actor=%s target=%s chat=%s repeated=%s",
            actor_id, target_id, chat_id, repeated_adjacent_self_poke,
        )
        await self.handle_message(msg_event)

    async def _handle_message_event(self, event: Dict[str, Any]) -> None:
        user_id = str(event.get("user_id"))
        if self.self_id and user_id == str(self.self_id):
            return  # ignore self

        message_type = event.get("message_type")  # "group" | "private"
        segments = event.get("message") or []
        parsed = _parse_segments(segments, self.self_id or 0)

        sender = event.get("sender") or {}
        sender_name = (
            sender.get("card") or sender.get("nickname") or user_id
        )
        raw_message_id = str(event.get("message_id", ""))

        if message_type == "group":
            group_id = str(event.get("group_id"))

            # Group allowlist — if set, ignore chats outside it
            if self._allowed_groups and group_id not in self._allowed_groups:
                return

            chat_id = f"{CHAT_ID_GROUP_PREFIX}{group_id}"
            chat_type = "group"
            chat_name = event.get("group_name") or f"QQ group {group_id}"
            source = self.build_source(
                chat_id=chat_id,
                chat_name=chat_name,
                chat_type=chat_type,
                user_id=user_id,
                user_name=sender_name,
            )

            # Engagement rule: @-mention or reply always triggers.  In
            # passive mode, EVERY message is forwarded and the LLM is
            # prompted to decide whether to respond (or emit ``[SILENT]``).
            triggered = parsed["mentioned_self"] or bool(parsed["reply_to"])
            if not triggered and not self._passive_group:
                return
            # Any message that reaches the bot — triggered OR passive-mode
            # forward — must reset the per-user "last interaction kind".
            # The poke-adjacency check at line 955 reads this to decide
            # whether a follow-up poke from the same user is still
            # "adjacent" (chained) or has been broken by a real message.
            # Without resetting on passive-forwarded messages, a casual
            # group reply from the user gets ignored by the adjacency
            # tracker and the next poke wrongly looks chained.
            self._set_last_to_bot_kind(source, user_id, "message")
        elif message_type == "private":
            chat_id = f"{CHAT_ID_PRIVATE_PREFIX}{user_id}"
            chat_type = "dm"
            chat_name = sender_name
            source = self.build_source(
                chat_id=chat_id,
                chat_name=chat_name,
                chat_type=chat_type,
                user_id=user_id,
                user_name=sender_name,
            )
            self._set_last_to_bot_kind(source, user_id, "message")
        else:
            return  # unknown type

        # Resolve every media item to an absolute local file path.
        # Strategy (per NapCat docs):
        #   1. If data.path points at an existing local file → use it
        #      directly (NapCat pre-downloaded, e.g. images).
        #   2. Else if data.url / data.file is an HTTP(S) URL or file:// →
        #      download / strip to local.
        #   3. Else fall back to the ``get_file`` / ``get_record`` OneBot
        #      action with ``file_id``, which triggers NapCat's async
        #      download and returns a local path.  Required for most
        #      record / video / file segments — NapCat doesn't auto-
        #      download those.
        local_media_paths: List[str] = []
        media_types: List[str] = []
        for item in parsed["media"]:
            local = None
            tried_action_resolve = False
            if item["kind"] == "voice" and item.get("file_id"):
                # DESIGN INVARIANT: QQ/NapCat voice cache paths are usually
                # AMR.  Prefer NapCat's get_record(out_format=mp3) so Hermes
                # STT receives a supported format.
                local = await self._resolve_via_get_file(item["file_id"], item["kind"])
                tried_action_resolve = True
            if not local:
                local = await self._resolve_media_ref(
                    item["ref"], item["kind"], item["ext"], item.get("filename", "")
                )
            if not local and item.get("file_id") and not tried_action_resolve:
                local = await self._resolve_via_get_file(item["file_id"], item["kind"])
            if local:
                local_media_paths.append(local)
                media_types.append(item["kind"])
            else:
                logger.warning(
                    "NapCat: couldn't resolve %s ref=%r file_id=%r",
                    item["kind"], item.get("ref", "")[:80], item.get("file_id", ""),
                )

        # QQ-specific: transcribe voice messages here so the agent sees
        # plain text and the gateway's auto-TTS-on-voice path (designed
        # for voice-first platforms like Discord channels) doesn't fire
        # and produce a redundant TTS bubble alongside the text reply.
        message_text = parsed["text"]
        voice_indices = [i for i, k in enumerate(media_types) if k == "voice"]
        if voice_indices:
            voice_paths = [local_media_paths[i] for i in voice_indices]
            transcripts = await transcribe_voice_files(voice_paths)
            message_text = compose_voice_message_text(
                parsed_text=parsed["text"],
                transcripts=transcripts,
            )
            voice_set = set(voice_indices)
            local_media_paths = [
                p for i, p in enumerate(local_media_paths) if i not in voice_set
            ]
            media_types = [
                k for i, k in enumerate(media_types) if i not in voice_set
            ]

        # MessageType reflects the first / primary media item.  Hermes
        # uses this to route (e.g. vision on PHOTO, STT on VOICE).
        primary_type = MessageType.TEXT
        if media_types:
            primary_type = {
                "photo": MessageType.PHOTO,
                "voice": MessageType.VOICE,
                "video": MessageType.VIDEO,
                "document": MessageType.DOCUMENT,
                "sticker": MessageType.STICKER,
            }.get(media_types[0], MessageType.TEXT)

        event_timestamp = datetime.now().astimezone()
        raw_time = event.get("time")
        if raw_time is not None:
            try:
                event_timestamp = datetime.fromtimestamp(float(raw_time), tz=timezone.utc)
            except Exception:
                pass

        reply_to_text = await self._resolve_reply_to_text(parsed["reply_to"])

        msg_event = MessageEvent(
            text=message_text,
            message_type=primary_type,
            source=source,
            message_id=raw_message_id,
            raw_message=event,
            reply_to_message_id=parsed["reply_to"],
            reply_to_text=reply_to_text,
            media_urls=local_media_paths,
            media_types=media_types,
            timestamp=event_timestamp,
            channel_prompt=build_poke_channel_prompt(),
        )

        await self.handle_message(msg_event)

    async def send_poke(self, chat_id: str, user_id: str) -> SendResult:
        params: Dict[str, Any] = {"user_id": int(user_id)}
        if chat_id.startswith(CHAT_ID_GROUP_PREFIX):
            params["group_id"] = int(chat_id[len(CHAT_ID_GROUP_PREFIX):])
        elif not chat_id.startswith(CHAT_ID_PRIVATE_PREFIX):
            return SendResult(success=False, error=f"bad chat_id {chat_id!r}")

        # TEMP DIAG: timestamped record of every outbound poke so we can
        # match against inbound phantom notify_poke events.
        logger.info(
            "POKE-DIAG outbound send_poke ts=%.3f chat=%s target=%s params=%s",
            time.monotonic(), chat_id, user_id, params,
        )
        try:
            resp = await self._call_action("send_poke", params, timeout=30)
        except asyncio.TimeoutError:
            logger.info("POKE-DIAG outbound send_poke TIMEOUT chat=%s target=%s", chat_id, user_id)
            return SendResult(success=False, error="OneBot action timeout", retryable=True)
        except Exception as exc:
            logger.info("POKE-DIAG outbound send_poke EXC chat=%s target=%s exc=%s", chat_id, user_id, exc)
            return SendResult(success=False, error=str(exc), retryable=True)
        logger.info(
            "POKE-DIAG outbound send_poke RESULT ts=%.3f chat=%s target=%s status=%s resp=%s",
            time.monotonic(), chat_id, user_id, (resp or {}).get("status"),
            json.dumps(resp, ensure_ascii=False, default=str)[:300],
        )
        if (resp or {}).get("status") == "ok":
            return SendResult(success=True, raw_response=resp)
        return SendResult(success=False, error=str((resp or {}).get("wording") or resp), raw_response=resp)

    async def _resolve_via_get_file(
        self,
        file_id: str,
        kind: str,
    ) -> Optional[str]:
        """Ask NapCat to download the media and return its local path.

        NapCat does NOT auto-download most received media — we must
        request it via the OneBot action:
          * ``get_record`` for voice (supports ``out_format``)
          * ``get_file`` for image / video / file
        Both accept ``file_id`` (preferred) or ``file``.
        """
        if not file_id:
            return None

        action = "get_record" if kind == "voice" else "get_file"
        params: Dict[str, Any] = {"file_id": file_id}
        if kind == "voice":
            params["out_format"] = "mp3"  # Hermes STT handles mp3 well

        try:
            resp = await self._call_action(action, params, timeout=60)
        except Exception as exc:
            logger.warning("NapCat: %s(%s) failed: %s", action, file_id[:40], exc)
            return None
        if not resp or resp.get("status") != "ok":
            logger.warning("NapCat: %s(%s) returned %s", action, file_id[:40], (resp or {}).get("message"))
            return None

        data = resp.get("data") or {}
        path = data.get("file") or data.get("path") or ""
        if path and os.path.isfile(path):
            return path
        logger.warning("NapCat: %s returned invalid path: %r", action, path)
        return None

    async def _resolve_media_ref(
        self,
        ref: str,
        kind: str,
        ext: str,
        filename: str,
    ) -> Optional[str]:
        """Turn a NapCat ref (``path``/``url``/``file://``/``base64://``) into a local file path.

        Returns the absolute local path on success, None on failure.  The
        *kind* (photo/voice/video/document/sticker) picks the right cache
        helper so Hermes's media pipeline knows what type it is.
        """
        if not ref:
            return None

        # 1) Already an absolute local path that exists — NapCat's
        #    preferred format (``data.path``).  Return as-is.
        if os.path.isabs(ref) and os.path.isfile(ref):
            return ref

        # 2) file:// URI — strip scheme.
        if ref.startswith("file://"):
            local = ref[len("file://"):]
            if os.path.isfile(local):
                return local
            logger.warning("NapCat: file:// ref points to missing file: %s", local)
            return None

        # 3) base64:// data URI — decode + cache.
        if ref.startswith("base64://"):
            try:
                import base64 as _b64
                raw = _b64.b64decode(ref[len("base64://"):])
            except Exception as exc:
                logger.warning("NapCat: base64 decode failed (%s): %s", kind, exc)
                return None
            return self._cache_bytes(raw, kind, ext, filename)

        # 4) HTTP(S) URL — download, validate, cache.
        if ref.startswith(("http://", "https://")):
            try:
                if kind == "photo" or kind == "sticker":
                    return await cache_image_from_url(ref, ext=ext or ".jpg")
                if kind == "voice":
                    return await cache_audio_from_url(ref, ext=ext or ".amr")
                # video / document don't have _from_url helpers, fall
                # through to generic httpx fetch + _from_bytes.
                import httpx
                async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as c:
                    r = await c.get(ref)
                    r.raise_for_status()
                    return self._cache_bytes(r.content, kind, ext, filename)
            except Exception as exc:
                logger.warning("NapCat: download failed for %s (%s): %s", kind, ref[:100], exc)
                return None

        logger.warning("NapCat: unrecognized media ref kind=%s: %r", kind, ref[:200])
        return None

    @staticmethod
    def _cache_bytes(data: bytes, kind: str, ext: str, filename: str) -> Optional[str]:
        """Dispatch raw bytes to the right Hermes cache helper."""
        try:
            if kind == "photo" or kind == "sticker":
                return cache_image_from_bytes(data, ext=ext or ".jpg")
            if kind == "voice":
                return cache_audio_from_bytes(data, ext=ext or ".amr")
            if kind == "video":
                return cache_video_from_bytes(data, ext=ext or ".mp4")
            if kind == "document":
                return cache_document_from_bytes(data, filename=filename or "file.bin")
        except Exception as exc:
            logger.warning("NapCat: cache_%s_from_bytes failed: %s", kind, exc)
        return None

    # -------------------------------------------------------------------
    # Outbound (BasePlatformAdapter interface)
    # -------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Any = None,
    ) -> SendResult:
        # Passive-group-mode silence marker.  When enabled, LLM sees every
        # group message; if it decides the message doesn't warrant a
        # reply, it outputs exactly ``[SILENT]`` (optionally followed by
        # any reasoning text which we also swallow).  Short-circuit
        # without hitting QQ so we don't send "[SILENT]" as a real
        # message.
        stripped = (content or "").strip()
        if stripped == "[SILENT]" or stripped.startswith("[SILENT]\n") or stripped.startswith("[SILENT] "):
            logger.info("NapCat: [SILENT] marker, skipping send to %s", chat_id)
            try:
                from gateway.sparrow_log import log_event as _sparrow_event
                _sparrow_event("silent", platform="onebot_napcat", chat=chat_id)
            except Exception:
                pass
            return SendResult(success=True, message_id=None)

        # POKE / REPLY directives are the assistant's real output and stay
        # in history.  At delivery time only, consume them into actions and
        # strip them from chat text.
        #
        # REPLY semantics:
        #   `REPLY:<message_id>` → quote-reply that specific message
        #   `REPLY:none`         → explicitly skip the auto-quote
        # If multiple REPLY: lines appear, the last one wins.  Without any
        # REPLY directive we fall back to chat-type defaults: groups quote
        # the triggering message (matches QQ etiquette / disambiguates
        # who's being addressed); DMs don't quote (the conversation is
        # already 1:1, so a quote is pure noise).
        poke_ids: List[str] = []
        reply_override: Optional[str] = None
        reply_override_set = False
        text_lines: List[str] = []
        for line in (content or "").splitlines():
            poke_match = re.fullmatch(r"\s*POKE\s*:\s*(\d+)\s*", line, flags=re.IGNORECASE)
            if poke_match:
                poke_ids.append(poke_match.group(1))
                continue
            reply_match = re.fullmatch(
                r"\s*REPLY\s*:\s*(\d+|none)\s*", line, flags=re.IGNORECASE,
            )
            if reply_match:
                arg = reply_match.group(1).lower()
                reply_override = None if arg == "none" else arg
                reply_override_set = True
                continue
            text_lines.append(line)
        content = "\n".join(text_lines).strip()

        last_result = SendResult(success=True)
        for poke_id in poke_ids:
            last_result = await self.send_poke(chat_id, poke_id)

        if not content:
            return last_result

        if reply_override_set:
            effective_reply_to = reply_override
        elif chat_id.startswith(CHAT_ID_PRIVATE_PREFIX):
            effective_reply_to = None
        else:
            effective_reply_to = reply_to

        segments = _expand_cq_codes_in_text(content)
        if effective_reply_to:
            segments = [{"type": "reply", "data": {"id": effective_reply_to}}] + segments
        return await self._send_segments(chat_id, segments)

    async def send_typing(self, chat_id: str, metadata: Any = None) -> None:
        # OneBot v11 has no typing indicator — no-op.
        return

    async def stop_typing(self, chat_id: str) -> None:
        return

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        segments: List[Dict[str, Any]] = []
        if reply_to:
            segments.append({"type": "reply", "data": {"id": reply_to}})
        if caption:
            segments.extend(_expand_cq_codes_in_text(caption))
        segments.append({"type": "image", "data": {"file": image_url}})
        return await self._send_segments(chat_id, segments)

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        # QQ treats animated GIFs as images — reuse send_image.
        return await self.send_image(chat_id, animation_url, caption, reply_to, metadata)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        # OneBot accepts ``file://`` URIs for local files.
        url = image_path if image_path.startswith("file://") else f"file://{os.path.abspath(image_path)}"
        return await self.send_image(chat_id, url, caption, reply_to=reply_to)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        url = audio_path if audio_path.startswith(("file://", "http://", "https://")) else f"file://{os.path.abspath(audio_path)}"
        segments: List[Dict[str, Any]] = []
        if reply_to:
            segments.append({"type": "reply", "data": {"id": reply_to}})
        # Caption separate from voice — OneBot record segment has no caption.
        if caption:
            segments.extend(_expand_cq_codes_in_text(caption))
        segments.append({"type": "record", "data": {"file": url}})
        return await self._send_segments(chat_id, segments)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        url = video_path if video_path.startswith(("file://", "http://", "https://")) else f"file://{os.path.abspath(video_path)}"
        segments: List[Dict[str, Any]] = []
        if reply_to:
            segments.append({"type": "reply", "data": {"id": reply_to}})
        if caption:
            segments.extend(_expand_cq_codes_in_text(caption))
        segments.append({"type": "video", "data": {"file": url}})
        return await self._send_segments(chat_id, segments)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        # OneBot ``file`` segment in groups maps to group-file upload; for
        # DMs it attaches.  The ``file_name`` field is a send-side override
        # (NapCat reads it from ``name``).  Caption has no native slot in
        # the file segment, so we send a separate text segment first if
        # the caller provided one.
        url = file_path if file_path.startswith(("file://", "http://", "https://")) else f"file://{os.path.abspath(file_path)}"
        segments: List[Dict[str, Any]] = []
        if reply_to:
            segments.append({"type": "reply", "data": {"id": reply_to}})
        if caption:
            segments.extend(_expand_cq_codes_in_text(caption))
        file_data: Dict[str, Any] = {"file": url}
        if file_name:
            file_data["name"] = file_name
        segments.append({"type": "file", "data": file_data})
        return await self._send_segments(chat_id, segments)

    async def send_combined(
        self,
        chat_id: str,
        text: str,
        images: Sequence[Tuple[str, str]] = (),
        media_files: Sequence[Tuple[str, bool]] = (),
        local_files: Sequence[str] = (),
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SendResult]:
        """OneBot v11 atomic delivery: build one segment array containing
        ``reply`` + caption text + every image / audio / video / file
        attachment, then send as a single ``send_group_msg`` /
        ``send_private_msg`` action.

        This produces the native QQ UX (single chat bubble; multiple
        images render as a grid) instead of one bubble per attachment.

        ``text`` follows the same POKE/REPLY/SILENT directive parsing as
        :meth:`send` — POKE lines fire side-effect actions before the
        main send, REPLY:<id> overrides ``reply_to``, [SILENT] short-
        circuits.  This keeps the combined path semantically aligned
        with the text-only path.
        """
        # Split off control directives the same way send() does, so a
        # combined response carrying POKE: / REPLY: / [SILENT] behaves
        # identically to the text-only path.
        stripped = (text or "").strip()
        if stripped == "[SILENT]" or stripped.startswith("[SILENT]\n") or stripped.startswith("[SILENT] "):
            logger.info("NapCat: [SILENT] marker, skipping combined send to %s", chat_id)
            try:
                from gateway.sparrow_log import log_event as _sparrow_event
                _sparrow_event("silent", platform="onebot_napcat", chat=chat_id)
            except Exception:
                pass
            return SendResult(success=True, message_id=None)

        poke_ids: List[str] = []
        reply_override: Optional[str] = None
        reply_override_set = False
        text_lines: List[str] = []
        for line in (text or "").splitlines():
            poke_match = re.fullmatch(r"\s*POKE\s*:\s*(\d+)\s*", line, flags=re.IGNORECASE)
            if poke_match:
                poke_ids.append(poke_match.group(1))
                continue
            reply_match = re.fullmatch(
                r"\s*REPLY\s*:\s*(\d+|none)\s*", line, flags=re.IGNORECASE,
            )
            if reply_match:
                arg = reply_match.group(1).lower()
                reply_override = None if arg == "none" else arg
                reply_override_set = True
                continue
            text_lines.append(line)
        clean_text = "\n".join(text_lines).strip()

        last_poke_result = SendResult(success=True)
        for poke_id in poke_ids:
            last_poke_result = await self.send_poke(chat_id, poke_id)

        if reply_override_set:
            effective_reply_to = reply_override
        elif chat_id.startswith(CHAT_ID_PRIVATE_PREFIX):
            effective_reply_to = None
        else:
            effective_reply_to = reply_to

        # If after directive stripping there's no text AND no media,
        # the whole response was directives (e.g. POKE only).  Return
        # the last action's result instead of sending an empty message.
        has_attachments = bool(images) or bool(media_files) or bool(local_files)
        if not clean_text and not has_attachments:
            return last_poke_result

        segments: List[Dict[str, Any]] = []
        if effective_reply_to:
            segments.append({"type": "reply", "data": {"id": effective_reply_to}})
        if clean_text:
            segments.extend(_expand_cq_codes_in_text(clean_text))

        # Image URLs (markdown ![alt](url) extracted by base.py).
        for image_url, _alt in images:
            segments.append({"type": "image", "data": {"file": image_url}})

        # MEDIA:<path> tags — route by extension.
        _AUDIO_EXTS = {'.ogg', '.opus', '.mp3', '.wav', '.m4a'}
        _VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'}
        _IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        for media_path, _is_voice in media_files:
            ext = os.path.splitext(media_path)[1].lower()
            url = media_path if media_path.startswith(("file://", "http://", "https://")) else f"file://{os.path.abspath(media_path)}"
            if ext in _AUDIO_EXTS:
                segments.append({"type": "record", "data": {"file": url}})
            elif ext in _VIDEO_EXTS:
                segments.append({"type": "video", "data": {"file": url}})
            elif ext in _IMAGE_EXTS:
                segments.append({"type": "image", "data": {"file": url}})
            else:
                segments.append({"type": "file", "data": {"file": url}})

        # Bare local file paths auto-detected in body — same routing
        # except no audio dispatch (extract_local_files never matches
        # audio extensions).
        for file_path in local_files:
            ext = os.path.splitext(file_path)[1].lower()
            url = file_path if file_path.startswith(("file://", "http://", "https://")) else f"file://{os.path.abspath(file_path)}"
            if ext in _IMAGE_EXTS:
                segments.append({"type": "image", "data": {"file": url}})
            elif ext in _VIDEO_EXTS:
                segments.append({"type": "video", "data": {"file": url}})
            else:
                segments.append({"type": "file", "data": {"file": url}})

        return await self._send_segments(chat_id, segments)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        if chat_id.startswith(CHAT_ID_GROUP_PREFIX):
            gid = chat_id[len(CHAT_ID_GROUP_PREFIX):]
            try:
                resp = await self._call_action("get_group_info", {"group_id": int(gid)})
                data = (resp or {}).get("data") or {}
                return {"chat_id": chat_id, "name": data.get("group_name", gid), "type": "group"}
            except Exception:
                return {"chat_id": chat_id, "name": gid, "type": "group"}
        elif chat_id.startswith(CHAT_ID_PRIVATE_PREFIX):
            uid = chat_id[len(CHAT_ID_PRIVATE_PREFIX):]
            try:
                resp = await self._call_action("get_stranger_info", {"user_id": int(uid)})
                data = (resp or {}).get("data") or {}
                return {"chat_id": chat_id, "name": data.get("nickname", uid), "type": "private"}
            except Exception:
                return {"chat_id": chat_id, "name": uid, "type": "private"}
        return {"chat_id": chat_id, "name": chat_id, "type": "unknown"}

    # -------------------------------------------------------------------
    # Internal: OneBot action RPC
    # -------------------------------------------------------------------

    @staticmethod
    def _summarize_segments(segments: List[Dict[str, Any]]) -> str:
        """Compact ``type:target`` representation for sparrow logging.

        Examples:
          [{type:"text", data:{text:"hi"}}]                     → "text"
          [{type:"reply", data:{id:"123"}},
           {type:"text", data:{text:"hi"}},
           {type:"image", data:{file:"file:///tmp/a.jpg"}}]     → "reply:123,text,image:file:///tmp/a.jpg"

        Text segments are listed without their content (full text is
        already on the [user_msg]/[assistant_msg] lines elsewhere in
        sparrow.log; duplicating it here would just bloat the timeline).
        """
        parts: List[str] = []
        for seg in segments or ():
            t = seg.get("type", "?")
            data = seg.get("data") or {}
            if t == "text":
                parts.append("text")
            elif t == "reply":
                parts.append(f"reply:{data.get('id', '?')}")
            elif t == "at":
                parts.append(f"at:{data.get('qq', '?')}")
            elif t in ("image", "record", "video", "file"):
                parts.append(f"{t}:{data.get('file', '?')}")
            else:
                parts.append(t)
        return ",".join(parts)

    async def _send_segments(
        self,
        chat_id: str,
        segments: List[Dict[str, Any]],
    ) -> SendResult:
        if chat_id.startswith(CHAT_ID_GROUP_PREFIX):
            action = "send_group_msg"
            params = {"group_id": int(chat_id[len(CHAT_ID_GROUP_PREFIX):]), "message": segments}
        elif chat_id.startswith(CHAT_ID_PRIVATE_PREFIX):
            action = "send_private_msg"
            params = {"user_id": int(chat_id[len(CHAT_ID_PRIVATE_PREFIX):]), "message": segments}
        else:
            return SendResult(success=False, error=f"bad chat_id {chat_id!r}")

        # Lazy import: log_event is a no-op before init_sparrow_log().
        try:
            from gateway.sparrow_log import log_event as _sparrow_event
        except Exception:
            def _sparrow_event(*_a, **_kw):  # type: ignore[no-redef]
                pass

        # Starting carries the context (chat / action / segments).
        # Finished carries only the outcome — operators read the pair
        # via adjacency, no need to repeat the bulky segment list.
        _sparrow_event(
            "napcat_send_starting",
            chat=chat_id, action=action,
            segments=self._summarize_segments(segments),
        )

        try:
            resp = await self._call_action(action, params, timeout=30)
        except asyncio.TimeoutError:
            _sparrow_event("napcat_send_finished", status="error", error="timeout")
            return SendResult(success=False, error="OneBot action timeout", retryable=True)
        except Exception as exc:
            _sparrow_event("napcat_send_finished", status="error", error=repr(exc))
            return SendResult(success=False, error=str(exc), retryable=True)

        if (resp or {}).get("status") == "ok":
            data = resp.get("data") or {}
            message_id = str(data.get("message_id", "")) or None
            _sparrow_event("napcat_send_finished", status="ok", message_id=message_id)
            return SendResult(success=True, message_id=message_id, raw_response=resp)

        err = (resp or {}).get("message", "OneBot action failed")
        _sparrow_event("napcat_send_finished", status="failed", error=err)
        return SendResult(success=False, error=err, raw_response=resp)

    async def _call_action(
        self,
        action: str,
        params: Dict[str, Any],
        timeout: float = 90,
    ) -> Optional[Dict[str, Any]]:
        """Invoke a OneBot action via HTTP — primary action transport.

        HTTP is preferred over WS because each request is independent:
        a single slow/wedged action doesn't block others like it does on
        WS (where actions share one socket).  The WS listener stays up
        for inbound events only.

        Returns the parsed JSON response or raises.
        """
        if self._http_client is None or self._http_client.closed:
            raise RuntimeError("OneBot HTTP client not initialized")

        url = f"{self.http_url}/{action}"
        try:
            async with self._http_client.post(
                url,
                json=params,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                # OneBot spec: always application/json; status=ok|failed.
                # Non-2xx is rare but possible (token mismatch, bad route).
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(
                        f"OneBot HTTP {resp.status} for {action}: {body[:200]}"
                    )
                return await resp.json()
        except asyncio.TimeoutError:
            raise RuntimeError(f"OneBot HTTP action {action} timed out after {timeout}s")
