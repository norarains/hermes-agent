"""Constants shared across the onebot_napcat adapter modules."""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Version — bump on functional changes
# ---------------------------------------------------------------------------
ONEBOT_NAPCAT_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Paths (inside the Hermes container — set by control/hermes-agent/Dockerfile.patched)
# ---------------------------------------------------------------------------
NAPCAT_PATH = os.environ.get("HERMES_NAPCAT_PATH", "/app")
QQ_BINARY = os.environ.get("HERMES_QQ_BINARY", "/opt/QQ/qq")

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
# OneBot v11 forward WebSocket endpoint that NapCat exposes — used by the
# adapter only for receiving inbound events (messages / notices / lifecycle).
# All outbound actions go via HTTP to keep them independent of WS state.
NAPCAT_WS_URL = os.environ.get("NAPCAT_WS_URL", "ws://127.0.0.1:3001")

# OneBot v11 HTTP server NapCat exposes.  Each request is independent —
# WS wedging doesn't affect HTTP throughput.  We use this for ALL actions
# (send_msg, get_file, get_login_info, …).
NAPCAT_HTTP_URL = os.environ.get("NAPCAT_HTTP_URL", "http://127.0.0.1:3000")

# WebUI port (HTTP) — NapCat's admin dashboard where users scan QR / manage
# login.  Exposed on the container by control/run.sh, bound to host's 127.0.0.1.
NAPCAT_WEBUI_PORT = int(os.environ.get("NAPCAT_WEBUI_PORT", "6099"))

# Optional shared-secret the adapter passes in the WS Authorization header
# (matches NapCat's onebot11.json `token` field).  Empty disables auth.
NAPCAT_ACCESS_TOKEN = os.environ.get("NAPCAT_ACCESS_TOKEN", "")

# ---------------------------------------------------------------------------
# Chat ID scheme
# ---------------------------------------------------------------------------
# Normalized chat_id format shared with the rest of Hermes (SessionSource,
# cron delivery, send_message tool).  We use a flat string with a prefix so
# callers don't need to know whether a chat is a group or DM.
CHAT_ID_GROUP_PREFIX = "group_"
CHAT_ID_PRIVATE_PREFIX = "private_"

# ---------------------------------------------------------------------------
# Messaging limits
# ---------------------------------------------------------------------------
# QQ server-side text cap is ~4500 UTF-8 chars.  Leave headroom for CQ codes.
MAX_MESSAGE_LENGTH = 4000

# ---------------------------------------------------------------------------
# Subprocess management
# ---------------------------------------------------------------------------
NAPCAT_SPAWN_TIMEOUT = 90.0  # seconds to wait for ws endpoint to come up
NAPCAT_SHUTDOWN_TIMEOUT = 15.0  # seconds to wait for clean termination
