"""
OneBot v11 / NapCat platform adapter for Hermes Agent.

Connects Hermes to a real QQ account via NapCat (https://github.com/NapNeko/NapCatQQ),
an NTQQ-based OneBot v11 implementation that runs as a subprocess inside the
Hermes container.  Unlike the official QQ Bot API (``gateway.platforms.qqbot``)
which exposes only a restricted bot identity with no group-join capability,
this adapter drives a normal QQ user account and can do everything the QQ
desktop client can: join groups, send/receive images/voice/files, manage
group admin state, etc.

⚠  Third-party QQ bot frameworks violate Tencent's ToS in spirit; accounts
   can be temporarily restricted or permanently banned.  Use a dedicated
   secondary account, not your primary.  Prefer residential IP addresses
   over datacenter IPs (DC IPs are heavily flagged post-2025-09).
"""

from .adapter import OneBotNapCatAdapter, check_onebot_napcat_requirements  # noqa: F401

__all__ = [
    "OneBotNapCatAdapter",
    "check_onebot_napcat_requirements",
]
