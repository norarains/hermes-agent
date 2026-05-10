"""NapCat poke helpers.

Keep QQ poke behavior out of gateway core so rebases against upstream Hermes
only have to carry the generic adapter hook points.
"""

from __future__ import annotations


def format_poke_event_text(
    *,
    actor_id: str,
    target_id: str,
    actor_is_self: bool = False,
    target_is_self: bool = False,
    repeated_adjacent: bool = False,
) -> str:
    del repeated_adjacent
    if actor_is_self:
        return f"you poked {target_id}"
    if target_is_self:
        return f"{actor_id} poked you"
    return f"{actor_id} poked {target_id}"
