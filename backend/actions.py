"""Action / automation catalog.

backend–frontend কানেকশন।

"""

from typing import Any, Dict, Optional, Tuple


def detect_action_intent(user_text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return (action_type, params) if user is asking for an automation.

    Conservative heuristic: only triggers on clear 'notify me' wording.
    """
    t = (user_text or "").strip().lower()
    if not t:
        return None

    notify_markers = [
        "notify", "remind", "alert", "update দিলে", "জানাবেন", "জানাবা", "মনে করিয়ে", "নোটিফাই",
    ]
    if not any(m in t for m in notify_markers):
        return None

    # simple mapping examples
    if "seat" in t and "plan" in t:
        return ("check_seatplan_updates", {})
    if "circular" in t or "নোটিশ" in t or "সার্কুলার" in t:
        return ("check_circular_updates", {})

    return ("generic_updates", {})
