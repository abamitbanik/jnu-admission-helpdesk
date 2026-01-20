"""Action / automation catalog.

এই ফাইলটা intentionally lightweight রাখা হয়েছে যাতে আপনার বর্তমান backend–frontend কানেকশন ভেঙে না যায়।

আপনি পরে চাইলে এখানে official site scrape/check logic যোগ করে:
  - subscriptions collection এ save
  - scheduler থেকে periodic check
  - notifications collection এ insert
করতে পারবেন।
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
