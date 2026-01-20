import os
import time
import uuid
from typing import Any, Dict, List, Optional


def _col_chat_messages(db):
    return db[os.getenv("MONGODB_CHAT_MESSAGES_COLLECTION", "chat_messages").strip()]


def _col_slot_state(db):
    return db[os.getenv("MONGODB_SLOT_STATE_COLLECTION", "slot_state").strip()]


def _col_notifications(db):
    return db[os.getenv("MONGODB_NOTIFICATIONS_COLLECTION", "notifications").strip()]


def ensure_session_id(session_id: Optional[str]) -> str:
    """Return a stable session_id. If missing, create a new UUID."""
    sid = (session_id or "").strip()
    return sid if sid else str(uuid.uuid4())


def save_chat_turn(
    db,
    session_id: str,
    user_message: str,
    bot_answer: str,
    intent_id: Optional[str],
    slots: Optional[Dict[str, Any]],
    source: str,
) -> None:
    """Persist one chat turn for memory (TTL is handled by MongoDB index if you set it)."""
    try:
        doc = {
            "session_id": session_id,
            "user_message": user_message,
            "bot_answer": bot_answer,
            "intent_id": intent_id,
            "slots": slots or {},
            "source": source,
            "created_at": int(time.time()),
        }
        _col_chat_messages(db).insert_one(doc)
    except Exception:
        # never break chat if memory write fails
        return


def load_recent_turns(db, session_id: str, limit: int = 8) -> List[Dict[str, Any]]:
    try:
        cur = (
            _col_chat_messages(db)
            .find({"session_id": session_id}, {"_id": 0})
            .sort("created_at", -1)
            .limit(max(1, int(limit)))
        )
        rows = list(cur)
        rows.reverse()
        return rows
    except Exception:
        return []


def load_slot_state(db, session_id: str) -> Dict[str, Any]:
    try:
        return _col_slot_state(db).find_one({"session_id": session_id}, {"_id": 0}) or {}
    except Exception:
        return {}


def upsert_slot_state(db, session_id: str, state: Dict[str, Any]) -> None:
    try:
        state = dict(state or {})
        state["session_id"] = session_id
        state["updated_at"] = int(time.time())
        _col_slot_state(db).update_one({"session_id": session_id}, {"$set": state}, upsert=True)
    except Exception:
        return


def set_last_unit(db, session_id: str, unit: str) -> None:
    """Persist the last resolved unit for this session (A/B/C/D/E)."""
    try:
        u = (unit or "").strip().upper()
        if u not in {"A", "B", "C", "D", "E"}:
            return
        _col_slot_state(db).update_one(
            {"session_id": session_id},
            {"$set": {"session_id": session_id, "last_unit": u, "updated_at": int(time.time())}},
            upsert=True,
        )
    except Exception:
        return


def clear_slot_state(db, session_id: str) -> None:
    """Clear only the active slot-filling flow but keep long-lived session context like last_unit."""
    try:
        _col_slot_state(db).update_one(
            {"session_id": session_id},
            {
                "$unset": {
                    "pending_intent": "",
                    "required_slots": "",
                    "slot_questions": "",
                    "filled_slots": "",
                    "missing_slots": "",
                },
                "$set": {"updated_at": int(time.time())},
            },
            upsert=True,
        )
    except Exception:
        return


def peek_notifications(db, session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Return newest notifications for this session. Does NOT mark as read (safe default)."""
    try:
        cur = (
            _col_notifications(db)
            .find({"session_id": session_id}, {"_id": 0})
            .sort("created_at", -1)
            .limit(max(1, int(limit)))
        )
        return list(cur)
    except Exception:
        return []
