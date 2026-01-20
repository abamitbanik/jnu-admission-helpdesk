import re
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Extractors
# -----------------------------


def normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[_\-]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def extract_unit_code(text: str):
    if not text:
        return None

    t = text.strip().upper()

        # single letter only (e.g., "a", "D")
    if re.fullmatch(r"[a-e]", t.strip()):
        return t.strip()
    
    # direct single-letter unit
    if t in ["A", "B", "C", "D", "E"]:
        return t

    # cases like: "a unit", "unit a"
    for u in ["A", "B", "C", "D", "E"]:
        if f"{u} UNIT" in t or f"UNIT {u}" in t:
            return u

    return None


def extract_year(text: str) -> Optional[str]:
    """Extract academic year in formats like 2025-26, 2025-2026, ২০২৫-২৬."""
    t = (text or "").strip()

    # English digits: 2025-26 or 2025-2026 or 2025/26
    m = re.search(r"\b(20\d{2})\s*[-/ ]\s*(\d{2,4})\b", t)
    if m:
        y1 = m.group(1)
        y2 = m.group(2)
        if len(y2) == 2:
            y2 = y1[:2] + y2
        return f"{y1}-{y2}"

    # Single year like 2026
    m = re.search(r"\b(20\d{2})\b", t)
    if m:
        return m.group(1)

    # Bangla digits (very common): ২০২৫-২৬
    bn_map = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
    t2 = t.translate(bn_map)
    m = re.search(r"\b(20\d{2})\s*[-/ ]\s*(\d{2,4})\b", t2)
    if m:
        y1 = m.group(1)
        y2 = m.group(2)
        if len(y2) == 2:
            y2 = y1[:2] + y2
        return f"{y1}-{y2}"

    return None


def extract_roll(text: str) -> Optional[str]:
    """Extract a likely roll / application ID (kept conservative)."""
    t = normalize(text)
    # patterns: roll 123456, application 12345
    m = re.search(r"\b(roll|application|app|id)\s*[:# ]\s*(\d{4,12})\b", t)
    if m:
        return m.group(2)
    # standalone long digits (avoid dates)
    m = re.search(r"\b(\d{6,12})\b", t)
    if m:
        return m.group(1)
    return None


EXTRACTORS = {
    "unit": extract_unit_code,
    "year": extract_year,
    "roll": extract_roll,
}


# -----------------------------
# Slot engine
# -----------------------------


def merge_slots(*slot_dicts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in slot_dicts:
        if not d:
            continue
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            out[k] = v
    return out


def infer_required_slots(intent_doc: Dict[str, Any]) -> List[str]:
    rs = intent_doc.get("required_slots") or []
    if isinstance(rs, list):
        return [str(x).strip() for x in rs if str(x).strip()]
    return []


def infer_slot_questions(intent_doc: Dict[str, Any]) -> Dict[str, str]:
    sq = intent_doc.get("slot_questions") or {}
    if isinstance(sq, dict):
        return {str(k).strip(): str(v).strip() for k, v in sq.items() if str(k).strip() and str(v).strip()}
    return {}


DEFAULT_SLOT_QUESTIONS = {
    "unit": "আপনি কোন ইউনিট (A/B/C/D/E) সম্পর্কে জানতে চাচ্ছেন?",
    "year": "কোন শিক্ষাবর্ষ/সেশন (যেমন 2025-2026)?",
    "roll": "আপনার রোল/অ্যাপ্লিকেশন আইডি কত?",
}


def fill_slots_from_text(text: str, required_slots: List[str]) -> Dict[str, Any]:
    filled: Dict[str, Any] = {}
    for s in required_slots:
        fn = EXTRACTORS.get(s)
        if not fn:
            continue
        v = fn(text)
        if v is not None:
            filled[s] = v
    return filled


def compute_missing_slots(required_slots: List[str], filled_slots: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for s in required_slots:
        v = filled_slots.get(s)
        if v is None:
            missing.append(s)
            continue
        if isinstance(v, str) and not v.strip():
            missing.append(s)
    return missing


def next_slot_question(
    missing_slots: List[str],
    intent_slot_questions: Dict[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    if not missing_slots:
        return None, None
    slot = missing_slots[0]
    q = intent_slot_questions.get(slot) or DEFAULT_SLOT_QUESTIONS.get(slot) or f"{slot} দিন"
    return slot, q
