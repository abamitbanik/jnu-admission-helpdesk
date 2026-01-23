import os
import re
import time
import json
import uuid
import logging
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient
from bson import ObjectId
from language_style import paraphrase_grounded_answer

from google import genai  # pip install google-genai

# --- New: memory + slots + actions (non-breaking additions) ---
from memory import (
    ensure_session_id,
    load_recent_turns,
    load_slot_state,
    upsert_slot_state,
    clear_slot_state,
    save_chat_turn,
    peek_notifications,
)
from slots import (
    infer_required_slots,
    infer_slot_questions,
    fill_slots_from_text,
    merge_slots,
    compute_missing_slots,
    next_slot_question,
)
from actions import detect_action_intent


# Force-load .env from backend/.env (no matter where uvicorn runs)
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


# ---------- Logging (production-friendly JSON logs) ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper().strip()
logging.basicConfig(level=LOG_LEVEL, format="%(message)s")
logger = logging.getLogger("jnu_helpdesk")


def log_event(event: str, payload: Dict[str, Any]) -> None:
    rec = {"event": event, **payload}
    try:
        logger.info(json.dumps(rec, ensure_ascii=False))
    except Exception:
        # fallback
        logger.info(f"{event}: {payload}")


# ---------- Simple TTL Cache (in-memory) ----------
class TTLCache:
    def __init__(self, ttl_seconds: int = 600, max_items: int = 2000):
        self.ttl = max(10, int(ttl_seconds))
        self.max_items = max(100, int(max_items))
        self._store: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        item = self._store.get(key)
        if not item:
            return None
        exp, value = item
        if exp < now:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Dict[str, Any]) -> None:
        now = time.time()
        # crude eviction
        if len(self._store) >= self.max_items:
            # remove expired first
            expired = [k for k, (exp, _) in self._store.items() if exp < now]
            for k in expired[:200]:
                self._store.pop(k, None)
            # if still big, pop arbitrary
            while len(self._store) >= self.max_items:
                self._store.pop(next(iter(self._store)))
        self._store[key] = (now + self.ttl, value)


CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600").strip() or "600")
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "2000").strip() or "2000")
cache = TTLCache(ttl_seconds=CACHE_TTL_SECONDS, max_items=CACHE_MAX_ITEMS)


# ---------- Helpers ----------
def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[’‘]", "'", text)
    text = re.sub(r"[_\-]+", " ", text)  # better for seat_plan / unit-e
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_typo_tolerant(text: str) -> str:
    """
    Small, safe typo-tolerance for Roman Bangla / common joining.
    Does NOT change meaning; only normalizes typical slips.
    """
    t = normalize(text)
    # common roman bangla typos
    t = re.sub(r"\bkoba\b", "kobe", t)
    t = re.sub(r"\bkobey\b", "kobe", t)
    t = re.sub(r"\bkobe\b", "kobe", t)
    # join/split common tokens
    t = re.sub(r"\badmitcard\b", "admit card", t)
    t = re.sub(r"\bseatplan\b", "seat plan", t)
    t = re.sub(r"\bexamdate\b", "exam date", t)
    # remove repeated punctuation spacing
    t = re.sub(r"\s+", " ", t).strip()
    return t


def similarity(a: str, b: str) -> float:
    # use typo-tolerant normalization for matching
    return SequenceMatcher(None, normalize_typo_tolerant(a), normalize_typo_tolerant(b)).ratio()


def get_db():
    uri = os.getenv("MONGODB_URI", "").strip()
    db_name = os.getenv("MONGODB_DB", "jnu_helpdesk").strip()
    if not uri:
        raise RuntimeError(f"MONGODB_URI missing. Please set it in .env (expected at: {ENV_PATH})")
    client = MongoClient(uri)
    return client[db_name]


def col_intents(db):
    return db[os.getenv("MONGODB_INTENTS_COLLECTION", "intents_faq").strip()]


def col_facts(db):
    return db[os.getenv("MONGODB_FACTS_COLLECTION", "admission_facts").strip()]


def col_chunks(db):
    return db[os.getenv("MONGODB_CHUNKS_COLLECTION", "knowledge_chunks").strip()]


def clean_chunk_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\.\.\.https?://\S+$", "", t).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------- Keyword helpers ----------
_EN_STOP = {
    "a", "an", "the", "to", "of", "in", "on", "for", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "being",
    "what", "when", "where", "why", "how", "can", "could", "should", "would",
    "i", "me", "my", "we", "you", "your", "our", "us",
}
_BN_STOP = {
    "কি", "কী", "কখন", "কিভাবে", "কোথায়", "কোথায়", "কেন", "কতো", "কত",
    "আমি", "আমার", "তুমি", "আপনি", "আমরা", "তোমাদের", "আপনাদের",
}
_ROMAN_STOP = {
    "ki", "kobe", "kivabe", "kothay", "koto", "ami", "tumi", "apni",
    "help", "please"
}


def tokenize_keywords(text: str) -> List[str]:
    """
    Extract keywords from Bangla/English/Roman text.
    Unique tokens; ignores very short tokens (unit letter handled separately).
    """
    t = normalize_typo_tolerant(text)
    raw = re.findall(r"[0-9A-Za-z_]+|[\u0980-\u09FF]+", t)
    out: List[str] = []
    seen = set()

    for w in raw:
        w = w.strip().lower()
        if len(w) < 2:
            continue
        if w in _EN_STOP or w in _BN_STOP or w in _ROMAN_STOP:
            continue
        if w not in seen:
            out.append(w)
            seen.add(w)

    return out


# ---------- Unit extraction (A–E) ----------
def extract_unit_code(text: str) -> Optional[str]:
    t = normalize_typo_tolerant(text)

    # NEW: user replies only the letter (a/b/c/d/e)
    if re.fullmatch(r"[a-e]", t.strip()):
        return t.strip()

    m = re.search(r"\b([a-e])\s*unit\b", t)
    if m:
        return m.group(1)

    m = re.search(r"\bunit\s*[-_ ]?\s*([a-e])\b", t)
    if m:
        return m.group(1)

    return None


def infer_intent_unit_code(intent_doc: Dict[str, Any]) -> Optional[str]:
    """
    Infer intent unit code from intent questions (no DB change needed).
    """
    for q in (intent_doc.get("questions") or []):
        u = extract_unit_code(str(q))
        if u:
            return u
    # also check intent name (optional)
    name = str(intent_doc.get("intent") or "")
    u = extract_unit_code(name)
    return u


def intent_keyword_set(intent_doc: Dict[str, Any]) -> set:
    kw = set()
    for q in (intent_doc.get("questions") or []):
        for w in tokenize_keywords(str(q)):
            kw.add(w)
    return kw


def match_intent_keywords_and_similarity(
    user_msg: str,
    intents: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], int, float, List[Tuple[int, float, Dict[str, Any]]]]:
    """
    Combined intent match:
      - keyword overlap count (primary)
      - similarity score (secondary / tie-break)
    Unit-safe patch:
      If user specifies a unit (a-e), only same-unit intents can match.
    """
    msg_n = normalize_typo_tolerant(user_msg)
    user_kws = set(tokenize_keywords(user_msg))
    user_unit = extract_unit_code(user_msg)

    top: List[Tuple[int, float, Dict[str, Any]]] = []

    for it in intents:
        intent_unit = infer_intent_unit_code(it)

        # HARD RULE: if user specified a unit, only match same unit intents
        if user_unit and intent_unit and user_unit != intent_unit:
            top.append((0, 0.0, it))
            continue

        it_kw = intent_keyword_set(it)
        kw_matches = len(user_kws.intersection(it_kw)) if user_kws else 0

        best_sim = 0.0
        for q in (it.get("questions") or []):
            qn = normalize_typo_tolerant(str(q))
            if not qn:
                continue
            if qn == msg_n:
                best_sim = 1.0
                break
            best_sim = max(best_sim, similarity(msg_n, qn))

        top.append((kw_matches, best_sim, it))

    top.sort(key=lambda x: (x[0], x[1]), reverse=True)

    if not top:
        return None, 0, 0.0, []

    best_kw, best_sim, best_intent = top[0]
    return best_intent, best_kw, best_sim, top[:5]


# ---------- Minimal context builder ----------
def build_context_from_fact_and_chunks(
    fact_doc: Optional[Dict[str, Any]],
    chunk_docs: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, str]], Optional[str]]:
    """
    Minimal context for Gemini + UI metadata.
    Token-minimized: includes only answer + up to 2 chunks.
    """
    sources: List[Dict[str, str]] = []
    updated_at: Optional[str] = None
    parts: List[str] = []

    if fact_doc:
        val = fact_doc.get("value") or {}
        ans = (val.get("answer") or "").strip()
        if ans:
            parts.append("FACT:\n" + ans)

        fs = val.get("sources") or []
        if isinstance(fs, list):
            for s in fs[:5]:
                if isinstance(s, dict) and s.get("title") and s.get("url"):
                    sources.append({"title": str(s["title"]), "url": str(s["url"])})

        updated_at = fact_doc.get("updated_at") or updated_at

    for c in (chunk_docs or [])[:2]:
        title = (c.get("title") or "").strip()
        text = clean_chunk_text(c.get("text", "") or c.get("content", ""))
        if title and text:
            parts.append(f"CHUNK:\n{title}\n{text}")
        elif text:
            parts.append("CHUNK:\n" + text)

        cs = c.get("sources") or []
        if isinstance(cs, list):
            for s in cs[:5]:
                if isinstance(s, dict) and s.get("title") and s.get("url"):
                    sources.append({"title": str(s["title"]), "url": str(s["url"])})

        updated_at = c.get("updated_at") or updated_at

    # de-dup sources
    uniq = []
    seen = set()
    for s in sources:
        key = (s.get("title", "").strip().lower(), s.get("url", "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)

    context_text = "\n\n".join([p for p in parts if p]).strip()
        #  Make updated_at JSON/Pydantic safe (datetime -> string)
    if updated_at is not None and not isinstance(updated_at, str):
        try:
            updated_at = updated_at.isoformat()
        except Exception:
            updated_at = str(updated_at)

    context_text = "\n\n".join([p for p in parts if p]).strip()
    return context_text, uniq, updated_at

    

# ---------- Gemini (timeout + retry + strict no-invent) ----------
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "8").strip() or "8")
GEMINI_RETRIES = int(os.getenv("GEMINI_RETRIES", "2").strip() or "2")


def detect_user_language(text: str) -> str:
    """
    Returns: "bn" | "roman_bn" | "en"
    """
    if not text:
        return "bn"

    # Bangla Unicode -> Bangla
    if any("\u0980" <= ch <= "\u09FF" for ch in text):
        return "bn"

    t = text.lower().strip()

    # Roman Bangla markers ONLY (avoid English words like unit/result/seat/admit)
    roman_markers = [
        "kobe", "kobay", "kothay", "kivabe", "ki", "kisu", "diben", "dibe",
        "hobe", "hocche", "korbo", "korte", "chai", "lagbe", "please",
        "apni", "ami", "tumi", "karon", "tarikh", "porikkha"
    ]
    if any(w in t for w in roman_markers):
        return "roman_bn"

    return "en"


def _gemini_call(model: str, api_key: str, prompt: str) -> str:
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=model, contents=prompt)
    return (getattr(resp, "text", "") or "").strip()


def gemini_compose_answer(user_question: str, context_text: str, official_site: str = "") -> str:
    """
    Gemini must not invent. Uses ONLY context_text.
    Timeout + retry using ThreadPoolExecutor (Windows-friendly).
    Enforces SAME LANGUAGE as user.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    if not api_key:
        return ""

    lang = detect_user_language(user_question)

    # Keep prompt small, strict + language-locked
    prompt = (
        "You are a JNU admission helpdesk assistant.\n"
        "RULES (STRICT):\n"
        "- Use ONLY the provided CONTEXT.\n"
        "- Do NOT add or invent any new facts.\n"
        "- Answer in the SAME LANGUAGE as the user.\n"
        "- If user language is bn → answer in Bangla script.\n"
        "- If user language is roman_bn → answer in Roman Bangla.\n"
        "- If user language is en → answer in English.\n"
        "- If CONTEXT has no answer, say exactly: 'info not available' and suggest official site.\n\n"
        f"USER_LANGUAGE: {lang}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{user_question}\n\n"
        f"{('Official site: ' + official_site) if official_site else ''}\n"
        "Answer:"
    ).strip()

    last_err = None
    for attempt in range(1, GEMINI_RETRIES + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_gemini_call, model, api_key, prompt)
                return fut.result(timeout=GEMINI_TIMEOUT_SECONDS)
        except FuturesTimeoutError as e:
            last_err = f"timeout_{attempt}"
        except Exception as e:
            last_err = f"error_{attempt}"
        time.sleep(0.2 * attempt)

    log_event("gemini_failed", {"reason": last_err or "unknown"})
    return ""


# ---------- FastAPI ----------
app = FastAPI(title="JNU Helpdesk Backend (Production Final)")

# ---------- Admin security (X-Admin-Key) ----------
def require_admin(x_admin_key: Optional[str] = Header(default=None)):
    expected = os.getenv("ADMIN_API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured")
    if (x_admin_key or "").strip() != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

def oid_str(doc: dict) -> dict:
    """Convert Mongo _id ObjectId to string for JSON."""
    if doc and "_id" in doc and isinstance(doc.get("_id"), ObjectId):
        doc["_id"] = str(doc["_id"])
    return doc


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://jnuadmissionhelpdesk.onrender.com"],  # production এ domain restrict করবেন
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatIn(BaseModel):
    message: str
    # optional: frontend can store this in localStorage and send it back
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    title: str
    url: str


class ChatOut(BaseModel):
    answer: str
    source: str  # "db" | "gemini" | "fallback"
    confidence: float = 0.0
    intent_id: Optional[str] = None

    # session continuity (frontend can ignore safely)
    session_id: Optional[str] = None

    # for frontend badges
    sources: List[SourceItem] = []
    updated_at: Optional[str] = None

    # debug
    used_fact_ids: List[str] = []
    used_chunk_ids: List[str] = []

    # optional: notifications peek (frontend can ignore safely)
    notifications: List[Dict[str, Any]] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn):
    request_id = str(uuid.uuid4())
    t0 = time.monotonic()

    msg = (payload.message or "").strip()
    if not msg:
        return ChatOut(answer="Please type a question.", source="fallback", confidence=0.0)

    session_id = ensure_session_id(payload.session_id)

    # Cache key: normalized message only (simple, effective)
    cache_key = "chat:" + session_id + ":" + normalize_typo_tolerant(msg)
    cached = cache.get(cache_key)
    if cached:
        cached_out = ChatOut(**cached)
        log_event("chat", {
            "request_id": request_id,
            "route": cached_out.source,
            "cache_hit": True,
            "latency_ms": int((time.monotonic() - t0) * 1000),
        })
        return cached_out

    db = get_db()
    intents_col = col_intents(db)
    facts_col = col_facts(db)
    chunks_col = col_chunks(db)

    intents = list(
        intents_col.find(
            {},
            {
                "_id": 0,
                "intent": 1,
                "questions": 1,
                "response_type": 1,
                "fact_key": 1,
                "chunk_ids": 1,
                "chunk_id": 1,

                #  Router support (MUST)
                "route_to_intent_by_unit": 1,

                #  New (optional, non-breaking): slot filling schema
                "required_slots": 1,
                "slot_questions": 1,
            },
        )
    )

    # quick map for pending intent lookup
    intent_by_name = {str(it.get("intent")): it for it in intents if it.get("intent")}

    # -------- Context memory: load slot_state (pending slots) + recent turns --------
    slot_state = load_slot_state(db, session_id)
    recent_turns = load_recent_turns(db, session_id, limit=8)

    best_intent, kw_matches, sim_score, top_hits = match_intent_keywords_and_similarity(msg, intents)
    official_site = os.getenv("OFFICIAL_WEBSITE_URL", "").strip()

    # If there is a pending intent (slot filling), continue that flow.
    pending_intent_name = (slot_state.get("pending_intent") or "").strip()
    if pending_intent_name:
        best_intent = intent_by_name.get(pending_intent_name) or best_intent

    # Detect automation request (does not block normal answering)
    _action = detect_action_intent(msg)

    # ---------- Production gating thresholds ----------
    DB_SIM_STRONG = float(os.getenv("DB_SIM_STRONG", "0.85").strip() or "0.85")
    GEMINI_SIM_MIN = float(os.getenv("GEMINI_SIM_MIN", "0.60").strip() or "0.60")
    GEMINI_SIM_MAX = float(os.getenv("GEMINI_SIM_MAX", "0.85").strip() or "0.85")

    db_direct = (kw_matches >= 3) or (sim_score >= DB_SIM_STRONG)
    gemini_compose = (kw_matches in (1, 2)) or (GEMINI_SIM_MIN <= sim_score < GEMINI_SIM_MAX)
    gemini_fallback_only = (kw_matches == 0) and (sim_score < GEMINI_SIM_MIN)

    intent_name = best_intent.get("intent") if best_intent else None
    user_unit = extract_unit_code(msg)
    best_intent_unit = infer_intent_unit_code(best_intent) if best_intent else None

    # If no intent found, force fallback-only
    if not best_intent:
        gemini_fallback_only = True
        db_direct = False
        gemini_compose = False

    # ---------- Slot Filling (generic) ----------
    if best_intent:
        required_slots = infer_required_slots(best_intent)
        slot_questions = infer_slot_questions(best_intent)

        # merge: existing filled (from slot_state) + newly extracted from this message
        existing_filled = slot_state.get("filled_slots") or {}
        extracted_now = fill_slots_from_text(msg, required_slots)
        filled_slots = merge_slots(existing_filled, extracted_now)

        # keep in-memory slot_state updated for router resolution
        try:
            slot_state["filled_slots"] = filled_slots
        except Exception:
            pass

        missing_slots = compute_missing_slots(required_slots, filled_slots)

        # If there is a pending flow OR intent requires slots, ask missing slot question.
        if (pending_intent_name or required_slots) and missing_slots:
            slot, question = next_slot_question(missing_slots, slot_questions)
            state = {
                "pending_intent": intent_name,
                "required_slots": required_slots,
                "slot_questions": slot_questions,
                "filled_slots": filled_slots,
                "missing_slots": missing_slots,
            }
            upsert_slot_state(db, session_id, state)

            out = ChatOut(
                answer=question or "আরও তথ্য দিন",
                source="db",
                confidence=sim_score,
                intent_id=intent_name,
                session_id=session_id,
                notifications=peek_notifications(db, session_id, limit=3),
            )

            cache.set(cache_key, out.model_dump())
            log_event("chat", {
                "request_id": request_id,
                "route": "slot_fill_clarify",
                "intent": intent_name,
                "missing": missing_slots,
                "cache_hit": False,
                "latency_ms": int((time.monotonic() - t0) * 1000),
            })
            save_chat_turn(db, session_id, msg, out.answer, intent_name, filled_slots, out.source)
            return out

        # If slots are satisfied and we were in a pending flow, clear it.
        if pending_intent_name:
            clear_slot_state(db, session_id)

    # ---------- Guardrail: user said "unit" but didn't specify A-E ----------
    # Avoid wrong unit guess in production.
    msg_norm = normalize_typo_tolerant(msg)
    mentions_unit = bool(re.search(r"\bunit\b", msg_norm))
    if mentions_unit and not user_unit:
        # If top intent is unit-specific OR user asked unit-wise topic, ask clarification.
        # (This prevents wrong A/E mistakes)
        if best_intent_unit or ("seat" in tokenize_keywords(msg) or "exam" in tokenize_keywords(msg)):
            answer = "আপনি কোন ইউনিট (A/B/C/D/E) সম্পর্কে জানতে চাচ্ছেন? ইউনিট লেটারটি লিখলে আমি সঠিক তথ্য দিতে পারবো।"
            out = ChatOut(
                answer=answer,
                source="db",
                confidence=sim_score,
                intent_id=intent_name,
                session_id=session_id,
                notifications=peek_notifications(db, session_id, limit=3),
            )
            cache.set(cache_key, out.model_dump())
            log_event("chat", {
                "request_id": request_id,
                "route": "guardrail_unit_clarify",
                "intent": intent_name,
                "kw": kw_matches,
                "sim": round(sim_score, 3),
                "user_unit": user_unit,
                "intent_unit": best_intent_unit,
                "cache_hit": False,
                "latency_ms": int((time.monotonic() - t0) * 1000),
            })
            return out

    # ---------- Router intent support (NEW) ----------
    # If an intent is a router, pick the correct unit-specific intent before fetching DB data.
    resolved_unit = None
    try:
        # slots engine may have filled it (lowercase a-e)
        # slot_state might be cleared already; so check both recent slot_state and direct extraction.
        resolved_unit = (
            (slot_state.get("filled_slots") or {}).get("unit")
            or extract_unit_code(msg)
        )
    except Exception:
        resolved_unit = extract_unit_code(msg)

    if best_intent:
        rtype = (best_intent.get("response_type") or "").strip().lower()
        route_map = best_intent.get("route_to_intent_by_unit") or {}

        if rtype == "router" or route_map:
            u = (resolved_unit or "").strip().upper()  # "a" -> "A"
            target_intent_name = route_map.get(u)

            if target_intent_name:
                routed = intent_by_name.get(target_intent_name)
                if routed:
                    best_intent = routed
                    intent_name = target_intent_name
                # else: target intent not found in DB -> will fallback

    # ---------- Fetch DB data based on response_type ----------
    fact_doc: Optional[Dict[str, Any]] = None
    chunk_docs: List[Dict[str, Any]] = []

    if best_intent:
        rtype = (best_intent.get("response_type") or "").strip().lower()

        if rtype in ("fact", "facts"):
            fk = (best_intent.get("fact_key") or "").strip()
            if fk:
                fact_doc = facts_col.find_one({"fact_key": fk}, {"_id": 0})

        elif rtype in ("chunk", "chunks"):
            ids = best_intent.get("chunk_ids") or []
            if not ids and best_intent.get("chunk_id"):
                ids = [best_intent.get("chunk_id")]
            ids = [str(x).strip() for x in ids if str(x).strip()]
            if ids:
                chunk_docs = list(chunks_col.find({"chunk_id": {"$in": ids}}, {"_id": 0}))

    context_text, ui_sources, ui_updated_at = build_context_from_fact_and_chunks(fact_doc, chunk_docs)

    #  Force DB answer if DB returned anything (ignore similarity thresholds)
    if fact_doc or chunk_docs:
        db_direct = True
        gemini_compose = False
        gemini_fallback_only = False


    # ---------- 1) DB direct ----------
    if db_direct:
        answer = ""
        used_fact_ids: List[str] = []
        used_chunk_ids: List[str] = []

        if fact_doc:
            used_fact_ids = [str(fact_doc.get("fact_key", ""))] if fact_doc.get("fact_key") else []
            answer = ((fact_doc.get("value") or {}).get("answer") or "").strip()

        if not answer:
            answer = "দুঃখিত—এই বিষয়ে পর্যাপ্ত তথ্য ডাটাবেজে পাওয়া যায়নি। অফিসিয়াল নোটিশ/ওয়েবসাইট দেখে নিশ্চিত করুন।"

        #  NEW: make DB answer sound natural (grounded rewrite)
        answer = paraphrase_grounded_answer(
            answer=answer,
            user_msg=msg,
            official_site=official_site,
            style="friendly",
        )

        out = ChatOut(
            answer=answer,
            source="db",
        )


        out = ChatOut(
            answer=answer,
            source="db",
            confidence=sim_score,
            intent_id=intent_name,
            session_id=session_id,
            sources=ui_sources,
            updated_at=ui_updated_at,
            used_fact_ids=used_fact_ids,
            used_chunk_ids=used_chunk_ids,
            notifications=peek_notifications(db, session_id, limit=3),
        )
        cache.set(cache_key, out.model_dump())
        log_event("chat", {
            "request_id": request_id,
            "route": "db_direct",
            "intent": intent_name,
            "kw": kw_matches,
            "sim": round(sim_score, 3),
            "user_unit": user_unit,
            "intent_unit": best_intent_unit,
            "cache_hit": False,
            "latency_ms": int((time.monotonic() - t0) * 1000),
        })
        # save memory (best effort)
        try:
            # if slot engine ran, it's in slot_state filled_slots. otherwise empty.
            filled = (slot_state.get("filled_slots") or {})
        except Exception:
            filled = {}
        save_chat_turn(db, session_id, msg, out.answer, intent_name, filled, out.source)
        return out

    # ---------- 2) Gemini compose (context-only) ----------
    if gemini_compose:
        ctx = context_text if context_text else "NO_DATA"
        ai = gemini_compose_answer(user_question=msg, context_text=ctx, official_site=official_site)
        if ai:
            used_fact_ids = [str(fact_doc.get("fact_key"))] if fact_doc and fact_doc.get("fact_key") else []
            used_chunk_ids = [str(c.get("chunk_id")) for c in chunk_docs if c.get("chunk_id")]
            out = ChatOut(
                answer=ai,
                source="gemini",
                confidence=sim_score,
                intent_id=intent_name,
                session_id=session_id,
                sources=ui_sources,
                updated_at=ui_updated_at,
                used_fact_ids=used_fact_ids,
                used_chunk_ids=used_chunk_ids,
                notifications=peek_notifications(db, session_id, limit=3),
            )
            cache.set(cache_key, out.model_dump())
            log_event("chat", {
                "request_id": request_id,
                "route": "gemini_compose",
                "intent": intent_name,
                "kw": kw_matches,
                "sim": round(sim_score, 3),
                "user_unit": user_unit,
                "intent_unit": best_intent_unit,
                "cache_hit": False,
                "latency_ms": int((time.monotonic() - t0) * 1000),
            })
            save_chat_turn(db, session_id, msg, out.answer, intent_name, (slot_state.get("filled_slots") or {}), out.source)
            return out

        out = ChatOut(
            answer="দুঃখিত—এই প্রশ্নের নির্ভরযোগ্য উত্তর এখন পাওয়া যাচ্ছে না। অফিসিয়াল ওয়েবসাইট দেখে নিশ্চিত করুন।",
            source="fallback",
            confidence=sim_score,
            intent_id=intent_name,
            session_id=session_id,
            notifications=peek_notifications(db, session_id, limit=3),
        )
        cache.set(cache_key, out.model_dump())
        log_event("chat", {
            "request_id": request_id,
            "route": "gemini_failed_fallback",
            "intent": intent_name,
            "kw": kw_matches,
            "sim": round(sim_score, 3),
            "latency_ms": int((time.monotonic() - t0) * 1000),
        })
        save_chat_turn(db, session_id, msg, out.answer, intent_name, (slot_state.get("filled_slots") or {}), out.source)
        return out

    # ---------- 3) Gemini fallback only ----------
    if gemini_fallback_only:
        ai = gemini_compose_answer(user_question=msg, context_text="NO_DATA", official_site=official_site)
        if ai:
            out = ChatOut(
                answer=ai,
                source="gemini",
                confidence=sim_score,
                intent_id=intent_name,
                session_id=session_id,
                notifications=peek_notifications(db, session_id, limit=3),
            )
            cache.set(cache_key, out.model_dump())
            log_event("chat", {
                "request_id": request_id,
                "route": "gemini_fallback_only",
                "kw": kw_matches,
                "sim": round(sim_score, 3),
                "latency_ms": int((time.monotonic() - t0) * 1000),
            })
            save_chat_turn(db, session_id, msg, out.answer, intent_name, (slot_state.get("filled_slots") or {}), out.source)
            return out

    out = ChatOut(
        answer="দুঃখিত—এই প্রশ্নের নির্ভরযোগ্য উত্তর আমার ডাটাবেজে নেই। অফিসিয়াল নোটিশ/ওয়েবসাইট দেখে নিশ্চিত করুন।",
        source="fallback",
        confidence=sim_score,
        intent_id=intent_name,
        session_id=session_id,
        notifications=peek_notifications(db, session_id, limit=3),
    )
    cache.set(cache_key, out.model_dump())
    log_event("chat", {
        "request_id": request_id,
        "route": "final_fallback",
        "kw": kw_matches,
        "sim": round(sim_score, 3),
        "latency_ms": int((time.monotonic() - t0) * 1000),
    })
    save_chat_turn(db, session_id, msg, out.answer, intent_name, (slot_state.get("filled_slots") or {}), out.source)
    return out


# ===================== Admin Endpoints =====================
# NOTE: These endpoints are protected by X-Admin-Key.
# They allow CRUD operations for the intents_faq collection (or your configured intents collection).

class IntentAdminItem(BaseModel):
    intent: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    tags: List[str] = []
    enabled: bool = True
    meta: Dict[str, Any] = {}


@app.get("/admin/intents", dependencies=[Depends(require_admin)])
def admin_list_intents(limit: int = 200):
    db = get_db()
    col = col_intents(db)
    docs = list(col.find({}).limit(int(limit)))
    return [oid_str(d) for d in docs]


@app.post("/admin/intents", dependencies=[Depends(require_admin)])
def admin_create_intent(item: IntentAdminItem):
    db = get_db()
    col = col_intents(db)
    doc = item.model_dump()
    res = col.insert_one(doc)
    doc["_id"] = str(res.inserted_id)
    return doc


@app.put("/admin/intents/{item_id}", dependencies=[Depends(require_admin)])
def admin_update_intent(item_id: str, item: IntentAdminItem):
    db = get_db()
    col = col_intents(db)
    try:
        _id = ObjectId(item_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")

    upd = item.model_dump()
    r = col.update_one({"_id": _id}, {"$set": upd})
    if r.matched_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    doc = col.find_one({"_id": _id})
    return oid_str(doc)


@app.delete("/admin/intents/{item_id}", dependencies=[Depends(require_admin)])
def admin_delete_intent(item_id: str):
    db = get_db()
    col = col_intents(db)
    try:
        _id = ObjectId(item_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")

    r = col.delete_one({"_id": _id})
    if r.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"deleted": True, "id": item_id}

# ===========================================================
