# backend/language_style.py
import os
import re
from google import genai  # pip install google-genai


# ---------- Language detection ----------
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

    # Roman Bangla markers ONLY (avoid English words like unit/result/seat)
    roman_markers = [
        "kobe", "kobay", "kothay", "kivabe", "ki", "kisu", "diben", "dibe",
        "hobe", "hocche", "korbo", "korte", "chai", "lagbe", "please",
        "apni", "ami", "tumi", "karon", "tarikh", "porikkha"
    ]
    if any(w in t for w in roman_markers):
        return "roman_bn"

    return "en"


def _has_bn_chars(text: str) -> bool:
    return any("\u0980" <= ch <= "\u09FF" for ch in (text or ""))


def _has_latin_letters(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def _digits_signature(text: str) -> str:
    """Signature of all numbers in order. If AI changes digits, fallback."""
    nums = re.findall(r"\d+", text or "")
    return "|".join(nums)


def _language_ok(lang: str, out: str) -> bool:
    """
    Hard guard:
      - bn -> must contain Bangla chars
      - en -> must NOT contain Bangla chars
      - roman_bn -> must NOT contain Bangla chars AND should contain latin letters
    """
    out = (out or "").strip()
    if not out:
        return False

    if lang == "bn":
        return _has_bn_chars(out)

    if lang == "en":
        return not _has_bn_chars(out)

    if lang == "roman_bn":
        return (not _has_bn_chars(out)) and _has_latin_letters(out)

    return True


# ---------- Grounded paraphrase ----------
def paraphrase_grounded_answer(
    answer: str,
    user_msg: str = "",
    official_site: str = "",
    style: str = "friendly",
) -> str:
    """
    Grounded paraphrasing:
    - Rephrase ONLY (no new facts)
    - Keep dates/numbers/unit names unchanged (digits guard)
    - Use SAME LANGUAGE as user (language guard)
    - Fallback to original answer on any error or mismatch
    """

    answer = (answer or "").strip()
    if not answer:
        return answer

    # Feature toggle
    if os.getenv("ENABLE_STYLE_REWRITE", "1").strip().lower() not in ("1", "true", "yes", "on"):
        return answer

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return answer

    model = os.getenv("GEMINI_MODEL_STYLE", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    lang = detect_user_language(user_msg)

    sig_in = _digits_signature(answer)

    prompt = (
        "You are a university admission helpdesk assistant.\n"
        "TASK: Rephrase the ANSWER to sound natural, polite, and human.\n"
        "RULES (STRICT):\n"
        "- Do NOT change any facts.\n"
        "- Do NOT change any dates, numbers, times, unit names (A/B/C/D/E), or day names.\n"
        "- Do NOT add new information.\n"
        "- Use the SAME LANGUAGE as the user (Bangla / Roman Bangla / English).\n"
        "- If unsure, return the original ANSWER unchanged.\n\n"
        f"USER_LANGUAGE: {lang}\n"
        f"STYLE: {style}\n\n"
        f"USER_QUESTION (for tone only):\n{user_msg}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "REPHRASED_ANSWER:"
    )

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model=model, contents=prompt)
        out = (getattr(resp, "text", "") or "").strip()
        if not out:
            return answer

        # Guard #1: digits must match exactly (dates/years/marks)
        if _digits_signature(out) != sig_in:
            return answer

        # Guard #2: output language must match user language
        if not _language_ok(lang, out):
            return answer

        return out

    except Exception:
        return answer
