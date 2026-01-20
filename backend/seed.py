import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

load_dotenv()

def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def main():
    uri = os.getenv("MONGODB_URI", "").strip()
    db_name = os.getenv("MONGODB_DB", "jnu_helpdesk").strip()
    col_name = os.getenv("MONGODB_COLLECTION", "faq").strip()

    if not uri:
        raise SystemExit("ERROR: MONGODB_URI missing. Create .env from .env.example and set it.")

    data_path = Path(__file__).parent / "data" / "jnu_helpdesk_dataset.json"
    if not data_path.exists():
        raise SystemExit(f"ERROR: Dataset not found at: {data_path}")

    dataset = json.loads(data_path.read_text(encoding="utf-8"))

    client = MongoClient(uri)
    col = client[db_name][col_name]

    # Create index for fast exact match
    try:
        col.create_index([("question_norm", ASCENDING)], unique=True)
    except Exception:
        # unique may fail if duplicates exist; still okay
        col.create_index([("question_norm", ASCENDING)])

    upserts = 0
    for item in dataset:
        q = item.get("question", "")
        if not q:
            continue

        doc = dict(item)
        doc["question_norm"] = normalize(q)

        # Ensure alt_questions is list
        alts = doc.get("alt_questions") or []
        if not isinstance(alts, list):
            alts = [str(alts)]
        doc["alt_questions"] = alts

        col.update_one(
            {"question_norm": doc["question_norm"]},
            {"$set": doc},
            upsert=True
        )
        upserts += 1

    print(f"✅ Seed completed. Upserted documents: {upserts}")
    print(f"DB: {db_name} | Collection: {col_name}")

if __name__ == "__main__":
    main()
