from pymongo import MongoClient
from datetime import datetime
import sys


MONGO_URI = "mongodb+srv://amitbanik:AMIT%40jnu%2316@jnu-admission-helpdesk.6jdcl0l.mongodb.net/?appName=jnu-admission-helpdesk"
DB_NAME = "jnu_helpdesk"                 # DB নাম
COLLECTION = "admission_facts"           # facts collection নাম
# ====================================================

def update_fact(fact_key: str, new_answer: str):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION]

    res = col.update_one(
        {"fact_key": fact_key},
        {
            "$set": {
                "value.answer": new_answer,
                "updated_at": datetime.utcnow(),  # ✅ auto update date
            }
        }
    )

    print("Matched:", res.matched_count, "Updated:", res.modified_count)
    client.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python update_fact.py <fact_key> <new_answer>")
        sys.exit(1)

    fact_key = sys.argv[1]
    new_answer = " ".join(sys.argv[2:])
    update_fact(fact_key, new_answer)
