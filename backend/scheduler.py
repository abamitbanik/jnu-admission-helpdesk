"""Scheduler skeleton (APScheduler).

✅ ডিফল্টভাবে এই scheduler চালু করা নেই (যাতে production এ surprise background thread না চলে)।

যদি আপনি automation চালু করতে চান:
  1) requirements.txt এ apscheduler install করুন
  2) main.py তে startup event এ start_scheduler() কল করুন
  3) jobs এর ভিতরে official sources check করে notifications collection update করুন
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:  # pragma: no cover
    BackgroundScheduler = None


def start_scheduler(db) -> Any:
    if BackgroundScheduler is None:
        return None

    interval_min = int(os.getenv("AUTOMATION_INTERVAL_MIN", "30").strip() or "30")
    sched = BackgroundScheduler(daemon=True)

    # Example job (no-op by default)
    def _heartbeat():
        # Put your periodic checks here
        _ = db
        _ = time.time()

    sched.add_job(_heartbeat, "interval", minutes=interval_min, id="heartbeat")
    sched.start()
    return sched
