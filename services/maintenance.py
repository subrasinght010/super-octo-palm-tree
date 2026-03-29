from __future__ import annotations

import os
import threading
import time

from app.services.knowledge_base import backfill_missing_embeddings, rebuild_faiss_index
from app.tools.scrape import clean_scrape_config
from app.trace import trace_event

_started = False
_lock = threading.Lock()


def _enabled() -> bool:
    value = os.getenv("ENABLE_MAINTENANCE_SCHEDULER", "false").lower()
    return value in {"1", "true", "yes", "on"}


def _interval_seconds() -> int:
    try:
        return max(300, int(os.getenv("MAINTENANCE_INTERVAL_SECONDS", "900")))
    except ValueError:
        return 900


def _maintenance_loop():
    interval = _interval_seconds()
    while True:
        try:
            trace_event("maintenance_cycle_started", interval_seconds=interval)
            clean_scrape_config()
            backfill_missing_embeddings()
            rebuild_faiss_index()
            trace_event("maintenance_cycle_completed", interval_seconds=interval)
        except Exception as exc:
            trace_event(
                "maintenance_cycle_failed",
                error=f"{exc.__class__.__name__}: {exc}",
                interval_seconds=interval,
            )
        time.sleep(interval)


def start_maintenance_scheduler():
    global _started
    if not _enabled():
        return

    with _lock:
        if _started:
            return
        thread = threading.Thread(target=_maintenance_loop, name="maintenance-scheduler", daemon=True)
        thread.start()
        _started = True
        trace_event("maintenance_scheduler_started", interval_seconds=_interval_seconds())
