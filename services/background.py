from __future__ import annotations

import os
import threading
import time

from app.tools.scrape import refresh_configured_sources
from app.trace import trace_event

_started = False
_lock = threading.Lock()


def _enabled() -> bool:
    value = os.getenv("ENABLE_BACKGROUND_SCRAPE", "false").lower()
    return value in {"1", "true", "yes", "on"}


def _interval_seconds() -> int:
    try:
        return max(300, int(os.getenv("SCRAPE_INTERVAL_SECONDS", "1800")))
    except ValueError:
        return 1800


def _scrape_loop():
    interval = _interval_seconds()
    while True:
        try:
            trace_event("background_scrape_cycle_started", interval_seconds=interval)
            refresh_configured_sources()
            trace_event("background_scrape_cycle_completed", interval_seconds=interval)
        except Exception as exc:
            trace_event(
                "background_scrape_cycle_failed",
                error=f"{exc.__class__.__name__}: {exc}",
                interval_seconds=interval,
            )
        time.sleep(interval)


def start_background_scraper():
    global _started
    if not _enabled():
        return

    with _lock:
        if _started:
            return
        thread = threading.Thread(target=_scrape_loop, name="background-scraper", daemon=True)
        thread.start()
        _started = True
        trace_event("background_scraper_started", interval_seconds=_interval_seconds())
