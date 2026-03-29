from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
TRACE_FILE = LOG_DIR / "trace.log"


def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_request_id() -> str:
    return uuid4().hex


def trace_event(event: str, **payload: Any) -> None:
    ensure_log_dir()
    record = {
        "event": event,
        "timestamp_utc": utc_now(),
        **payload,
    }
    with TRACE_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
