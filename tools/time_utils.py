from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Kolkata")


def now() -> datetime:
    return datetime.now(ZoneInfo(DEFAULT_TIMEZONE))


def _normalize_time(hour: int, minute: int, meridiem: str | None):
    if meridiem:
        meridiem = meridiem.lower()
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
    return hour, minute


def parse_relative_datetime(query: str):
    lowered = (query or "").lower()
    current = now()

    if "any time" in lowered or "whenever" in lowered or "as soon as possible" in lowered:
        return None, "as soon as possible"

    match = re.search(
        r"\b(today|tomorrow)\b(?:\s+at)?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        lowered,
    )
    if match:
        day_word, hour, minute, meridiem = match.groups()
        hour = int(hour)
        minute = int(minute or 0)
        hour, minute = _normalize_time(hour, minute, meridiem)
        target = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if day_word == "tomorrow":
            target += timedelta(days=1)
        elif target <= current:
            target += timedelta(days=1)
        return target, None

    match = re.search(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", lowered)
    if match:
        hour, minute, meridiem = match.groups()
        hour = int(hour)
        minute = int(minute or 0)
        hour, minute = _normalize_time(hour, minute, meridiem)
        target = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= current:
            target += timedelta(days=1)
        return target, None

    if "tonight" in lowered:
        target = current.replace(hour=20, minute=0, second=0, microsecond=0)
        if target <= current:
            target += timedelta(days=1)
        return target, None

    if "tomorrow" in lowered:
        target = current + timedelta(days=1)
        return target.replace(hour=9, minute=0, second=0, microsecond=0), None

    if "today" in lowered:
        target = current.replace(hour=17, minute=0, second=0, microsecond=0)
        if target <= current:
            target += timedelta(days=1)
        return target, None

    return None, None
