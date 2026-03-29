from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.trace import trace_event

DEFAULT_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Kolkata")


def _now():
    return datetime.now(ZoneInfo(DEFAULT_TIMEZONE))


def _normalize_time(hour: int, minute: int, meridiem: str | None):
    if meridiem:
        meridiem = meridiem.lower()
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
    return hour, minute


def _parse_relative_datetime(query: str):
    lowered = query.lower()
    now = _now()

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
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if day_word == "tomorrow":
            target += timedelta(days=1)
        elif target <= now:
            target += timedelta(days=1)
        return target, None

    match = re.search(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", lowered)
    if match:
        hour, minute, meridiem = match.groups()
        hour = int(hour)
        minute = int(minute or 0)
        hour, minute = _normalize_time(hour, minute, meridiem)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target, None

    if "tonight" in lowered:
        target = now.replace(hour=20, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target, None

    if "tomorrow" in lowered:
        target = now + timedelta(days=1)
        return target.replace(hour=9, minute=0, second=0, microsecond=0), None

    if "today" in lowered:
        target = now.replace(hour=17, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target, None

    return None, None


def _detect_channel(query: str):
    lowered = query.lower()
    if "whatsapp" in lowered or "whats app" in lowered or re.search(r"\bwa\b", lowered):
        return "whatsapp"
    if "email" in lowered or "mail" in lowered or "e-mail" in lowered:
        return "email"
    if "meeting" in lowered or "calendar" in lowered or "invite" in lowered:
        return "meeting"
    return "share"


def _suggest_endpoint(channel: str):
    if channel == "email":
        return "/dummy/email"
    if channel == "whatsapp":
        return "/dummy/whatsapp"
    return "/dummy/email"


def _extract_content(query: str):
    lowered = query.lower()
    for keyword in ["send", "share", "email", "mail", "whatsapp", "meeting", "schedule"]:
        if keyword in lowered:
            parts = re.split(rf"\b{keyword}\b", query, flags=re.I, maxsplit=1)
            if len(parts) > 1:
                tail = parts[1].strip(" :-,")
                if tail:
                    return tail
    return query.strip()


def share_details(query: str):
    query = (query or "").strip()
    if not query:
        return "No share request provided."

    channel = _detect_channel(query)
    scheduled_for, time_note = _parse_relative_datetime(query)
    content = _extract_content(query)

    needs = []
    if not scheduled_for and not time_note:
        time_note = "No time mentioned; treat as immediate unless you want it scheduled."
    if channel in {"email", "whatsapp", "meeting"} and not any(
        token in query.lower() for token in ["to ", "for ", "@", "recipient", "client", "team", "group"]
    ):
        needs.append("recipient")
    if len(content) < 15:
        needs.append("message content")

    lines = [f"Action: {channel}"]
    lines.append(f"Suggested endpoint: {_suggest_endpoint(channel)}")
    if scheduled_for:
        lines.append(f"Schedule: {scheduled_for.isoformat()}")
        lines.append(f"Timezone: {DEFAULT_TIMEZONE}")
    elif time_note:
        lines.append(f"Schedule: {time_note}")

    lines.append(f"Draft: {content}")
    if needs:
        lines.append("Missing details: " + ", ".join(needs))
    else:
        lines.append("Status: ready to hand off to an email, WhatsApp, or calendar integration.")

    trace_event(
        "share_plan_created",
        query=query,
        channel=channel,
        scheduled_for=scheduled_for.isoformat() if scheduled_for else None,
        suggested_endpoint=_suggest_endpoint(channel),
        missing_details=needs,
    )

    return "\n".join(lines)
