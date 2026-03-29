from __future__ import annotations

import re

from app.services.ingestion_state import enqueue_share_task
from app.services.scheduler import start_scheduler
from app.services.delivery import queue_dummy_delivery
from app.trace import trace_event
from app.tools.time_utils import DEFAULT_TIMEZONE, parse_relative_datetime
from app.services.ingestion_state import chat_history_transcript
from app.tools.utils import summarize_text


def _detect_channel(query: str):
    lowered = query.lower()
    if "whatsapp" in lowered or "whats app" in lowered or re.search(r"\bwa\b", lowered):
        return "whatsapp"
    if "email" in lowered or "mail" in lowered or "e-mail" in lowered:
        return "email"
    if any(keyword in lowered for keyword in ["meeting", "calendar", "invite", "schedule", "task", "todo", "remind"]):
        return None
    return None


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


def _extract_recipient(query: str):
    email_match = re.search(r"[\w.\-+]+@[\w.\-]+\.[A-Za-z]{2,}", query)
    if email_match:
        return email_match.group(0)

    lower = query.lower()
    for marker in [" to ", " for "]:
        if marker in lower:
            start = lower.index(marker) + len(marker)
            remainder = query[start:].strip(" :-,")
            if remainder:
                return remainder.split(",")[0].split(" and ")[0].strip()
    return None


def _wants_summary_share(query: str) -> bool:
    lowered = (query or "").lower()
    has_share_intent = any(keyword in lowered for keyword in ["share", "send", "email", "mail", "whatsapp", "whats app"])
    has_summary_intent = any(keyword in lowered for keyword in ["summary", "summarize", "summarise", "brief", "short version"])
    return has_share_intent and has_summary_intent


def _wants_full_conversation_share(query: str) -> bool:
    lowered = (query or "").lower()
    has_share_intent = any(keyword in lowered for keyword in ["share", "send", "email", "mail", "whatsapp", "whats app"])
    has_conversation_intent = any(keyword in lowered for keyword in ["conversation", "chat history", "our conversation", "full conversation", "whole conversation"])
    return has_share_intent and has_conversation_intent and not _wants_summary_share(query)


def parse_share_request(query: str):
    query = (query or "").strip()
    if not query:
        return {
            "error": "No share request provided.",
            "channel": None,
            "scheduled_for": None,
            "time_note": None,
            "content": "",
            "recipient": None,
            "endpoint": None,
            "missing": ["request text"],
        }

    channel = _detect_channel(query)
    scheduled_for, time_note = parse_relative_datetime(query)
    content = _extract_content(query)
    recipient = _extract_recipient(query)

    if not channel:
        return {
            "error": "Share agent only handles email or WhatsApp requests. Use the scheduler agent for meetings, reminders, and task scheduling.",
            "channel": None,
            "scheduled_for": scheduled_for.isoformat() if scheduled_for else None,
            "time_note": time_note,
            "content": content,
            "recipient": recipient,
            "endpoint": None,
            "missing": ["channel"],
        }

    needs = []
    if not scheduled_for and not time_note:
        time_note = "No time mentioned; treat as immediate unless you want it scheduled."
    if channel in {"email", "whatsapp"} and not recipient:
        needs.append("recipient")
    if len(content) < 15:
        needs.append("message content")

    return {
        "query": query,
        "channel": channel,
        "scheduled_for": scheduled_for.isoformat() if scheduled_for else None,
        "time_note": time_note,
        "content": content,
        "recipient": recipient,
        "endpoint": _suggest_endpoint(channel),
        "missing": needs,
    }


def _required_share_fields(plan: dict) -> list[str]:
    missing = []
    channel = plan.get("channel")
    recipient = (plan.get("recipient") or "").strip() if plan.get("recipient") else None
    content = (plan.get("content") or "").strip()

    if not channel:
        missing.append("channel")
    if channel in {"email", "whatsapp"} and not recipient:
        missing.append("recipient")
    if len(content) < 8 or content.lower() in {"details", "this", "that", "it", "message"}:
        missing.append("message content")

    for item in plan.get("missing", []):
        if item not in missing:
            missing.append(item)
    return missing


def share_details(query: str, context: str | None = None, *, source: str = "chat", actor: str = "user"):
    plan = parse_share_request(query)
    if plan.get("error"):
        return plan["error"]

    if context:
        context = context.strip()
        if context:
            if _wants_summary_share(query):
                plan["content"] = context
            elif len(plan.get("content") or "") < 15 or (plan.get("content") or "").lower() in {"this", "that", "it", "details", "summary"}:
                plan["content"] = context
    elif _wants_summary_share(query):
        plan["content"] = summarize_text(query)
    elif _wants_full_conversation_share(query):
        transcript = chat_history_transcript(limit=24)
        if transcript:
            plan["content"] = transcript

    missing = _required_share_fields(plan)
    if missing:
        lines = ["Share request needs a bit more detail before I can proceed."]
        if plan.get("channel"):
            lines.append(f"Channel: {plan['channel']}")
        if plan.get("scheduled_for"):
            lines.append(f"Schedule: {plan['scheduled_for']}")
        if plan.get("time_note") and not plan.get("scheduled_for"):
            lines.append(f"Schedule note: {plan['time_note']}")
        lines.append("Missing details: " + ", ".join(missing))
        lines.append("Please provide the missing details, then I can share now or schedule it.")
        trace_event(
            "share_plan_missing_details",
            query=plan["query"],
            channel=plan["channel"],
            scheduled_for=plan["scheduled_for"],
            missing_details=missing,
        )
        return "\n".join(lines)

    if plan.get("scheduled_for"):
        if _wants_full_conversation_share(query) and not plan.get("content"):
            transcript = chat_history_transcript(limit=24)
            if transcript:
                plan["content"] = transcript
        task = enqueue_share_task(
            {
                "kind": "share",
                "channel": plan["channel"],
                "scheduled_for": plan.get("scheduled_for"),
                "status": "scheduled",
                "source": source,
                "actor": actor,
                "query": plan["query"],
                "recipient": plan.get("recipient"),
                "subject": plan.get("content")[:64] or None,
                "message": plan.get("content"),
                "time_note": plan.get("time_note"),
            }
        )
        start_scheduler()

        lines = [f"Action: {plan['channel']}"]
        lines.append("State: saved to share_task list")
        lines.append(f"Task id: {task['id']}")
        lines.append(f"Schedule: {plan['scheduled_for']}")
        lines.append(f"Timezone: {DEFAULT_TIMEZONE}")
        if plan.get("recipient"):
            lines.append(f"Recipient: {plan['recipient']}")
        lines.append(f"Draft: {plan['content']}")
        lines.append("Status: ready in shared state for later processing.")
    else:
        if _wants_full_conversation_share(query) and not plan.get("content"):
            transcript = chat_history_transcript(limit=24)
            if transcript:
                plan["content"] = transcript
        delivery = execute_share_action(
            {
                "id": None,
                "kind": "share",
                "channel": plan["channel"],
                "scheduled_for": None,
                "status": "completed",
                "source": source,
                "actor": actor,
                "query": plan["query"],
                "recipient": plan.get("recipient"),
                "subject": plan.get("content")[:64] or None,
                "message": plan["content"],
                "time_note": plan.get("time_note"),
            }
        )

        lines = [f"Action: {plan['channel']}"]
        lines.append("State: shared immediately")
        if plan.get("recipient"):
            lines.append(f"Recipient: {plan['recipient']}")
        lines.append(f"Draft: {plan['content']}")
        lines.append(f"Delivery id: {delivery['delivery_id']}")
        lines.append("Status: completed")
        if plan["time_note"]:
            lines.append(f"Schedule note: {plan['time_note']}")

    trace_event(
        "share_plan_created",
        query=plan["query"],
        channel=plan["channel"],
        scheduled_for=plan["scheduled_for"],
        queue_id=task["id"] if plan.get("scheduled_for") else delivery["delivery_id"],
        missing_details=missing,
        source=source,
        actor=actor,
    )

    return "\n".join(lines)


def execute_share_action(action: dict):
    channel = str(action.get("channel") or "email")
    recipient = action.get("recipient")
    subject = action.get("subject")
    message = str(action.get("message") or action.get("query") or "")
    scheduled_for = action.get("scheduled_for")
    trace_event(
        "share_action_executed",
        action_id=action.get("id"),
        channel=channel,
        recipient=recipient,
        scheduled_for=scheduled_for,
        source=action.get("source"),
        actor=action.get("actor"),
    )
    return queue_dummy_delivery(
        channel=channel,
        recipient=recipient,
        subject=subject,
        scheduled_for=scheduled_for,
        message=message,
    )
