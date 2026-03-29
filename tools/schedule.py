from __future__ import annotations

import re

from app.services.ingestion_state import enqueue_scheduled_task
from app.services.scheduler import start_scheduler
from app.tools.time_utils import DEFAULT_TIMEZONE, parse_relative_datetime
from app.trace import trace_event


def _extract_task(query: str):
    lowered = query.lower()
    candidates = [
        "schedule",
        "set up",
        "set",
        "arrange",
        "book",
        "plan",
        "remind me to",
        "remind me",
        "remind",
        "create task",
        "task",
        "todo",
        "meeting",
        "calendar",
        "invite",
        "appointment",
    ]
    task = query.strip()
    for keyword in candidates:
        if keyword in lowered:
            parts = re.split(rf"\b{re.escape(keyword)}\b", query, flags=re.I, maxsplit=1)
            if len(parts) > 1:
                tail = parts[1].strip(" :-,")
                if tail:
                    task = tail
                    break
    return task


def parse_schedule_request(query: str):
    query = (query or "").strip()
    if not query:
        return {
            "error": "No schedule request provided.",
            "channel": None,
            "scheduled_for": None,
            "time_note": None,
            "task": "",
            "endpoint": None,
            "missing": ["request text"],
        }

    scheduled_for, time_note = parse_relative_datetime(query)
    task = _extract_task(query)
    lower = query.lower()
    kind = "task"
    if any(keyword in lower for keyword in ["meeting", "calendar", "invite"]):
        kind = "meeting"
    elif any(keyword in lower for keyword in ["remind", "reminder"]):
        kind = "reminder"

    missing = []
    if not scheduled_for and not time_note:
        time_note = "No time mentioned; ask for a time or treat as pending."
        missing.append("time")
    if len(task) < 10:
        missing.append("task description")

    return {
        "query": query,
        "kind": kind,
        "scheduled_for": scheduled_for.isoformat() if scheduled_for else None,
        "time_note": time_note,
        "task": task,
        "endpoint": "/dummy/task",
        "missing": missing,
        "timezone": DEFAULT_TIMEZONE,
    }


def schedule_task(query: str, *, source: str = "chat", actor: str = "user"):
    plan = parse_schedule_request(query)
    if plan.get("error"):
        return plan["error"]

    if not plan["scheduled_for"]:
        return (
            "Schedule requests need a concrete time.\n"
            f"Task: {plan['task']}\n"
            "Try something like: 'schedule a meeting tomorrow at 10 am'."
        )

    task = enqueue_scheduled_task(
        {
            "kind": "task",
            "channel": "task",
            "scheduled_for": plan["scheduled_for"],
            "status": "scheduled",
            "source": source,
            "actor": actor,
            "query": plan["query"],
            "subject": plan["kind"],
            "message": plan["task"],
            "time_note": plan["time_note"],
        }
    )
    start_scheduler()

    lines = [f"Action: schedule_{plan['kind']}"]
    lines.append("State: saved to scheduled_task list")
    lines.append(f"Task id: {task['id']}")
    if plan["scheduled_for"]:
        lines.append(f"Schedule: {plan['scheduled_for']}")
        lines.append(f"Timezone: {plan['timezone']}")
    elif plan["time_note"]:
        lines.append(f"Schedule: {plan['time_note']}")
    lines.append(f"Task: {plan['task']}")
    if plan["missing"]:
        lines.append("Missing details: " + ", ".join(plan["missing"]))
    else:
        lines.append("Status: ready to hand off to a reminder or calendar integration.")

    trace_event(
        "schedule_plan_created",
        query=plan["query"],
        kind=plan["kind"],
        scheduled_for=plan["scheduled_for"],
        task_id=task["id"],
        missing_details=plan["missing"],
        source=source,
        actor=actor,
    )

    return "\n".join(lines)


def queue_scheduled_task(
    *,
    query: str,
    request_id: str | None = None,
    context: str | None = None,
    source: str = "chat",
    actor: str = "user",
):
    plan = parse_schedule_request(query)
    if plan.get("error"):
        trace_event("schedule_agent_error", request_id=request_id, query=query, error=plan["error"])
        return plan["error"]
    if not plan["scheduled_for"]:
        trace_event(
            "schedule_agent_pending",
            request_id=request_id,
            query=query,
            kind=plan["kind"],
        )
        return schedule_task(query)

    task = enqueue_scheduled_task(
        {
            "kind": "task",
            "channel": "task",
            "scheduled_for": plan["scheduled_for"],
            "status": "scheduled",
            "source": source,
            "actor": actor,
            "query": plan["query"],
            "subject": plan["kind"],
            "message": context.strip() if context else plan["task"],
            "time_note": plan["time_note"],
        }
    )
    start_scheduler()
    trace_event(
        "schedule_agent_queued",
        request_id=request_id,
        kind=plan["kind"],
        scheduled_for=plan["scheduled_for"],
        task_id=task["id"],
        source=source,
        actor=actor,
    )
    return (
        f"Schedule agent used: {plan['kind']}\n"
        f"State: saved to scheduled_task list\n"
        f"Task id: {task['id']}\n"
        f"Schedule: {plan['scheduled_for']}\n"
        f"Task:\n{context.strip() if context else plan['task']}"
    )
