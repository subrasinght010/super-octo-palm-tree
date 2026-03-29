from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.services.ingestion_state import (
    claim_next_scheduled_task,
    claim_next_share_task,
    complete_scheduled_scrape,
    fail_scheduled_scrape,
    list_scheduled_tasks,
    list_share_tasks,
    load_state,
    next_scheduled_share_time,
    next_scheduled_task_time,
    scheduled_task_summary,
    share_task_summary,
    start_scheduled_scrape,
    update_scheduled_task_status,
    update_share_task_status,
)
from app.tools.scrape import refresh_configured_sources
from app.tools.time_utils import DEFAULT_TIMEZONE
from app.trace import new_request_id, trace_event

_lock = threading.Lock()
_started = False
_active_jobs: set[str] = set()


def _parse_iso_datetime(value: str | None):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(DEFAULT_TIMEZONE))
    return parsed


def _scrape_state() -> dict:
    state = load_state()
    return {
        "pending": bool(state.get("scheduled_scrape_pending")),
        "scheduled_at": state.get("scheduled_scrape_at"),
        "status": state.get("scheduled_scrape_status"),
        "task_type": state.get("scheduled_scrape_task_type"),
        "scope": state.get("scheduled_scrape_scope"),
        "trigger": state.get("scheduled_scrape_trigger"),
        "source": state.get("scheduled_scrape_source"),
        "actor": state.get("scheduled_scrape_actor"),
    }


def _next_scheduled_scrape_time() -> str | None:
    scrape = _scrape_state()
    if scrape["status"] not in {"pending", "scheduled"}:
        return None
    parsed = _parse_iso_datetime(scrape["scheduled_at"])
    if parsed is None:
        return None
    now = datetime.now(parsed.tzinfo or timezone.utc)
    if parsed <= now:
        return parsed.isoformat()
    return parsed.isoformat()


def _execute_share_task(task: dict) -> dict:
    from app.tools.share import execute_share_action

    return execute_share_action(task)


def _execute_scheduled_task(task: dict) -> dict:
    kind = str(task.get("kind") or "task")
    message = str(task.get("message") or task.get("query") or "")
    subject = task.get("subject") or kind
    scheduled_for = task.get("scheduled_for")

    return queue_dummy_delivery(
        channel="task",
        recipient=task.get("recipient"),
        subject=subject,
        scheduled_for=scheduled_for,
        message=message,
    )


def _mark_active(job_key: str):
    with _lock:
        _active_jobs.add(job_key)


def _clear_active(job_key: str):
    with _lock:
        _active_jobs.discard(job_key)


def process_next_share_task(*, mode: str = "scheduled_only") -> dict:
    task = claim_next_share_task(mode=mode)
    if not task:
        return {
            "status": "idle",
            "reason": "no_due_tasks",
            "summary": share_task_summary(),
            "queue": list_share_tasks(),
            "mode": mode,
        }

    job_key = f"share:{task.get('id')}"
    _mark_active(job_key)
    trace_event(
        "share_task_claimed",
        task_id=task.get("id"),
        kind=task.get("kind"),
        channel=task.get("channel"),
        status=task.get("status"),
        scheduled_for=task.get("scheduled_for"),
    )

    try:
        result = _execute_share_task(task)
    except Exception as exc:
        update_share_task_status(task["id"], "failed", error=f"{exc.__class__.__name__}: {exc}")
        trace_event(
            "share_task_failed",
            task_id=task.get("id"),
            kind=task.get("kind"),
            error=f"{exc.__class__.__name__}: {exc}",
        )
        return {"status": "failed", "task": task, "error": f"{exc.__class__.__name__}: {exc}", "mode": mode}
    finally:
        _clear_active(job_key)

    updated = update_share_task_status(task["id"], "completed", result=result)
    trace_event(
        "share_task_completed",
        task_id=task.get("id"),
        kind=task.get("kind"),
        channel=task.get("channel"),
        scheduled_for=task.get("scheduled_for"),
    )
    return {"status": "completed", "task": updated or task, "result": result, "summary": share_task_summary(), "mode": mode}


def process_next_scheduled_task() -> dict:
    task = claim_next_scheduled_task()
    if not task:
        return {
            "status": "idle",
            "reason": "no_due_tasks",
            "summary": scheduled_task_summary(),
            "queue": list_scheduled_tasks(),
        }

    job_key = f"task:{task.get('id')}"
    _mark_active(job_key)
    trace_event(
        "scheduled_task_claimed",
        task_id=task.get("id"),
        kind=task.get("kind"),
        status=task.get("status"),
        scheduled_for=task.get("scheduled_for"),
    )

    try:
        result = _execute_scheduled_task(task)
    except Exception as exc:
        update_scheduled_task_status(task["id"], "failed", error=f"{exc.__class__.__name__}: {exc}")
        trace_event(
            "scheduled_task_failed",
            task_id=task.get("id"),
            kind=task.get("kind"),
            error=f"{exc.__class__.__name__}: {exc}",
        )
        return {"status": "failed", "task": task, "error": f"{exc.__class__.__name__}: {exc}"}
    finally:
        _clear_active(job_key)

    updated = update_scheduled_task_status(task["id"], "completed", result=result)
    trace_event(
        "scheduled_task_completed",
        task_id=task.get("id"),
        kind=task.get("kind"),
        scheduled_for=task.get("scheduled_for"),
    )
    return {"status": "completed", "task": updated or task, "result": result, "summary": scheduled_task_summary()}


def process_due_scheduled_scrape() -> dict:
    state = load_state()
    if not state.get("scheduled_scrape_pending"):
        return {
            "status": "idle",
            "reason": "no_due_tasks",
            "state": _scrape_state(),
        }

    status = str(state.get("scheduled_scrape_status") or "scheduled")
    if status not in {"pending", "scheduled"}:
        return {
            "status": "idle",
            "reason": "not_scheduled",
            "state": _scrape_state(),
        }

    target = _parse_iso_datetime(state.get("scheduled_scrape_at"))
    if target is None:
        return {
            "status": "idle",
            "reason": "missing_schedule_time",
            "state": _scrape_state(),
        }

    now = datetime.now(target.tzinfo or timezone.utc)
    if target > now:
        return {
            "status": "idle",
            "reason": "not_due_yet",
            "next_run_at": target.isoformat(),
            "state": _scrape_state(),
        }

    request_id = new_request_id()
    job_key = f"scrape:{request_id}"
    _mark_active(job_key)
    trace_event(
        "scheduled_scrape_job_claimed",
        request_id=request_id,
        scheduled_for=target.isoformat(),
        scope=state.get("scheduled_scrape_scope"),
        source=state.get("scheduled_scrape_source"),
        actor=state.get("scheduled_scrape_actor"),
    )

    try:
        start_scheduled_scrape()
        results = refresh_configured_sources(scope=state.get("scheduled_scrape_scope") or "all")
        complete_scheduled_scrape()
        trace_event(
            "scheduled_scrape_job_completed",
            request_id=request_id,
            scheduled_for=target.isoformat(),
            source_count=len(results),
            scope=state.get("scheduled_scrape_scope") or "all",
            source=state.get("scheduled_scrape_source"),
            actor=state.get("scheduled_scrape_actor"),
        )
        return {
            "status": "completed",
            "request_id": request_id,
            "result_count": len(results),
            "state": _scrape_state(),
        }
    except Exception as exc:
        fail_scheduled_scrape()
        trace_event(
            "scheduled_scrape_job_failed",
            request_id=request_id,
            scheduled_for=target.isoformat(),
            error=f"{exc.__class__.__name__}: {exc}",
            scope=state.get("scheduled_scrape_scope") or "all",
            source=state.get("scheduled_scrape_source"),
            actor=state.get("scheduled_scrape_actor"),
        )
        return {
            "status": "failed",
            "request_id": request_id,
            "error": f"{exc.__class__.__name__}: {exc}",
            "state": _scrape_state(),
        }
    finally:
        _clear_active(job_key)


def _next_due_time() -> str | None:
    next_times = []
    for candidate in (next_scheduled_share_time(), next_scheduled_task_time(), _next_scheduled_scrape_time()):
        parsed = _parse_iso_datetime(candidate)
        if parsed is not None and parsed > datetime.now(parsed.tzinfo or timezone.utc):
            next_times.append(parsed)
    if not next_times:
        return None
    return min(next_times).isoformat()


def _worker():
    global _started
    try:
        while True:
            outcome = process_next_share_task(mode="scheduled_only")
            if outcome.get("status") == "completed":
                trace_event("common_scheduler_step_completed", kind="share")
                continue

            outcome = process_next_scheduled_task()
            if outcome.get("status") == "completed":
                trace_event("common_scheduler_step_completed", kind="task")
                continue

            outcome = process_due_scheduled_scrape()
            if outcome.get("status") == "completed":
                trace_event("common_scheduler_step_completed", kind="scrape")
                continue

            next_time = _next_due_time()
            if not next_time:
                break

            try:
                target = datetime.fromisoformat(next_time)
                if target.tzinfo is None:
                    target = target.replace(tzinfo=ZoneInfo(DEFAULT_TIMEZONE))
                delay = max(1.0, (target - datetime.now(target.tzinfo)).total_seconds())
            except Exception:
                delay = 30.0
            trace_event("common_scheduler_waiting", next_scheduled_for=next_time, delay_seconds=round(delay, 2))
            time.sleep(min(delay, 60.0))
    finally:
        with _lock:
            _started = False
        trace_event("common_scheduler_stopped", state=scheduler_status())


def start_scheduler():
    global _started
    with _lock:
        if _started:
            return False
        _started = True
    thread = threading.Thread(target=_worker, name="common-scheduler", daemon=True)
    thread.start()
    trace_event("common_scheduler_started")
    return True


def scheduler_status() -> dict:
    state = load_state()
    share = {
        "summary": share_task_summary(),
        "queue": list_share_tasks(),
        "next_run_at": next_scheduled_share_time(),
    }
    task = {
        "summary": scheduled_task_summary(),
        "queue": list_scheduled_tasks(),
        "next_run_at": next_scheduled_task_time(),
    }
    scrape = _scrape_state()
    scrape["next_run_at"] = _next_scheduled_scrape_time()
    next_candidates = [share["next_run_at"], task["next_run_at"], scrape["next_run_at"]]
    parsed_candidates = [ts for ts in (_parse_iso_datetime(value) for value in next_candidates) if ts is not None]
    return {
        "started": _started,
        "active_jobs": sorted(_active_jobs),
        "share": share,
        "task": task,
        "scrape": scrape,
        "next_run_at": min(parsed_candidates).isoformat() if parsed_candidates else None,
        "ingestion_state": state,
    }
