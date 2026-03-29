from __future__ import annotations

from app.services.ingestion_state import schedule_scrape_run
from app.services.scheduler import scheduler_status, start_scheduler
from app.trace import trace_event


# Compatibility helper that registers a scrape job with the shared scheduler.
def schedule_scrape_job(
    *,
    scheduled_for: str,
    scope: str = "all",
    query: str | None = None,
    source: str = "dashboard",
    actor: str = "admin",
) -> dict:
    scope = (scope or "all").strip().lower()
    task_type = "scrape_pending" if scope == "pending" else "scrape_all"
    state = schedule_scrape_run(
        scheduled_for=scheduled_for,
        query=query,
        task_type=task_type,
        trigger="admin",
        scope=scope,
        source=source,
        actor=actor,
    )
    started = start_scheduler()
    trace_event(
        "scheduled_scrape_job_registered",
        scheduled_for=scheduled_for,
        scope=scope,
        source=source,
        actor=actor,
        scheduler_started=started,
    )
    return {
        "job_id": f"scrape-{state.get('scheduled_scrape_at') or scheduled_for}",
        "scheduled_for": state.get("scheduled_scrape_at") or scheduled_for,
        "state": state,
        "time_zone": "Asia/Kolkata",
        "task_type": task_type,
        "scope": scope,
        "source": source,
        "actor": actor,
    }


def scrape_scheduler_status() -> dict:
    status = scheduler_status()
    scrape = status.get("scrape", {})
    return {
        "started": status.get("started", False),
        "active_jobs": status.get("active_jobs", []),
        "pending": scrape.get("pending", False),
        "scheduled_at": scrape.get("scheduled_at"),
        "status": scrape.get("status"),
        "task_type": scrape.get("task_type"),
        "scope": scrape.get("scope"),
        "trigger": scrape.get("trigger"),
        "source": scrape.get("source"),
        "actor": scrape.get("actor"),
        "next_run_at": scrape.get("next_run_at"),
    }
