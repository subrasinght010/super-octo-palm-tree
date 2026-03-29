from __future__ import annotations

from app.services.scheduler import scheduler_status, start_scheduler


# Compatibility wrapper around the shared scheduler service.
def start_task_scheduler():
    return start_scheduler()


def task_scheduler_status() -> dict:
    status = scheduler_status()
    task = status.get("task", {})
    return {
        "started": status.get("started", False),
        "active_jobs": status.get("active_jobs", []),
        "summary": task.get("summary", {}),
        "queue": task.get("queue", []),
        "next_run_at": task.get("next_run_at"),
    }
