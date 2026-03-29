from __future__ import annotations

from app.services.scheduler import scheduler_status, start_scheduler


# Compatibility wrapper around the shared scheduler service.
def start_share_scheduler():
    return start_scheduler()


def share_scheduler_status() -> dict:
    status = scheduler_status()
    share = status.get("share", {})
    return {
        "started": status.get("started", False),
        "active_jobs": status.get("active_jobs", []),
        "summary": share.get("summary", {}),
        "queue": share.get("queue", []),
        "next_run_at": share.get("next_run_at"),
    }
