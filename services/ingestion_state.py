from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable
from uuid import uuid4

STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "ingestion_state.json"

_DEFAULT_STATE: Dict[str, Any] = {
    "refresh_pending": False,
    "scheduled_scrape_pending": False,
    "scheduled_scrape_at": None,
    "scheduled_scrape_task_type": None,
    "scheduled_scrape_scope": None,
    "scheduled_scrape_trigger": None,
    "scheduled_scrape_source": None,
    "scheduled_scrape_actor": None,
    "scheduled_scrape_status": None,
    "scheduled_task": [],
    "share_task": [],
    "chat_history": [],
    "updated_at": None,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> Dict[str, Any]:
    return {**_DEFAULT_STATE, "updated_at": utc_now()}


def _prune_state(state: Dict[str, Any]) -> Dict[str, Any]:
    base = _default_state()
    return {key: state.get(key, base.get(key)) for key in base}


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return _default_state()

    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            merged = _default_state()
            merged.update({k: v for k, v in data.items() if k in merged})
            return merged
    except Exception:
        pass
    return _default_state()


def save_state(state: Dict[str, Any]) -> Dict[str, Any]:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = _prune_state(state)
    state["updated_at"] = utc_now()
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state


def mark_refresh_pending(*, reason: str, urls: Iterable[str | None] = (), query: str | None = None) -> Dict[str, Any]:
    _ = reason, query
    _ = urls
    state = load_state()
    state["refresh_pending"] = True
    return save_state(state)


def _action_now() -> str:
    return utc_now()


def _normalize_share_task(action: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "id": str(action.get("id") or uuid4().hex),
        "kind": str(action.get("kind") or "share").strip(),
        "channel": action.get("channel"),
        "scheduled_for": action.get("scheduled_for"),
        "status": action.get("status") or ("scheduled" if action.get("scheduled_for") else "pending"),
        "source": action.get("source") or "dashboard",
        "actor": action.get("actor") or "admin",
        "query": action.get("query"),
        "recipient": action.get("recipient"),
        "subject": action.get("subject"),
        "message": action.get("message"),
        "time_note": action.get("time_note"),
        "created_at": action.get("created_at") or _action_now(),
        "updated_at": action.get("updated_at") or _action_now(),
        "result": action.get("result"),
        "error": action.get("error"),
    }
    return normalized


def _normalize_scheduled_task(task: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "id": str(task.get("id") or uuid4().hex),
        "kind": str(task.get("kind") or "task").strip(),
        "channel": task.get("channel"),
        "scheduled_for": task.get("scheduled_for"),
        "status": task.get("status") or ("scheduled" if task.get("scheduled_for") else "pending"),
        "source": task.get("source") or "dashboard",
        "actor": task.get("actor") or "admin",
        "query": task.get("query"),
        "recipient": task.get("recipient"),
        "subject": task.get("subject"),
        "message": task.get("message"),
        "time_note": task.get("time_note"),
        "created_at": task.get("created_at") or _action_now(),
        "updated_at": task.get("updated_at") or _action_now(),
        "result": task.get("result"),
        "error": task.get("error"),
    }
    return normalized


def enqueue_share_task(action: Dict[str, Any]) -> Dict[str, Any]:
    state = load_state()
    queue = state.get("share_task", [])
    if not isinstance(queue, list):
        queue = []
    item = _normalize_share_task(action)
    queue.append(item)
    state["share_task"] = queue
    save_state(state)
    return item


def enqueue_scheduled_task(task: Dict[str, Any]) -> Dict[str, Any]:
    state = load_state()
    queue = state.get("scheduled_task", [])
    if not isinstance(queue, list):
        queue = []
    item = _normalize_scheduled_task(task)
    queue.append(item)
    state["scheduled_task"] = queue
    save_state(state)
    return item


def append_chat_turn(
    *,
    query: str,
    response: str,
    mode: str | None = None,
    route: str | None = None,
    username: str | None = None,
    role: str | None = None,
    limit: int = 24,
) -> Dict[str, Any]:
    state = load_state()
    history = state.get("chat_history", [])
    if not isinstance(history, list):
        history = []

    item = {
        "id": str(uuid4().hex),
        "query": query,
        "response": response,
        "mode": mode,
        "route": route,
        "username": username,
        "role": role,
        "created_at": _action_now(),
    }
    history.append(item)
    if len(history) > max(1, limit):
        history = history[-max(1, limit):]
    state["chat_history"] = history
    save_state(state)
    return item


def list_chat_history(limit: int = 24) -> list[Dict[str, Any]]:
    state = load_state()
    history = state.get("chat_history", [])
    if not isinstance(history, list):
        return []
    if limit and limit > 0:
        history = history[-limit:]
    return [dict(item) for item in history if isinstance(item, dict)]


def chat_history_transcript(limit: int = 24) -> str:
    history = list_chat_history(limit=limit)
    if not history:
        return ""
    lines = []
    for item in history:
        query = str(item.get("query") or "").strip()
        response = str(item.get("response") or "").strip()
        if query:
            lines.append(f"User: {query}")
        if response:
            lines.append(f"Assistant: {response}")
    return "\n".join(lines).strip()


def list_scheduled_tasks() -> list[Dict[str, Any]]:
    state = load_state()
    queue = state.get("scheduled_task", [])
    return [dict(item) for item in queue if isinstance(item, dict)]


def scheduled_task_summary() -> Dict[str, int]:
    queue = list_scheduled_tasks()
    summary = {
        "total": len(queue),
        "pending": 0,
        "scheduled": 0,
        "processing": 0,
        "completed": 0,
        "failed": 0,
    }
    for item in queue:
        status = str(item.get("status") or "pending")
        if status in summary:
            summary[status] += 1
    return summary


def _parse_task_time(value: str | None):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def claim_next_scheduled_task() -> Dict[str, Any] | None:
    state = load_state()
    queue = state.get("scheduled_task", [])
    if not isinstance(queue, list) or not queue:
        return None

    now = datetime.now(timezone.utc)
    due_indexes = []
    for index, item in enumerate(queue):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "pending")
        if status in {"completed", "failed", "processing"}:
            continue
        scheduled_for = _parse_task_time(item.get("scheduled_for"))
        if scheduled_for is not None and scheduled_for <= now:
            due_indexes.append((scheduled_for, index))

    if not due_indexes:
        return None

    due_indexes.sort(key=lambda pair: pair[0])
    _, index = due_indexes[0]
    item = dict(queue[index])
    item["status"] = "processing"
    item["updated_at"] = _action_now()
    queue[index] = item
    state["scheduled_task"] = queue
    save_state(state)
    return item


def update_scheduled_task_status(
    task_id: str,
    status: str,
    *,
    result: Any | None = None,
    error: str | None = None,
) -> Dict[str, Any] | None:
    state = load_state()
    queue = state.get("scheduled_task", [])
    if not isinstance(queue, list):
        return None

    updated_item = None
    for index, item in enumerate(queue):
        if not isinstance(item, dict) or str(item.get("id")) != str(task_id):
            continue
        item = dict(item)
        item["status"] = status
        item["updated_at"] = _action_now()
        if result is not None:
            item["result"] = result
        if error is not None:
            item["error"] = error
        queue[index] = item
        updated_item = item
        break

    if updated_item is None:
        return None

    state["scheduled_task"] = queue
    save_state(state)
    return updated_item


def next_scheduled_task_time() -> str | None:
    queue = list_scheduled_tasks()
    upcoming = []
    now = datetime.now(timezone.utc)
    for item in queue:
        status = str(item.get("status") or "pending")
        if status not in {"pending", "scheduled"}:
            continue
        scheduled_for = item.get("scheduled_for")
        if not scheduled_for:
            continue
        parsed = _parse_task_time(scheduled_for)
        if parsed is None:
            continue
        if parsed > now:
            upcoming.append(parsed)
    if not upcoming:
        return None
    return min(upcoming).isoformat()


def list_share_tasks() -> list[Dict[str, Any]]:
    state = load_state()
    queue = state.get("share_task", [])
    return [dict(item) for item in queue if isinstance(item, dict)]


def share_task_summary() -> Dict[str, int]:
    queue = list_share_tasks()
    summary = {
        "total": len(queue),
        "pending": 0,
        "scheduled": 0,
        "processing": 0,
        "completed": 0,
        "failed": 0,
    }
    for item in queue:
        status = str(item.get("status") or "pending")
        if status in summary:
            summary[status] += 1
    return summary


def _parse_action_time(value: str | None):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def claim_next_share_task(*, mode: str = "all") -> Dict[str, Any] | None:
    state = load_state()
    queue = state.get("share_task", [])
    if not isinstance(queue, list) or not queue:
        return None

    now = datetime.now(timezone.utc)
    due_indexes = []
    for index, item in enumerate(queue):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "pending")
        if status in {"completed", "failed", "processing"}:
            continue
        scheduled_for = _parse_action_time(item.get("scheduled_for"))
        if mode == "scheduled_only":
            if scheduled_for is not None and scheduled_for <= now:
                due_indexes.append((scheduled_for, index))
            continue
        if scheduled_for is None or scheduled_for <= now:
            due_indexes.append(
                (
                    scheduled_for or _parse_action_time(item.get("created_at")) or now,
                    index,
                )
            )

    if not due_indexes:
        return None

    due_indexes.sort(key=lambda pair: pair[0])
    _, index = due_indexes[0]
    item = dict(queue[index])
    item["status"] = "processing"
    item["updated_at"] = _action_now()
    queue[index] = item
    state["share_task"] = queue
    save_state(state)
    return item


def update_share_task_status(task_id: str, status: str, *, result: Any | None = None, error: str | None = None) -> Dict[str, Any] | None:
    state = load_state()
    queue = state.get("share_task", [])
    if not isinstance(queue, list):
        return None

    updated_item = None
    for index, item in enumerate(queue):
        if not isinstance(item, dict) or str(item.get("id")) != str(task_id):
            continue
        item = dict(item)
        item["status"] = status
        item["updated_at"] = _action_now()
        if result is not None:
            item["result"] = result
        if error is not None:
            item["error"] = error
        queue[index] = item
        updated_item = item
        break

    if updated_item is None:
        return None

    state["share_task"] = queue
    save_state(state)
    return updated_item


def next_scheduled_share_time() -> str | None:
    queue = list_share_tasks()
    upcoming = []
    now = datetime.now(timezone.utc)
    for item in queue:
        status = str(item.get("status") or "pending")
        if status not in {"pending", "scheduled"}:
            continue
        scheduled_for = item.get("scheduled_for")
        if not scheduled_for:
            continue
        parsed = _parse_action_time(scheduled_for)
        if parsed is None:
            continue
        if parsed > now:
            upcoming.append(parsed)
    if not upcoming:
        return None
    return min(upcoming).isoformat()


def record_web_search(*, query: str, result_count: int, added_urls: Iterable[str | None] = ()) -> Dict[str, Any]:
    _ = query, result_count
    _ = added_urls
    state = load_state()
    state["refresh_pending"] = True
    return save_state(state)


def record_scrape_run(
    *,
    query: str,
    source_count: int,
    scope: str = "all",
    urls: Iterable[str | None] = (),
) -> Dict[str, Any]:
    _ = query, source_count
    state = load_state()
    state.update(
        {
            "refresh_pending": False,
        }
    )
    return save_state(state)


def clear_refresh_pending() -> Dict[str, Any]:
    state = load_state()
    state["refresh_pending"] = False
    return save_state(state)


def schedule_scrape_run(
    *,
    scheduled_for: str,
    query: str | None = None,
    task_type: str = "scrape",
    trigger: str = "admin",
    scope: str = "all",
    source: str | None = None,
    actor: str | None = None,
) -> Dict[str, Any]:
    _ = query
    state = load_state()
    state.update(
        {
            "scheduled_scrape_pending": True,
            "scheduled_scrape_at": scheduled_for,
            "scheduled_scrape_task_type": task_type,
            "scheduled_scrape_scope": scope,
            "scheduled_scrape_trigger": trigger,
            "scheduled_scrape_source": source,
            "scheduled_scrape_actor": actor,
            "scheduled_scrape_status": "scheduled",
        }
    )
    return save_state(state)


def start_scheduled_scrape(*, query: str | None = None) -> Dict[str, Any]:
    _ = query
    state = load_state()
    state.update(
        {
            "scheduled_scrape_pending": True,
            "scheduled_scrape_status": "running",
            "scheduled_scrape_task_type": state.get("scheduled_scrape_task_type") or "scrape",
            "scheduled_scrape_scope": state.get("scheduled_scrape_scope") or "all",
            "scheduled_scrape_source": state.get("scheduled_scrape_source"),
            "scheduled_scrape_actor": state.get("scheduled_scrape_actor"),
        }
    )
    return save_state(state)


def complete_scheduled_scrape() -> Dict[str, Any]:
    state = load_state()
    state.update(
        {
            "scheduled_scrape_pending": False,
            "scheduled_scrape_status": "completed",
            "scheduled_scrape_task_type": state.get("scheduled_scrape_task_type") or "scrape",
            "scheduled_scrape_scope": state.get("scheduled_scrape_scope") or "all",
            "scheduled_scrape_source": state.get("scheduled_scrape_source"),
            "scheduled_scrape_actor": state.get("scheduled_scrape_actor"),
        }
    )
    return save_state(state)


def fail_scheduled_scrape() -> Dict[str, Any]:
    state = load_state()
    state.update(
        {
            "scheduled_scrape_pending": True,
            "scheduled_scrape_status": "failed",
            "scheduled_scrape_task_type": state.get("scheduled_scrape_task_type") or "scrape",
            "scheduled_scrape_scope": state.get("scheduled_scrape_scope") or "all",
            "scheduled_scrape_source": state.get("scheduled_scrape_source"),
            "scheduled_scrape_actor": state.get("scheduled_scrape_actor"),
        }
    )
    return save_state(state)
