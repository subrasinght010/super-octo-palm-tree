from __future__ import annotations

from typing import Any, Dict

from app.trace import new_request_id, trace_event


def queue_dummy_delivery(
    *,
    channel: str,
    message: str,
    recipient: str | None = None,
    subject: str | None = None,
    scheduled_for: str | None = None,
    request_id: str | None = None,
) -> Dict[str, Any]:
    delivery_id = new_request_id()
    payload = {
        "delivery_id": delivery_id,
        "channel": channel,
        "status": "queued",
        "mode": "dummy",
        "recipient": recipient,
        "subject": subject,
        "message": message,
        "scheduled_for": scheduled_for,
    }
    trace_event(
        "dummy_delivery_queued",
        request_id=request_id,
        delivery_id=delivery_id,
        channel=channel,
        recipient=recipient,
        subject=subject,
        scheduled_for=scheduled_for,
        message=message,
    )
    return payload

