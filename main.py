from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Literal

# Allow running from either the repo root or the app/ directory.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.agent import run_agent
from app.langgraph import build_graph
from app.mcp_registry import list_tools
from app.trace import new_request_id, trace_event

app = FastAPI(title="AI Research Assistant")
graph = build_graph()
FRONTEND_PATH = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


class QueryRequest(BaseModel):
    query: str
    mode: Literal["auto", "agent", "graph"] = "auto"


class DummyDeliveryRequest(BaseModel):
    recipient: str | None = None
    message: str
    subject: str | None = None
    scheduled_for: str | None = None


def _chunk_text(text: str, size: int = 48):
    if not text:
        return
    for index in range(0, len(text), size):
        yield text[index : index + size]


async def _resolve_chat(req: QueryRequest, request_id: str):
    trace_event("chat_started", request_id=request_id, query=req.query, mode=req.mode)

    if req.mode == "graph":
        result = await run_in_threadpool(graph.invoke, {"query": req.query, "request_id": request_id})
        trace_event(
            "chat_completed",
            request_id=request_id,
            mode="graph",
            route=result.get("route"),
        )
        return {
            "mode": "graph",
            "query": req.query,
            "route": result.get("route"),
            "result": result.get("result", ""),
        }

    answer = await run_in_threadpool(run_agent, req.query, request_id)
    trace_event("chat_completed", request_id=request_id, mode=req.mode)
    return {
        "mode": req.mode,
        "query": req.query,
        "result": answer,
    }


@app.post("/chat")
async def chat(req: QueryRequest):
    request_id = new_request_id()
    trace_event("chat_received", request_id=request_id, query=req.query, mode=req.mode)
    return await _resolve_chat(req, request_id)


@app.post("/chat/stream")
async def chat_stream(req: QueryRequest):
    request_id = new_request_id()
    trace_event("chat_received", request_id=request_id, query=req.query, mode=req.mode, stream=True)
    resolved = await _resolve_chat(req, request_id)
    result_text = resolved.get("result", "")
    display_lines = []
    if resolved.get("mode"):
        display_lines.append(f"Mode: {resolved['mode']}")
    if resolved.get("route"):
        display_lines.append(f"Route: {resolved['route']}")
    display_lines.append("")
    display_lines.append(result_text)
    display_text = "\n".join(display_lines).strip()

    async def streamer():
        trace_event(
            "chat_stream_started",
            request_id=request_id,
            mode=resolved.get("mode"),
            route=resolved.get("route"),
            output_length=len(display_text),
        )
        for chunk in _chunk_text(display_text):
            yield chunk
            await asyncio.sleep(0.01)
        trace_event(
            "chat_stream_completed",
            request_id=request_id,
            mode=resolved.get("mode"),
            route=resolved.get("route"),
            output_length=len(display_text),
        )

    return StreamingResponse(streamer(), media_type="text/plain; charset=utf-8")


@app.get("/tools")
async def tools():
    return {"tools": list_tools()}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/dummy/email")
async def dummy_email(req: DummyDeliveryRequest):
    trace_event(
        "dummy_email_queued",
        request_id=new_request_id(),
        recipient=req.recipient,
        subject=req.subject,
        scheduled_for=req.scheduled_for,
        message=req.message,
    )
    return {
        "channel": "email",
        "status": "queued",
        "mode": "dummy",
        "recipient": req.recipient,
        "subject": req.subject or "(no subject)",
        "message": req.message,
        "scheduled_for": req.scheduled_for,
        "note": "This is a mock endpoint for testing the email flow.",
    }


@app.post("/dummy/whatsapp")
async def dummy_whatsapp(req: DummyDeliveryRequest):
    trace_event(
        "dummy_whatsapp_queued",
        request_id=new_request_id(),
        recipient=req.recipient,
        scheduled_for=req.scheduled_for,
        message=req.message,
    )
    return {
        "channel": "whatsapp",
        "status": "queued",
        "mode": "dummy",
        "recipient": req.recipient,
        "message": req.message,
        "scheduled_for": req.scheduled_for,
        "note": "This is a mock endpoint for testing the WhatsApp flow.",
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return FRONTEND_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
