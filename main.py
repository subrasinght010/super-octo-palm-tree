from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import sys
import threading
from pathlib import Path
from typing import Literal

# Allow running from either the repo root or the app/ directory.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from fastapi import Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.langgraph import build_graph
from app.mcp_registry import list_tools
from app.services.auth import (
    AuthUser,
    authenticate_user,
    clear_session_cookie,
    register_user,
    require_min_role,
    set_session_cookie,
)
from app.services.background import start_background_scraper
from app.services.scheduler import scheduler_status, start_scheduler
from app.services.delivery import queue_dummy_delivery
from app.services.evaluation import load_eval_set, run_evaluation_suite
from app.services.ingestion_state import append_chat_turn, load_state
from app.services.maintenance import start_maintenance_scheduler
from app.services.scrape_scheduler import schedule_scrape_job
from app.services.knowledge_base import (
    EMBEDDING_MODEL,
    FAISS_DIRTY_PATH,
    confidence_label,
    format_hits,
    backfill_missing_embeddings,
    rebuild_faiss_index,
    initialize_db,
    search_knowledge,
    seed_corpus,
)
from app.multiagent import run_agent as run_multiagent, run_research_agent
from app.tools.rag import extract_citations
from app.trace import new_request_id, trace_event

def _bootstrap_background():
    try:
        trace_event("bootstrap_started")
        initialize_db()

        from app.services.knowledge_base import connect

        with connect() as conn:
            has_documents = conn.execute(
                "SELECT 1 FROM documents LIMIT 1"
            ).fetchone() is not None

        if not has_documents:
            seed_corpus(generate_embeddings=False)

        backfill_missing_embeddings()

        from app.tools.scrape import clean_scrape_config

        clean_scrape_config()
        rebuild_faiss_index()
        start_background_scraper()
        start_scheduler()
        start_maintenance_scheduler()
        trace_event("bootstrap_completed", seeded=not has_documents)
    except Exception as exc:
        trace_event("bootstrap_failed", error=f"{exc.__class__.__name__}: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_bootstrap_background, name="bootstrap-background", daemon=True).start()
    yield


app = FastAPI(title="AI Research Assistant", lifespan=lifespan)
graph = build_graph()
FRONTEND_PATH = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _render_frontend() -> str:
    return FRONTEND_PATH.read_text(encoding="utf-8")


class QueryRequest(BaseModel):
    query: str
    mode: Literal["auto", "agent", "multi", "graph"] = "auto"
    intent_choice: Literal["summary_only", "share_full", "summary_share"] | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str | None = None


class DummyDeliveryRequest(BaseModel):
    recipient: str | None = None
    message: str
    subject: str | None = None
    scheduled_for: str | None = None


class ScrapeNowRequest(BaseModel):
    scope: Literal["pending", "all"] = "pending"
    source: str = "dashboard"
    actor: str = "admin"


class ScrapeScheduleRequest(BaseModel):
    scheduled_for: str
    query: str | None = None
    scope: Literal["pending", "all"] = "pending"
    source: str = "dashboard"
    actor: str = "admin"


def _requires_privileged_action(query: str) -> str | None:
    lowered = (query or "").lower()
    if any(keyword in lowered for keyword in ["scrape", "crawl", "extract from site", "configured sites", "given web", "web given in config"]):
        return "scraping"
    if any(keyword in lowered for keyword in ["email", "mail", "whatsapp", "whats app", "share", "send"]):
        return "sharing"
    if any(keyword in lowered for keyword in ["meeting", "schedule", "calendar", "invite", "remind", "reminder", "task", "todo", "appointment"]):
        return "scheduling"
    return None


def _chunk_text(text: str, size: int = 48):
    if not text:
        return
    for index in range(0, len(text), size):
        yield text[index : index + size]


async def _resolve_chat(req: QueryRequest, request_id: str, current_user: AuthUser):
    trace_event(
        "chat_started",
        request_id=request_id,
        query=req.query,
        mode=req.mode,
        username=current_user.username,
        role=current_user.role,
    )

    privileged_action = _requires_privileged_action(req.query)
    if current_user.role == "user" and privileged_action:
        trace_event(
            "chat_forbidden",
            request_id=request_id,
            query=req.query,
            mode=req.mode,
            username=current_user.username,
            role=current_user.role,
            action=privileged_action,
        )
        raise HTTPException(
            status_code=403,
            detail=f"{privileged_action.title()} requests require an admin or super admin account.",
        )

    if req.mode == "graph":
        result = await run_in_threadpool(
            graph.invoke,
            {"query": req.query, "request_id": request_id, "intent_choice": req.intent_choice},
        )
        citations = extract_citations(result.get("result", "")) if isinstance(result.get("result", ""), str) else []
        append_chat_turn(
            query=req.query,
            response=str(result.get("result", "")),
            mode="graph",
            route=result.get("route"),
            username=current_user.username,
            role=current_user.role,
        )
        trace_event(
            "chat_completed",
            request_id=request_id,
            mode="graph",
            route=result.get("route"),
            username=current_user.username,
            role=current_user.role,
        )
        return {
            "mode": "graph",
            "query": req.query,
            "route": result.get("route"),
            "result": result.get("result", ""),
            "citations": citations,
        }

    if req.mode == "agent":
        answer = await run_in_threadpool(run_research_agent, req.query, request_id)
    else:
        answer = await run_in_threadpool(
            run_multiagent,
            req.query,
            request_id,
            req.intent_choice,
            source="chat",
            actor=current_user.username,
        )
    citations = extract_citations(answer) if isinstance(answer, str) else []
    append_chat_turn(
        query=req.query,
        response=str(answer),
        mode=req.mode,
        username=current_user.username,
        role=current_user.role,
    )
    trace_event(
        "chat_completed",
        request_id=request_id,
        mode=req.mode,
        username=current_user.username,
        role=current_user.role,
    )
    return {
        "mode": req.mode,
        "query": req.query,
        "result": answer,
        "citations": citations,
    }


@app.post("/auth/login")
async def auth_login(req: LoginRequest, response: Response):
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    set_session_cookie(response, user)
    trace_event("auth_login_success", username=user.username, role=user.role)
    return {"status": "ok", "user": user.to_dict()}


@app.post("/auth/register")
async def auth_register(req: RegisterRequest, response: Response):
    user = register_user(req.username, req.password, req.display_name)
    set_session_cookie(response, user)
    trace_event("auth_register_success", username=user.username, role=user.role)
    return {"status": "ok", "user": user.to_dict()}


@app.post("/auth/logout")
async def auth_logout(response: Response):
    clear_session_cookie(response)
    trace_event("auth_logout")
    return {"status": "ok"}


@app.get("/auth/me")
async def auth_me(current_user: AuthUser = Depends(require_min_role("user"))):
    return {"status": "ok", "user": current_user.to_dict()}


@app.post("/chat")
async def chat(req: QueryRequest, current_user: AuthUser = Depends(require_min_role("user"))):
    request_id = new_request_id()
    trace_event(
        "chat_received",
        request_id=request_id,
        query=req.query,
        mode=req.mode,
        username=current_user.username,
        role=current_user.role,
    )
    return await _resolve_chat(req, request_id, current_user)


@app.post("/chat/stream")
async def chat_stream(req: QueryRequest, current_user: AuthUser = Depends(require_min_role("user"))):
    request_id = new_request_id()
    trace_event(
        "chat_received",
        request_id=request_id,
        query=req.query,
        mode=req.mode,
        stream=True,
        username=current_user.username,
        role=current_user.role,
    )
    resolved = await _resolve_chat(req, request_id, current_user)
    result_text = resolved.get("result", "")
    display_lines = []
    if resolved.get("mode"):
        display_lines.append(f"Mode: {resolved['mode']}")
    if resolved.get("route"):
        display_lines.append(f"Route: {resolved['route']}")
    display_lines.append("")
    display_lines.append(result_text)
    citations = resolved.get("citations") or []
    if citations:
        display_lines.append("")
        display_lines.append("Citations:")
        for citation in citations[:5]:
            display_lines.append(
                f"- {citation.get('title')} [{citation.get('source')}] "
                f"(score: {citation.get('score')}, {citation.get('retrieval_method')})"
            )
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
async def tools(current_user: AuthUser = Depends(require_min_role("super_admin"))):
    return {"tools": list_tools()}


@app.get("/knowledge")
async def knowledge(current_user: AuthUser = Depends(require_min_role("admin"))):
    initialize_db()
    from app.services.knowledge_base import connect

    with connect() as conn:
        documents = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()["count"]
        chunks = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
        embeddings = conn.execute(
            "SELECT COUNT(*) AS count FROM chunks WHERE embedding_json IS NOT NULL AND embedding_json != ''"
        ).fetchone()["count"]
    return {"documents": documents, "chunks": chunks, "embedded_chunks": embeddings}


@app.get("/knowledge/search")
async def knowledge_search(
    q: str = "",
    limit: int = Query(4, ge=1, le=10),
    current_user: AuthUser = Depends(require_min_role("admin")),
):
    initialize_db()
    hits = search_knowledge(q, limit=limit) if q.strip() else []
    top_score = hits[0]["score"] if hits else None
    return {
        "query": q,
        "limit": limit,
        "hit_count": len(hits),
        "confidence": confidence_label(top_score),
        "top_score": top_score,
        "embedding_model": EMBEDDING_MODEL,
        "formatted": format_hits(hits) if hits else "",
        "hits": hits,
    }


@app.get("/ingestion/state")
async def ingestion_state(current_user: AuthUser = Depends(require_min_role("admin"))):
    return load_state()


@app.get("/dashboard")
async def dashboard(current_user: AuthUser = Depends(require_min_role("admin"))):
    initialize_db()
    from app.services.knowledge_base import connect
    from app.tools.scrape import inspect_scrape_config
    from app.services.evaluation import load_eval_set
    config_path = Path(__file__).resolve().parents[1] / "config" / "scrape_sources.json"
    import json

    config_stats = inspect_scrape_config()
    with connect() as conn:
        documents = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()["count"]
        chunks = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
        embedded_chunks = conn.execute(
            "SELECT COUNT(*) AS count FROM chunks WHERE embedding_json IS NOT NULL AND embedding_json != ''"
        ).fetchone()["count"]
        by_type = conn.execute(
            """
            SELECT doc_type, COUNT(*) AS count
            FROM documents
            GROUP BY doc_type
            ORDER BY count DESC, doc_type ASC
            """
        ).fetchall()
        recent_docs = conn.execute(
            """
            SELECT title, source, doc_type, url, updated_at
            FROM documents
            ORDER BY updated_at DESC, id DESC
            LIMIT 5
            """
        ).fetchall()
    config_payload = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {"sites": []}
    pending_config_sources = sum(
        1
        for site in config_payload.get("sites", [])
        if isinstance(site, dict) and bool(site.get("is_pending"))
    )
    from app.services.knowledge_base import _faiss_manifest  # local import to avoid clutter

    scheduler = scheduler_status()
    return {
        "documents": documents,
        "chunks": chunks,
        "embedded_chunks": embedded_chunks,
        "embedding_model": EMBEDDING_MODEL,
        "config_sources": len(config_payload.get("sites", [])),
        "pending_config_sources": pending_config_sources,
        "evaluation_cases": len(load_eval_set()),
        "config_stats": config_stats,
        "documents_by_type": [dict(row) for row in by_type],
        "recent_documents": [dict(row) for row in recent_docs],
        "faiss_index_ready": Path(__file__).resolve().parents[1].joinpath("data", "knowledge.faiss.index").exists(),
        "faiss_dirty": FAISS_DIRTY_PATH.exists(),
        "faiss_manifest": _faiss_manifest(),
        "ingestion_state": load_state(),
        "scheduler": scheduler,
        "share_scheduler": scheduler.get("share", {}),
        "task_scheduler": scheduler.get("task", {}),
        "scrape_scheduler": scheduler.get("scrape", {}),
    }


@app.post("/admin/config-clean")
async def config_clean(current_user: AuthUser = Depends(require_min_role("super_admin"))):
    from app.tools.scrape import clean_scrape_config

    return {"status": "ok", "result": clean_scrape_config()}


@app.get("/evaluation")
async def evaluation(current_user: AuthUser = Depends(require_min_role("admin"))):
    return {"cases": load_eval_set()}


@app.post("/evaluation/run")
async def evaluation_run(current_user: AuthUser = Depends(require_min_role("admin"))):
    return run_evaluation_suite()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/dummy/email")
async def dummy_email(req: DummyDeliveryRequest, current_user: AuthUser = Depends(require_min_role("super_admin"))):
    return {
        **queue_dummy_delivery(
            channel="email",
            recipient=req.recipient,
            subject=req.subject,
            scheduled_for=req.scheduled_for,
            message=req.message,
        ),
        "note": "This is a mock endpoint for testing the email flow.",
    }


@app.post("/dummy/whatsapp")
async def dummy_whatsapp(req: DummyDeliveryRequest, current_user: AuthUser = Depends(require_min_role("super_admin"))):
    return {
        **queue_dummy_delivery(
            channel="whatsapp",
            recipient=req.recipient,
            scheduled_for=req.scheduled_for,
            message=req.message,
        ),
        "note": "This is a mock endpoint for testing the WhatsApp flow.",
    }


@app.post("/dummy/meeting")
async def dummy_meeting(req: DummyDeliveryRequest, current_user: AuthUser = Depends(require_min_role("super_admin"))):
    return {
        **queue_dummy_delivery(
            channel="meeting",
            recipient=req.recipient,
            subject=req.subject,
            scheduled_for=req.scheduled_for,
            message=req.message,
        ),
        "note": "This is a mock endpoint for testing the meeting scheduling flow.",
    }


@app.post("/dummy/task")
async def dummy_task(req: DummyDeliveryRequest, current_user: AuthUser = Depends(require_min_role("super_admin"))):
    return {
        **queue_dummy_delivery(
            channel="task",
            recipient=req.recipient,
            subject=req.subject,
            scheduled_for=req.scheduled_for,
            message=req.message,
        ),
        "note": "This is a mock endpoint for testing the task scheduling flow.",
    }


@app.get("/sources")
async def sources(current_user: AuthUser = Depends(require_min_role("super_admin"))):
    config_path = Path(__file__).resolve().parents[1] / "config" / "scrape_sources.json"
    import json

    return json.loads(config_path.read_text(encoding="utf-8"))


@app.post("/admin/scrape-now")
async def scrape_now(req: ScrapeNowRequest | None = None, current_user: AuthUser = Depends(require_min_role("super_admin"))):
    from app.tools.scrape import refresh_configured_sources
    job_id = new_request_id()
    scope = (req.scope if req else "pending")
    source = (req.source if req else "dashboard")
    actor = (req.actor if req else "admin")

    def worker():
        try:
            trace_event("manual_scrape_job_started", request_id=job_id, scope=scope, source=source, actor=actor)
            results = refresh_configured_sources(scope=scope)
            trace_event(
                "manual_scrape_job_completed",
                request_id=job_id,
                source_count=len(results),
                scope=scope,
                source=source,
                actor=actor,
            )
        except Exception as exc:
            trace_event(
                "manual_scrape_job_failed",
                request_id=job_id,
                error=f"{exc.__class__.__name__}: {exc}",
                scope=scope,
                source=source,
                actor=actor,
            )

    threading.Thread(target=worker, name=f"manual-scrape-{job_id[:8]}", daemon=True).start()
    return {"status": "started", "job_id": job_id, "scope": scope, "source": source, "actor": actor, "state": load_state()}


@app.post("/admin/scrape-schedule")
async def scrape_schedule(req: ScrapeScheduleRequest, current_user: AuthUser = Depends(require_min_role("super_admin"))):
    scheduled = schedule_scrape_job(
        scheduled_for=req.scheduled_for,
        scope=req.scope,
        query=req.query,
        source=req.source,
        actor=req.actor,
    )
    return {"status": "scheduled", **scheduled}


@app.get("/admin/share-queue")
async def admin_share_queue(current_user: AuthUser = Depends(require_min_role("super_admin"))):
    return scheduler_status().get("share", {})


@app.get("/admin/scheduler")
async def admin_scheduler(current_user: AuthUser = Depends(require_min_role("super_admin"))):
    return scheduler_status()


@app.get("/", response_class=HTMLResponse)
async def index():
    return _render_frontend()


@app.get("/app/home", response_class=HTMLResponse)
async def app_home():
    return _render_frontend()


@app.get("/app/chat", response_class=HTMLResponse)
async def app_chat():
    return _render_frontend()


@app.get("/app/dashboard", response_class=HTMLResponse)
async def app_dashboard():
    return _render_frontend()


@app.get("/app/admin", response_class=HTMLResponse)
async def app_admin():
    return _render_frontend()


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return _render_frontend()


@app.get("/register", response_class=HTMLResponse)
async def register_page():
    return _render_frontend()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
