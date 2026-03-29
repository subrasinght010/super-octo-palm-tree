from __future__ import annotations

import re

from app.mcp_registry import route_query, run_tool
from app.trace import trace_event


def _format_tool_result(tool_name: str, query: str, request_id: str | None = None) -> str:
    result = run_tool(tool_name, query)
    trace_event(
        "tool_executed",
        request_id=request_id,
        tool_name=tool_name,
        decision_source="keyword_router",
        query=query,
    )
    return f"Tool used: {tool_name}\n\n{result}"


def _has_time_hint(query: str) -> bool:
    lowered = (query or "").lower()
    return (
        any(keyword in lowered for keyword in ["later", "tomorrow", "today", "tonight", "after"])
        or re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", lowered) is not None
        or re.search(r"\bat\b", lowered) is not None
    )


def run_agent(query: str, request_id: str | None = None):
    query = (query or "").strip()
    if not query:
        return "No query provided."

    trace_event("agent_started", request_id=request_id, query=query)

    routed_tool = route_query(query)
    if routed_tool == "scrape_configured_sites" and _has_time_hint(query):
        trace_event(
            "agent_scrape_schedule_redirected",
            request_id=request_id,
            query=query,
            decision_source="keyword_router",
        )
        return (
            "Scrape scheduling is manual from the admin panel.\n"
            "Choose the scope in the dashboard: Pending URLs only or All config URLs.\n"
            "Then use Run now to scrape immediately or Run later to schedule a one-off job."
        )
    if routed_tool in {"share_details", "schedule_task", "scrape_configured_sites"}:
        trace_event(
            "agent_tool_short_circuit",
            request_id=request_id,
            query=query,
            tool_name=routed_tool,
            decision_source="keyword_router",
        )
        return _format_tool_result(routed_tool, query, request_id)

    from app.tools.rag import search_docs
    from app.tools.web import web_search
    from app.tools.utils import summarize_text

    rag_answer = search_docs(query)
    if not rag_answer.startswith("No strong match found"):
        trace_event(
            "agent_answer_from_db",
            request_id=request_id,
            query=query,
            route="search_docs",
        )
        return rag_answer

    trace_event(
        "agent_db_miss",
        request_id=request_id,
        query=query,
        route="web_search",
    )
    fallback_reason = "No strong match was found in the database; live web search was used."
    web_answer = web_search(query)
    if web_answer.startswith("No live web results could be parsed") or web_answer.startswith("Live web search failed"):
        trace_event(
            "agent_web_failed",
            request_id=request_id,
            query=query,
        )
        return web_answer

    # Give the freshly fetched web material one small generation step so the user gets an answer,
    # and the content stays in the knowledge base for future questions.
    final_answer = summarize_text(
        f"Question: {query}\n\nRelevant web evidence:\n{web_answer}"
    )
    trace_event(
        "agent_web_fallback_completed",
        request_id=request_id,
        query=query,
        route="web_search",
    )
    return f"Reason: {fallback_reason}\n\nWeb fallback answer:\n{final_answer}\n\nEvidence:\n{web_answer}"
