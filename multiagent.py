from __future__ import annotations

import re
from typing import Dict, List

from app.agent import run_agent as _run_research_agent_single
from app.tools.share import parse_share_request, share_details
from app.tools.schedule import parse_schedule_request, schedule_task
from app.tools.scrape import scrape_configured_sites
from app.tools.utils import summarize_text
from app.trace import trace_event


def _has_keywords(query: str, keywords: list[str]) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in keywords)


def _wants_summary(query: str) -> bool:
    return _has_keywords(
        query,
        [
            "summary",
            "summarize",
            "summarise",
            "brief",
            "short version",
            "tldr",
            "tl;dr",
        ],
    )


def _wants_share(query: str) -> bool:
    return _has_keywords(query, ["email", "mail", "whatsapp", "whats app", "share", "send", "@"])


def _summary_share_clarification(query: str) -> str | None:
    if _wants_summary(query) and _wants_share(query):
        return (
            "I found both summary and share intent in your request.\n"
            "Please choose one:\n"
            "1. Summarize in chat only.\n"
            "2. Share the full conversation without summarizing.\n"
            "3. Summarize first and then share the summary.\n"
            "Reply with the option you want."
        )
    return None


def _build_plan(query: str) -> List[str]:
    plan: List[str] = []

    if _has_keywords(query, ["scrape", "crawl", "extract from site", "extract from websites", "configured sites", "given web", "web given in config"]):
        plan.append("scrape")

    if _has_keywords(query, ["latest", "current", "today", "news", "recent", "breaking", "rag", "pdf", "document", "docs", "paper", "note", "corpus", "research"]):
        plan.append("research")

    if _has_keywords(query, ["meeting", "schedule", "calendar", "invite", "remind", "reminder", "task", "todo", "appointment"]):
        plan.append("schedule")

    if _wants_summary(query):
        plan.append("summary")

    if _has_keywords(query, ["email", "mail", "whatsapp", "whats app"]):
        plan.append("share")

    if not plan:
        plan.append("research")

    # Ensure sharing happens last when it is part of a compound request.
    if "share" in plan and len(plan) > 1:
        plan = [item for item in plan if item != "share"] + ["share"]

    # Avoid duplicate agent work.
    ordered: List[str] = []
    for item in plan:
        if item not in ordered:
            ordered.append(item)
    return ordered


def _strip_share_noise(query: str) -> str:
    tokens = re.findall(r"[a-z0-9@.\-+]+", query.lower())
    noise = {
        "send",
        "share",
        "email",
        "mail",
        "whatsapp",
        "whats",
        "app",
        "meeting",
        "schedule",
        "calendar",
        "invite",
        "today",
        "tomorrow",
        "tonight",
        "at",
        "am",
        "pm",
        "any",
        "time",
        "whenever",
        "asap",
        "as",
        "soon",
        "possible",
        "by",
        "to",
        "for",
    }
    filtered = [token for token in tokens if token not in noise and not token.isdigit()]
    return " ".join(filtered).strip() or query


def _format_agent_output(agent_name: str, result: str) -> str:
    return f"[{agent_name}]\n{result.strip()}"


def _summarize_if_needed(query: str, outputs: List[Dict[str, str]]) -> str:
    combined = "\n\n".join(f"{item['agent']}:\n{item['result']}" for item in outputs)
    if len(outputs) == 1 or not _wants_summary(query):
        return outputs[0]["result"]

    summary = summarize_text(combined)
    return (
        "Multi-agent summary:\n"
        f"{summary}\n\n"
        "Agent details:\n"
        f"{combined}"
    )


def run_summary_agent(query: str, request_id: str | None = None, context: str | None = None):
    text = (context or query or "").strip()
    if not text:
        trace_event("summary_agent_error", request_id=request_id, query=query, error="No text provided.")
        return "Summary agent needs text to summarize."
    trace_event("summary_agent_started", request_id=request_id, query=query, has_context=bool(context))
    result = summarize_text(text)
    trace_event("summary_agent_completed", request_id=request_id, query=query, output_length=len(result or ""))
    return result


def run_share_agent(
    query: str,
    request_id: str | None = None,
    context: str | None = None,
    *,
    source: str = "chat",
    actor: str = "user",
):
    plan = parse_share_request(query)
    if plan.get("error"):
        trace_event("share_agent_error", request_id=request_id, query=query, error=plan["error"])
        return plan["error"]
    trace_event("share_agent_planned", request_id=request_id, query=query, channel=plan["channel"])
    return share_details(query, context=context, source=source, actor=actor)


def run_schedule_agent(
    query: str,
    request_id: str | None = None,
    context: str | None = None,
    *,
    source: str = "chat",
    actor: str = "user",
):
    plan = parse_schedule_request(query)
    if plan.get("error"):
        trace_event("schedule_agent_error", request_id=request_id, query=query, error=plan["error"])
        return plan["error"]
    trace_event("schedule_agent_planned", request_id=request_id, query=query, kind=plan["kind"])
    return schedule_task(query, source=source, actor=actor)


def run_scrape_agent(query: str, request_id: str | None = None):
    lowered = (query or "").lower()
    if any(keyword in lowered for keyword in ["later", "tomorrow", "today", "tonight", "after"]) or re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", lowered):
        trace_event("scrape_agent_schedule_redirected", request_id=request_id, query=query)
        return (
            "Scrape scheduling is handled manually from the admin panel.\n"
            "Use /admin/scrape-schedule or the dashboard time picker to schedule it, or /admin/scrape-now to run immediately."
        )
    trace_event("scrape_agent_started", request_id=request_id, query=query)
    result = scrape_configured_sites(query)
    trace_event("scrape_agent_completed", request_id=request_id, query=query)
    return result

def run_research_agent(query: str, request_id: str | None = None):
    trace_event("research_agent_started", request_id=request_id, query=query)
    result = _run_research_agent_single(query, request_id=request_id)
    trace_event("research_agent_completed", request_id=request_id, query=query)
    return result


def run_multiagent(
    query: str,
    request_id: str | None = None,
    intent_choice: str | None = None,
    *,
    source: str = "chat",
    actor: str = "user",
):
    query = (query or "").strip()
    if not query:
        return "No query provided."

    if intent_choice == "summary_only":
        trace_event("multiagent_summary_choice", request_id=request_id, query=query, choice=intent_choice)
        return run_summary_agent(query, request_id=request_id)

    clarification = None if intent_choice else _summary_share_clarification(query)
    if clarification:
        trace_event("multiagent_summary_share_clarification", request_id=request_id, query=query)
        return clarification

    trace_event("multiagent_started", request_id=request_id, query=query)
    if intent_choice == "share_full":
        plan = ["share"]
    elif intent_choice == "summary_share":
        plan = ["summary", "share"]
    else:
        plan = _build_plan(query)
    trace_event("multiagent_plan_created", request_id=request_id, query=query, plan=plan)

    outputs: List[Dict[str, str]] = []
    context_parts: List[str] = []
    share_last = "share" in plan and len(plan) > 1
    working_query = _strip_share_noise(query) if share_last else query

    for step in plan:
        if step == "scrape":
            result = run_scrape_agent(working_query, request_id=request_id)
            outputs.append({"agent": "scrape", "result": result})
            context_parts.append(result)
        elif step == "research":
            result = run_research_agent(working_query, request_id=request_id)
            outputs.append({"agent": "research", "result": result})
            context_parts.append(result)
        elif step == "schedule":
            context = "\n\n".join(context_parts) if context_parts else None
            result = run_schedule_agent(query, request_id=request_id, context=context, source=source, actor=actor)
            outputs.append({"agent": "schedule", "result": result})
            context_parts.append(result)
        elif step == "summary":
            context = "\n\n".join(context_parts) if context_parts else None
            result = run_summary_agent(query, request_id=request_id, context=context)
            outputs.append({"agent": "summary", "result": result})
            context_parts.append(result)
        elif step == "share":
            context = "\n\n".join(context_parts) if context_parts else None
            result = run_share_agent(query, request_id=request_id, context=context, source=source, actor=actor)
            outputs.append({"agent": "share", "result": result})
            context_parts.append(result)

    final = _summarize_if_needed(query, outputs)
    if len(outputs) > 1 and not _wants_summary(query):
        final = "\n\n".join(_format_agent_output(item["agent"], item["result"]) for item in outputs)
    trace_event(
        "multiagent_completed",
        request_id=request_id,
        query=query,
        plan=plan,
        output_count=len(outputs),
    )
    return final


run_agent = run_multiagent
