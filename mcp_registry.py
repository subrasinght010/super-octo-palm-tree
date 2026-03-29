from __future__ import annotations

from app.tools.rag import search_docs
from app.tools.share import share_details
from app.tools.schedule import schedule_task
from app.tools.scrape import scrape_configured_sites
from app.tools.utils import summarize_text
from app.tools.web import web_search


TOOLS = [
    {
        "name": "search_docs",
        "description": "Search the local research corpus with RAG-style retrieval.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Question or topic to search for in the knowledge base.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "handler": search_docs,
    },
    {
        "name": "web_search",
        "description": "Search the live web for recent or current information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for the web.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "handler": web_search,
    },
    {
        "name": "summarize_text",
        "description": "Summarize long text or compress a topic into a short brief.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text or topic to summarize.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "handler": summarize_text,
    },
    {
        "name": "share_details",
        "description": "Prepare a sharing plan for email or WhatsApp requests.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language email or WhatsApp share request.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "handler": share_details,
    },
    {
        "name": "schedule_task",
        "description": "Prepare a task, reminder, or meeting schedule plan from the query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language task scheduling request.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "handler": schedule_task,
    },
    {
        "name": "scrape_configured_sites",
        "description": "Scrape configured websites from config and return useful excerpts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for while scraping configured sources.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "handler": scrape_configured_sites,
    },
]


def available_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in TOOLS
    ]


def list_tools():
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        for tool in TOOLS
    ]


def run_tool(name: str, query: str) -> str:
    for tool in TOOLS:
        if tool["name"] == name:
            return tool["handler"](query)
    raise KeyError(f"Unknown tool: {name}")


def route_query(query: str) -> str:
    lowered = query.lower()
    if any(keyword in lowered for keyword in ["scrape", "crawl", "extract from site", "extract from websites", "configured sites", "given web", "web given in config"]):
        return "scrape_configured_sites"
    if any(keyword in lowered for keyword in ["meeting", "schedule", "calendar", "invite", "remind", "reminder", "task", "todo", "appointment"]):
        return "schedule_task"
    if (
        any(keyword in lowered for keyword in ["summary", "summarize", "summarise", "brief", "short version", "tldr", "tl;dr"])
        and any(keyword in lowered for keyword in ["email", "mail", "whatsapp", "whats app", "share", "send", "@"])
    ):
        return "clarify_summary_share"
    if any(keyword in lowered for keyword in ["email", "mail", "whatsapp", "whats app"]) or "@" in lowered:
        return "share_details"
    if any(keyword in lowered for keyword in ["summary", "summarize", "summarise", "brief", "short version", "tldr", "tl;dr"]):
        return "summarize_text"
    if any(keyword in lowered for keyword in ["latest", "current", "today", "news", "recent", "breaking"]):
        return "web_search"
    if any(keyword in lowered for keyword in ["rag", "pdf", "document", "docs", "paper", "note", "corpus", "research"]):
        return "search_docs"
    return "search_docs"
