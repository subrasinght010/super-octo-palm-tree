from __future__ import annotations

import json
import os

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

from app.mcp_registry import available_tools, route_query, run_tool
from app.trace import trace_event

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _format_tool_result(tool_name: str, query: str, request_id: str | None = None) -> str:
    result = run_tool(tool_name, query)
    trace_event(
        "tool_executed",
        request_id=request_id,
        tool_name=tool_name,
        route_source="direct" if tool_name == route_query(query) else "forced",
        query=query,
    )
    return f"Tool used: {tool_name}\n\n{result}"


def _local_agent(query: str, request_id: str | None = None) -> str:
    tool_name = route_query(query)
    trace_event(
        "local_route_selected",
        request_id=request_id,
        query=query,
        tool_name=tool_name,
        decision_source="keyword_router",
    )
    return _format_tool_result(tool_name, query, request_id)


def run_agent(query: str, request_id: str | None = None):
    query = (query or "").strip()
    if not query:
        return "No query provided."

    trace_event("agent_started", request_id=request_id, query=query)

    if route_query(query) == "share_details":
        trace_event(
            "agent_tool_short_circuit",
            request_id=request_id,
            query=query,
            tool_name="share_details",
            decision_source="keyword_router",
        )
        return _format_tool_result("share_details", query, request_id)

    client = _client()
    if client is None:
        trace_event(
            "agent_fallback",
            request_id=request_id,
            query=query,
            reason="missing_openai_client",
        )
        return _local_agent(query, request_id)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI research assistant. "
                "Use the available tools when they help answer the user's question. "
                "Prefer search_docs for local project knowledge, web_search for current information, "
                "summarize_text for compression, and share_details for email, WhatsApp, "
                "or meeting scheduling requests."
            ),
        },
        {"role": "user", "content": query},
    ]

    for _ in range(4):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=available_tools(),
            tool_choice="auto",
        )

        message = response.choices[0].message
        if not message.tool_calls:
            trace_event(
                "llm_answered_without_tool",
                request_id=request_id,
                query=query,
                model=MODEL_NAME,
            )
            return message.content or _local_agent(query, request_id)

        chosen_tools = [tool_call.function.name for tool_call in message.tool_calls]
        trace_event(
            "llm_selected_tools",
            request_id=request_id,
            query=query,
            model=MODEL_NAME,
            chosen_tools=chosen_tools,
            tool_count=len(chosen_tools),
            available_tools=[tool["function"]["name"] for tool in available_tools()],
        )

        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ],
        }
        messages.append(assistant_message)

        for tool_call in message.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_query = arguments.get("query", query)
            tool_result = run_tool(tool_call.function.name, tool_query)
            trace_event(
                "tool_executed",
                request_id=request_id,
                query=query,
                tool_name=tool_call.function.name,
                tool_query=tool_query,
                decision_source="llm_tool_call",
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    trace_event("agent_fallback", request_id=request_id, query=query, reason="tool_loop_exhausted")
    return _local_agent(query, request_id)
