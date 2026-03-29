from __future__ import annotations

from typing import TypedDict

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - optional dependency
    END = "__end__"
    StateGraph = None

from app.mcp_registry import route_query, run_tool
from app.trace import trace_event
from app.tools.schedule import schedule_task
from app.tools.scrape import scrape_configured_sites


class AgentState(TypedDict, total=False):
    query: str
    route: str
    result: str
    request_id: str
    intent_choice: str


def decide_tool(state: AgentState):
    decision_source = "keyword_router"
    choice = state.get("intent_choice")
    if choice == "summary_only":
        route = "summarize_text"
        decision_source = "ui_choice"
    elif choice == "share_full":
        route = "share_details"
        decision_source = "ui_choice"
    elif choice == "summary_share":
        route = "clarify_summary_share"
        decision_source = "ui_choice"
    else:
        route = route_query(state["query"])
    trace_event(
        "graph_route_decided",
        request_id=state.get("request_id"),
        query=state["query"],
        route=route,
        decision_source=decision_source,
    )
    return route


def router_node(state: AgentState):
    return {"route": decide_tool(state)}


def web_node(state: AgentState):
    result = run_tool("web_search", state["query"])
    trace_event(
        "tool_executed",
        request_id=state.get("request_id"),
        query=state["query"],
        tool_name="web_search",
        decision_source="graph_router",
    )
    return {"result": result}


def rag_node(state: AgentState):
    result = run_tool("search_docs", state["query"])
    trace_event(
        "tool_executed",
        request_id=state.get("request_id"),
        query=state["query"],
        tool_name="search_docs",
        decision_source="graph_router",
    )
    return {"result": result}


def summarize_node(state: AgentState):
    result = run_tool("summarize_text", state["query"])
    trace_event(
        "tool_executed",
        request_id=state.get("request_id"),
        query=state["query"],
        tool_name="summarize_text",
        decision_source="graph_router",
    )
    return {"result": result}


def share_node(state: AgentState):
    result = run_tool("share_details", state["query"])
    trace_event(
        "tool_executed",
        request_id=state.get("request_id"),
        query=state["query"],
        tool_name="share_details",
        decision_source="graph_router",
    )
    return {"result": result}


def schedule_node(state: AgentState):
    result = schedule_task(state["query"])
    trace_event(
        "tool_executed",
        request_id=state.get("request_id"),
        query=state["query"],
        tool_name="schedule_task",
        decision_source="graph_router",
    )
    return {"result": result}


def scrape_node(state: AgentState):
    result = scrape_configured_sites(state["query"])
    trace_event(
        "tool_executed",
        request_id=state.get("request_id"),
        query=state["query"],
        tool_name="scrape_configured_sites",
        decision_source="graph_router",
    )
    return {"result": result}


def clarify_summary_share_node(state: AgentState):
    result = (
        "I found both summary and share intent in your request.\n"
        "Please choose one:\n"
        "1. Summarize in chat only.\n"
        "2. Share the full conversation without summarizing.\n"
        "3. Summarize first and then share the summary.\n"
        "Reply with the option you want."
    )
    trace_event(
        "tool_executed",
        request_id=state.get("request_id"),
        query=state["query"],
        tool_name="clarify_summary_share",
        decision_source="graph_router",
    )
    return {"result": result}


class _FallbackGraph:
    def invoke(self, state: AgentState):
        trace_event(
            "graph_invoked",
            request_id=state.get("request_id"),
            query=state["query"],
            mode="fallback",
        )
        routed_state = {**state, **router_node(state)}
        route = routed_state["route"]

        if route == "web_search":
            routed_state.update(web_node(routed_state))
        elif route == "search_docs":
            routed_state.update(rag_node(routed_state))
        elif route == "share_details":
            routed_state.update(share_node(routed_state))
        elif route == "schedule_task":
            routed_state.update(schedule_node(routed_state))
        elif route == "scrape_configured_sites":
            routed_state.update(scrape_node(routed_state))
        elif route == "clarify_summary_share":
            routed_state.update(clarify_summary_share_node(routed_state))
        else:
            routed_state.update(rag_node(routed_state))

        return routed_state


def build_graph():
    if StateGraph is None:
        return _FallbackGraph()

    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("web", web_node)
    graph.add_node("rag", rag_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("share", share_node)
    graph.add_node("schedule", schedule_node)
    graph.add_node("scrape", scrape_node)
    graph.add_node("clarify", clarify_summary_share_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        decide_tool,
        {
            "web_search": "web",
            "search_docs": "rag",
            "summarize_text": "summarize",
            "share_details": "share",
            "schedule_task": "schedule",
            "scrape_configured_sites": "scrape",
            "search_docs": "rag",
            "clarify_summary_share": "clarify",
        },
    )

    graph.add_edge("web", END)
    graph.add_edge("rag", END)
    graph.add_edge("summarize", END)
    graph.add_edge("share", END)
    graph.add_edge("schedule", END)
    graph.add_edge("scrape", END)
    graph.add_edge("clarify", END)

    return graph.compile()
