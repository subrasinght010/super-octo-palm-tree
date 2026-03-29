from __future__ import annotations

from typing import TypedDict

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - optional dependency
    END = "__end__"
    StateGraph = None

from app.mcp_registry import route_query, run_tool
from app.trace import trace_event


class AgentState(TypedDict, total=False):
    query: str
    route: str
    result: str
    request_id: str


def decide_tool(state: AgentState):
    route = route_query(state["query"])
    trace_event(
        "graph_route_decided",
        request_id=state.get("request_id"),
        query=state["query"],
        route=route,
        decision_source="keyword_router",
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
        else:
            routed_state.update(summarize_node(routed_state))

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

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        decide_tool,
        {
            "web_search": "web",
            "search_docs": "rag",
            "summarize_text": "summarize",
            "share_details": "share",
        },
    )

    graph.add_edge("web", END)
    graph.add_edge("rag", END)
    graph.add_edge("summarize", END)
    graph.add_edge("share", END)

    return graph.compile()
