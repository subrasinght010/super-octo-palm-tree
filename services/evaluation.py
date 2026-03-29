from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict, List

from app.agent import run_agent as run_research_agent
from app.langgraph import build_graph
from app.multiagent import run_agent as run_multiagent

EVAL_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_set.json"


def load_eval_set() -> List[Dict[str, Any]]:
    if not EVAL_PATH.exists():
        return []
    try:
        data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _run_case(query: str, mode: str) -> str:
    mode = (mode or "agent").lower()
    if mode == "graph":
        result = build_graph().invoke({"query": query, "request_id": "eval"})
        return result.get("result", "") if isinstance(result, dict) else str(result)
    if mode == "multi":
        return run_multiagent(query, request_id="eval")
    return run_research_agent(query, request_id="eval")


def run_evaluation_suite() -> Dict[str, Any]:
    cases = load_eval_set()
    results = []
    passed = 0

    for case in cases:
        query = str(case.get("query", "")).strip()
        mode = str(case.get("mode", "agent")).strip() or "agent"
        if case.get("fresh"):
            query = f"{query} {uuid4().hex[:8]}"
        output = _run_case(query, mode)
        must_contain = case.get("must_contain", [])
        if isinstance(must_contain, str):
            must_contain = [must_contain]
        missing = [item for item in must_contain if item not in output]
        ok = not missing
        passed += int(ok)
        results.append(
            {
                "name": case.get("name", query[:40] or "case"),
                "query": query,
                "mode": mode,
                "passed": ok,
                "missing": missing,
                "output": output[:1200],
            }
        )

    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
