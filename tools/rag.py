from __future__ import annotations

import re

from app.services.knowledge_base import (
    build_context,
    confidence_label,
    format_hits,
    knowledge_miss,
    search_knowledge,
)
from app.tools.utils import summarize_text
from app.trace import trace_event


def search_docs(query: str):
    query = (query or "").strip()
    if not query:
        return "No query provided for document search."

    hits = search_knowledge(query, limit=4)
    if not hits or knowledge_miss(query):
        trace_event("rag_miss", query=query)
        return "No strong match found in the knowledge base."

    context = build_context(hits)
    answer = summarize_text(
        f"Question: {query}\n\nUse the following knowledge base context to answer clearly:\n{context}"
    )
    if not answer:
        answer = "I found relevant knowledge base passages, but could not generate a concise answer."

    top_score = hits[0]["score"] if hits else None
    confidence = confidence_label(top_score)
    retrieval_mode = hits[0].get("retrieval_method", "lexical") if hits else "lexical"

    trace_event(
        "rag_hit",
        query=query,
        top_score=top_score,
        confidence=confidence,
        retrieval_mode=retrieval_mode,
        hit_count=len(hits),
    )

    return (
        f"Answer:\n{answer}\n\n"
        f"Retrieval mode: {retrieval_mode}\n"
        f"Retrieval confidence: {confidence}"
        + (f" (top score: {top_score:.3f})" if top_score is not None else "")
        + "\n\n"
        f"Sources:\n{format_hits(hits)}"
    )


def extract_citations(answer_text: str):
    if not answer_text or "Sources:" not in answer_text:
        return []

    sources_block = answer_text.split("Sources:", 1)[1].strip()
    lines = [line.rstrip() for line in sources_block.splitlines()]
    citations = []
    current = None

    header_pattern = re.compile(r"^\s*(\d+)\.\s+(.*?)\s+\[(.*?)\]\s+\(score:\s*([0-9.]+),\s*(.*?)\)\s*$")
    for line in lines:
        match = header_pattern.match(line)
        if match:
            if current:
                citations.append(current)
            current = {
                "rank": int(match.group(1)),
                "title": match.group(2).strip(),
                "source": match.group(3).strip(),
                "score": float(match.group(4)),
                "retrieval_method": match.group(5).strip(),
                "excerpt": "",
            }
            continue
        if current is not None and line.strip():
            text = line.strip()
            if text.startswith("Title:") or text.startswith("Source:") or text.startswith("URL:") or text.startswith("Content:"):
                continue
            if current["excerpt"]:
                current["excerpt"] += " " + text
            else:
                current["excerpt"] = text

    if current:
        citations.append(current)

    return citations
