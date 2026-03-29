from __future__ import annotations

import html
import re
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from app.services.ingestion_state import load_state, record_web_search
from app.services.knowledge_base import upsert_document
from app.tools.scrape import remember_sources_from_web_results
from app.trace import trace_event


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value))


def web_search(query: str):
    query = (query or "").strip()
    if not query:
        return "No query provided for web search."

    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    request = Request(
        search_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Codex Research Assistant)",
        },
    )

    try:
        with urlopen(request, timeout=10) as response:
            page = response.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        return (
            f"Live web search failed for '{query}'. "
            f"Fallback note: {exc.__class__.__name__}: {exc}"
        )

    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        re.S,
    )

    results = []
    for match in pattern.finditer(page):
        results.append(
            {
                "title": _strip_tags(match.group("title")),
                "url": html.unescape(match.group("url")),
                "snippet": _strip_tags(match.group("snippet")),
            }
        )

    if not results:
        return f"No live web results could be parsed for: {query}"

    added_urls = remember_sources_from_web_results(query, results)

    lines = [f"Live web results for: {query}"]
    lines.append("Reason: No strong match was found in the knowledge base, so live web search was used.")
    for index, result in enumerate(results[:3], start=1):
        lines.append(
            f"{index}. {result['title']}\n"
            f"   {result['url']}\n"
            f"   {result['snippet']}"
        )
    formatted = "\n".join(lines)

    upsert_document(
        source="web search",
        title=f"Web search: {query}",
        content=formatted,
        url=search_url,
        doc_type="web",
        metadata={"query": query, "result_count": len(results)},
    )
    trace_event(
        "web_search_saved",
        query=query,
        result_count=len(results),
        added_config_urls=added_urls,
        url=search_url,
    )
    record_web_search(query=query, result_count=len(results), added_urls=added_urls)

    if added_urls:
        formatted += "\n\nAdded to scrape config:\n" + "\n".join(f"- {url}" for url in added_urls)
    else:
        formatted += "\n\nAdded to scrape config: none"
    state = load_state()
    if state.get("refresh_pending"):
        formatted += "\nPending scrape refresh: yes"
    formatted += "\nSaved to knowledge base: yes"
    return formatted
