from __future__ import annotations

import html
import re
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


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

    lines = [f"Live web results for: {query}"]
    for index, result in enumerate(results[:3], start=1):
        lines.append(
            f"{index}. {result['title']}\n"
            f"   {result['url']}\n"
            f"   {result['snippet']}"
        )
    return "\n".join(lines)

