from __future__ import annotations

import json
import re
from functools import lru_cache
from html import unescape
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from urllib.error import URLError
from urllib.request import Request, urlopen

from app.services.ingestion_state import load_state, record_scrape_run
from app.services.knowledge_base import upsert_document
from app.trace import trace_event

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "scrape_sources.json"

DEFAULT_SOURCES = [
    {
        "name": "TechCrunch AI",
        "url": "https://techcrunch.com/category/artificial-intelligence/",
        "notes": "AI technology news and startups",
        "is_pending": False,
    },
    {
        "name": "Reuters AI",
        "url": "https://www.reuters.com/technology/artificial-intelligence/",
        "notes": "Business and policy AI news",
        "is_pending": False,
    },
]

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "with",
    "from",
    "by",
    "about",
    "data",
    "scrape",
    "scraping",
    "extract",
    "latest",
    "new",
    "show",
    "find",
    "me",
    "please",
}


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def _strip_noise_blocks(html: str) -> str:
    html = re.sub(r"<script\b.*?</script>", " ", html, flags=re.I | re.S)
    html = re.sub(r"<style\b.*?</style>", " ", html, flags=re.I | re.S)
    html = re.sub(r"<noscript\b.*?</noscript>", " ", html, flags=re.I | re.S)
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.S)
    return html


def _keywords(query: str):
    return [token for token in re.findall(r"[a-z0-9]+", query.lower()) if token not in STOPWORDS]


@lru_cache(maxsize=1)
def _load_sources():
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            sites = data.get("sites") if isinstance(data, dict) else data
            if isinstance(sites, list) and sites:
                return sites
        except Exception:
            pass
    return DEFAULT_SOURCES


def _filter_sources(query: str):
    matches = _matching_sources(query)
    return matches or _load_sources()


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "pending"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _pending_sources():
    sources = _load_sources()
    return [source for source in sources if isinstance(source, dict) and _coerce_bool(source.get("is_pending"), default=False)]


def sources_for_scope(scope: str = "all", query: str | None = None):
    scope = (scope or "all").strip().lower()
    if scope in {"pending", "pending_only", "pending-urls"}:
        return _pending_sources()
    if scope in {"query", "matched"} and query:
        return _filter_sources(query)
    return _load_sources()


def _matching_sources(query: str):
    sources = _load_sources()
    terms = _keywords(query)
    return [
        source
        for source in sources
        if any(term in f"{source.get('name', '')} {source.get('url', '')} {source.get('notes', '')}".lower() for term in terms)
    ]


def configured_sources_for_query(query: str):
    return _matching_sources(query)


def _normalize_discovered_url(url: str | None) -> str | None:
    if not url:
        return None

    cleaned = unquote(url.strip())
    parsed = urlparse(cleaned)
    if parsed.scheme in {"http", "https"}:
        return cleaned

    if parsed.netloc.endswith("duckduckgo.com") or "duckduckgo.com" in parsed.path:
        query = parse_qs(parsed.query)
        if "uddg" in query and query["uddg"]:
            target = unquote(query["uddg"][0])
            if target.startswith(("http://", "https://")):
                return target

    return cleaned if cleaned.startswith(("http://", "https://")) else None


def _read_config_payload():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"sites": _load_sources()}
    return {"sites": _load_sources()}


def _write_config_sources(sites: list[dict]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps({"sites": sites}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _load_sources.cache_clear()


def remember_sources_from_web_results(query: str, results: list[dict], limit: int = 3) -> list[str]:
    if not results:
        return []

    payload = _read_config_payload()
    existing_sites = payload.get("sites") if isinstance(payload, dict) else payload
    if not isinstance(existing_sites, list):
        existing_sites = []

    existing_urls = {
        str(site.get("url", "")).strip()
        for site in existing_sites
        if isinstance(site, dict) and site.get("url")
    }

    added_urls: list[str] = []
    for result in results[:limit]:
        url = _normalize_discovered_url(result.get("url"))
        if not url or url in existing_urls:
            continue

        title = (result.get("title") or "Discovered source").strip()
        existing_sites.append(
            {
                "name": title[:120],
                "url": url,
                "notes": f"Discovered from web search for: {query}",
                "is_pending": True,
            }
        )
        existing_urls.add(url)
        added_urls.append(url)

    if added_urls:
        _write_config_sources(existing_sites)
        trace_event(
            "scrape_config_updated",
            query=query,
            added_count=len(added_urls),
            added_urls=added_urls,
        )

    return added_urls


def clean_scrape_config() -> dict:
    payload = _read_config_payload()
    sites = payload.get("sites") if isinstance(payload, dict) else payload
    if not isinstance(sites, list):
        sites = []

    cleaned_sites = []
    seen_urls = set()
    removed = 0

    for site in sites:
        if not isinstance(site, dict):
            removed += 1
            continue

        name = str(site.get("name", "")).strip()
        url = _normalize_discovered_url(site.get("url"))
        notes = str(site.get("notes", "")).strip()
        is_pending = _coerce_bool(site.get("is_pending"), default=False)
        if not name or not url:
            removed += 1
            continue
        if url in seen_urls:
            removed += 1
            continue

        seen_urls.add(url)
        cleaned_sites.append(
            {
                "name": name[:120],
                "url": url,
                "notes": notes[:240] if notes else "Curated scrape source",
                "is_pending": is_pending,
            }
        )

    if cleaned_sites != sites:
        _write_config_sources(cleaned_sites)
        trace_event(
            "scrape_config_cleaned",
            removed_count=removed,
            kept_count=len(cleaned_sites),
        )

    return {"removed_count": removed, "kept_count": len(cleaned_sites)}


def inspect_scrape_config() -> dict:
    payload = _read_config_payload()
    sites = payload.get("sites") if isinstance(payload, dict) else payload
    if not isinstance(sites, list):
        sites = []

    kept = 0
    pending = 0
    invalid = 0
    seen_urls = set()

    for site in sites:
        if not isinstance(site, dict):
            invalid += 1
            continue

        name = str(site.get("name", "")).strip()
        url = _normalize_discovered_url(site.get("url"))
        if not name or not url:
            invalid += 1
            continue
        if url in seen_urls:
            invalid += 1
            continue

        seen_urls.add(url)
        kept += 1
        if _coerce_bool(site.get("is_pending"), default=False):
            pending += 1

    return {
        "total_count": len(sites),
        "kept_count": kept,
        "removed_count": invalid,
        "pending_count": pending,
    }


def _set_config_pending_flags(urls: list[str], pending: bool) -> None:
    payload = _read_config_payload()
    sites = payload.get("sites") if isinstance(payload, dict) else payload
    if not isinstance(sites, list):
        return

    normalized_urls = {str(url).strip() for url in urls if str(url).strip()}
    if not normalized_urls:
        return

    updated = False
    for site in sites:
        if not isinstance(site, dict):
            continue
        url = str(site.get("url", "")).strip()
        if url in normalized_urls:
            site["is_pending"] = pending
            updated = True

    if updated:
        _write_config_sources(sites)
        trace_event(
            "scrape_config_pending_flags_updated",
            url_count=len(normalized_urls),
            pending=pending,
        )


def _fetch_page(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (Codex Research Assistant)"})
    with urlopen(request, timeout=12) as response:
        return response.read().decode("utf-8", errors="ignore")


def _extract_page(html: str):
    html = _strip_noise_blocks(html)
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    title = _clean_text(title_match.group(1)) if title_match else "Untitled page"
    body_match = re.search(r"<main\b[^>]*>(.*?)</main>", html, re.I | re.S)
    if not body_match:
        body_match = re.search(r"<article\b[^>]*>(.*?)</article>", html, re.I | re.S)
    if body_match:
        text = _clean_text(body_match.group(1))
    else:
        text = _clean_text(html)
    return title, text


def _summarize_page(text: str, keywords: list[str]) -> str:
    lowered = text.lower()
    focus = None
    for keyword in keywords:
        position = lowered.find(keyword)
        if position != -1:
            focus = text[max(0, position - 140) : position + 320]
            break
    excerpt = focus or text[:460]
    return excerpt


def _scrape_sources(sources, query_label: str, scope: str = "all"):
    keywords = _keywords(query_label)
    results = []

    for source in sources[:4]:
        url = source.get("url")
        name = source.get("name", url)
        try:
            html = _fetch_page(url)
            title, text = _extract_page(html)
            summary = _summarize_page(text, keywords)
            content = f"{title}\n{summary}\n\n{text[:6000]}"
            upsert_document(
                source=name,
                title=title,
                content=content,
                url=url,
                doc_type="scrape",
                metadata={"query_label": query_label, "source_name": name},
            )
            results.append(
                {
                    "name": name,
                    "url": url,
                    "summary": f"{title}\n{summary}",
                    "stored": True,
                }
            )
            _set_config_pending_flags([url], False)
        except URLError as exc:
            results.append(
                {
                    "name": name,
                    "url": url,
                    "summary": f"Failed to fetch page: {exc.reason if hasattr(exc, 'reason') else exc}",
                    "stored": False,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": name,
                    "url": url,
                    "summary": f"Failed to scrape page: {exc.__class__.__name__}: {exc}",
                    "stored": False,
                }
            )

    trace_event(
        "scrape_completed",
        query=query_label,
        source_count=len(results),
        sources=[item["url"] for item in results],
        scope=scope,
    )
    return results


def refresh_configured_sources(scope: str = "all", query: str | None = None):
    clean_scrape_config()
    sources = sources_for_scope(scope=scope, query=query)
    scope_label = (scope or "all").strip().lower()
    if scope_label in {"pending", "pending_only", "pending-urls"}:
        query_label = "pending refresh"
    elif query:
        query_label = query
    else:
        query_label = "background refresh"
    results = _scrape_sources(sources, query_label=query_label, scope=scope_label)
    record_scrape_run(
        query=query_label,
        source_count=len(results),
        scope=scope_label,
        urls=[item.get("url") for item in results],
    )
    return results


def scrape_configured_sites(query: str, scope: str = "all"):
    query = (query or "").strip()
    if not query:
        return "No scrape query provided."

    clean_scrape_config()
    sources = sources_for_scope(scope=scope, query=query)
    results = _scrape_sources(sources, query_label=query, scope=scope)
    record_scrape_run(
        query=query,
        source_count=len(results),
        scope=scope,
        urls=[item.get("url") for item in results],
    )

    lines = ["Scrape results:"]
    for index, item in enumerate(results, start=1):
        lines.append(
            f"{index}. {item['name']}\n"
            f"   {item['url']}\n"
            f"   {item['summary']}"
        )
    return "\n".join(lines)
