from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

try:
    import faiss
except ImportError:  # pragma: no cover - optional dependency
    faiss = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DB_PATH = DATA_DIR / "knowledge.db"
SEED_PATH = DATA_DIR / "corpus.json"
FAISS_INDEX_PATH = DATA_DIR / "knowledge.faiss.index"
FAISS_META_PATH = DATA_DIR / "knowledge.faiss.meta.json"
FAISS_DIRTY_PATH = DATA_DIR / "knowledge.faiss.dirty"
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
LOCAL_EMBEDDING_DIM = max(128, int(os.getenv("RAG_LOCAL_EMBEDDING_DIM", "256")))

DEFAULT_SEED_CORPUS = [
    {
        "id": "seed-rag",
        "title": "RAG basics",
        "source": "project notes",
        "content": "Retrieval augmented generation combines retrieval and generation. The retriever finds relevant passages from a corpus, and the generator uses those passages to answer the user's question with grounding.",
    },
    {
        "id": "seed-tool",
        "title": "Tool calling loop",
        "source": "project notes",
        "content": "Tool calling works best when the model gets a strict schema, the app executes the tool, and the result is passed back into the conversation. Multi step tool use needs a loop instead of a single call.",
    },
    {
        "id": "seed-mcp",
        "title": "MCP style registry",
        "source": "project notes",
        "content": "An MCP style registry keeps tool names, descriptions, JSON schemas, and handlers in one place. The UI, agent, and router can all read from the same registry so the system stays consistent.",
    },
    {
        "id": "seed-fastapi",
        "title": "FastAPI research app",
        "source": "project notes",
        "content": "A FastAPI backend makes it easy to expose chat, tool registry, and health endpoints. Keep long running work off the event loop and return structured JSON so the frontend can render it cleanly.",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embedding_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _embedding_supported() -> bool:
    return True


def _faiss_supported() -> bool:
    return faiss is not None and np is not None


def _mark_faiss_dirty(reason: str = "knowledge base changed") -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIRTY_PATH.write_text(
        json.dumps({"reason": reason, "timestamp_utc": utc_now()}, ensure_ascii=False),
        encoding="utf-8",
    )


def _clear_faiss_dirty() -> None:
    if FAISS_DIRTY_PATH.exists():
        FAISS_DIRTY_PATH.unlink()


def _faiss_is_dirty() -> bool:
    return FAISS_DIRTY_PATH.exists()


def _serialize_embedding(values: List[float] | None) -> str | None:
    if not values:
        return None
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _deserialize_embedding(raw: str | None) -> List[float] | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except Exception:
        return None
    if isinstance(value, list) and value and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    return None


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return dot / (left_norm * right_norm)


def _local_embedding(text: str) -> List[float]:
    vector = [0.0] * LOCAL_EMBEDDING_DIM
    tokens = _tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % LOCAL_EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + (len(token) / 10.0)
        vector[bucket] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _embed_texts(texts: List[str]) -> List[List[float] | None]:
    client = _embedding_client()
    if not texts:
        return []

    if client is not None:
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        except Exception:
            response = None
        else:
            embeddings: List[List[float] | None] = [None for _ in texts]
            for item in response.data:
                embeddings[item.index] = [float(value) for value in item.embedding]
            return embeddings

    return [_local_embedding(text) for text in texts]


def _chunk_text(text: str, *, max_words: int = 120, overlap: int = 30) -> List[str]:
    words = re.findall(r"\S+", (text or "").strip())
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks = []
    start = 0
    step = max(1, max_words - overlap)
    while start < len(words):
        end = min(len(words), start + max_words)
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT,
                doc_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                embedding_json TEXT,
                embedding_model TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(document_id, chunk_index)
            );
            """
        )
        existing_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        if "embedding_json" not in existing_columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN embedding_json TEXT")
        if "embedding_model" not in existing_columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN embedding_model TEXT")


def _normalize_metadata(metadata: Optional[Dict[str, Any]]) -> str:
    if not metadata:
        return "{}"
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


def upsert_document(
    *,
    source: str,
    title: str,
    content: str,
    url: str | None = None,
    doc_type: str = "manual",
    metadata: Optional[Dict[str, Any]] = None,
    generate_embeddings: bool = False,
) -> Dict[str, Any]:
    initialize_db()
    raw_content = (content or "").strip()
    if not raw_content:
        return {"document_id": None, "created": False}

    content_hash = _sha256(f"{source}|{title}|{url or ''}|{doc_type}|{raw_content}")
    now = utc_now()

    with connect() as conn:
        existing = conn.execute(
            "SELECT id FROM documents WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE documents SET updated_at = ? WHERE id = ?",
                (now, existing["id"]),
            )
            _mark_faiss_dirty()
            return {"document_id": existing["id"], "created": False}

        cursor = conn.execute(
            """
            INSERT INTO documents (
                content_hash, source, title, url, doc_type, content, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content_hash,
                source,
                title,
                url,
                doc_type,
                raw_content,
                _normalize_metadata(metadata),
                now,
                now,
            ),
        )
        document_id = cursor.lastrowid

        chunks = _chunk_text(raw_content)
        embeddings = _embed_texts(chunks) if generate_embeddings else [None for _ in chunks]
        for chunk_index, chunk_text in enumerate(chunks):
            embedding = embeddings[chunk_index] if chunk_index < len(embeddings) else None
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks (
                    document_id, chunk_index, chunk_text, token_count, embedding_json, embedding_model, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    chunk_index,
                    chunk_text,
                    len(_tokenize(chunk_text)),
                    _serialize_embedding(embedding),
                    EMBEDDING_MODEL if embedding else None,
                    now,
                ),
            )
        _mark_faiss_dirty()

    return {"document_id": document_id, "created": True}


def _load_seed_corpus() -> List[Dict[str, str]]:
    if SEED_PATH.exists():
        try:
            data = json.loads(SEED_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    return DEFAULT_SEED_CORPUS


def seed_corpus(force: bool = False, generate_embeddings: bool = False) -> int:
    initialize_db()
    loaded = _load_seed_corpus()
    inserted = 0
    for item in loaded:
        result = upsert_document(
            source=item.get("source", "project notes"),
            title=item.get("title", "Untitled"),
            content=item.get("content", ""),
            url=item.get("url"),
            doc_type="seed",
            metadata={"seed_id": item.get("id")},
            generate_embeddings=generate_embeddings,
        )
        if result.get("created"):
            inserted += 1
    return inserted


def _load_chunks() -> List[Dict[str, Any]]:
    initialize_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.chunk_index,
                c.chunk_text,
                c.token_count,
                d.id AS document_id,
                d.source,
                d.title,
                d.url,
                d.doc_type,
                d.content,
                d.metadata_json,
                d.created_at,
                d.updated_at,
                c.embedding_json,
                c.embedding_model
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY d.updated_at DESC, d.id DESC, c.chunk_index ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _fetch_chunks_by_ids(chunk_ids: List[int]) -> List[Dict[str, Any]]:
    if not chunk_ids:
        return []

    initialize_db()
    placeholders = ",".join("?" for _ in chunk_ids)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT
                c.id AS chunk_id,
                c.chunk_index,
                c.chunk_text,
                c.token_count,
                d.id AS document_id,
                d.source,
                d.title,
                d.url,
                d.doc_type,
                d.content,
                d.metadata_json,
                d.created_at,
                d.updated_at,
                c.embedding_json,
                c.embedding_model
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholders})
            """,
            tuple(chunk_ids),
        ).fetchall()
    by_id = {row["chunk_id"]: dict(row) for row in rows}
    return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]


def _faiss_manifest() -> Dict[str, Any]:
    initialize_db()
    with connect() as conn:
        stats = conn.execute(
            """
            SELECT
                COUNT(*) AS chunk_count,
                COALESCE(MAX(d.updated_at), '') AS latest_document_update
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding_json IS NOT NULL AND c.embedding_json != ''
            """
        ).fetchone()
    return {
        "chunk_count": int(stats["chunk_count"] or 0),
        "latest_document_update": stats["latest_document_update"] or "",
        "embedding_model": EMBEDDING_MODEL,
    }


def rebuild_faiss_index(force: bool = False) -> Dict[str, Any]:
    if not _faiss_supported():
        return {"built": False, "reason": "faiss_unavailable"}

    initialize_db()
    if not force and not _faiss_is_dirty() and FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists():
        try:
            meta = json.loads(FAISS_META_PATH.read_text(encoding="utf-8"))
            if meta.get("embedding_model") == EMBEDDING_MODEL and meta.get("chunk_count", 0) > 0:
                return {"built": False, "reason": "already_fresh"}
        except Exception:
            pass

    rows = _load_chunks()
    embedded_rows = []
    embeddings = []
    for row in rows:
        embedding = _deserialize_embedding(row.get("embedding_json"))
        if embedding:
            embedded_rows.append(row)
            embeddings.append(embedding)

    if not embedded_rows:
        return {"built": False, "reason": "no_embeddings"}

    matrix = np.asarray(embeddings, dtype="float32")
    if matrix.ndim != 2 or matrix.size == 0:
        return {"built": False, "reason": "invalid_embedding_matrix"}

    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    meta = {
        **_faiss_manifest(),
        "chunk_ids": [int(row["chunk_id"]) for row in embedded_rows],
        "built_at": utc_now(),
        "index_type": "IndexFlatIP",
    }
    FAISS_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _clear_faiss_dirty()
    return {"built": True, "chunk_count": len(embedded_rows)}


def _load_persistent_faiss_index():
    if not _faiss_supported():
        return None
    if _faiss_is_dirty() or not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
        return None

    try:
        meta = json.loads(FAISS_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None

    if meta.get("embedding_model") != EMBEDDING_MODEL:
        return None

    manifest = _faiss_manifest()
    if meta.get("chunk_count") != manifest["chunk_count"]:
        return None
    if meta.get("latest_document_update") != manifest["latest_document_update"]:
        return None

    try:
        index = faiss.read_index(str(FAISS_INDEX_PATH))
    except Exception:
        return None

    chunk_ids = meta.get("chunk_ids")
    if not isinstance(chunk_ids, list) or not chunk_ids:
        return None

    return index, [int(chunk_id) for chunk_id in chunk_ids if isinstance(chunk_id, (int, float))]


def _build_search_result(row: Dict[str, Any], score: float, retrieval_method: str) -> Dict[str, Any]:
    return {
        "score": round(score, 4),
        "chunk_text": row["chunk_text"],
        "chunk_index": row["chunk_index"],
        "document_id": row["document_id"],
        "source": row["source"],
        "title": row["title"],
        "url": row.get("url"),
        "doc_type": row["doc_type"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "retrieval_method": retrieval_method,
        "embedding_model": row.get("embedding_model"),
    }


def _search_with_faiss(
    query_embedding: List[float],
    index: Any,
    chunk_ids: List[int],
    query_text: str,
    limit: int,
) -> List[Dict[str, Any]]:
    if index is None or not chunk_ids:
        return []

    query_vector = np.asarray([query_embedding], dtype="float32")
    faiss.normalize_L2(query_vector)

    search_limit = min(len(chunk_ids), max(limit * 4, limit))
    scores, indices = index.search(query_vector, search_limit)

    selected_chunk_ids: List[int] = []
    for index_value in indices[0]:
        if index_value < 0 or index_value >= len(chunk_ids):
            continue
        selected_chunk_ids.append(chunk_ids[index_value])

    rows = _fetch_chunks_by_ids(selected_chunk_ids)
    row_by_chunk_id = {row["chunk_id"]: row for row in rows}

    scored: List[Dict[str, Any]] = []
    for score, index_value in zip(scores[0], indices[0]):
        if index_value < 0 or index_value >= len(chunk_ids):
            continue
        chunk_id = chunk_ids[index_value]
        row = row_by_chunk_id.get(chunk_id)
        if not row:
            continue
        adjusted_score = float(score)
        lowered_chunk = row["chunk_text"].lower()
        lowered_title = row["title"].lower()
        if query_text and query_text in lowered_chunk:
            adjusted_score += 0.6
        if query_text and query_text in lowered_title:
            adjusted_score += 0.2
        if row.get("url") and query_text and query_text in row["url"].lower():
            adjusted_score += 0.05
        if adjusted_score <= 0:
            continue
        scored.append(_build_search_result(row, adjusted_score, "faiss"))

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]


def search_knowledge(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    initialize_db()
    query = (query or "").strip()
    if not query:
        return []

    query_tokens = Counter(_tokenize(query))
    if not query_tokens:
        return []

    query_text = " ".join(query_tokens.elements()).strip().lower()

    persistent_index = _load_persistent_faiss_index()
    if persistent_index is not None:
        query_embedding = _embed_texts([query])[0]
        if query_embedding is not None:
            index, chunk_ids = persistent_index
            scored = _search_with_faiss(query_embedding, index, chunk_ids, query_text, limit)
            if scored:
                return scored

    rows = _load_chunks()
    total_docs = max(len(rows), 1)
    document_frequency = Counter()
    chunk_tokens_list: List[Counter] = []
    for row in rows:
        tokens = Counter(_tokenize(row["chunk_text"]))
        chunk_tokens_list.append(tokens)
        document_frequency.update(set(tokens))

    query_vector = {}
    for token, count in query_tokens.items():
        idf = math.log((1 + total_docs) / (1 + document_frequency.get(token, 0))) + 1.0
        query_vector[token] = count * idf
    query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values())) or 1.0

    scored: List[Dict[str, Any]] = []
    for row, chunk_tokens in zip(rows, chunk_tokens_list):
        if not chunk_tokens:
            continue

        lexical_score = 0.0
        for token, count in chunk_tokens.items():
            idf = math.log((1 + total_docs) / (1 + document_frequency.get(token, 0))) + 1.0
            lexical_score += query_vector.get(token, 0.0) * (count * idf)
        chunk_norm = math.sqrt(
            sum((count * (math.log((1 + total_docs) / (1 + document_frequency.get(token, 0))) + 1.0)) ** 2 for token, count in chunk_tokens.items())
        ) or 1.0
        lexical_score /= query_norm * chunk_norm

        lowered_chunk = row["chunk_text"].lower()
        lowered_title = row["title"].lower()
        boosted_score = lexical_score
        if query_text and query_text in lowered_chunk:
            boosted_score += 0.6
        if query_text and query_text in lowered_title:
            boosted_score += 0.2
        if row.get("url") and query_text and query_text in row["url"].lower():
            boosted_score += 0.05

        if boosted_score <= 0:
            continue

        scored.append(_build_search_result(row, boosted_score, "lexical"))

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]


def knowledge_miss(query: str, threshold: float = 0.18) -> bool:
    hits = search_knowledge(query, limit=1)
    if not hits:
        return True
    top_hit = hits[0]
    if top_hit.get("retrieval_method") == "embedding":
        if top_hit["score"] < max(threshold, 0.4):
            return True
    elif top_hit["score"] < threshold:
        return True
    return not hit_is_fresh_enough(top_hit, query)


def query_needs_freshness(query: str) -> bool:
    lowered = (query or "").lower()
    return any(keyword in lowered for keyword in ["latest", "current", "today", "recent", "breaking", "news", "now"])


def hit_is_fresh_enough(hit: Dict[str, Any], query: str, max_age_days: int = 7) -> bool:
    if not query_needs_freshness(query):
        return True

    raw_updated = hit.get("updated_at") or hit.get("created_at")
    if not raw_updated:
        return False

    try:
        updated_at = datetime.fromisoformat(raw_updated.replace("Z", "+00:00"))
    except Exception:
        return False

    age_days = (datetime.now(timezone.utc) - updated_at.astimezone(timezone.utc)).total_seconds() / 86400
    return age_days <= max_age_days


def backfill_missing_embeddings(batch_size: int = 32) -> int:
    if not _embedding_supported():
        return 0

    initialize_db()
    updated = 0

    while True:
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT id, chunk_text
                FROM chunks
                WHERE embedding_json IS NULL OR embedding_json = ''
                ORDER BY id ASC
                LIMIT ?
                """,
                (batch_size,),
            ).fetchall()

        if not rows:
            break

        embeddings = _embed_texts([row["chunk_text"] for row in rows])
        with connect() as conn:
            for row, embedding in zip(rows, embeddings):
                if not embedding:
                    continue
                conn.execute(
                    """
                    UPDATE chunks
                    SET embedding_json = ?, embedding_model = ?
                    WHERE id = ?
                    """,
                    (
                        _serialize_embedding(embedding),
                        EMBEDDING_MODEL,
                        row["id"],
                    ),
                )
                updated += 1

    if updated:
        _mark_faiss_dirty("embeddings_backfilled")

    return updated


def format_hits(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for index, hit in enumerate(hits, start=1):
        excerpt = " ".join(hit["chunk_text"].split())
        if len(excerpt) > 360:
            excerpt = excerpt[:357].rsplit(" ", 1)[0] + "..."
        method = hit.get("retrieval_method", "lexical")
        lines.append(
            f"{index}. {hit['title']} [{hit['source']}] (score: {hit['score']:.3f}, {method})\n"
            f"   {excerpt}"
        )
    return "\n".join(lines)


def confidence_label(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.45:
        return "high"
    if score >= 0.25:
        return "medium"
    if score >= 0.18:
        return "low"
    return "very low"


def build_context(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(
            f"Title: {hit['title']}\n"
            f"Source: {hit['source']}\n"
            f"URL: {hit.get('url') or '(none)'}\n"
            f"Content: {hit['chunk_text']}"
        )
    return "\n\n".join(blocks)
