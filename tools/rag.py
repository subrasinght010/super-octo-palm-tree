from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus.json"

DEFAULT_CORPUS = [
    {
        "id": "fallback-1",
        "title": "RAG basics",
        "source": "fallback",
        "content": "Retrieval augmented generation combines retrieval and generation. The retriever finds relevant passages and the generator uses them to answer the question.",
    },
    {
        "id": "fallback-2",
        "title": "Tool calling loop",
        "source": "fallback",
        "content": "Tool calling works when the app sends a schema, executes the tool, and feeds the result back into the model.",
    },
    {
        "id": "fallback-3",
        "title": "MCP registry",
        "source": "fallback",
        "content": "A shared registry keeps tool definitions and handlers consistent across the backend and the frontend.",
    },
]


def _load_corpus() -> List[Dict[str, str]]:
    if DATA_PATH.exists():
        with DATA_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list) and data:
            return data
    return DEFAULT_CORPUS


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _excerpt(text: str, limit: int = 220) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rsplit(" ", 1)[0] + "..."


@lru_cache(maxsize=1)
def _build_index():
    corpus = _load_corpus()
    tokenized_docs = []
    document_frequency = Counter()

    for item in corpus:
        tokens = _tokenize(f"{item['title']} {item['content']}")
        tokenized_docs.append(tokens)
        document_frequency.update(set(tokens))

    total_docs = len(corpus)
    idf = {
        token: math.log((1 + total_docs) / (1 + df)) + 1.0
        for token, df in document_frequency.items()
    }

    doc_vectors = []
    for tokens in tokenized_docs:
        term_frequency = Counter(tokens)
        vector = {token: term_frequency[token] * idf.get(token, 1.0) for token in term_frequency}
        magnitude = math.sqrt(sum(weight * weight for weight in vector.values())) or 1.0
        normalized = {token: weight / magnitude for token, weight in vector.items()}
        doc_vectors.append(normalized)

    return corpus, doc_vectors, idf


def _score_query(query: str, idf: Dict[str, float]) -> Dict[str, float]:
    tokens = _tokenize(query)
    term_frequency = Counter(tokens)
    vector = {token: term_frequency[token] * idf.get(token, 0.5) for token in term_frequency}
    magnitude = math.sqrt(sum(weight * weight for weight in vector.values())) or 1.0
    return {token: weight / magnitude for token, weight in vector.items()}


def _rank_documents(query: str):
    corpus, doc_vectors, idf = _build_index()
    query_vector = _score_query(query, idf)
    scored = []

    for item, doc_vector in zip(corpus, doc_vectors):
        score = sum(query_vector.get(token, 0.0) * weight for token, weight in doc_vector.items())
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:3]


def search_docs(query: str):
    query = (query or "").strip()
    if not query:
        return "No query provided for document search."

    ranked = _rank_documents(query)
    if not ranked or ranked[0][0] <= 0:
        return "No strong match found in the local corpus."

    lines = ["Local research matches:"]
    for index, (score, item) in enumerate(ranked, start=1):
        excerpt = _excerpt(item["content"])
        lines.append(
            f"{index}. {item['title']} [{item['source']}] (score: {score:.3f})\n"
            f"   {excerpt}"
        )

    return "\n".join(lines)

