from __future__ import annotations

import os
import re

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _local_summary(text: str) -> str:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return "No text provided to summarize."

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    if len(sentences) == 1:
        return cleaned[:300] + ("..." if len(cleaned) > 300 else "")

    summary = " ".join(sentences[:2]).strip()
    return summary[:400] + ("..." if len(summary) > 400 else "")


def summarize_text(query: str):
    text = (query or "").strip()
    if not text:
        return "No text provided to summarize."

    client = _client()
    if client is None:
        return _local_summary(text)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the input clearly, keeping the answer short and useful.",
                },
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content or _local_summary(text)
    except Exception:
        return _local_summary(text)
