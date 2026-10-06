"""Ollama embedder + cosine similarity."""
from __future__ import annotations

import math

from ..adapters.base import post_json
from ..errors import MetricError


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise MetricError("Embedding vectors must be the same non-zero length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        self.model, self.host = model, host

    def embed(self, text: str) -> list[float]:
        data = post_json(f"{self.host}/api/embeddings", {"model": self.model, "prompt": text})
        vec = data.get("embedding")
        if not vec:
            raise MetricError(f"No embedding returned: {data}")
        return [float(x) for x in vec]
