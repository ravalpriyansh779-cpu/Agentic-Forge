"""LLM-as-a-Judge (local Ollama by default)."""
from __future__ import annotations

import json
import re

from ..adapters.base import post_json
from ..errors import MetricError

RUBRICS = {
    "faithfulness": "Is every claim in the ANSWER supported by the CONTEXT? 1 = mostly invented, 5 = fully grounded.",
    "relevance": "Does the ANSWER directly address the QUESTION? 1 = off-topic, 5 = fully on point.",
    "helpfulness": "How helpful, clear and complete is the ANSWER for the QUESTION? 1 = useless, 5 = excellent.",
}

_TEMPLATE = """You are a strict evaluator. {rubric}
Rate on an integer scale 1-5. Reply with JSON only: {{"score": <1-5>, "reason": "<short>"}}

QUESTION: {question}
{context_block}ANSWER: {answer}
"""


def model_family(model: str | None) -> str:
    """'llama3:8b' -> 'llama3', 'gpt-4o-mini' -> 'gpt'."""
    return re.split(r"[:\-_/.]", (model or "").lower())[0]


def parse_score(text: str) -> float:
    """Extract a 1-5 score from a judge reply and normalise to 0-1."""
    m = re.search(r"\{.*\}", text, re.S)
    raw = None
    if m:
        try:
            raw = json.loads(m.group(0)).get("score")
        except (json.JSONDecodeError, AttributeError):
            raw = None
    if raw is None:
        n = re.search(r"\b([1-5])\b", text)
        raw = n.group(1) if n else None
    try:
        score = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as e:
        raise MetricError(f"Could not parse judge score from: {text[:120]!r}") from e
    if not 1 <= score <= 5:
        raise MetricError(f"Judge score out of range: {score}")
    return (score - 1) / 4


class OllamaJudge:
    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        self.model, self.host = model, host

    def complete(self, prompt: str) -> str:
        data = post_json(
            f"{self.host}/api/generate",
            {"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        )
        return str(data.get("response", ""))

    def score(self, rubric: str, question: str, answer: str, context: str | None = None) -> float:
        if rubric not in RUBRICS:
            raise MetricError(f"Unknown rubric {rubric!r}")
        block = f"CONTEXT: {context}\n" if context else ""
        prompt = _TEMPLATE.format(rubric=RUBRICS[rubric], question=question, context_block=block, answer=answer)
        return parse_score(self.complete(prompt))
