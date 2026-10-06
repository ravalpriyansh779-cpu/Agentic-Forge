"""Deterministic built-in metrics + names of model-backed ones."""
from __future__ import annotations

from typing import Any, Callable

from ..errors import MetricError

MetricFn = Callable[..., float]


def exact_match(*, output: str, expected: str | None = None, **_: Any) -> float:
    if expected is None:
        raise MetricError("exact_match needs expected_output")
    return 1.0 if output.strip() == str(expected).strip() else 0.0


def contains_expected(*, output: str, expected: str | None = None, **_: Any) -> float:
    if expected is None:
        raise MetricError("contains_expected needs expected_output")
    return 1.0 if str(expected).strip().lower() in output.lower() else 0.0


DETERMINISTIC: dict[str, MetricFn] = {
    "exact_match": exact_match,
    "contains_expected": contains_expected,
}
JUDGE_METRICS = ("faithfulness", "relevance", "helpfulness")
SEMANTIC_METRICS = ("semantic_similarity",)
