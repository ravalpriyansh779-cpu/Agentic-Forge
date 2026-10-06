"""Plain dataclass data models (no third-party dependencies)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Sample:
    input: str
    expected_output: str | None = None
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    sample: Sample
    output: str | None
    scores: dict[str, float | None]
    latency_s: float = 0.0
    error: str | None = None


@dataclass
class RunResult:
    agent: str
    metrics: list[str]
    results: list[SampleResult]
    model: str | None = None
    created_at: str = field(default_factory=_now)

    def aggregate_scores(self) -> dict[str, float]:
        """Mean score per metric, ignoring samples where it failed."""
        agg: dict[str, float] = {}
        for m in self.metrics:
            vals = [r.scores[m] for r in self.results if r.scores.get(m) is not None]
            if vals:
                agg[m] = sum(vals) / len(vals)  # type: ignore[arg-type]
        return agg

    def overall(self) -> float:
        agg = self.aggregate_scores()
        return sum(agg.values()) / len(agg) if agg else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["aggregate"] = self.aggregate_scores()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunResult":
        results = [
            SampleResult(
                sample=Sample(**r["sample"]),
                output=r["output"],
                scores=r["scores"],
                latency_s=r.get("latency_s", 0.0),
                error=r.get("error"),
            )
            for r in d["results"]
        ]
        return cls(
            agent=d["agent"],
            metrics=d["metrics"],
            results=results,
            model=d.get("model"),
            created_at=d.get("created_at", _now()),
        )

    def save(self, path: str) -> None:
        from .persistence import save_json

        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str) -> "RunResult":
        from .persistence import load_json

        return cls.from_dict(load_json(path))
