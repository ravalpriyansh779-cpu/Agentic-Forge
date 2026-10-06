"""Multi-agent comparison and leaderboards."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import RunResult


@dataclass
class ComparisonResult:
    runs: dict[str, RunResult]
    metrics: list[str]

    def leaderboard(self) -> list[dict[str, Any]]:
        rows = []
        for name, run in self.runs.items():
            agg = run.aggregate_scores()
            rows.append({"agent": name, "overall": run.overall(), "scores": agg})
        rows.sort(key=lambda r: r["overall"], reverse=True)
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        return rows

    def leaderboard_text(self) -> str:
        w = max([5] + [len(n) for n in self.runs]) + 2
        head = f"{'#':<3}{'agent':<{w}}{'overall':>9}" + "".join(f"{m[:18]:>20}" for m in self.metrics)
        lines = [head, "-" * len(head)]
        for r in self.leaderboard():
            cells = "".join(
                f"{r['scores'][m]:>20.3f}" if m in r["scores"] else f"{'n/a':>20}" for m in self.metrics
            )
            lines.append(f"{r['rank']:<3}{r['agent']:<{w}}{r['overall']:>9.3f}{cells}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "leaderboard": self.leaderboard(),
            "runs": {k: v.to_dict() for k, v in self.runs.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ComparisonResult":
        return cls({k: RunResult.from_dict(v) for k, v in d["runs"].items()}, d["metrics"])

    def save(self, path: str) -> None:
        from .persistence import save_json

        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str) -> "ComparisonResult":
        from .persistence import load_json

        return cls.from_dict(load_json(path))
