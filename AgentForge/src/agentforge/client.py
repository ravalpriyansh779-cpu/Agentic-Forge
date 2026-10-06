"""Main entry point: the Client class."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

from .adapters import Adapter, build_adapter
from .comparison import ComparisonResult
from .datasets import load_dataset
from .embeddings import OllamaEmbedder, cosine
from .errors import ConfigError, MetricError
from .evaluate import run_evaluation
from .judges import RUBRICS, OllamaJudge, model_family
from .metrics import DETERMINISTIC, JUDGE_METRICS, SEMANTIC_METRICS
from .models import RunResult


class Client:
    """Register agents and metrics, then evaluate or compare them.

    Judge and embedding models default to a local Ollama server, so the whole
    loop can run offline with no API cost.
    """

    def __init__(
        self,
        judge_model: str = "llama3",
        embed_model: str = "nomic-embed-text",
        ollama_host: str = "http://localhost:11434",
        judge: Any = None,
        embedder: Any = None,
    ):
        self.judge = judge or OllamaJudge(judge_model, ollama_host)
        self.embedder = embedder or OllamaEmbedder(embed_model, ollama_host)
        self._agents: dict[str, Adapter] = {}
        self._metrics: dict[str, Callable[..., float]] = {}

    # ---- registration -------------------------------------------------
    def register_agent(
        self,
        name: str,
        adapter: str = "python",
        model: str | None = None,
        config: dict[str, Any] | None = None,
        target: Any = None,
    ) -> None:
        self._agents[name] = build_adapter(adapter, model, config, target)

    def agent(self, name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: wrap a plain Python function as an agent."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register_agent(name or fn.__name__, "python", target=fn)
            return fn

        return deco

    def metric(self, name: str | None = None) -> Callable[[Callable[..., float]], Callable[..., float]]:
        """Decorator: register a custom metric fn(*, prompt, output, **kw) -> float in [0, 1]."""

        def deco(fn: Callable[..., float]) -> Callable[..., float]:
            self._metrics[name or fn.__name__] = fn
            return fn

        return deco

    def list_agents(self) -> list[str]:
        return sorted(self._agents)

    # ---- metric resolution --------------------------------------------
    def _resolve(self, name: str, agent_model: str | None) -> Callable[..., float]:
        if name in self._metrics:
            return self._metrics[name]
        if name in DETERMINISTIC:
            return DETERMINISTIC[name]
        if name in SEMANTIC_METRICS:
            return self._semantic
        if name in JUDGE_METRICS:
            self._warn_same_family(agent_model)
            return lambda *, prompt, output, context=None, **_: self.judge.score(
                name, prompt, output, context
            )
        known = sorted({*self._metrics, *DETERMINISTIC, *SEMANTIC_METRICS, *JUDGE_METRICS})
        raise ConfigError(f"Unknown metric {name!r}. Available: {', '.join(known)}")

    def _semantic(self, *, output: str, expected: str | None = None, **_: Any) -> float:
        if expected is None:
            raise MetricError("semantic_similarity needs expected_output")
        sim = cosine(self.embedder.embed(output), self.embedder.embed(str(expected)))
        return max(0.0, sim)

    def _warn_same_family(self, agent_model: str | None) -> None:
        jm = getattr(self.judge, "model", None)
        if jm and agent_model and model_family(jm) == model_family(agent_model):
            warnings.warn(
                f"Judge model {jm!r} and agent model {agent_model!r} are from the same family; "
                "LLM judges tend to favour their own family's output (self-preference bias).",
                stacklevel=3,
            )

    # ---- evaluation ----------------------------------------------------
    def evaluate(
        self, agent: str, dataset: str | Path | Iterable[Any], metrics: list[str]
    ) -> RunResult:
        if agent not in self._agents:
            raise ConfigError(f"Unknown agent {agent!r}. Registered: {', '.join(self.list_agents()) or 'none'}")
        if not metrics:
            raise ConfigError("Provide at least one metric")
        adapter = self._agents[agent]
        fns = {m: self._resolve(m, adapter.model) for m in metrics}
        return run_evaluation(agent, adapter, load_dataset(dataset), fns)

    def compare(
        self, agents: list[str], dataset: str | Path | Iterable[Any], metrics: list[str]
    ) -> ComparisonResult:
        samples = load_dataset(dataset)  # load once so every agent sees identical data
        runs = {a: self.evaluate(a, samples, metrics) for a in agents}
        return ComparisonResult(runs, list(metrics))
