"""The evaluation loop."""
from __future__ import annotations

import time
from typing import Callable

from .adapters import Adapter
from .errors import AgentForgeError
from .models import RunResult, Sample, SampleResult


def run_evaluation(
    agent_name: str,
    adapter: Adapter,
    samples: list[Sample],
    metric_fns: dict[str, Callable[..., float]],
) -> RunResult:
    results: list[SampleResult] = []
    for s in samples:
        t0 = time.perf_counter()
        try:
            output = adapter.generate(s.input, s.context)
            err = None
        except AgentForgeError as e:
            output, err = None, str(e)
        latency = time.perf_counter() - t0

        scores: dict[str, float | None] = {}
        for name, fn in metric_fns.items():
            if output is None:
                scores[name] = None
                continue
            try:
                val = float(fn(prompt=s.input, output=output, expected=s.expected_output, context=s.context))
                scores[name] = min(1.0, max(0.0, val))
            except Exception as e:  # a bad metric must not kill the whole run
                scores[name] = None
                err = (err + "; " if err else "") + f"{name}: {e}"
        results.append(SampleResult(s, output, scores, latency, err))
    return RunResult(agent=agent_name, metrics=list(metric_fns), results=results, model=adapter.model)
