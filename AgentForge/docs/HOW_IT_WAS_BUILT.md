# How a project like this is built (build order)

1. **Define the problem & contract** – "score any agent on any dataset with any metric." Core contract: an agent is `prompt -> text`; a metric is `(prompt, output, expected, context) -> float`.
2. **Errors + data models** (`errors.py`, `models.py`) – dataclasses for Sample / SampleResult / RunResult, plus `aggregate_scores()`.
3. **Adapters** – one tiny class per provider behind a common `generate()`; a stdlib `post_json` helper; a factory (`build_adapter`).
4. **Metrics** – deterministic first (easy to test), then embeddings (cosine), then LLM-as-judge (rubric prompt -> JSON score -> normalise to 0-1).
5. **Evaluation loop** – iterate samples, time the call, record errors as data, clamp scores to [0,1].
6. **Client facade** – registers agents/metrics, resolves metric names, exposes `evaluate()` and `compare()`; `@client.agent` / `@client.metric` decorators.
7. **Comparison + persistence** – leaderboard, JSON save/load.
8. **CLI** – `argparse` over the same API; config is a Python file exposing `client`.
9. **Tests** – fake judge/embedder/HTTP so everything runs offline; CLI tested end to end.
10. **Packaging & docs** – `pyproject.toml` (src layout, console script), README, CHANGELOG, LICENSE.
