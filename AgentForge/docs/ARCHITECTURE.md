# Architecture

```
Client ──register──▶ Adapter (prompt -> answer)       src/agentforge/adapters
   │
   ├─evaluate()──▶ load_dataset ─▶ run_evaluation ─▶ RunResult
   │                                   │
   │                                   └─ metric fns (deterministic | semantic | judge | custom)
   └─compare()───▶ evaluate() per agent over the SAME samples ─▶ ComparisonResult (leaderboard)
```

Design choices
- **Zero runtime dependencies.** HTTP uses `urllib`; models are dataclasses.
- **Adapters are tiny:** one method, `generate(prompt, context) -> str`.
- **Metrics are plain functions** `fn(*, prompt, output, expected, context) -> float in [0,1]`.
- **Failures are data.** An adapter/metric error is stored on the sample, not raised, so one bad row never kills a run.
- **Judge = local Ollama by default** (offline, free). A warning fires if the judge and agent share a model family.
