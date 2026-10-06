# Usage reference

## Client
`Client(judge_model="llama3", embed_model="nomic-embed-text", ollama_host=..., judge=None, embedder=None)`

| Method | Purpose |
|---|---|
| `register_agent(name, adapter="python", model=None, config=None, target=None)` | add an agent |
| `@client.agent(name)` | wrap a function as an agent |
| `@client.metric(name)` | register a custom metric |
| `evaluate(agent, dataset, metrics)` | returns `RunResult` |
| `compare(agents, dataset, metrics)` | returns `ComparisonResult` |

## Adapters
`python`, `ollama`, `openai` (config: `api_key`, `base_url`), `anthropic`, `gemini`, `huggingface`,
`langchain` (target=runnable), `langgraph` (target=graph), `http` (config: `url`, `response_key`, `headers`).
API keys can also come from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`.

## Datasets
List of dicts, `.json`, or `.jsonl`. Fields: `input` (or `question`), `expected_output` (or `expected`), `context`.

## Results
`run.aggregate_scores()`, `run.save(path)`, `RunResult.load(path)`,
`cmp.leaderboard()`, `cmp.leaderboard_text()`, `cmp.save(path)`.
