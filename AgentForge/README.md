# 🔨 AgentForge (clean-room rebuild)

Forge, test and compare AI agents. Framework- and model-agnostic. **No runtime dependencies.**

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest
```

## Python API
```python
from agentforge import Client
client = Client()

@client.agent(name="rule-bot")
def rule_bot(q: str) -> str:
    return "Paris" if "France" in q else "unknown"

client.register_agent(name="llama3", adapter="ollama", model="llama3")   # local, free

run = client.evaluate("rule-bot",
    [{"input": "Capital of France?", "expected_output": "Paris"}],
    metrics=["exact_match"])
print(run.aggregate_scores())

cmp = client.compare(["rule-bot", "llama3"], "examples/capitals.jsonl", ["exact_match", "relevance"])
print(cmp.leaderboard_text())
```

## CLI
```bash
agentforge run     --config examples/myclient.py --agent rule-bot --dataset examples/capitals.jsonl --metrics exact_match
agentforge compare --config examples/myclient.py --agents rule-bot,lazy-bot --dataset examples/capitals.jsonl \
                   --metrics exact_match,contains_expected --output cmp.json
agentforge inspect cmp.json
```

## Features
| Area | Included |
|---|---|
| Adapters (9) | python, ollama, openai, anthropic, gemini, huggingface, langchain, langgraph, http |
| Metrics | exact_match, contains_expected, semantic_similarity, faithfulness, relevance, helpfulness, custom |
| Extras | compare + leaderboard, JSON persistence, .json/.jsonl datasets, same-family judge-bias warning |

See `docs/USAGE.md` and `docs/ARCHITECTURE.md`. License: Apache-2.0.
