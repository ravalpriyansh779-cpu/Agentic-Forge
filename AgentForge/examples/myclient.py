"""Config file for the CLI. Run:

    agentforge compare --config examples/myclient.py --agents rule-bot,lazy-bot \
        --dataset examples/capitals.jsonl --metrics exact_match,contains_expected
"""
from agentforge import Client

client = Client()

CAPITALS = {"France": "Paris", "Japan": "Tokyo", "India": "New Delhi", "Italy": "Rome"}


@client.agent(name="rule-bot")
def rule_bot(question: str) -> str:
    for country, city in CAPITALS.items():
        if country in question:
            return city
    return "unknown"


@client.agent(name="lazy-bot")
def lazy_bot(question: str) -> str:
    return "Paris" if "France" in question else "I don't know"


# Local LLMs (needs `ollama serve` + `ollama pull llama3`):
# client.register_agent(name="llama3", adapter="ollama", model="llama3")
