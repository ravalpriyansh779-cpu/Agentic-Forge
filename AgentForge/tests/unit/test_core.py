import json
import warnings

import pytest

from agentforge import Client, ComparisonResult, ConfigError, DatasetError, RunResult
from agentforge.adapters import build_adapter
from agentforge.datasets import load_dataset
from agentforge.embeddings import cosine
from agentforge.judges import model_family, parse_score
from agentforge.errors import AdapterError, MetricError

DATA = [
    {"input": "Capital of France?", "expected_output": "Paris"},
    {"input": "Capital of Japan?", "expected_output": "Tokyo"},
]


class FakeJudge:
    model = "judge-x"

    def score(self, rubric, question, answer, context=None):
        return 0.75


class FakeEmbedder:
    def embed(self, text):
        return [1.0, 0.0] if "Paris" in text else [0.0, 1.0]


def make_client():
    c = Client(judge=FakeJudge(), embedder=FakeEmbedder())

    @c.agent(name="good")
    def good(q):
        return "Paris" if "France" in q else "Tokyo"

    @c.agent(name="bad")
    def bad(q):
        return "unknown"

    return c


def test_exact_match_and_aggregate():
    c = make_client()
    run = c.evaluate("good", DATA, ["exact_match", "contains_expected"])
    assert run.aggregate_scores() == {"exact_match": 1.0, "contains_expected": 1.0}
    assert c.evaluate("bad", DATA, ["exact_match"]).aggregate_scores()["exact_match"] == 0.0


def test_contains_is_case_insensitive():
    c = Client(judge=FakeJudge(), embedder=FakeEmbedder())
    c.register_agent("a", target=lambda q: "It is PARIS, of course")
    assert c.evaluate("a", DATA[:1], ["contains_expected"]).aggregate_scores()["contains_expected"] == 1.0


def test_custom_metric():
    c = make_client()

    @c.metric(name="one_word")
    def one_word(*, prompt, output, **kw):
        return 1.0 if len(output.split()) == 1 else 0.0

    assert c.evaluate("good", DATA, ["one_word"]).aggregate_scores() == {"one_word": 1.0}


def test_semantic_and_judge_metrics_use_injected_services():
    c = make_client()
    run = c.evaluate("good", DATA[:1], ["semantic_similarity", "relevance"])
    assert run.aggregate_scores()["semantic_similarity"] == 1.0
    assert run.aggregate_scores()["relevance"] == 0.75


def test_agent_error_is_recorded_not_raised():
    c = Client(judge=FakeJudge(), embedder=FakeEmbedder())

    def boom(q):
        raise RuntimeError("kaput")

    c.register_agent("boom", target=boom)
    run = c.evaluate("boom", DATA, ["exact_match"])
    assert run.results[0].output is None and "kaput" in run.results[0].error
    assert run.aggregate_scores() == {}


def test_missing_expected_is_recorded_as_metric_error():
    c = make_client()
    run = c.evaluate("good", [{"input": "hi"}], ["exact_match"])
    assert run.results[0].scores["exact_match"] is None


def test_compare_leaderboard_order_and_persistence(tmp_path):
    c = make_client()
    cmp = c.compare(["bad", "good"], DATA, ["exact_match"])
    lb = cmp.leaderboard()
    assert [r["agent"] for r in lb] == ["good", "bad"] and lb[0]["rank"] == 1
    assert "good" in cmp.leaderboard_text()
    p = tmp_path / "cmp.json"
    cmp.save(str(p))
    loaded = ComparisonResult.load(str(p))
    assert loaded.leaderboard()[0]["agent"] == "good"


def test_run_roundtrip(tmp_path):
    run = make_client().evaluate("good", DATA, ["exact_match"])
    p = tmp_path / "run.json"
    run.save(str(p))
    assert RunResult.load(str(p)).aggregate_scores() == run.aggregate_scores()


def test_same_family_warning():
    c = Client(judge=FakeJudge(), embedder=FakeEmbedder())
    c.judge.model = "llama3:8b"
    c.register_agent("l", adapter="ollama", model="llama3")
    with pytest.warns(UserWarning, match="same family"):
        c._resolve("relevance", "llama3")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c._resolve("relevance", "mistral")


def test_unknown_agent_and_metric():
    c = make_client()
    with pytest.raises(ConfigError):
        c.evaluate("nope", DATA, ["exact_match"])
    with pytest.raises(ConfigError):
        c.evaluate("good", DATA, ["nonsense"])


def test_dataset_loading(tmp_path):
    (tmp_path / "d.jsonl").write_text('{"input":"a","expected":"b"}\n\n{"question":"c"}\n')
    (tmp_path / "d.json").write_text(json.dumps(DATA))
    s = load_dataset(str(tmp_path / "d.jsonl"))
    assert s[0].expected_output == "b" and s[1].input == "c"
    assert len(load_dataset(str(tmp_path / "d.json"))) == 2
    with pytest.raises(DatasetError):
        load_dataset(str(tmp_path / "missing.jsonl"))
    with pytest.raises(DatasetError):
        load_dataset([])
    with pytest.raises(DatasetError):
        load_dataset([{"nope": 1}])


def test_helpers():
    assert cosine([1, 0], [1, 0]) == 1.0 and cosine([1, 0], [0, 1]) == 0.0
    assert model_family("llama3:8b") == "llama3" and model_family("gpt-4o-mini") == "gpt"
    assert parse_score('{"score": 5, "reason": "x"}') == 1.0
    assert parse_score('Sure! {"score": 1}') == 0.0
    assert parse_score("I'd say 3 out of 5") == 0.5
    with pytest.raises(MetricError):
        parse_score("no idea")


def test_adapter_factory_and_http(monkeypatch):
    with pytest.raises(ConfigError):
        build_adapter("nope", "m")
    with pytest.raises(ConfigError):
        build_adapter("openai")  # model required

    import agentforge.adapters.builtin as b

    monkeypatch.setattr(b, "post_json", lambda url, payload, headers=None, **k: {"choices": [{"message": {"content": " hi "}}]})
    assert build_adapter("openai", "gpt-x", {"api_key": "k"}).generate("q") == "hi"
    monkeypatch.setattr(b, "post_json", lambda url, payload, headers=None, **k: {"data": {"answer": "yes"}})
    ad = build_adapter("http", config={"url": "http://x", "response_key": "data.answer"})
    assert ad.generate("q") == "yes"
    monkeypatch.setattr(b, "post_json", lambda *a, **k: {"response": " ok "})
    assert build_adapter("ollama", "llama3").generate("q", "ctx") == "ok"
    monkeypatch.setattr(b, "post_json", lambda *a, **k: {"content": [{"type": "text", "text": "A"}]})
    assert build_adapter("anthropic", "m", {"api_key": "k"}).generate("q") == "A"
    with pytest.raises(ConfigError):
        build_adapter("gemini", "m").generate("q")  # no key in env


def test_langchain_and_langgraph_wrappers():
    class Msg:
        content = "from-chain"

    class Runnable:
        def invoke(self, x):
            return Msg()

    class Graph:
        def invoke(self, state):
            return {"messages": [("user", "x"), Msg()]}

    assert build_adapter("langchain", target=Runnable(), model="m").generate("q") == "from-chain"
    assert build_adapter("langgraph", target=Graph(), model="m").generate("q") == "from-chain"


def test_python_adapter_passes_context_when_supported():
    seen = {}

    def agent(q, ctx):
        seen["ctx"] = ctx
        return "ok"

    build_adapter("python", target=agent).generate("q", "the-context")
    assert seen["ctx"] == "the-context"
    with pytest.raises(AdapterError):
        build_adapter("python", target=lambda q: 1 / 0).generate("q")
