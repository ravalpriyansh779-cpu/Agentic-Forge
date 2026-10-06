from pathlib import Path

from agentforge.cli import main

EX = Path(__file__).resolve().parents[2] / "examples"


def test_cli_run_compare_inspect(tmp_path, capsys):
    out = tmp_path / "cmp.json"
    base = ["--config", str(EX / "myclient.py"), "--dataset", str(EX / "capitals.jsonl")]
    assert main(["run", *base, "--agent", "rule-bot", "--metrics", "exact_match"]) == 0
    assert "exact_match" in capsys.readouterr().out
    assert main(["compare", *base, "--agents", "rule-bot,lazy-bot", "--metrics", "exact_match", "--output", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.index("rule-bot") < text.index("lazy-bot")
    assert main(["inspect", str(out)]) == 0
    assert "rule-bot" in capsys.readouterr().out


def test_cli_errors_return_nonzero(capsys):
    assert main(["run", "--config", str(EX / "myclient.py"), "--agent", "ghost",
                 "--dataset", str(EX / "capitals.jsonl"), "--metrics", "exact_match"]) == 1
    assert "error:" in capsys.readouterr().err
