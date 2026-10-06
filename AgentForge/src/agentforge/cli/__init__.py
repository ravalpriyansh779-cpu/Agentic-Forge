"""Command line: agentforge run | compare | inspect."""
from __future__ import annotations

import argparse
import runpy
import sys

from ..client import Client
from ..comparison import ComparisonResult
from ..errors import AgentForgeError
from ..persistence import load_json


def _load_client(path: str) -> Client:
    ns = runpy.run_path(path)
    client = ns.get("client")
    if not isinstance(client, Client):
        raise AgentForgeError(f"{path} must define a variable named 'client' (an agentforge.Client)")
    return client


def _metrics(s: str) -> list[str]:
    return [m.strip() for m in s.split(",") if m.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentforge", description="Forge, test and compare AI agents.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="evaluate one agent")
    r.add_argument("--config", required=True)
    r.add_argument("--agent", required=True)
    r.add_argument("--dataset", required=True)
    r.add_argument("--metrics", required=True)
    r.add_argument("--output")

    c = sub.add_parser("compare", help="compare several agents")
    c.add_argument("--config", required=True)
    c.add_argument("--agents", required=True)
    c.add_argument("--dataset", required=True)
    c.add_argument("--metrics", required=True)
    c.add_argument("--output")

    i = sub.add_parser("inspect", help="show a saved run or comparison")
    i.add_argument("path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "run":
            client = _load_client(args.config)
            run = client.evaluate(args.agent, args.dataset, _metrics(args.metrics))
            print(f"Agent: {run.agent}  samples: {len(run.results)}")
            for m, v in run.aggregate_scores().items():
                print(f"  {m:<22}{v:.3f}")
            if args.output:
                run.save(args.output)
        elif args.cmd == "compare":
            client = _load_client(args.config)
            cmp = client.compare(_metrics(args.agents), args.dataset, _metrics(args.metrics))
            print(cmp.leaderboard_text())
            if args.output:
                cmp.save(args.output)
        else:
            data = load_json(args.path)
            if "leaderboard" in data:
                print(ComparisonResult.from_dict(data).leaderboard_text())
            else:
                print(f"Agent: {data['agent']}  samples: {len(data['results'])}")
                for m, v in data.get("aggregate", {}).items():
                    print(f"  {m:<22}{v:.3f}")
    except AgentForgeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
