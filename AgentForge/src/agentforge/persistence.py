"""JSON save / load helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import AgentForgeError


def save_json(data: dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise AgentForgeError(f"File not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))
