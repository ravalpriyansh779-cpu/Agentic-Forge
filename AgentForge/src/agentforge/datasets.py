"""Dataset loading: in-memory lists, .jsonl and .json files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .errors import DatasetError
from .models import Sample


def _to_sample(row: Any, idx: int) -> Sample:
    if isinstance(row, Sample):
        return row
    if not isinstance(row, dict):
        raise DatasetError(f"Row {idx} must be an object, got {type(row).__name__}")
    inp = row.get("input", row.get("question"))
    if inp is None:
        raise DatasetError(f"Row {idx} is missing an 'input' field")
    known = {"input", "question", "expected_output", "expected", "context"}
    return Sample(
        input=str(inp),
        expected_output=row.get("expected_output", row.get("expected")),
        context=row.get("context"),
        metadata={k: v for k, v in row.items() if k not in known},
    )


def load_dataset(source: str | Path | Iterable[Any]) -> list[Sample]:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise DatasetError(f"Dataset file not found: {source}")
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif p.suffix == ".json":
            rows = json.loads(text)
            if not isinstance(rows, list):
                raise DatasetError("A .json dataset must be a list of objects")
        else:
            raise DatasetError(f"Unsupported dataset format: {p.suffix}")
    else:
        rows = list(source)
    if not rows:
        raise DatasetError("Dataset is empty")
    return [_to_sample(r, i) for i, r in enumerate(rows)]
