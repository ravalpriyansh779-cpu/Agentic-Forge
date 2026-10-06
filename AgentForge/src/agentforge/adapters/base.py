"""Adapter base class and a tiny stdlib HTTP helper."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from ..errors import AdapterError


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> Any:
    """POST JSON and return decoded JSON. Raises AdapterError on any failure."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **(headers or {})}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        raise AdapterError(f"HTTP {e.code} from {url}: {body}") from e
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise AdapterError(f"Request to {url} failed: {e}") from e


class Adapter(ABC):
    """Turns a prompt (plus optional context) into an answer string."""

    def __init__(self, model: str | None = None, config: dict[str, Any] | None = None):
        self.model = model
        self.config = config or {}

    @abstractmethod
    def generate(self, prompt: str, context: str | None = None) -> str: ...

    @staticmethod
    def with_context(prompt: str, context: str | None) -> str:
        return f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
