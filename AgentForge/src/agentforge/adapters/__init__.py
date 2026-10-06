from __future__ import annotations

from typing import Any

from ..errors import ConfigError
from .base import Adapter
from .builtin import (
    AnthropicAdapter,
    GeminiAdapter,
    HttpAdapter,
    HuggingFaceAdapter,
    LangChainAdapter,
    LangGraphAdapter,
    OllamaAdapter,
    OpenAIAdapter,
    PythonAdapter,
)

ADAPTER_NAMES = (
    "python", "ollama", "openai", "anthropic", "gemini",
    "huggingface", "langchain", "langgraph", "http",
)


def build_adapter(
    name: str,
    model: str | None = None,
    config: dict[str, Any] | None = None,
    target: Any = None,
) -> Adapter:
    """Create an adapter. `target` is the callable / runnable / graph for python, langchain, langgraph."""
    config = config or {}
    wrappers = {"python": PythonAdapter, "langchain": LangChainAdapter, "langgraph": LangGraphAdapter}
    if name in wrappers:
        if target is None:
            raise ConfigError(f"{name} adapter needs a target object/callable")
        return wrappers[name](target, model, config)
    table = {
        "ollama": OllamaAdapter, "openai": OpenAIAdapter, "anthropic": AnthropicAdapter,
        "gemini": GeminiAdapter, "huggingface": HuggingFaceAdapter, "http": HttpAdapter,
    }
    if name not in table:
        raise ConfigError(f"Unknown adapter {name!r}. Choose from: {', '.join(ADAPTER_NAMES)}")
    if name != "http" and not model:
        raise ConfigError(f"adapter {name!r} requires model=...")
    return table[name](model, config)


__all__ = ["Adapter", "build_adapter", "ADAPTER_NAMES"]
