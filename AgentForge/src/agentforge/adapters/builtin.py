"""All nine adapters."""
from __future__ import annotations

import inspect
import os
from typing import Any, Callable

from ..errors import AdapterError, ConfigError
from .base import Adapter, post_json


def _key(config: dict[str, Any], cfg_name: str, env: str) -> str:
    k = config.get(cfg_name) or os.environ.get(env)
    if not k:
        raise ConfigError(f"Missing API key: pass config['{cfg_name}'] or set {env}")
    return k


def _accepts_two(fn: Callable[..., Any]) -> bool:
    try:
        params = [
            p for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        return len(params) >= 2
    except (TypeError, ValueError):
        return False


class PythonAdapter(Adapter):
    def __init__(self, fn: Callable[..., Any], model: str | None = None, config=None):
        super().__init__(model, config)
        self.fn = fn

    def generate(self, prompt: str, context: str | None = None) -> str:
        try:
            if context is not None and _accepts_two(self.fn):
                out = self.fn(prompt, context)
            else:
                out = self.fn(prompt)
        except Exception as e:  # user code can raise anything
            raise AdapterError(f"Python agent raised: {e}") from e
        return str(out)


class OllamaAdapter(Adapter):
    def generate(self, prompt: str, context: str | None = None) -> str:
        host = self.config.get("host", "http://localhost:11434")
        data = post_json(
            f"{host}/api/generate",
            {"model": self.model, "prompt": self.with_context(prompt, context), "stream": False},
        )
        return str(data.get("response", "")).strip()


class OpenAIAdapter(Adapter):
    def generate(self, prompt: str, context: str | None = None) -> str:
        base = self.config.get("base_url", "https://api.openai.com/v1")
        key = _key(self.config, "api_key", "OPENAI_API_KEY")
        data = post_json(
            f"{base}/chat/completions",
            {"model": self.model, "messages": [{"role": "user", "content": self.with_context(prompt, context)}]},
            {"Authorization": f"Bearer {key}"},
        )
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise AdapterError(f"Unexpected OpenAI response: {data}") from e


class AnthropicAdapter(Adapter):
    def generate(self, prompt: str, context: str | None = None) -> str:
        key = _key(self.config, "api_key", "ANTHROPIC_API_KEY")
        data = post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "model": self.model,
                "max_tokens": self.config.get("max_tokens", 1024),
                "messages": [{"role": "user", "content": self.with_context(prompt, context)}],
            },
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        try:
            return "".join(b.get("text", "") for b in data["content"]).strip()
        except (KeyError, TypeError) as e:
            raise AdapterError(f"Unexpected Anthropic response: {data}") from e


class GeminiAdapter(Adapter):
    def generate(self, prompt: str, context: str | None = None) -> str:
        key = _key(self.config, "api_key", "GEMINI_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={key}"
        data = post_json(url, {"contents": [{"parts": [{"text": self.with_context(prompt, context)}]}]})
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as e:
            raise AdapterError(f"Unexpected Gemini response: {data}") from e


class HuggingFaceAdapter(Adapter):
    def generate(self, prompt: str, context: str | None = None) -> str:
        key = _key(self.config, "api_key", "HF_TOKEN")
        data = post_json(
            f"https://api-inference.huggingface.co/models/{self.model}",
            {"inputs": self.with_context(prompt, context)},
            {"Authorization": f"Bearer {key}"},
        )
        try:
            return str(data[0]["generated_text"]).strip()
        except (KeyError, IndexError, TypeError) as e:
            raise AdapterError(f"Unexpected Hugging Face response: {data}") from e


class HttpAdapter(Adapter):
    """Generic endpoint: POSTs {"input": ...}, reads config['response_key'] (default 'output')."""

    def generate(self, prompt: str, context: str | None = None) -> str:
        url = self.config.get("url")
        if not url:
            raise ConfigError("http adapter needs config['url']")
        payload: dict[str, Any] = {self.config.get("request_key", "input"): prompt}
        if context is not None:
            payload["context"] = context
        data = post_json(url, payload, self.config.get("headers"))
        val: Any = data
        for part in str(self.config.get("response_key", "output")).split("."):
            try:
                val = val[int(part)] if isinstance(val, list) else val[part]
            except (KeyError, IndexError, ValueError, TypeError) as e:
                raise AdapterError(f"Response key not found: {part!r} in {data}") from e
        return str(val).strip()


def _content(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    c = getattr(obj, "content", None)
    if c is not None:
        return c if isinstance(c, str) else str(c)
    if isinstance(obj, dict):
        return str(obj.get("output", obj.get("content", obj)))
    return str(obj)


class LangChainAdapter(Adapter):
    """Wraps any LangChain Runnable (anything with .invoke)."""

    def __init__(self, runnable: Any, model: str | None = None, config=None):
        super().__init__(model, config)
        self.runnable = runnable

    def generate(self, prompt: str, context: str | None = None) -> str:
        try:
            return _content(self.runnable.invoke(self.with_context(prompt, context))).strip()
        except Exception as e:
            raise AdapterError(f"LangChain runnable raised: {e}") from e


class LangGraphAdapter(Adapter):
    """Wraps a compiled LangGraph graph with a 'messages' state."""

    def __init__(self, graph: Any, model: str | None = None, config=None):
        super().__init__(model, config)
        self.graph = graph

    def generate(self, prompt: str, context: str | None = None) -> str:
        try:
            state = self.graph.invoke({"messages": [("user", self.with_context(prompt, context))]})
        except Exception as e:
            raise AdapterError(f"LangGraph graph raised: {e}") from e
        msgs = state.get("messages") if isinstance(state, dict) else None
        return _content(msgs[-1]).strip() if msgs else _content(state).strip()
