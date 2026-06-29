from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


Message = Mapping[str, str]


class ChatBackend(Protocol):
    """Minimal protocol Actor needs from a chat model backend."""

    def chat(self, messages: Sequence[Message], *, temperature: float | None = None) -> str:
        ...


@dataclass(slots=True)
class LiteLLMBackend:
    """LiteLLM chat backend for local and remote models.

    The ``model`` string is passed directly to LiteLLM, so callers can use
    provider-prefixed model names such as ``openai/gpt-4o-mini``,
    ``anthropic/claude-3-5-sonnet-latest``, ``ollama/llama3.1``, or
    ``hosted_vllm/my-model``.
    """

    model: str = "openai/gpt-4o-mini"
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.0
    timeout: float | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def chat(self, messages: Sequence[Message], *, temperature: float | None = None) -> str:
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError(
                "LiteLLMBackend requires the 'litellm' package. "
                "Install this project with `uv sync` or add `litellm` to your environment."
            ) from exc

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
        }
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        kwargs.update(self.extra_kwargs)

        response = completion(**kwargs)
        return response.choices[0].message.content or ""

@dataclass(slots=True)
class StaticBackend:
    """Test backend that returns a fixed response."""

    response: str

    def chat(self, messages: Sequence[Message], *, temperature: float | None = None) -> str:
        return self.response
