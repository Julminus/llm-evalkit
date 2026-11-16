"""Model providers and the ``name:model`` spec the CLI accepts.

==============  ==========================  ======================================
spec            backend                     needs
==============  ==========================  ======================================
``mock``        deterministic, offline      nothing (used in CI)
``ollama:M``    Ollama ``/api/chat``        a running Ollama server with ``M``
``openai:M``    any OpenAI-compatible API   ``OPENAI_BASE_URL`` and/or a key
``anthropic:M`` Claude via the official SDK ``llm-evalkit[anthropic]`` + a key
==============  ==========================  ======================================

The spec is split at the first colon only, so Ollama tags such as
``ollama:qwen2.5:7b`` keep their own colon.
"""

from __future__ import annotations

from evalkit.prompts import Prompt
from evalkit.providers.anthropic_api import AnthropicProvider
from evalkit.providers.base import (
    Completion,
    FinishReason,
    GenerationParams,
    Provider,
    ProviderConfigError,
    ProviderError,
)
from evalkit.providers.mock import MockProvider
from evalkit.providers.ollama import OllamaProvider
from evalkit.providers.openai_compat import OpenAICompatProvider

__all__ = [
    "PROVIDER_NAMES",
    "AnthropicProvider",
    "Completion",
    "FinishReason",
    "GenerationParams",
    "MockProvider",
    "OllamaProvider",
    "OpenAICompatProvider",
    "Prompt",
    "Provider",
    "ProviderConfigError",
    "ProviderError",
    "make_provider",
    "parse_spec",
]

PROVIDER_NAMES = ("mock", "ollama", "openai", "anthropic")


def parse_spec(spec: str) -> tuple[str, str]:
    """Split ``"ollama:qwen2.5:7b"`` into ``("ollama", "qwen2.5:7b")``."""
    name, _, model = spec.strip().partition(":")
    name = name.lower()
    if name not in PROVIDER_NAMES:
        raise ProviderConfigError(
            f"unknown provider {name!r} in {spec!r}; expected one of: {', '.join(PROVIDER_NAMES)}"
        )
    if name == "mock":
        return name, model or "default"
    if not model:
        examples = {
            "ollama": "ollama:qwen2.5:7b",
            "openai": "openai:gpt-4o-mini",
            "anthropic": "anthropic:claude-opus-5-5",
        }
        raise ProviderConfigError(f"{spec!r} needs a model name, e.g. {examples[name]}")
    return name, model


def make_provider(spec: str, *, timeout: float = 300.0, base_url: str | None = None) -> Provider:
    """Build a provider from a CLI spec. ``base_url`` overrides the env default.

    For the mock with canned responses use :meth:`MockProvider.for_suite`.
    """
    name, model = parse_spec(spec)
    if name == "mock":
        return MockProvider(model)
    if name == "ollama":
        return OllamaProvider(model, host=base_url, timeout=timeout)
    if name == "openai":
        return OpenAICompatProvider(model, base_url=base_url, timeout=timeout)
    return AnthropicProvider(model, base_url=base_url, timeout=timeout)
