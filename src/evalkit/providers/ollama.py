"""Ollama's native chat API (``POST /api/chat``), the default local backend."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

import httpx

from evalkit.providers._http import optional_int, parse_response, post_json
from evalkit.providers.base import (
    Completion,
    FinishReason,
    GenerationParams,
    Prompt,
    Provider,
    ProviderError,
)

__all__ = ["DEFAULT_OLLAMA_HOST", "OllamaProvider", "normalize_host"]

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

_FINISH: dict[str, FinishReason] = {"stop": "stop", "length": "length"}

# evalkit name -> Ollama "options" name
_OPTIONS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "max_tokens": "num_predict",
    "seed": "seed",
    "stop": "stop",
}


def normalize_host(host: str) -> str:
    """Accept ``OLLAMA_HOST`` the way Ollama does.

    ``localhost``, ``localhost:11434`` and ``http://localhost:11434/`` all mean
    the same server. A bare host gets Ollama's default port, and the bind-all
    address ``0.0.0.0`` (a common server setting) becomes loopback, because
    a client can't connect to it on every OS.
    """
    host = host.strip().rstrip("/")
    if "://" not in host:
        parts = urlsplit(f"http://{host}")
        port = parts.port or 11434
        hostname = parts.hostname or "127.0.0.1"
        host = f"http://{hostname}:{port}"
    parts = urlsplit(host)
    if parts.hostname == "0.0.0.0":
        host = host.replace("0.0.0.0", "127.0.0.1", 1)
    return host


class OllamaProvider(Provider):
    """A model served by a local (or remote) Ollama server.

    The host comes from ``host=``, else ``OLLAMA_HOST``, else
    ``http://127.0.0.1:11434``. The cache key uses the model *tag*: if you
    re-pull a tag and get new weights, clear the cache.
    """

    name = "ollama"
    supported_params = frozenset(_OPTIONS)

    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        timeout: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(model)
        self.host = normalize_host(host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST)
        self._client = httpx.AsyncClient(
            base_url=self.host,
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
        )

    def _options(self, params: GenerationParams) -> dict[str, Any]:
        return {_OPTIONS[k]: v for k, v in self.effective_params(params).items()}

    async def preflight(self) -> None:
        """Check that the server is up and the model is pulled."""
        try:
            resp = await self._client.post("/api/show", json={"model": self.model})
        except httpx.TransportError as exc:
            raise ProviderError(
                f"ollama: no server at {self.host} ({type(exc).__name__}). "
                "Start it with `ollama serve`, or set OLLAMA_HOST.",
                retryable=False,
            ) from exc
        if resp.status_code == 404:
            raise ProviderError(
                f"ollama: model {self.model!r} is not pulled. Run `ollama pull {self.model}`.",
                retryable=False,
                status=404,
            )
        parse_response(resp, provider=self.name)

    async def generate(self, prompt: Prompt, params: GenerationParams) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": prompt.messages(),
            "stream": False,
            "options": self._options(params),
        }
        data = await post_json(self._client, "/api/chat", body, provider=self.name)
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderError("ollama: response has no message content", retryable=False)
        raw = data.get("done_reason")
        raw = raw if isinstance(raw, str) else None
        return Completion(
            text=message["content"],
            model=str(data.get("model") or self.model),
            finish_reason=_FINISH.get(raw or "", "other"),
            raw_finish_reason=raw,
            input_tokens=optional_int(data.get("prompt_eval_count")),
            output_tokens=optional_int(data.get("eval_count")),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
