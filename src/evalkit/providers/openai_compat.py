"""Any server that speaks the OpenAI Chat Completions API.

That covers OpenAI itself and most local and hosted inference servers: vLLM,
llama.cpp's server, LM Studio, and Ollama's own ``/v1`` endpoint. The key is
optional, because local servers usually don't need one.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

import httpx

from evalkit.providers._http import optional_int, post_json
from evalkit.providers.base import (
    Completion,
    FinishReason,
    GenerationParams,
    Prompt,
    Provider,
    ProviderConfigError,
    ProviderError,
)

__all__ = ["DEFAULT_OPENAI_BASE_URL", "OpenAICompatProvider"]

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

_FINISH: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "content_filter": "refusal",
}


class OpenAICompatProvider(Provider):
    """``POST {base_url}/chat/completions``.

    ``base_url`` comes from the argument, else ``OPENAI_BASE_URL``, else
    OpenAI's API. ``api_key`` comes from the argument, else ``OPENAI_API_KEY``;
    it is required only for ``api.openai.com``. Sends ``max_tokens``, which
    compatible servers accept; OpenAI's reasoning models want
    ``max_completion_tokens`` instead and are not supported here.
    """

    name = "openai"
    supported_params = frozenset({"temperature", "top_p", "max_tokens", "seed", "stop"})

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(model)
        url = base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
        self.base_url = url.strip().rstrip("/")
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not key and urlsplit(self.base_url).hostname == "api.openai.com":
            raise ProviderConfigError(
                "openai: OPENAI_API_KEY is not set. Set it (see .env.example), or point "
                "OPENAI_BASE_URL at a local server such as http://127.0.0.1:11434/v1."
            )
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
        )

    def identity(self) -> dict[str, str]:
        # Two servers can serve different weights under the same model name.
        return {**super().identity(), "base_url": self.base_url}

    async def generate(self, prompt: Prompt, params: GenerationParams) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": prompt.messages(),
            **self.effective_params(params),
        }
        data = await post_json(self._client, "/chat/completions", body, provider=self.name)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderError("openai: response has no choices", retryable=False)
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ProviderError("openai: response choice has no message", retryable=False)
        content = message.get("content")
        refusal = message.get("refusal")
        raw = choice.get("finish_reason")
        raw = raw if isinstance(raw, str) else None
        if isinstance(content, str) and content:
            text, finish = content, _FINISH.get(raw or "", "other")
        elif isinstance(refusal, str) and refusal:
            text, finish = refusal, "refusal"
        else:
            text, finish = "", _FINISH.get(raw or "", "other")
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        return Completion(
            text=text,
            model=str(data.get("model") or self.model),
            finish_reason=finish,
            raw_finish_reason=raw,
            input_tokens=optional_int(usage.get("prompt_tokens")),
            output_tokens=optional_int(usage.get("completion_tokens")),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
