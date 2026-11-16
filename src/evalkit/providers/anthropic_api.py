"""Claude models through the official ``anthropic`` SDK (optional extra).

Install with ``pip install "llm-evalkit[anthropic]"``. Credentials are found
the way the SDK finds them: ``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN`` or
an ``ant auth login`` profile.

Two deliberate choices:

* **No sampling settings.** API version ``2026-10-21`` (sent by SDK 2.x)
  rejects ``temperature``, ``top_p`` and ``top_k`` on every model, and there
  is no ``seed``. Claude runs therefore sample at the API's defaults and are
  not reproducible run to run; the response cache is what makes a re-run
  return the same text. The CLI says so when these settings are given.
* **No refusal fallback.** The API can re-run a refused request on another
  model. In an evaluation that would replace the refusal of the model under
  test with a different model's answer, and over-refusal is one of the failure
  modes being measured, so a refusal is returned as a result
  (``finish_reason="refusal"``).
"""

from __future__ import annotations

from typing import Any

from evalkit.providers.base import (
    RETRYABLE_STATUS,
    Completion,
    FinishReason,
    GenerationParams,
    Prompt,
    Provider,
    ProviderConfigError,
    ProviderError,
    parse_retry_after,
)

try:
    import anthropic

    _HAVE_SDK = True
except ImportError:  # pragma: no cover - only without the [anthropic] extra
    _HAVE_SDK = False

__all__ = ["DEFAULT_MAX_TOKENS", "AnthropicProvider"]

DEFAULT_MAX_TOKENS = 16000
"""Used when ``max_tokens`` isn't set. Current Claude models think before they
answer and the thinking counts toward this limit, so a small value would cut
answers off; 16k keeps a non-streaming request well inside the SDK timeout."""

_FINISH: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "refusal": "refusal",
}


def _status_message(exc: anthropic.APIStatusError) -> str:
    body = exc.body
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return str(err["message"])
    return exc.message


class AnthropicProvider(Provider):
    """``client.messages.create`` with SDK retries off (the runner retries)."""

    name = "anthropic"
    supported_params = frozenset({"max_tokens", "stop"})

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 300.0,
        http_client: Any = None,
    ) -> None:
        super().__init__(model)
        if not _HAVE_SDK:  # pragma: no cover - only without the [anthropic] extra
            raise ProviderConfigError(
                "anthropic: the SDK is not installed. "
                'Install the extra: pip install "llm-evalkit[anthropic]"'
            )
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )

    async def generate(self, prompt: Prompt, params: GenerationParams) -> Completion:
        p = self.effective_params(params)
        try:
            msg = await self._client.messages.create(
                model=self.model,
                max_tokens=p.get("max_tokens", DEFAULT_MAX_TOKENS),
                system=prompt.system if prompt.system else anthropic.omit,
                stop_sequences=p.get("stop", anthropic.omit),
                messages=[{"role": "user", "content": prompt.user}],
            )
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"anthropic: HTTP {exc.status_code}: {_status_message(exc)}",
                retryable=exc.status_code in RETRYABLE_STATUS,
                status=exc.status_code,
                retry_after=parse_retry_after(exc.response.headers.get("retry-after")),
            ) from exc
        except anthropic.APIConnectionError as exc:  # includes APITimeoutError
            raise ProviderError(f"anthropic: {exc}", retryable=True) from exc
        except TypeError as exc:
            # The SDK resolves credentials lazily and raises TypeError at
            # request time when it finds none.
            if "authentication" not in str(exc):
                raise
            raise ProviderError(
                "anthropic: no credentials found. Set ANTHROPIC_API_KEY (see .env.example) "
                "or run `ant auth login`.",
                retryable=False,
            ) from exc
        # Thinking blocks (empty by default) and any other block types are not
        # part of the answer being evaluated.
        text = "".join(block.text for block in msg.content if block.type == "text")
        raw = msg.stop_reason
        return Completion(
            text=text,
            model=msg.model,
            finish_reason=_FINISH.get(raw or "", "other"),
            raw_finish_reason=raw,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )

    async def aclose(self) -> None:
        await self._client.close()
