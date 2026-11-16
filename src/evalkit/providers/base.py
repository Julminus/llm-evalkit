"""The provider interface: one async call that turns a prompt into a completion.

Every provider (mock, Ollama, OpenAI-compatible, Anthropic) takes the same
:class:`Prompt` and :class:`GenerationParams` and returns the same
:class:`Completion`, so the runner, the cache and the reports never need to
know which backend produced a response.

Errors are normalised too: a provider raises :class:`ProviderError` with a
``retryable`` flag, and the runner alone decides whether to try again. No
provider retries on its own, so retries never multiply across layers.
"""

from __future__ import annotations

import email.utils
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from evalkit.prompts import Prompt

__all__ = [
    "RETRYABLE_STATUS",
    "Completion",
    "FinishReason",
    "GenerationParams",
    "Prompt",
    "Provider",
    "ProviderConfigError",
    "ProviderError",
    "parse_retry_after",
]

FinishReason = Literal["stop", "length", "refusal", "other"]
"""Why generation ended, normalised across providers.

``length`` matters for evaluation: a response cut off at the token limit can
fail a check for a reason that has nothing to do with the model's ability.
"""

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
"""HTTP statuses worth retrying: timeouts, conflicts, rate limits and server
errors (529 is Anthropic's "overloaded"). Any other 4xx is a bug in the
request or the configuration, and retrying it would only repeat the error."""


class GenerationParams(BaseModel):
    """Sampling settings. ``None`` means "use the provider's default".

    Not every provider accepts every setting (the Anthropic API no longer
    accepts ``temperature``, ``top_p`` or ``seed``); see
    :meth:`Provider.ignored_params`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None
    stop: tuple[str, ...] = ()

    def explicit(self) -> dict[str, Any]:
        """The settings that were actually set (not ``None`` / empty)."""
        data = self.model_dump(exclude_defaults=True)
        if "stop" in data:
            data["stop"] = list(data["stop"])
        return data


class Completion(BaseModel):
    """What a provider returns for one prompt."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    """The model name the server reports. Usually the requested one; it can
    differ when the server resolves an alias."""
    finish_reason: FinishReason
    raw_finish_reason: str | None = None
    """The provider's own value (``end_turn``, ``done``, ``content_filter``...)."""
    input_tokens: int | None = None
    output_tokens: int | None = None


class ProviderError(Exception):
    """A request failed. ``retryable`` says whether trying again could help."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after
        """Seconds the server asked us to wait (``Retry-After``), if it said."""


class ProviderConfigError(Exception):
    """A provider cannot be built: unknown name, missing key or missing extra.

    Raised before any request is sent. The message is user-facing.
    """


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse a ``Retry-After`` header: delta-seconds or an HTTP date."""
    if value is None or not value.strip():
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:  # pragma: no cover - parsedate_to_datetime sets UTC for "GMT"
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


class Provider(ABC):
    """Base class for model backends.

    Subclasses set :attr:`name` and :attr:`supported_params` and implement
    :meth:`generate`. Providers hold an HTTP client, so use them as async
    context managers (or call :meth:`aclose`).
    """

    name: ClassVar[str]
    supported_params: ClassVar[frozenset[str]]

    def __init__(self, model: str) -> None:
        if not model:
            raise ProviderConfigError(f"{self.name}: a model name is required")
        self.model = model

    @property
    def spec(self) -> str:
        """``name:model``, the same string the CLI accepts."""
        return f"{self.name}:{self.model}"

    def identity(self) -> dict[str, str]:
        """Everything except the prompt and params that decides the output.

        Part of the cache key. Subclasses add fields such as an endpoint URL.
        """
        return {"provider": self.name, "model": self.model}

    def effective_params(self, params: GenerationParams) -> dict[str, Any]:
        """The settings this provider will send: explicit ones it supports."""
        return {k: v for k, v in params.explicit().items() if k in self.supported_params}

    def ignored_params(self, params: GenerationParams) -> list[str]:
        """Explicit settings this provider cannot apply, so the user can be told."""
        return sorted(k for k in params.explicit() if k not in self.supported_params)

    async def preflight(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Fail fast on configuration problems before a run starts.

        Raises :class:`ProviderError`. The default does nothing; hosted
        providers keep it that way so a check never costs a paid request.
        """

    @abstractmethod
    async def generate(self, prompt: Prompt, params: GenerationParams) -> Completion:
        """Send one prompt and return the completion, or raise ProviderError."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release network resources."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
