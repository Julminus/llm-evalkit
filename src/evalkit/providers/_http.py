"""Shared JSON-over-HTTP plumbing for the Ollama and OpenAI-compatible providers.

Turns every transport or HTTP failure into a :class:`ProviderError` with the
right ``retryable`` flag, so the providers themselves only deal with the
happy-path response shape.
"""

from __future__ import annotations

from typing import Any

import httpx

from evalkit.providers.base import RETRYABLE_STATUS, ProviderError, parse_retry_after

_MAX_DETAIL = 300


def error_detail(resp: httpx.Response) -> str:
    """The most useful one-line explanation in an error response body."""
    try:
        data = resp.json()
    except ValueError:
        text = resp.text.strip()
        return text[:_MAX_DETAIL] if text else resp.reason_phrase
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, str):  # Ollama: {"error": "model 'x' not found"}
            return err[:_MAX_DETAIL]
        if isinstance(err, dict) and isinstance(err.get("message"), str):  # OpenAI shape
            return str(err["message"])[:_MAX_DETAIL]
    return str(data)[:_MAX_DETAIL]


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    *,
    provider: str,
) -> dict[str, Any]:
    """POST ``body`` as JSON and return the decoded JSON object."""
    try:
        resp = await client.post(url, json=body)
    except httpx.TimeoutException as exc:
        raise ProviderError(f"{provider}: request timed out", retryable=True) from exc
    except httpx.TransportError as exc:  # connection refused, reset, DNS, protocol errors
        raise ProviderError(
            f"{provider}: cannot reach {client.base_url} ({type(exc).__name__})", retryable=True
        ) from exc
    return parse_response(resp, provider=provider)


def parse_response(resp: httpx.Response, *, provider: str) -> dict[str, Any]:
    """Raise ProviderError for an error status, else return the JSON object."""
    if resp.status_code >= 400:
        raise ProviderError(
            f"{provider}: HTTP {resp.status_code}: {error_detail(resp)}",
            retryable=resp.status_code in RETRYABLE_STATUS,
            status=resp.status_code,
            retry_after=parse_retry_after(resp.headers.get("retry-after")),
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise ProviderError(f"{provider}: response is not JSON", retryable=False) from exc
    if not isinstance(data, dict):
        raise ProviderError(f"{provider}: expected a JSON object in the response", retryable=False)
    return data


def optional_int(value: object) -> int | None:
    """Token counts are optional in every API; keep them only if they are ints."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None
