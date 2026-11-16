"""The Anthropic provider, run through the real SDK against a mock transport."""

import asyncio
import json
from typing import Any

import anthropic
import httpx2
import pytest

from evalkit.prompts import Prompt
from evalkit.providers import (
    AnthropicProvider,
    GenerationParams,
    ProviderConfigError,
    ProviderError,
)
from evalkit.providers.anthropic_api import DEFAULT_MAX_TOKENS


def message(
    *, stop: str = "end_turn", content: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content
        if content is not None
        else [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Halo, "},
            {"type": "text", "text": "apa kabar?"},
        ],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 5},
    }


def error(
    status: int, kind: str, msg: str, headers: dict[str, str] | None = None
) -> httpx2.Response:
    return httpx2.Response(
        status, headers=headers, json={"type": "error", "error": {"type": kind, "message": msg}}
    )


def generate(
    handler,
    *,
    params: GenerationParams | None = None,
    prompt: Prompt | None = None,
    api_key: str | None = "sk-test",
):
    async def go():
        client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
        async with AnthropicProvider("claude-opus-5-5", api_key=api_key, http_client=client) as p:
            return await p.generate(
                prompt or Prompt(user="Halo", system="Jawab singkat."), params or GenerationParams()
            )

    return asyncio.run(go())


def test_request_has_no_sampling_params() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        seen["version"] = req.headers.get("anthropic-version")
        return httpx2.Response(200, json=message())

    params = GenerationParams(temperature=0, top_p=0.5, seed=1, max_tokens=256, stop=("END",))
    out = generate(handler, params=params)
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["version"] == "2026-10-21"
    assert seen["body"] == {
        "model": "claude-opus-5-5",
        "max_tokens": 256,
        "system": "Jawab singkat.",
        "stop_sequences": ["END"],
        "messages": [{"role": "user", "content": "Halo"}],
    }
    # Thinking blocks are dropped; text blocks are joined.
    assert out.text == "Halo, apa kabar?"
    assert out.finish_reason == "stop"
    assert (out.input_tokens, out.output_tokens) == (12, 5)


def test_identity_includes_the_endpoint() -> None:
    direct = AnthropicProvider("claude-opus-5-5", api_key="k")
    proxied = AnthropicProvider("claude-opus-5-5", api_key="k", base_url="https://gw.example.com/")
    assert direct.identity() == {
        "provider": "anthropic",
        "model": "claude-opus-5-5",
        "base_url": "https://api.anthropic.com",
    }
    assert proxied.identity()["base_url"] == "https://gw.example.com"


def test_scheme_less_base_url_is_a_config_error() -> None:
    with pytest.raises(ProviderConfigError, match="must start with http"):
        AnthropicProvider("claude-opus-5-5", api_key="k", base_url="gw.example.com")


def test_defaults_omit_system_and_stop() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx2.Request) -> httpx2.Response:
        seen["body"] = json.loads(req.content)
        return httpx2.Response(200, json=message())

    generate(handler, prompt=Prompt(user="Hi"))
    assert seen["body"] == {
        "model": "claude-opus-5-5",
        "max_tokens": DEFAULT_MAX_TOKENS,
        "messages": [{"role": "user", "content": "Hi"}],
    }


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
        ("refusal", "refusal"),
        ("stop_sequence", "stop"),
        ("pause_turn", "other"),
    ],
)
def test_stop_reasons(stop: str, expected: str) -> None:
    out = generate(lambda req: httpx2.Response(200, json=message(stop=stop, content=[])))
    assert out.finish_reason == expected
    assert out.raw_finish_reason == stop
    assert out.text == ""


@pytest.mark.parametrize(
    ("response", "retryable", "retry_after"),
    [
        (error(429, "rate_limit_error", "slow down", {"retry-after": "7"}), True, 7.0),
        (error(529, "overloaded_error", "overloaded"), True, None),
        (error(500, "api_error", "internal"), True, None),
        (error(400, "invalid_request_error", "bad model"), False, None),
        (error(401, "authentication_error", "invalid x-api-key"), False, None),
    ],
)
def test_status_errors(
    response: httpx2.Response, retryable: bool, retry_after: float | None
) -> None:
    with pytest.raises(ProviderError) as info:
        generate(lambda req: response)
    err = info.value
    assert err.retryable is retryable
    assert err.status == response.status_code
    assert err.retry_after == retry_after
    assert json.loads(response.content)["error"]["message"] in str(err)


def test_status_error_without_json_body() -> None:
    with pytest.raises(ProviderError, match="HTTP 502"):
        generate(lambda req: httpx2.Response(502, text="bad gateway"))


def test_connection_error_is_retryable() -> None:
    def handler(req: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=req)

    with pytest.raises(ProviderError) as info:
        generate(handler)
    assert info.value.retryable is True


def test_missing_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    # Keep the SDK from finding a real `ant auth login` profile on this machine.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    with pytest.raises(ProviderError, match="no credentials found") as info:
        generate(lambda req: httpx2.Response(200, json=message()), api_key=None)
    assert info.value.retryable is False


def test_other_type_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(*args: Any, **kwargs: Any) -> None:
        raise TypeError("something else")

    async def go() -> None:
        p = AnthropicProvider("claude-opus-5-5", api_key="k")
        monkeypatch.setattr(p._client.messages, "create", broken)
        await p.generate(Prompt(user="x"), GenerationParams())

    with pytest.raises(TypeError, match="something else"):
        asyncio.run(go())
