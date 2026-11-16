import asyncio
import json
from typing import Any

import httpx
import pytest

from evalkit.prompts import Prompt
from evalkit.providers import (
    GenerationParams,
    OpenAICompatProvider,
    ProviderConfigError,
    ProviderError,
)

BASE = "http://llm.test:8000/v1"


def ok(content: Any = "Hello!", finish: Any = "stop", **message: Any) -> dict[str, Any]:
    return {
        "model": "served-name",
        "choices": [
            {
                "message": {"role": "assistant", "content": content, **message},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 2},
    }


def generate(handler, *, api_key: str | None = None, params: GenerationParams | None = None):
    async def go():
        async with OpenAICompatProvider(
            "my-model", base_url=BASE + "/", api_key=api_key, transport=httpx.MockTransport(handler)
        ) as p:
            return await p.generate(
                Prompt(user="Hi", system="Be brief."), params or GenerationParams()
            )

    return asyncio.run(go())


def test_request_shape_and_auth_header() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=ok())

    out = generate(
        handler, api_key="sk-test", params=GenerationParams(temperature=0, seed=3, max_tokens=20)
    )
    assert seen["url"] == BASE + "/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"] == {
        "model": "my-model",
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        "temperature": 0,
        "seed": 3,
        "max_tokens": 20,
    }
    assert out.text == "Hello!"
    assert out.model == "served-name"
    assert (out.input_tokens, out.output_tokens) == (9, 2)
    assert out.finish_reason == "stop"


def test_no_auth_header_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json=ok())

    generate(handler)
    assert seen["auth"] is None


@pytest.mark.parametrize(
    ("finish", "expected"),
    [("length", "length"), ("content_filter", "refusal"), ("tool_calls", "other"), (None, "other")],
)
def test_finish_reasons(finish: Any, expected: str) -> None:
    out = generate(lambda req: httpx.Response(200, json=ok(finish=finish)))
    assert out.finish_reason == expected


def test_refusal_field() -> None:
    out = generate(
        lambda req: httpx.Response(200, json=ok(content=None, refusal="I can't help with that."))
    )
    assert out.text == "I can't help with that."
    assert out.finish_reason == "refusal"


def test_null_content_without_refusal() -> None:
    out = generate(lambda req: httpx.Response(200, json=ok(content=None, finish="length")))
    assert out.text == ""
    assert out.finish_reason == "length"


def test_missing_usage() -> None:
    body = ok()
    del body["usage"]
    out = generate(lambda req: httpx.Response(200, json=body))
    assert out.input_tokens is None


@pytest.mark.parametrize(
    "body",
    [
        {"choices": []},
        {"choices": "x"},
        {"choices": [{"finish_reason": "stop"}]},
        {"choices": ["x"]},
    ],
)
def test_malformed(body: dict[str, Any]) -> None:
    with pytest.raises(ProviderError) as info:
        generate(lambda req: httpx.Response(200, json=body))
    assert info.value.retryable is False


def test_openai_error_shape() -> None:
    body = {"error": {"message": "Rate limit reached", "type": "requests"}}
    with pytest.raises(ProviderError, match="HTTP 429: Rate limit reached") as info:
        generate(lambda req: httpx.Response(429, json=body))
    assert info.value.retryable is True


class TestConfig:
    def test_key_required_for_openai_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY is not set"):
            OpenAICompatProvider("gpt-4o-mini")

    def test_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1/")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        p = OpenAICompatProvider("qwen2.5:7b")
        assert p.base_url == "http://127.0.0.1:11434/v1"
        assert p.identity() == {
            "provider": "openai",
            "model": "qwen2.5:7b",
            "base_url": "http://127.0.0.1:11434/v1",
        }

    def test_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        assert OpenAICompatProvider("gpt-4o-mini").base_url == "https://api.openai.com/v1"
