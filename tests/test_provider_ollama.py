import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from evalkit.prompts import Prompt
from evalkit.providers import GenerationParams, OllamaProvider, ProviderError
from evalkit.providers.ollama import normalize_host

Handler = Callable[[httpx.Request], httpx.Response]

OK_BODY = {
    "model": "qwen2.5:7b",
    "message": {"role": "assistant", "content": "Halo!"},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 21,
    "eval_count": 3,
}


def provider(handler: Handler, **kwargs: Any) -> OllamaProvider:
    return OllamaProvider(
        "qwen2.5:7b",
        host="http://ollama.test:11434",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def generate(
    p: OllamaProvider, params: GenerationParams | None = None, prompt: Prompt | None = None
):
    async def go():
        async with p:
            return await p.generate(
                prompt or Prompt(user="Halo", system="Singkat."), params or GenerationParams()
            )

    return asyncio.run(go())


def test_request_shape_and_option_mapping() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=OK_BODY)

    params = GenerationParams(temperature=0, seed=7, max_tokens=50, top_p=0.9, stop=("###",))
    out = generate(provider(handler), params)
    assert seen["url"] == "http://ollama.test:11434/api/chat"
    assert seen["body"] == {
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "system", "content": "Singkat."},
            {"role": "user", "content": "Halo"},
        ],
        "stream": False,
        "options": {"temperature": 0, "seed": 7, "num_predict": 50, "top_p": 0.9, "stop": ["###"]},
    }
    assert out.text == "Halo!"
    assert out.finish_reason == "stop"
    assert (out.input_tokens, out.output_tokens) == (21, 3)
    assert out.model == "qwen2.5:7b"


def test_unset_params_are_not_sent() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=OK_BODY)

    generate(provider(handler))
    assert seen["body"]["options"] == {}


@pytest.mark.parametrize(
    ("raw", "expected"), [("length", "length"), ("load", "other"), (None, "other")]
)
def test_finish_reason_mapping(raw: str | None, expected: str) -> None:
    body = {**OK_BODY, "done_reason": raw}
    out = generate(provider(lambda req: httpx.Response(200, json=body)))
    assert out.finish_reason == expected
    assert out.raw_finish_reason == raw


def test_missing_token_counts_are_none() -> None:
    body = {"message": {"content": "x"}, "done_reason": "stop", "eval_count": True}
    out = generate(provider(lambda req: httpx.Response(200, json=body)))
    assert out.input_tokens is None
    assert out.output_tokens is None  # a bool is not a token count
    assert out.model == "qwen2.5:7b"  # falls back to the requested model


@pytest.mark.parametrize(
    ("status", "retryable"), [(404, False), (400, False), (429, True), (500, True), (503, True)]
)
def test_http_errors_are_classified(status: int, retryable: bool) -> None:
    handler = lambda req: httpx.Response(status, json={"error": "model 'x' not found"})  # noqa: E731
    with pytest.raises(ProviderError) as info:
        generate(provider(handler))
    assert info.value.retryable is retryable
    assert info.value.status == status
    assert "model 'x' not found" in str(info.value)


def test_retry_after_is_passed_through() -> None:
    handler = lambda req: httpx.Response(429, headers={"Retry-After": "4"}, text="busy")  # noqa: E731
    with pytest.raises(ProviderError) as info:
        generate(provider(handler))
    assert info.value.retry_after == 4.0
    assert "busy" in str(info.value)


@pytest.mark.parametrize(
    "exc",
    [httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), httpx.RemoteProtocolError("x")],
)
def test_transport_errors_are_retryable(exc: Exception) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise exc

    with pytest.raises(ProviderError) as info:
        generate(provider(handler))
    assert info.value.retryable is True


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=["a", "list"]),
        httpx.Response(200, json={"done": True}),
        httpx.Response(200, json={"message": {"content": None}}),
    ],
)
def test_malformed_responses_are_not_retryable(response: httpx.Response) -> None:
    with pytest.raises(ProviderError) as info:
        generate(provider(lambda req: response))
    assert info.value.retryable is False


def test_error_detail_falls_back_to_reason_phrase() -> None:
    with pytest.raises(ProviderError, match="HTTP 502: Bad Gateway"):
        generate(provider(lambda req: httpx.Response(502)))


def test_error_detail_with_unexpected_json() -> None:
    with pytest.raises(ProviderError, match=r"HTTP 500: \{'oops': 1\}"):
        generate(provider(lambda req: httpx.Response(500, json={"oops": 1})))


class TestPreflight:
    def run(self, handler: Handler) -> None:
        async def go():
            async with provider(handler) as p:
                await p.preflight()

        asyncio.run(go())

    def test_ok(self) -> None:
        seen: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["path"] = req.url.path
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json={"details": {}})

        self.run(handler)
        assert seen == {"path": "/api/show", "body": {"model": "qwen2.5:7b"}}

    def test_model_not_pulled(self) -> None:
        with pytest.raises(ProviderError, match=r"Run `ollama pull qwen2\.5:7b`"):
            self.run(lambda req: httpx.Response(404, json={"error": "not found"}))

    def test_server_down(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        with pytest.raises(ProviderError, match="ollama serve"):
            self.run(handler)

    def test_other_error(self) -> None:
        with pytest.raises(ProviderError, match="HTTP 500"):
            self.run(lambda req: httpx.Response(500, json={"error": "boom"}))


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("localhost", "http://localhost:11434"),
        ("localhost:8080", "http://localhost:8080"),
        ("0.0.0.0", "http://127.0.0.1:11434"),
        ("http://0.0.0.0:11434/", "http://127.0.0.1:11434"),
        ("https://ollama.example.com", "https://ollama.example.com"),
        ("  http://box:11434/  ", "http://box:11434"),
    ],
)
def test_normalize_host(given: str, expected: str) -> None:
    assert normalize_host(given) == expected


def test_host_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box:11500")
    assert OllamaProvider("m").host == "http://gpu-box:11500"
    monkeypatch.delenv("OLLAMA_HOST")
    assert OllamaProvider("m").host == "http://127.0.0.1:11434"
