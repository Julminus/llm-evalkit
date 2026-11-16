import subprocess
import sys
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from evalkit.prompts import Prompt, build_prompt
from evalkit.providers import (
    Completion,
    GenerationParams,
    MockProvider,
    ProviderConfigError,
    make_provider,
    parse_spec,
)
from evalkit.providers.base import parse_retry_after
from evalkit.schema import EvalCase


class TestPrompt:
    def test_messages_without_system(self) -> None:
        assert Prompt(user="hi").messages() == [{"role": "user", "content": "hi"}]

    def test_messages_with_system(self) -> None:
        assert Prompt(user="hi", system="be brief").messages() == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]

    def test_hashable_and_equal_by_value(self) -> None:
        assert {Prompt(user="a"): 1}[Prompt(user="a")] == 1


class TestBuildPrompt:
    def test_plain_case(self) -> None:
        case = EvalCase(id="c1", language="en", prompt="Say hi.", system="Be kind.")
        assert build_prompt(case) == Prompt(user="Say hi.", system="Be kind.")

    def test_context_goes_first_in_language_neutral_tags(self) -> None:
        case = EvalCase(id="c1", language="id", prompt="Kapan?", context="  Pasal 2: 14 hari.\n")
        assert build_prompt(case).user == "<context>\nPasal 2: 14 hari.\n</context>\n\nKapan?"


class TestGenerationParams:
    def test_explicit_drops_unset(self) -> None:
        assert GenerationParams().explicit() == {}
        assert GenerationParams(temperature=0, stop=("END",)).explicit() == {
            "temperature": 0,
            "stop": ["END"],
        }

    @pytest.mark.parametrize(
        "kwargs",
        [{"temperature": -0.1}, {"temperature": 2.5}, {"top_p": 0}, {"max_tokens": 0}],
    )
    def test_rejects_out_of_range(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValidationError):
            GenerationParams(**kwargs)

    def test_rejects_unknown_setting(self) -> None:
        with pytest.raises(ValidationError):
            GenerationParams(temprature=0)  # type: ignore[call-arg]


class TestRetryAfter:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(None, None), ("", None), ("7", 7.0), (" 1.5 ", 1.5), ("-3", 0.0), ("soon", None)],
    )
    def test_seconds(self, value: str | None, expected: float | None) -> None:
        assert parse_retry_after(value) == expected

    def test_http_date(self) -> None:
        now = datetime(2026, 11, 16, 12, 0, 0, tzinfo=UTC)
        assert parse_retry_after("Mon, 16 Nov 2026 12:00:30 GMT", now=now) == 30.0
        assert parse_retry_after("Mon, 16 Nov 2026 11:00:00 GMT", now=now) == 0.0


class TestSpec:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("ollama:qwen2.5:7b", ("ollama", "qwen2.5:7b")),
            ("OpenAI:gpt-4o-mini", ("openai", "gpt-4o-mini")),
            ("anthropic:claude-opus-5-5", ("anthropic", "claude-opus-5-5")),
            ("mock", ("mock", "default")),
            ("mock:fixed", ("mock", "fixed")),
        ],
    )
    def test_parse(self, spec: str, expected: tuple[str, str]) -> None:
        assert parse_spec(spec) == expected

    def test_unknown_provider(self) -> None:
        with pytest.raises(ProviderConfigError, match="unknown provider 'gpt'"):
            parse_spec("gpt:4")

    def test_missing_model_gives_example(self) -> None:
        with pytest.raises(ProviderConfigError, match=r"e\.g\. ollama:qwen2\.5:7b"):
            parse_spec("ollama")

    def test_make_provider_types(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert make_provider("mock").spec == "mock:default"
        assert make_provider("ollama:qwen2.5:7b", base_url="localhost").spec == "ollama:qwen2.5:7b"
        assert make_provider("openai:local-model").spec == "openai:local-model"
        assert make_provider("anthropic:claude-opus-5-5").spec == "anthropic:claude-opus-5-5"

    def test_base_class_requires_model(self) -> None:
        with pytest.raises(ProviderConfigError, match="model name is required"):
            MockProvider("")


def test_completion_text_never_holds_a_lone_surrogate() -> None:
    # A JSON escape like "\ud83d" from a server would otherwise crash the run log.
    completion = Completion(text="ok \ud83d!", model="m", finish_reason="stop")
    assert completion.text == "ok \N{REPLACEMENT CHARACTER}!"
    assert Completion(text="ok \U0001f600", model="m", finish_reason="stop").text == "ok \U0001f600"


def test_effective_and_ignored_params() -> None:
    from evalkit.providers import AnthropicProvider

    params = GenerationParams(temperature=0, seed=1, max_tokens=64)
    claude = AnthropicProvider("claude-opus-5-5", api_key="k")
    assert claude.effective_params(params) == {"max_tokens": 64}
    assert claude.ignored_params(params) == ["seed", "temperature"]
    assert MockProvider().ignored_params(params) == []
    # The cap the API requires is part of the sent settings, so manifests show it.
    assert claude.effective_params(GenerationParams()) == {"max_tokens": 16000}


def test_cli_does_not_import_the_anthropic_sdk() -> None:
    # The SDK takes ~0.6 s to import; every command paid for it before.
    # A fresh interpreter, because this test session has already imported it.
    code = "import sys, evalkit.cli; print('anthropic' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_unknown_attribute_still_raises() -> None:
    import evalkit.providers

    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        _ = evalkit.providers.Nope  # type: ignore[attr-defined]
