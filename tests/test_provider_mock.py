import asyncio

import pytest

from evalkit.prompts import Prompt
from evalkit.providers import GenerationParams, MockProvider, ProviderConfigError
from evalkit.schema import parse_suite

P0 = GenerationParams()


def gen(provider: MockProvider, prompt: Prompt, params: GenerationParams = P0):
    return asyncio.run(provider.generate(prompt, params))


def test_placeholder_is_deterministic_and_wordless() -> None:
    a = gen(MockProvider(), Prompt(user="hello"))
    b = gen(MockProvider(), Prompt(user="hello"))
    c = gen(MockProvider(), Prompt(user="hello", system="sys"))
    assert a.text == b.text
    assert a.text != c.text
    assert a.text.startswith("mock-")
    assert len(a.text.split()) == 1
    assert a.finish_reason == "stop"


def test_canned_response_by_prompt() -> None:
    provider = MockProvider(responses={Prompt(user="q"): "the answer is 42"})
    out = gen(provider, Prompt(user="q"))
    assert out.text == "the answer is 42"
    assert out.output_tokens == 4
    assert out.input_tokens == 1


def test_max_tokens_truncates_by_words() -> None:
    provider = MockProvider(responses={Prompt(user="q"): "one two three four"})
    out = gen(provider, Prompt(user="q"), GenerationParams(max_tokens=2))
    assert out.text == "one two"
    assert out.finish_reason == "length"


def test_identity_changes_with_response_table() -> None:
    a = MockProvider(responses={Prompt(user="q"): "x"}).identity()
    b = MockProvider(responses={Prompt(user="q"): "y"}).identity()
    assert a != b
    assert a == MockProvider(responses={Prompt(user="q"): "x"}).identity()


SUITE = parse_suite(
    {
        "name": "t",
        "cases": [
            {"id": "a", "language": "en", "prompt": "Say hi."},
            {"id": "b", "language": "en", "prompt": "Say bye."},
            {"id": "c", "language": "en", "prompt": "Say hi."},  # same prompt as "a"
        ],
    }
)


def test_for_suite_maps_case_ids_to_prompts() -> None:
    provider = MockProvider.for_suite(SUITE, {"a": "hi there", "b": "bye now", "c": "hi there"})
    assert gen(provider, Prompt(user="Say bye.")).text == "bye now"
    assert gen(provider, Prompt(user="Say hi.")).text == "hi there"


def test_for_suite_rejects_conflicting_responses_for_same_prompt() -> None:
    with pytest.raises(ProviderConfigError, match="cases 'a' and 'c' have the same prompt"):
        MockProvider.for_suite(SUITE, {"a": "hi", "c": "hello"})


def test_context_manager_is_a_no_op() -> None:
    async def use() -> str:
        async with MockProvider() as p:
            return (await p.generate(Prompt(user="x"), P0)).text

    assert asyncio.run(use()).startswith("mock-")
