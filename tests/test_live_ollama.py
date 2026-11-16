"""Live tests against a real Ollama server. Opt-in: they need a GPU and a pulled model.

    EVALKIT_LIVE_OLLAMA=1 uv run pytest -m live
    EVALKIT_LIVE_MODEL=llama3.1:8b EVALKIT_LIVE_OLLAMA=1 uv run pytest -m live

Comparisons start each run from a fresh model load. At temperature 0 Ollama's
output also depends on what the server processed since the model was loaded
(see the README's Results), so two back-to-back runs on a warm server are
not guaranteed to match, and a test that compared them would be flaky.
"""

import asyncio
import os

import httpx
import pytest

from evalkit.prompts import Prompt
from evalkit.providers import GenerationParams, OllamaProvider, OpenAICompatProvider, Provider
from evalkit.runner import RunResult, run_suite
from evalkit.schema import parse_suite

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("EVALKIT_LIVE_OLLAMA") != "1", reason="set EVALKIT_LIVE_OLLAMA=1"
    ),
]

MODEL = os.environ.get("EVALKIT_LIVE_MODEL", "qwen2.5:7b")
HOST = OllamaProvider(MODEL).host  # same OLLAMA_HOST parsing as the provider
PARAMS = GenerationParams(temperature=0, seed=0, max_tokens=60)
SUITE = parse_suite(
    {
        "name": "live",
        "cases": [
            {"id": "en", "language": "en", "prompt": "Name one planet in one sentence."},
            {"id": "id", "language": "id", "prompt": "Sebutkan satu planet dalam satu kalimat."},
        ],
    }
)


async def unload() -> None:
    async with httpx.AsyncClient(base_url=HOST, timeout=60) as client:
        await client.post("/api/generate", json={"model": MODEL, "keep_alive": 0})
        for _ in range(100):
            loaded = (await client.get("/api/ps")).json().get("models", [])
            if not any(m.get("name") == MODEL for m in loaded):
                return
            await asyncio.sleep(0.2)
    raise AssertionError(f"{MODEL} did not unload")


async def fresh_run(provider: Provider) -> RunResult:
    await unload()
    return await run_suite(SUITE, provider, params=PARAMS, concurrency=1, out_dir=None)


def test_preflight_and_generate() -> None:
    async def go():
        async with OllamaProvider(MODEL, host=HOST) as p:
            await p.preflight()
            return await p.generate(Prompt(user="Reply with the word: ok"), PARAMS)

    out = asyncio.run(go())
    assert out.text.strip()
    assert out.output_tokens


def test_fresh_loads_give_identical_responses() -> None:
    async def go():
        async with OllamaProvider(MODEL, host=HOST) as p:
            return await fresh_run(p), await fresh_run(p)

    a, b = asyncio.run(go())
    assert not a.errors
    assert a.responses() == b.responses()


def test_openai_compatible_endpoint_matches_native() -> None:
    async def go():
        async with OllamaProvider(MODEL, host=HOST) as native:
            n = await fresh_run(native)
        async with OpenAICompatProvider(MODEL, base_url=f"{HOST}/v1", api_key="") as compat:
            c = await fresh_run(compat)
        return n, c

    n, c = asyncio.run(go())
    assert not n.errors
    assert n.responses() == c.responses()
