"""Measure the providers and the runner on real local models.

Runs ``examples/suites/everyday.yaml`` (40 cases, 20 EN + 20 ID) through
Ollama for each model with the settings used to build the S1 language corpus
(temperature 0, seed 0, max_tokens 200). Per model, in this order:

    fresh   unload the model, load it with one warm-up request, run (c=1)
    warm    run again on the same, now warm, server (c=1)
    c4      run again with 4 requests in flight
    replay  run again from the response cache
    fresh2  unload, warm up, run (c=1)
    v1      unload, warm up, run through the OpenAI-compatible /v1 endpoint (c=1)

and records:

1. fidelity: ``fresh`` vs the S1 corpus, which a separate, minimal urllib
   client generated (scripts/build_lang_corpus.py);
2. repeatability: ``fresh`` vs ``fresh2`` (both from a fresh load) and
   ``fresh`` vs ``warm`` (same server, later), plus how much the warm re-run
   moves the deterministic-check pass rate;
3. concurrency: wall time and per-request latency at c=1 vs c=4;
4. cache: wall time of a re-run served entirely from the cache;
5. the OpenAI-compatible provider: ``v1`` vs ``fresh`` (both from a fresh load,
   so the endpoint is not confounded with server state).

The warm-up request keeps model loading out of every timed run. Writes a JSON
file and prints a Markdown summary.

Usage (needs Ollama with the models pulled, and a GPU nothing else is using):
    uv run python scripts/measure_runner.py --out docs/results/runner-s2.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evalkit import __version__
from evalkit.cache import ResponseCache
from evalkit.evaluate import CheckSummary, check_suite
from evalkit.prompts import Prompt
from evalkit.providers import GenerationParams, OllamaProvider, OpenAICompatProvider, Provider
from evalkit.runner import RunResult, run_suite
from evalkit.schema import Suite, load_suite

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "examples" / "suites" / "everyday.yaml"
CORPUS = ROOT / "tests" / "fixtures" / "langdetect_heldout.jsonl"
PARAMS = GenerationParams(temperature=0, seed=0, max_tokens=200)  # = the S1 corpus settings
DEFAULT_MODELS = ["qwen2.5:7b", "llama3.1:8b"]


def compare(a: dict[str, str], b: dict[str, str]) -> dict[str, Any]:
    """How many shared ids have byte-identical text, and which differ."""
    differ = [k for k in a if k in b and a[k] != b[k]]
    return {"same": sum(1 for k in a if k in b and a[k] == b[k]), "of": len(a), "differ": differ}


def latency_stats(result: RunResult) -> dict[str, float]:
    lat = sorted(r.latency_s for r in result.records if r.latency_s is not None and not r.cached)
    if not lat:
        return {}
    p95 = lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))]
    return {"p50_s": round(statistics.median(lat), 3), "p95_s": round(p95, 3)}


def run_info(result: RunResult) -> dict[str, Any]:
    _, out_tokens = result.tokens()
    rate = round(out_tokens / result.duration_s, 1) if result.duration_s > 0.1 else None
    return {
        "wall_s": result.duration_s,
        "errors": len(result.errors),
        "cached": result.n_cached,
        "output_tokens": out_tokens,
        "output_tok_per_s": rate,
        **latency_stats(result),
    }


def check_info(summary: CheckSummary) -> dict[str, Any]:
    return {
        "pass_rate": round(summary.pass_rate, 4),
        "pass_rate_by_language": {
            k: round(v, 4) for k, v in summary.pass_rate_by_language().items()
        },
        "counts": {s.value: n for s, n in summary.counts.items()},
        "failure_modes": {m.value: n for m, n in summary.failure_modes().items()},
        "failed_cases": [r.case_id for r in summary.reports if r.status == "fail"],
    }


def corpus_texts(model: str) -> dict[str, str]:
    """S1 corpus responses for ``model``, keyed by the everyday suite's case ids."""
    family = model.split(":")[0]
    out = {}
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["model"] == model:
            out[row["id"].removeprefix(f"{family}-")] = row["text"]
    return out


async def unload(host: str, model: str) -> None:
    """Ask Ollama to unload ``model`` and wait until it is gone."""
    async with httpx.AsyncClient(base_url=host, timeout=60) as client:
        await client.post("/api/generate", json={"model": model, "keep_alive": 0})
        for _ in range(100):
            loaded = (await client.get("/api/ps")).json().get("models", [])
            if not any(m.get("name") == model for m in loaded):
                return
            await asyncio.sleep(0.2)
    raise RuntimeError(f"{model} did not unload")


async def measure_model(model: str, suite: Suite, host: str, cache_root: Path) -> dict[str, Any]:
    async def go(
        provider: Provider, concurrency: int, cache: ResponseCache | None = None
    ) -> RunResult:
        return await run_suite(
            suite, provider, params=PARAMS, concurrency=concurrency, cache=cache, out_dir=None
        )

    async def fresh(provider: Provider) -> None:
        await unload(host, model)
        await provider.generate(Prompt(user="Hi"), PARAMS)  # load the model before timing

    cache = ResponseCache(cache_root / model.replace(":", "-"))  # empty: fresh run fills it
    async with OllamaProvider(model, host=host) as native:
        await native.preflight()
        await fresh(native)
        r_fresh = await go(native, 1, cache)
        r_warm = await go(native, 1)
        r_c4 = await go(native, 4)
        r_replay = await go(native, 4, cache)
        await fresh(native)
        r_fresh2 = await go(native, 1)
    async with OpenAICompatProvider(model, base_url=f"{host}/v1", api_key="") as compat:
        await fresh(compat)
        r_v1 = await go(compat, 1)

    base = r_fresh.responses()
    stripped = {k: v.strip() for k, v in base.items()}  # the corpus script stored .strip()ped text
    checks_fresh = check_suite(suite, base)
    checks_warm = check_suite(suite, r_warm.responses())
    return {
        "runs": {
            "fresh": run_info(r_fresh),
            "warm": run_info(r_warm),
            "concurrency_4": run_info(r_c4),
            "cached_replay": run_info(r_replay),
            "fresh2": run_info(r_fresh2),
            "openai_compat_v1": run_info(r_v1),
        },
        "speedup_c4_vs_c1": round(r_fresh.duration_s / r_c4.duration_s, 2),
        "identical": {
            "fresh_vs_s1_corpus": compare(stripped, corpus_texts(model)),
            "fresh_vs_fresh2": compare(base, r_fresh2.responses()),
            "fresh_vs_warm": compare(base, r_warm.responses()),
            "warm_vs_concurrency_4": compare(r_warm.responses(), r_c4.responses()),
            "openai_v1_vs_native_fresh": compare(base, r_v1.responses()),
        },
        "truncated_at_200_tokens": r_fresh.truncated,
        "checks": {"fresh": check_info(checks_fresh), "warm": check_info(checks_warm)},
        "responses": {"fresh": base, "warm": r_warm.responses()},
    }


def gpu_info() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def markdown(results: dict[str, Any]) -> str:
    models = results["models"]
    rows = ["| | " + " | ".join(f"`{m}`" for m in models) + " |", "|---|" + "---|" * len(models)]

    def row(label: str, fn: Any) -> None:
        rows.append(f"| {label} | " + " | ".join(fn(m) for m in models.values()) + " |")

    def same(key: str) -> Any:
        return lambda m: f"{m['identical'][key]['same']}/{m['identical'][key]['of']}"

    def rate(run: str) -> Any:
        def fmt(m: dict[str, Any]) -> str:
            c = m["checks"][run]
            langs = " / ".join(f"{v:.0%}" for v in c["pass_rate_by_language"].values())
            return f"{c['pass_rate']:.1%} ({langs})"

        return fmt

    row("Identical to the S1 corpus (fresh load)", same("fresh_vs_s1_corpus"))
    row("Identical, fresh load vs another fresh load", same("fresh_vs_fresh2"))
    row("Identical, fresh load vs a 2nd run on the warm server", same("fresh_vs_warm"))
    row("Check pass rate, fresh run (EN / ID)", rate("fresh"))
    row("Check pass rate, warm re-run (EN / ID)", rate("warm"))
    row(
        "OpenAI-compatible `/v1` identical to native (both fresh)",
        same("openai_v1_vs_native_fresh"),
    )
    row("Wall time, 40 cases, concurrency 1", lambda m: f"{m['runs']['fresh']['wall_s']:.1f} s")
    row("Wall time, concurrency 4", lambda m: f"{m['runs']['concurrency_4']['wall_s']:.1f} s")
    row(
        "Median request latency, c=1 / c=4",
        lambda m: (
            f"{m['runs']['fresh']['p50_s']:.2f} / {m['runs']['concurrency_4']['p50_s']:.2f} s"
        ),
    )
    row("Output tokens/s, c=1", lambda m: f"{m['runs']['fresh']['output_tok_per_s']:.0f}")
    row("Re-run from the cache", lambda m: f"{m['runs']['cached_replay']['wall_s'] * 1000:.0f} ms")
    return "\n".join(rows)


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    suite = load_suite(SUITE)
    async with httpx.AsyncClient(base_url=args.host) as client:
        ollama_version = (await client.get("/api/version")).json().get("version")
    results: dict[str, Any] = {
        "meta": {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "evalkit_version": __version__,
            "ollama_version": ollama_version,
            "gpu_at_start": gpu_info(),
            "python": platform.python_version(),
            "suite": f"{suite.name} v{suite.version} ({len(suite.cases)} cases)",
            "params": PARAMS.explicit(),
        },
        "models": {},
    }
    with tempfile.TemporaryDirectory(prefix="evalkit-measure-") as tmp:
        for model in args.models:
            print(f"measuring {model} ...", file=sys.stderr)
            results["models"][model] = await measure_model(model, suite, args.host, Path(tmp))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()
    results = asyncio.run(main_async(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(markdown(results))
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
