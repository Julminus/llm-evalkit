"""Measure how far to trust the judge: position bias, self-consistency, judge vs judge.

Input: the 80 responses saved by the S2 measurement (``docs/results/runner-s2.json``,
``responses.fresh``: qwen2.5:7b and llama3.1:8b on the 40-case everyday suite), and
the S3 judgments of those responses (``docs/results/judge-s3-prompt-v1.json``).

Conditions, each from a fresh model load (unload every model, load the judge
with one warm-up request), concurrency 1, no cache, max 1024 tokens:

    pairwise_<judge>      A = qwen2.5:7b, B = llama3.1:8b, 40 pairs, both orders;
                          temperature 0, seed 0
    control_<judge>       each model's responses against themselves (80 pairs):
                          the only right verdict is a tie
    consistency_<judge>   the rubric judge on all 80 responses, 5 samples each,
                          temperature 0.7, seeds 0-4

for the judges qwen2.5:14b and llama3.1:8b. Judge-vs-judge agreement needs no
GPU: it is computed from the S3 file (qwen2.5:14b vs llama3.1:8b, and
qwen2.5:14b vs its own second fresh load).

Every record is saved, so ``--summary FILE`` recomputes every number from the
file with the library's own code (``PairwiseSummary``, ``consistency``,
``compare``), with no model.

Usage (needs Ollama with the models pulled, and a GPU nothing else is using):
    uv run python scripts/measure_reliability.py --out docs/results/reliability-s4.json
    uv run python scripts/measure_reliability.py --summary docs/results/reliability-s4.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evalkit import __version__
from evalkit.agreement import Rating, Ratings, compare, consistency
from evalkit.judge import JUDGE_PROMPT_VERSION, Judge, JudgeRecord, judge_suite
from evalkit.judge.pairwise import (
    PAIRWISE_PROMPT_VERSION,
    PairRecord,
    PairwiseSummary,
    judge_pairs,
)
from evalkit.prompts import Prompt
from evalkit.providers import GenerationParams, OllamaProvider
from evalkit.schema import Dimension, load_suite
from evalkit.taxonomy import CHECKABLE, FailureMode

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "examples" / "suites" / "everyday.yaml"
S2_RESULTS = ROOT / "docs" / "results" / "runner-s2.json"
S3_RESULTS = ROOT / "docs" / "results" / "judge-s3-prompt-v1.json"
A, B = "qwen2.5:7b", "llama3.1:8b"
JUDGES = ["qwen2.5:14b", "llama3.1:8b"]
ALL_MODELS = ["qwen2.5:14b", "qwen2.5:7b", "llama3.1:8b"]
PAIRWISE_PARAMS = GenerationParams(temperature=0, seed=0, max_tokens=1024)
SAMPLES = 5
SAMPLE_PARAMS = GenerationParams(temperature=0.7, seed=0, max_tokens=1024)


def s2_responses() -> dict[str, dict[str, str]]:
    data = json.loads(S2_RESULTS.read_text(encoding="utf-8"))
    return {m: data["models"][m]["responses"]["fresh"] for m in (A, B)}


def gpu_info() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


async def unload_all(host: str) -> None:
    async with httpx.AsyncClient(base_url=host, timeout=60) as client:
        for model in ALL_MODELS:
            await client.post("/api/generate", json={"model": model, "keep_alive": 0})
        for _ in range(150):
            if not (await client.get("/api/ps")).json().get("models"):
                return
            await asyncio.sleep(0.2)
    raise RuntimeError("models did not unload")


async def fresh_judge(
    model: str, host: str, params: GenerationParams
) -> tuple[OllamaProvider, Judge]:
    await unload_all(host)
    provider = OllamaProvider(model, host=host)
    await provider.__aenter__()
    await provider.preflight()
    await provider.generate(Prompt(user="Hi"), params)  # load before timing
    return provider, Judge(provider, params=params, concurrency=1)


async def run_pairwise(
    judge_model: str, host: str, responses: dict[str, dict[str, str]]
) -> dict[str, Any]:
    suite = load_suite(SUITE)
    provider, judge = await fresh_judge(judge_model, host, PAIRWISE_PARAMS)
    try:
        main = await judge_pairs(
            suite, responses[A], responses[B], judge, names=(A, B), out_dir=None
        )
        controls = {}
        for model in (A, B):
            controls[model] = await judge_pairs(
                suite, responses[model], responses[model], judge, names=(model, model), out_dir=None
            )
    finally:
        await provider.__aexit__(None, None, None)
    print(f"  pairwise {judge_model}: {main.duration_s:.0f} s + controls", file=sys.stderr)
    return {
        "judge": judge_model,
        "params": PAIRWISE_PARAMS.explicit(),
        "wall_s": round(main.duration_s + sum(c.duration_s for c in controls.values()), 1),
        "pairs": [r.model_dump(mode="json") for r in main.records],
        "controls": {
            m: [r.model_dump(mode="json") for r in c.records] for m, c in controls.items()
        },
    }


async def run_consistency(
    judge_model: str, host: str, responses: dict[str, dict[str, str]]
) -> dict[str, Any]:
    suite = load_suite(SUITE)
    provider, judge = await fresh_judge(judge_model, host, SAMPLE_PARAMS)
    records: dict[str, list[dict[str, Any]]] = {}
    wall = 0.0
    try:
        for model in (A, B):
            result = await judge_suite(
                suite, responses[model], judge, out_dir=None, samples=SAMPLES
            )
            wall += result.duration_s
            records[model] = [r.model_dump(mode="json") for r in result.records]
    finally:
        await provider.__aexit__(None, None, None)
    print(f"  consistency {judge_model}: {wall:.0f} s", file=sys.stderr)
    return {
        "judge": judge_model,
        "params": SAMPLE_PARAMS.explicit(),
        "samples": SAMPLES,
        "wall_s": round(wall, 1),
        "records": records,
    }


# ---------------------------------------------------------------- analysis


def _pairs(rows: list[dict[str, Any]]) -> list[PairRecord]:
    return [PairRecord.model_validate(r) for r in rows]


def pairwise_analysis(cond: dict[str, Any]) -> dict[str, Any]:
    main = PairwiseSummary(records=_pairs(cond["pairs"]), name_a=A, name_b=B)
    out: dict[str, Any] = {"main": main.to_dict()}
    # Flip rate split by what the deterministic checks say about the pair:
    # same verdict on both sides, or one side passes and the other fails.
    same = [r for r in main.usable if r.check_status_a == r.check_status_b]
    pass_fail = [r for r in main.usable if {r.check_status_a, r.check_status_b} == {"pass", "fail"}]
    out["by_checks"] = {
        "same_status": {
            "pairs": len(same),
            "flips": sum(r.verdict == "inconsistent" for r in same),
        },
        "pass_vs_fail": {
            "pairs": len(pass_fail),
            "flips": sum(r.verdict == "inconsistent" for r in pass_fail),
            "picked_passing": sum(
                r.verdict == ("a" if r.check_status_a == "pass" else "b") for r in pass_fail
            ),
        },
        "other": len(main.usable) - len(same) - len(pass_fail),
    }
    controls = [r for rows in cond["controls"].values() for r in _pairs(rows)]
    ctrl = PairwiseSummary(records=controls, name_a="same", name_b="same")
    # In a control pair both orders are the same prompt, sent twice (no cache):
    # how often two identical requests at temperature 0 got different answers.
    usable = ctrl.usable
    out["control"] = {
        "pairs": len(controls),
        "usable": len(usable),
        "verdicts": ctrl.verdicts(),
        "flips": ctrl.flips(),
        "positions": ctrl.positions(),
        "cached_orders": sum(o.cached for r in controls for o in r.orders),
        "replies_differ": sum(r.orders[0].replies != r.orders[1].replies for r in usable),
        "positions_differ": sum(r.orders[0].position != r.orders[1].position for r in usable),
    }
    out["wall_s"] = cond["wall_s"]
    return out


def consistency_analysis(cond: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pooled: list[JudgeRecord] = []
    for model, rows in cond["records"].items():
        records = [JudgeRecord.model_validate(r) for r in rows]
        out[model] = consistency(records).to_dict()
        # Both models pooled: case ids repeat across models, so key them apart.
        pooled += [r.model_copy(update={"id": f"{model}|{r.id}"}) for r in records]
    out["pooled"] = consistency(pooled).to_dict()
    out["wall_s"] = cond["wall_s"]
    return out


def _s3_ratings(items: list[dict[str, Any]], name: str) -> Ratings:
    ratings: dict[str, Rating] = {}
    for i in items:
        if i["error"] is not None:
            continue
        counted = {
            *i["check_modes"],
            *(m for m in i["judge_modes"] if FailureMode(m) not in CHECKABLE),
        }
        ratings[f"{i['model']}|{i['id']}"] = Rating(
            id=f"{i['model']}|{i['id']}",
            scores={Dimension(d): s for d, s in i["scores"].items()},
            failure_modes=frozenset(FailureMode(m) for m in counted),
            language=i["language"],
        )
    return Ratings(name, ratings)


def judge_vs_judge() -> dict[str, Any]:
    s3 = json.loads(S3_RESULTS.read_text(encoding="utf-8"))["conditions"]
    qwen = _s3_ratings(s3["qwen14b"]["items"], "qwen2.5:14b")
    return {
        "source": S3_RESULTS.name,
        "llama8b_vs_qwen14b": compare(
            _s3_ratings(s3["llama8b"]["items"], "llama3.1:8b"), qwen
        ).to_dict(),
        "qwen14b_again_vs_qwen14b": compare(
            _s3_ratings(s3["qwen14b_again"]["items"], "qwen2.5:14b (2nd load)"), qwen
        ).to_dict(),
    }


def analyse(results: dict[str, Any]) -> dict[str, Any]:
    c = results["conditions"]
    return {
        **{
            f"pairwise_{j}": pairwise_analysis(c[f"pairwise_{j}"])
            for j in JUDGES
            if f"pairwise_{j}" in c
        },
        **{
            f"consistency_{j}": consistency_analysis(c[f"consistency_{j}"])
            for j in JUDGES
            if f"consistency_{j}" in c
        },
        "judge_vs_judge": judge_vs_judge(),
    }


def markdown(results: dict[str, Any]) -> str:
    a = analyse(results)
    lines = ["| | " + " | ".join(JUDGES) + " |", "|---|" + "---|" * len(JUDGES)]

    def row(label: str, fn: Any) -> None:
        lines.append(f"| {label} | " + " | ".join(str(fn(j)) for j in JUDGES) + " |")

    def p(j: str) -> dict[str, Any]:
        return a[f"pairwise_{j}"]

    row("pairs usable (of 40)", lambda j: p(j)["main"]["usable"])
    row(
        "verdicts A/B/tie/flip",
        lambda j: "/".join(str(v) for v in p(j)["main"]["verdicts"].values()),
    )
    row("flip rate", lambda j: p(j)["main"]["flip_rate"])
    row(
        "flips: pos1 / pos2 / tie once",
        lambda j: "/".join(str(v) for v in p(j)["main"]["flips"].values()),
    )
    row(
        "positions chosen 1/2/tie",
        lambda j: "/".join(str(v) for v in p(j)["main"]["positions"].values()),
    )
    row("win rate A (qwen2.5:7b)", lambda j: p(j)["main"]["win_rate_a"])

    def checks(j: str, group: str, key: str) -> str:
        g = p(j)["by_checks"][group]
        return f"{g[key]}/{g['pairs']}"

    row("flips, checks same on both", lambda j: checks(j, "same_status", "flips"))
    row("flips, one passes one fails", lambda j: checks(j, "pass_vs_fail", "flips"))
    row("picked the passing side", lambda j: checks(j, "pass_vs_fail", "picked_passing"))

    def control(j: str) -> str:
        c = p(j)["control"]
        f = c["flips"]
        ties = f"{c['verdicts']['tie']}/{c['usable']} ties"
        return f"{ties}; flips pos1 {f['first']}, pos2 {f['second']}, tie once {f['tie_once']}"

    def repeat(j: str) -> str:
        c = p(j)["control"]
        n = c["usable"]
        return (
            f"reply differs {c['replies_differ']}/{n}, position differs {c['positions_differ']}/{n}"
            f" ({c['cached_orders']} cached)"
        )

    row("control (same text)", control)
    row("same prompt twice, temp 0", repeat)
    for dim in Dimension:

        def cons(j: str, key: str = dim.value) -> str:
            d = a[f"consistency_{j}"]["pooled"]["dimensions"][key]
            return (
                f"{d['all_same']}/{d['cases']} same, mean SD {d['mean_sd']}, "
                f"range>=2 {d['range_2_plus']}"
            )

        row(f"consistency {dim.value}", cons)
    jj = a["judge_vs_judge"]["llama8b_vs_qwen14b"]["dimensions"]
    lines.append("")
    lines.append("Judge vs judge (llama3.1:8b vs qwen2.5:14b, S3 temperature 0):")
    for key, d in jj.items():
        lines.append(
            f"- {key}: n {d['n']}, exact {d['exact']}, within 1 {d['within_1']}, "
            f"kappa {d['kappa']}, quadratic {d['kappa_quadratic']} {d['kappa_quadratic_ci']}, "
            f"Spearman {d['spearman']}"
        )
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    responses = s2_responses()
    async with httpx.AsyncClient(base_url=args.host, timeout=10) as client:
        ollama = (await client.get("/api/version")).json().get("version")
    conditions: dict[str, Any] = {}
    for judge_model in JUDGES:
        conditions[f"pairwise_{judge_model}"] = await run_pairwise(
            judge_model, args.host, responses
        )
        conditions[f"consistency_{judge_model}"] = await run_consistency(
            judge_model, args.host, responses
        )
    await unload_all(args.host)
    return {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "evalkit_version": __version__,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "pairwise_prompt_version": PAIRWISE_PROMPT_VERSION,
        "environment": {
            "ollama": ollama,
            "gpu": gpu_info(),
            "python": platform.python_version(),
            "os": platform.platform(),
        },
        "inputs": {"suite": SUITE.name, "responses": S2_RESULTS.name, "a": A, "b": B},
        "conditions": {k: conditions[k] for k in sorted(conditions)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("reliability-results.json"))
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--summary", type=Path, help="print the Markdown summary of this file")
    args = parser.parse_args()
    if args.summary:
        results = json.loads(args.summary.read_text(encoding="utf-8"))
        print(markdown(results))
        return 0
    results = asyncio.run(main_async(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Compact: 800 judge records with every reply come to 1.4 MB even so.
    text = json.dumps(results, ensure_ascii=False, separators=(",", ":"))
    args.out.write_text(text + "\n", encoding="utf-8")
    print(markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
