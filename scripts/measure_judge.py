"""Measure the LLM judge on real responses, against what is known for certain.

Input: the 80 responses saved by the S2 measurement (``docs/results/runner-s2.json``,
``responses.fresh``: qwen2.5:7b and llama3.1:8b on the 40-case everyday suite).
They come with three kinds of ground truth that need no human opinion:

1. **Constraint violations.** The deterministic checks failed 10 of them (9
   sentence counts, 1 missing required word). Those checks are exact, so they
   say which responses break a stated constraint and which don't.
2. **Truncation.** 9 responses stopped at the 200-token limit mid-answer, so
   they are unfinished (``partial_completion``).
3. **Language.** Every response is in the language it was asked in (I read and
   labelled all 80 for the S1 language corpus), so any ``language_mismatch`` tag
   from the judge is a false alarm.

Judge conditions, each from a fresh model load (unload every model, load the
judge with one warm-up request), concurrency 1, no cache, temperature 0, seed 0,
max 1024 tokens:

    qwen14b         qwen2.5:14b, check results in the prompt (the default)
    qwen14b_nochk   qwen2.5:14b, without check results (``--no-checks``)
    qwen14b_again   qwen2.5:14b, default, a second fresh load (repeatability)
    llama8b         llama3.1:8b as the judge, default

Writes every judge reply and record to a JSON file and prints a Markdown
summary. Re-grading the saved replies needs no GPU: ``--regrade FILE``.

Usage (needs Ollama with the models pulled, and a GPU nothing else is using):
    uv run python scripts/measure_judge.py --out docs/results/judge-s3-prompt-v1.json
    uv run python scripts/measure_judge.py --regrade docs/results/judge-s3-prompt-v1.json
    uv run python scripts/measure_judge.py --summary docs/results/judge-s3-prompt-v1.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evalkit import __version__
from evalkit.judge import JUDGE_PROMPT_VERSION, Judge, JudgeRecord, judge_suite, parse_judgment
from evalkit.prompts import Prompt
from evalkit.providers import GenerationParams, OllamaProvider
from evalkit.schema import Dimension, Suite, load_suite
from evalkit.taxonomy import FailureMode

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "examples" / "suites" / "everyday.yaml"
S2_RESULTS = ROOT / "docs" / "results" / "runner-s2.json"
PARAMS = GenerationParams(temperature=0, seed=0, max_tokens=1024)
MODELS_UNDER_TEST = ["qwen2.5:7b", "llama3.1:8b"]
CONDITIONS: dict[str, tuple[str, bool]] = {
    "qwen14b": ("qwen2.5:14b", True),
    "qwen14b_nochk": ("qwen2.5:14b", False),
    "qwen14b_again": ("qwen2.5:14b", True),
    "llama8b": ("llama3.1:8b", True),
}
ALL_MODELS = ["qwen2.5:14b", "qwen2.5:7b", "llama3.1:8b"]
CV = FailureMode.CONSTRAINT_VIOLATION.value
PC = FailureMode.PARTIAL_COMPLETION.value
LM = FailureMode.LANGUAGE_MISMATCH.value


def s2_inputs() -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    data = json.loads(S2_RESULTS.read_text(encoding="utf-8"))
    responses = {m: data["models"][m]["responses"]["fresh"] for m in MODELS_UNDER_TEST}
    truncated = {m: data["models"][m]["truncated_at_200_tokens"] for m in MODELS_UNDER_TEST}
    return responses, truncated


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


def item(model: str, r: JudgeRecord, truncated: list[str]) -> dict[str, Any]:
    return {
        "model": model,
        "id": r.id,
        "language": r.language,
        "truncated": r.id in truncated,
        "check_status": r.check_status.value,
        "check_modes": [m.value for m in r.check_failure_modes],
        "judge_modes": [m.value for m in r.judge_failure_modes],
        "scores": {d.value: s for d, s in r.scores.items()},
        "rationale": r.rationale,
        "error": r.error,
        "parse": r.parse_outcome,
        "repairs": r.repairs,
        "reasks": r.reasks,
        "latency_s": r.latency_s,
        "output_tokens": r.output_tokens,
        "replies": r.replies,
    }


async def run_condition(
    name: str,
    suite: Suite,
    responses: dict[str, dict[str, str]],
    truncated: dict[str, list[str]],
    host: str,
) -> dict[str, Any]:
    judge_model, include_checks = CONDITIONS[name]
    await unload_all(host)
    items: list[dict[str, Any]] = []
    wall = 0.0
    async with OllamaProvider(judge_model, host=host) as provider:
        await provider.preflight()
        await provider.generate(Prompt(user="Hi"), PARAMS)  # load before timing
        judge = Judge(provider, params=PARAMS, concurrency=1, include_checks=include_checks)
        for model in MODELS_UNDER_TEST:
            result = await judge_suite(suite, responses[model], judge, out_dir=None)
            wall += result.duration_s
            items += [item(model, r, truncated[model]) for r in result.records]
    lat = sorted(i["latency_s"] for i in items if i["error"] is None)
    toks = sorted(i["output_tokens"] or 0 for i in items)
    print(f"  {name}: {len(items)} judged in {wall:.1f} s", file=sys.stderr)
    return {
        "judge": judge_model,
        "include_checks": include_checks,
        "wall_s": round(wall, 2),
        "latency_p50_s": round(statistics.median(lat), 3) if lat else None,
        "latency_max_s": round(lat[-1], 3) if lat else None,
        "output_tokens_p50": statistics.median(toks),
        "items": items,
    }


# ---------------------------------------------------------------- analysis


def _key(i: dict[str, Any]) -> tuple[str, str]:
    return i["model"], i["id"]


def _ok(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [i for i in items if i["error"] is None]


def ground_truth(items: list[dict[str, Any]]) -> dict[str, Any]:
    """How the judge's tags line up with what the checks and truncation say for certain."""
    ok = _ok(items)
    cv_true = [i for i in ok if CV in i["check_modes"]]
    cv_clean = [i for i in ok if i["check_status"] == "pass"]
    trunc = [i for i in ok if i["truncated"]]
    not_trunc = [i for i in ok if not i["truncated"]]

    def tagged(group: list[dict[str, Any]], mode: str) -> int:
        return sum(mode in i["judge_modes"] for i in group)

    def mean_if(group: list[dict[str, Any]]) -> float | None:
        vals = [i["scores"]["instruction_following"] for i in group]
        return round(statistics.fmean(vals), 2) if vals else None

    return {
        "constraint_violation": {
            "judge_tagged_of_check_failures": [tagged(cv_true, CV), len(cv_true)],
            "judge_tagged_of_check_passes": [tagged(cv_clean, CV), len(cv_clean)],
            "mean_if_check_failures": mean_if(cv_true),
            "mean_if_check_passes": mean_if(cv_clean),
        },
        "partial_completion": {
            "judge_tagged_of_truncated": [tagged(trunc, PC), len(trunc)],
            "judge_tagged_of_complete": [tagged(not_trunc, PC), len(not_trunc)],
        },
        "language_mismatch_false_alarms": [tagged(ok, LM), len(ok)],
        "judge_tag_counts": dict(Counter(m for i in ok for m in i["judge_modes"]).most_common()),
    }


def agreement(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    """Per dimension: exact and within-one agreement on items both judged."""
    bi = {_key(i): i for i in _ok(b)}
    pairs = [(i, bi[_key(i)]) for i in _ok(a) if _key(i) in bi]
    out: dict[str, Any] = {"items": len(pairs)}
    for d in Dimension:
        xs = [(x["scores"][d.value], y["scores"][d.value]) for x, y in pairs]
        out[d.value] = {
            "exact": sum(p == q for p, q in xs),
            "within_1": sum(abs(p - q) <= 1 for p, q in xs),
            "mean_a": round(statistics.fmean(p for p, _ in xs), 2),
            "mean_b": round(statistics.fmean(q for _, q in xs), 2),
        }
    same_tags = sum(set(x["judge_modes"]) == set(y["judge_modes"]) for x, y in pairs)
    out["identical_tag_sets"] = same_tags
    return out


def repeatability(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    bi = {_key(i): i for i in b}
    return {
        "items": len(a),
        "identical_replies": sum(i["replies"] == bi[_key(i)]["replies"] for i in a),
        "identical_verdicts": sum(
            (i["scores"], i["judge_modes"]) == (bi[_key(i)]["scores"], bi[_key(i)]["judge_modes"])
            for i in a
        ),
    }


def by_model(items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model in MODELS_UNDER_TEST:
        group = [i for i in _ok(items) if i["model"] == model]
        out[model] = {
            d.value: round(statistics.fmean(i["scores"][d.value] for i in group), 2)
            for d in Dimension
        }
    return out


def analyse(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    items = {name: c["items"] for name, c in conditions.items()}
    return {
        "parse": {name: dict(Counter(i["parse"] for i in its)) for name, its in items.items()},
        "ground_truth": {name: ground_truth(its) for name, its in items.items()},
        "repeatability_qwen14b": repeatability(items["qwen14b"], items["qwen14b_again"]),
        "checks_in_prompt_vs_not": agreement(items["qwen14b"], items["qwen14b_nochk"]),
        "qwen14b_vs_llama8b": agreement(items["qwen14b"], items["llama8b"]),
        "mean_scores_by_model_under_test": {name: by_model(its) for name, its in items.items()},
    }


def regrade(path: Path) -> int:
    """Re-parse every saved reply with the current parser; report any change."""
    data = json.loads(path.read_text(encoding="utf-8"))
    suite = load_suite(SUITE)
    changed = 0
    for name, cond in data["conditions"].items():
        for i in cond["items"]:
            if i["error"] is not None:
                continue
            case = suite.case(i["id"])
            dims = case.rubric.dimensions if case.rubric else list(Dimension)
            parsed = parse_judgment(i["replies"][-1], dims)
            scores = {d.value: s for d, s in parsed.scores.items()}
            modes = [m.value for m in parsed.failure_modes]
            if (scores, modes, parsed.repairs) != (i["scores"], i["judge_modes"], i["repairs"]):
                changed += 1
                print(f"{name} {i['model']} {i['id']}: changed", file=sys.stderr)
    print(f"re-graded {path}: {changed} verdicts changed")
    return 1 if changed else 0


def fraction(pair: list[int]) -> str:
    return f"{pair[0]}/{pair[1]}"


def markdown(results: dict[str, Any]) -> str:
    a = results["analysis"]
    conds = results["conditions"]
    names = list(conds)
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]

    def row(label: str, fn: Any) -> None:
        lines.append(f"| {label} | " + " | ".join(str(fn(n)) for n in names) + " |")

    def cv(n: str) -> dict[str, Any]:
        return dict(a["ground_truth"][n]["constraint_violation"])

    def pc(n: str) -> dict[str, Any]:
        return dict(a["ground_truth"][n]["partial_completion"])

    row("parse outcomes", lambda n: ", ".join(f"{k} {v}" for k, v in a["parse"][n].items()))
    row("wall time (80 judgments)", lambda n: f"{conds[n]['wall_s']} s")
    row("median latency", lambda n: f"{conds[n]['latency_p50_s']} s")
    row("CV tagged / check failures", lambda n: fraction(cv(n)["judge_tagged_of_check_failures"]))
    row("CV tagged / check passes", lambda n: fraction(cv(n)["judge_tagged_of_check_passes"]))
    row(
        "mean IF: check fail / pass",
        lambda n: f"{cv(n)['mean_if_check_failures']} / {cv(n)['mean_if_check_passes']}",
    )
    row("partial_completion / truncated", lambda n: fraction(pc(n)["judge_tagged_of_truncated"]))
    row("partial_completion / complete", lambda n: fraction(pc(n)["judge_tagged_of_complete"]))
    row(
        "language_mismatch false alarms",
        lambda n: fraction(a["ground_truth"][n]["language_mismatch_false_alarms"]),
    )
    row("judge tags", lambda n: a["ground_truth"][n]["judge_tag_counts"])
    rep = a["repeatability_qwen14b"]
    lines.append("")
    lines.append(
        "Repeatability (qwen14b vs qwen14b_again): "
        f"identical replies {rep['identical_replies']}/{rep['items']}, "
        f"identical verdicts {rep['identical_verdicts']}/{rep['items']}"
    )
    pairs = (
        ("checks vs no checks", "checks_in_prompt_vs_not"),
        ("qwen14b vs llama8b", "qwen14b_vs_llama8b"),
    )
    for label, key in pairs:
        ag = a[key]
        parts = []
        for d in Dimension:
            x = ag[d.value]
            parts.append(
                f"{d.value} exact {x['exact']}, +-1 {x['within_1']} "
                f"(means {x['mean_a']} vs {x['mean_b']})"
            )
        lines.append(
            f"{label} ({ag['items']} items): "
            + "; ".join(parts)
            + f"; identical tag sets {ag['identical_tag_sets']}"
        )
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    suite = load_suite(SUITE)
    responses, truncated = s2_inputs()
    async with httpx.AsyncClient(base_url=args.host, timeout=10) as client:
        ollama_version = (await client.get("/api/version")).json().get("version")
    results: dict[str, Any] = {
        "meta": {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "evalkit_version": __version__,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "ollama_version": ollama_version,
            "gpu_at_start": gpu_info(),
            "python": platform.python_version(),
            "suite": f"{suite.name} v{suite.version} ({len(suite.cases)} cases)",
            "responses": "runner-s2.json responses.fresh (qwen2.5:7b, llama3.1:8b)",
            "params": PARAMS.explicit(),
        },
        "conditions": {},
    }
    for name in CONDITIONS:
        results["conditions"][name] = await run_condition(
            name, suite, responses, truncated, args.host
        )
    await unload_all(args.host)
    results["analysis"] = analyse(results["conditions"])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=False, default=Path("judge-results.json"))
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--regrade", type=Path, help="re-parse the replies saved in this file")
    parser.add_argument("--summary", type=Path, help="print the Markdown summary of this file")
    args = parser.parse_args()
    if args.regrade:
        return regrade(args.regrade)
    if args.summary:
        print(markdown(json.loads(args.summary.read_text(encoding="utf-8"))))
        return 0
    results = asyncio.run(main_async(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(markdown(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
