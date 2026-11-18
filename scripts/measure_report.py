"""Measure the report tools on real runs: a regression diff and a model-vs-model report.

Input: the responses saved by the S2 measurement (``docs/results/runner-s2.json``):
qwen2.5:7b and llama3.1:8b on the 40-case everyday suite, at temperature 0 and seed 0,
once from a fresh model load (``fresh``) and once more on the already-warm server
(``warm``). The warm run reworded 12 of qwen2.5:7b's answers and 4 of llama3.1:8b's.

Steps:

1. Write the four response sets as JSONL files under ``runs/s5/``.
2. Judge them with qwen2.5:14b from a fresh load (prompt v1, temperature 0, seed 0,
   at most 1024 tokens, concurrency 1, the warm-up request S3 used), the fresh sets
   first in S3's order, then the warm ones, all through one new response cache. A
   warm answer identical to its fresh one then replays the fresh judgment, as a CI
   run with a cache would. The fresh replies are compared with S3's.
3. ``diff_runs`` (what ``evalkit compare`` prints), fresh vs warm per model: on the
   checks alone, then on the judged runs.
4. ``build_report`` (what ``evalkit report`` writes) of the two fresh judged runs,
   model vs model, into ``docs/results/report-s5/``.

Every judge record is saved, so ``--summary FILE`` recomputes the diffs from the
file and the S2 responses, with no model.

Usage, from the repository root (needs Ollama with the models pulled and an idle GPU):
    uv run python scripts/measure_report.py --out docs/results/report-s5.json
    uv run python scripts/measure_report.py --summary docs/results/report-s5.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx

from evalkit import __version__
from evalkit.cache import ResponseCache
from evalkit.files import write_text_atomic
from evalkit.judge import JUDGE_PROMPT_VERSION, Judge, JudgeRecord, judge_suite
from evalkit.prompts import Prompt
from evalkit.providers import GenerationParams, OllamaProvider
from evalkit.report import (
    RunDiff,
    RunView,
    build_report,
    case_views,
    diff_runs,
    load_runs,
    render_html,
    render_markdown,
)
from evalkit.schema import Dimension, Suite, load_suite
from evalkit.taxonomy import FailureMode

SUITE = Path("examples/suites/everyday.yaml")
S2_RESULTS = Path("docs/results/runner-s2.json")
S3_RESULTS = Path("docs/results/judge-s3-prompt-v1.json")
WORK = Path("runs/s5")  # relative, so the manifests and the report hold no local paths
REPORT_DIR = Path("docs/results/report-s5")
MODELS = ["qwen2.5:7b", "llama3.1:8b"]
RUNS = ["fresh", "warm"]
JUDGE = "qwen2.5:14b"
ALL_MODELS = ["qwen2.5:14b", "qwen2.5:7b", "llama3.1:8b"]
PARAMS = GenerationParams(temperature=0, seed=0, max_tokens=1024)


def slug(model: str) -> str:
    return model.replace(":", "-")


def s2_responses() -> dict[tuple[str, str], dict[str, str]]:
    data = json.loads(S2_RESULTS.read_text(encoding="utf-8"))
    return {(m, r): data["models"][m]["responses"][r] for m in MODELS for r in RUNS}


def write_inputs(suite: Suite) -> dict[tuple[str, str], Path]:
    paths = {}
    for (model, run), responses in s2_responses().items():
        path = WORK / f"{slug(model)}-{run}.jsonl"
        rows = [{"id": c.id, "response": responses[c.id]} for c in suite.cases]
        write_text_atomic(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        paths[model, run] = path
    return paths


def judged_dir(model: str, run: str) -> Path:
    return WORK / f"judged-{slug(model)}-{run}"


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


async def unload_all(client: httpx.AsyncClient) -> None:
    for model in ALL_MODELS:
        await client.post("/api/generate", json={"model": model, "keep_alive": 0})
    for _ in range(150):
        if not (await client.get("/api/ps")).json().get("models"):
            return
        await asyncio.sleep(0.2)
    raise RuntimeError("models did not unload")


async def judge_all(
    suite: Suite, paths: dict[tuple[str, str], Path], host: str
) -> dict[str, dict[str, Any]]:
    async with httpx.AsyncClient(base_url=host, timeout=60) as client:
        await unload_all(client)
    out: dict[str, dict[str, Any]] = {}
    async with OllamaProvider(JUDGE, host=host) as provider:
        await provider.preflight()
        await provider.generate(Prompt(user="Hi"), PARAMS)  # load before timing, as in S3
        judge = Judge(provider, params=PARAMS, concurrency=1, cache=ResponseCache(WORK / "cache"))
        for run in RUNS:
            for model in MODELS:
                d = judged_dir(model, run)
                result = await judge_suite(
                    suite,
                    s2_responses()[model, run],
                    judge,
                    out_dir=d.parent,
                    run_id=d.name,
                    suite_path=SUITE,
                    responses_path=paths[model, run],
                )
                counts = result.counts()
                print(
                    f"  {model} {run}: {counts['requests']} requests, "
                    f"{counts['cached']} from cache, {result.duration_s:.1f} s",
                    file=sys.stderr,
                )
                out[f"{model}/{run}"] = {
                    "wall_s": result.duration_s,
                    "counts": counts,
                    "records": [r.model_dump(mode="json") for r in result.records],
                }
    return out


def s3_identical(judged: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Fresh judgments whose replies are byte-identical to S3's qwen2.5:14b condition."""
    s3 = json.loads(S3_RESULTS.read_text(encoding="utf-8"))["conditions"]["qwen14b"]["items"]
    out = {}
    for model in MODELS:
        ours = {r["id"]: r["replies"] for r in judged[f"{model}/fresh"]["records"]}
        theirs = {i["id"]: i["replies"] for i in s3 if i["model"] == model}
        same = sum(ours[i] == theirs[i] for i in theirs)
        out[model] = f"{same}/{len(theirs)}"
    return out


def views(
    suite: Suite, judged: dict[str, dict[str, Any]]
) -> dict[tuple[str, str], tuple[RunView, RunView]]:
    """(checks only, judged) per model and run, rebuilt from the S2 responses and the
    saved records, so --summary needs neither the runs directory nor a model."""
    out = {}
    for (model, run), texts in s2_responses().items():
        records = [JudgeRecord.model_validate(r) for r in judged[f"{model}/{run}"]["records"]]
        rated = case_views(suite, texts, {r.id: r for r in records})
        name = f"{model} {run}"
        out[model, run] = (
            RunView(name, "s2", "responses", case_views(suite, texts)),
            RunView(name, "s2", "judgments", rated),
        )
    return out


def near_identical() -> dict[str, set[str]]:
    """Changed answers that kept most of their text. The 16 changed answers have
    difflib ratios of 0.17-0.51 or 0.75-0.98, so any cut in between gives the same split."""
    responses = s2_responses()
    out = {}
    for model in MODELS:
        fresh, warm = responses[model, "fresh"], responses[model, "warm"]
        out[model] = {
            i
            for i in fresh
            if fresh[i] != warm[i] and SequenceMatcher(None, fresh[i], warm[i]).ratio() >= 0.75
        }
    return out


def diff_summary(diff: RunDiff) -> dict[str, Any]:
    payload = diff.to_dict()
    payload["worse_and_better"] = [c.id for c in diff.changes if c.worse and c.better]
    payload["score_changes"] = sum(len(c.scores) for c in diff.changes)
    return payload


def diffs(suite: Suite, judged: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    v = views(suite, judged)
    out = {}
    for model in MODELS:
        (checks_f, judged_f), (checks_w, judged_w) = v[model, "fresh"], v[model, "warm"]
        out[model] = {
            "checks": diff_summary(diff_runs(checks_f, checks_w)),
            "judged": diff_summary(diff_runs(judged_f, judged_w)),
        }
    return out


def write_report(suite: Suite) -> dict[str, Any]:
    """The model-vs-model report of the fresh judged runs, as `evalkit report` writes it."""
    t0 = time.perf_counter()
    runs = load_runs([judged_dir(m, "fresh") for m in MODELS], suite, MODELS)
    report = build_report(suite, runs)
    html, md = render_html(report), render_markdown(report)
    elapsed = time.perf_counter() - t0
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_text_atomic(REPORT_DIR / "report.html", html)
    write_text_atomic(REPORT_DIR / "report.md", md)
    notes = [n for r in runs for n in r.notes]
    return {"seconds": round(elapsed, 3), "html_bytes": len(html.encode()), "notes": notes}


def summary(data: dict[str, Any]) -> None:
    suite = load_suite(SUITE)
    recomputed = diffs(suite, data["judged"])
    if recomputed != data["diffs"]:
        raise SystemExit("the saved diffs differ from the ones recomputed from the saved records")
    print("| fresh -> warm | " + " | ".join(MODELS) + " |")
    print("|---|" + "---|" * len(MODELS))

    def row(label: str, fn: Any) -> None:
        print(f"| {label} | " + " | ".join(str(fn(m)) for m in MODELS) + " |")

    d = recomputed
    row("cases in both runs", lambda m: d[m]["checks"]["shared"])
    row("response text changed", lambda m: d[m]["checks"]["text_changed"])
    row("checks: verdict changed", lambda m: len(d[m]["checks"]["changes"]))
    row(
        "judge requests for the warm run",
        lambda m: data["judged"][f"{m}/warm"]["counts"]["requests"],
    )
    near = near_identical()
    row("... near-identical / different content", lambda m: f"{len(near[m])} / "
        f"{d[m]['checks']['text_changed'] - len(near[m])}")  # fmt: skip
    judged_ids = {m: {c["id"] for c in d[m]["judged"]["changes"]} for m in MODELS}
    row("judged: cases changed", lambda m: len(judged_ids[m]))
    row("... of them near-identical", lambda m: len(judged_ids[m] & near[m]))
    row("... worse / better / both", lambda m: "{} / {} / {}".format(
        len(d[m]["judged"]["worse"]), len(d[m]["judged"]["better"]),
        len(d[m]["judged"]["worse_and_better"])))  # fmt: skip
    row("... worse and near-identical", lambda m: ", ".join(
        i for i in d[m]["judged"]["worse"] if i in near[m]))  # fmt: skip
    moves = {m: [abs(n - b) for c in d[m]["judged"]["changes"] for b, n in c["scores"].values()]
             for m in MODELS}  # fmt: skip
    row("... score moves of 1 point / 2 or more", lambda m: f"{moves[m].count(1)} / "
        f"{len(moves[m]) - moves[m].count(1)}")  # fmt: skip
    near_moves = {m: [abs(n - b) for c in d[m]["judged"]["changes"] if c["id"] in near[m]
                      for b, n in c["scores"].values()] for m in MODELS}  # fmt: skip
    row("... of them on near-identical answers: 1 point / 2 or more", lambda m: (
        f"{near_moves[m].count(1)} / {len(near_moves[m]) - near_moves[m].count(1)}"))  # fmt: skip
    v = views(suite, data["judged"])
    coarse = {m: diff_runs(v[m, "fresh"][1], v[m, "warm"][1], min_change=2) for m in MODELS}
    row("worse with --min-change 2", lambda m: ", ".join(c.id for c in coarse[m].worse))
    for dim in Dimension:
        row(f"mean {dim.value} fresh -> warm", lambda m, dim=dim: "{base:.2f} -> {new:.2f}".format(
            **d[m]["judged"]["mean_scores"][dim.value]))  # fmt: skip
    hallucination = FailureMode.HALLUCINATION.value
    row("hallucination tags fresh -> warm", lambda m: "{base} -> {new}".format(
        **d[m]["judged"]["failure_modes"].get(hallucination, {"base": 0, "new": 0})))  # fmt: skip
    row("fresh replies identical to S3", lambda m: data["s3_identical_replies"][m])
    print(f"report of the fresh runs built in {data['report']['seconds']} s")


async def measure(host: str) -> dict[str, Any]:
    suite = load_suite(SUITE)
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    paths = write_inputs(suite)
    gpu = gpu_info()
    async with httpx.AsyncClient(base_url=host, timeout=10) as client:
        ollama = (await client.get("/api/version")).json().get("version")
    judged = await judge_all(suite, paths, host)
    return {
        "meta": {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "evalkit_version": __version__,
            "judge": JUDGE,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "params": PARAMS.explicit(),
            "ollama_version": ollama,
            "gpu_at_start": gpu,
            "python": platform.python_version(),
            "suite": SUITE.as_posix(),
            "responses": "runner-s2.json responses.fresh and responses.warm",
        },
        "judged": judged,
        "s3_identical_replies": s3_identical(judged),
        "diffs": diffs(suite, judged),
        "report": write_report(suite),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="write the results here")
    parser.add_argument("--summary", type=Path, help="recompute and print a results file")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    if args.summary:
        summary(json.loads(args.summary.read_text(encoding="utf-8")))
        return
    if not args.out:
        parser.error("--out or --summary is required")
    data = asyncio.run(measure(args.host))
    write_text_atomic(args.out, json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    summary(data)


if __name__ == "__main__":
    main()
