"""Run the benchmark suite on local models, judge the answers and write the report.

Steps (needs Ollama with the models pulled and an idle GPU):

1. For each model under test, from a fresh load (unload everything, one warm-up
   request): run ``examples/suites/benchmark.yaml`` at temperature 0, seed 0, at
   most 1024 tokens, one request at a time, and save the answers as
   ``examples/responses/benchmark-<model>.jsonl``.
2. Judge both answer sets with qwen2.5:14b from a fresh load (prompt v1, the same
   settings), writing the judge runs to ``docs/results/benchmark-s6/judged-<model>/``.
3. ``evalkit report`` of the two judged runs into ``docs/results/benchmark-s6/``.

Everything is saved in the results file, so ``--summary FILE`` rebuilds every
table from it with no model: check pass rates and judge means by group and
language, the reasoning-trap answers against the suite's reference answers, the
judge's failure modes, and agreement with ``examples/human_labels.csv``.

Usage, from the repository root:
    uv run python scripts/measure_benchmark.py --out docs/results/benchmark-s6.json
    uv run python scripts/measure_benchmark.py --summary docs/results/benchmark-s6.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from measure_report import gpu_info, slug, unload_all

from evalkit import __version__
from evalkit.agreement import Rating, Ratings, compare, load_labels
from evalkit.cache import ResponseCache
from evalkit.evaluate import check_case
from evalkit.files import write_text_atomic
from evalkit.judge import JUDGE_PROMPT_VERSION, Judge, JudgeRecord, JudgeSummary, judge_suite
from evalkit.prompts import Prompt
from evalkit.providers import GenerationParams, OllamaProvider
from evalkit.report import build_report, load_runs, render_html, render_markdown
from evalkit.runner import run_suite
from evalkit.schema import Dimension, EvalCase, Suite, load_suite

SUITE = Path("examples/suites/benchmark.yaml")
MODELS = ["qwen2.5:7b", "llama3.1:8b"]
JUDGE = "qwen2.5:14b"
PARAMS = GenerationParams(temperature=0, seed=0, max_tokens=1024)
WORK = Path("runs/s6")  # caches only
OUT = Path("docs/results/benchmark-s6")  # relative, so manifests and the report hold no local paths
LABELS = Path("examples/human_labels.csv")
LABELLED = "qwen2.5:7b"  # the model whose answers the sheet labels
# The run before the post-review fixes to five cases' checks and criteria.
BEFORE = Path("docs/results/benchmark-s6-before-review.json")
GROUPS = ["multi-constraint", "ambiguous", "edge-case", "reasoning-trap"]
LANGS = ["en", "id"]
ANSWER_LINE = re.compile(r"^[ \t*_#>]*(?:answer|jawaban)[ \t*_]*:(.*)$", re.I | re.M)


def responses_path(model: str) -> Path:
    return Path(f"examples/responses/benchmark-{slug(model)}.jsonl")


def judged_dir(model: str) -> Path:
    return OUT / f"judged-{slug(model)}"


async def fresh_load(provider: OllamaProvider, host: str) -> None:
    async with httpx.AsyncClient(base_url=host, timeout=60) as client:
        await unload_all(client)
    await provider.preflight()
    await provider.generate(Prompt(user="Hi"), PARAMS)  # load before timing, as in S2-S5


async def generate(suite: Suite, host: str) -> dict[str, Any]:
    out = {}
    for model in MODELS:
        async with OllamaProvider(model, host=host) as provider:
            await fresh_load(provider, host)
            cache = ResponseCache(WORK / "cache" / slug(model))
            result = await run_suite(
                suite, provider, params=PARAMS, concurrency=1, cache=cache, out_dir=None
            )
        if result.errors:
            raise SystemExit(f"{model}: {len(result.errors)} case(s) failed: {result.errors[0]}")
        rows = [{"id": r.id, "response": r.response} for r in result.records]
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        write_text_atomic(responses_path(model), text)
        print(f"  {model}: {result.counts()}, {result.duration_s:.1f} s", file=sys.stderr)
        out[model] = {
            "wall_s": result.duration_s,
            "counts": result.counts(),
            "records": [r.model_dump(mode="json") for r in result.records],
        }
    return out


async def judge_all(suite: Suite, generated: dict[str, Any], host: str) -> dict[str, Any]:
    out = {}
    async with OllamaProvider(JUDGE, host=host) as provider:
        await fresh_load(provider, host)
        judge = Judge(provider, params=PARAMS, concurrency=1, cache=ResponseCache(WORK / "judge"))
        for model in MODELS:
            shutil.rmtree(judged_dir(model), ignore_errors=True)
            result = await judge_suite(
                suite,
                responses(generated, model),
                judge,
                out_dir=OUT,
                run_id=judged_dir(model).name,
                suite_path=SUITE,
                responses_path=responses_path(model),
            )
            print(f"  judge {model}: {result.counts()}, {result.duration_s:.1f} s", file=sys.stderr)
            out[model] = {
                "wall_s": result.duration_s,
                "counts": result.counts(),
                "records": [r.model_dump(mode="json") for r in result.records],
            }
    return out


def write_report(suite: Suite) -> dict[str, Any]:
    runs = load_runs([judged_dir(m) for m in MODELS], suite, MODELS)
    report = build_report(suite, runs)
    html = render_html(report)
    write_text_atomic(OUT / "report.html", html)
    write_text_atomic(OUT / "report.md", render_markdown(report))
    return {"html_bytes": len(html.encode()), "notes": [n for r in runs for n in r.notes]}


def responses(data: dict[str, Any], model: str) -> dict[str, str]:
    return {r["id"]: r["response"] for r in data[model]["records"]}


def final_answer(text: str) -> str | None:
    """The text after the last "Answer:" / "Jawaban:" label, either language."""
    found = ANSWER_LINE.findall(text)
    return found[-1].strip(" \t*_") if found else None


def is_correct(case: EvalCase, answer: str | None) -> bool:
    """The right value appears on the Answer line: a number anywhere on it, separators
    ignored ("Rp 100.000 (96.000 / 0.96)" = 100000, "11.00 WIT" = 11:00), or the word.
    shortcut: a hedged line ("07:00 or 11:00", "Thursday, not Monday") or working that passes
    through the right value ("15 + 5 = 20") counts as right, and a decimal ("Rp 100.000,00")
    as wrong; the summary prints every answer next to its reference so a misread shows.
    Pick a stricter rule if a model starts hedging or putting working on the Answer line."""
    reference = case.rubric.reference_answer if case.rubric else None
    if reference is None or answer is None:
        return False
    if any(ch.isdigit() for ch in reference):
        target = re.sub(r"\D", "", reference)
        return any(re.sub(r"\D", "", n) == target for n in re.findall(r"\d[\d.,:]*", answer))
    return re.search(rf"\b{re.escape(reference)}\b", answer, re.I) is not None


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "-"


def summary(data: dict[str, Any]) -> None:
    suite = load_suite(SUITE)

    def select(group: str | None, lang: str | None) -> list[EvalCase]:
        return [
            c
            for c in suite.cases
            if (group is None or group in c.tags) and (lang is None or c.language == lang)
        ]

    print("## Checks: cases passing every check")
    print("| group | " + " | ".join(f"{m} {lang}" for m in MODELS for lang in LANGS) + " |")
    for group in [*GROUPS, None]:
        cells = []
        for model in MODELS:
            texts = responses(data["generated"], model)
            for lang in LANGS:
                reports = [check_case(c, texts[c.id]) for c in select(group, lang)]
                decided = [r for r in reports if r.status.value != "skip"]
                cells.append(pct(sum(r.status.value == "pass" for r in decided), len(decided)))
        print(f"| {group or 'all'} | " + " | ".join(cells) + " |")

    print("\n## Checks that failed")
    for model in MODELS:
        texts = responses(data["generated"], model)
        for c in suite.cases:
            for r in check_case(c, texts[c.id]).results:
                if r.status.value == "fail":
                    print(f"  {model} {c.id}: {r.check}: {r.message}")

    print("\n## Reasoning traps: final answer vs reference")
    for model in MODELS:
        texts = responses(data["generated"], model)
        for lang in LANGS:
            traps = select("reasoning-trap", lang)
            right = [c.id for c in traps if is_correct(c, final_answer(texts[c.id]))]
            print(f"  {model} {lang}: {pct(len(right), len(traps))}")
            for c in traps:
                answer = final_answer(texts[c.id])
                ok = "ok   " if c.id in right else "WRONG"
                ref = c.rubric.reference_answer if c.rubric else None
                print(f"    {ok} {c.id}: {answer!r} (reference {ref!r})")

    print("\n## Judge means (1-5) by group and language")
    print("| model | group | lang | n | " + " | ".join(d.value for d in Dimension) + " |")
    for model in MODELS:
        records = [JudgeRecord.model_validate(r) for r in data["judged"][model]["records"]]
        for group in [*GROUPS, None]:
            for lang in [*LANGS, None]:
                picked = [
                    r
                    for r in records
                    if r.ok
                    and (group is None or group in r.tags)
                    and (lang is None or r.language == lang)
                ]
                means = JudgeSummary(records=picked).mean_scores()
                cells = [f"{v:.2f}" if v is not None else "-" for v in means.values()]
                row = f"| {model} | {group or 'all'} | {lang or 'all'} | {len(picked)} | "
                print(row + " | ".join(cells) + " |")

    print("\n## Failure modes (checks decide the checkable three, the judge the rest)")
    for model in MODELS:
        records = [JudgeRecord.model_validate(r) for r in data["judged"][model]["records"]]
        counts = JudgeSummary(records=records).failure_modes()
        print(f"  {model}: " + ", ".join(f"{m.value} {c}" for m, c in counts.items()))
        for r in records:
            if r.failure_modes:
                modes = ", ".join(m.value for m in r.failure_modes)
                print(f"    {r.id}: {modes}")

    print("\n## Judge mechanics")
    for model in MODELS:
        j = data["judged"][model]
        print(f"  {model}: {j['counts']}, {j['wall_s']:.1f} s")
    for model in MODELS:
        g = data["generated"][model]
        print(f"  generation {model}: {g['counts']}, {g['wall_s']:.1f} s")

    if BEFORE.exists():
        print(f"\n## Against {BEFORE.as_posix()}: answers, and judge replies that changed")
        before = json.loads(BEFORE.read_text(encoding="utf-8"))
        for model in MODELS:
            old, new = responses(before["generated"], model), responses(data["generated"], model)
            shared = [i for i in new if i in old]  # a case added since has nothing to compare
            same = sum(old[i] == new[i] for i in shared)
            print(f"  {model}: answers identical {same}/{len(shared)} ({len(new)} cases now)")
            olds = {
                r["id"]: JudgeRecord.model_validate(r) for r in before["judged"][model]["records"]
            }
            for b in (JudgeRecord.model_validate(r) for r in data["judged"][model]["records"]):
                a = olds.get(b.id)
                if a is not None and a.replies != b.replies:
                    print(
                        f"    {b.id}: checks {a.check_status.value} -> {b.check_status.value}; "
                        f"scores {list(a.scores.values())} -> {list(b.scores.values())}; "
                        f"tags {sorted(m.value for m in a.judge_failure_modes)} -> "
                        f"{sorted(m.value for m in b.judge_failure_modes)}"
                    )

    if LABELS.exists():
        # The sheet labels one model's answers; the other model's judgments share its ids.
        print(f"\n## Judge vs {LABELS.as_posix()} ({LABELLED}'s answers; labeler: see the sheet)")
        judged, labels = judged_ratings(data), load_labels(LABELS)
        report = compare(judged, labels)
        for line in [*report.notes, *report.warnings]:
            print(f"  note: {line}")
        if report.matched:
            print(f"  {report.matched} labelled cases")
            for lang, dims in report.by_language.items():
                kappas = ", ".join(f"{d.value} {a.kappa_quadratic}" for d, a in dims.items())
                print(f"    {lang}: quadratic kappa {kappas}")
            for dim, a in report.dimensions.items():
                print(
                    f"    {dim.value}: n={a.n}, exact {a.exact}, within one {a.within_1}, "
                    f"quadratic kappa {a.kappa_quadratic} {a.kappa_quadratic_ci}, "
                    f"means label {a.mean_reference} / judge {a.mean_judge}"
                )
            # The cases compare() counts modes on: both sides labelled failure modes.
            both = [
                i
                for i in judged.ratings.keys() & labels.ratings.keys()
                if judged.ratings[i].failure_modes is not None
                and labels.ratings[i].failure_modes is not None
            ]
            for mode, m in report.failure_modes.items():
                if m.both or m.judge_only or m.reference_only:
                    j, lab = (
                        {i for i in both if mode in (side.ratings[i].failure_modes or ())}
                        for side in (judged, labels)
                    )
                    print(
                        f"    {mode.value} ({m.decided_by}): both {m.both}, "
                        f"judge only {sorted(j - lab)}, labels only {sorted(lab - j)}"
                    )


def judged_ratings(data: dict[str, Any]) -> Ratings:
    """The labelled model's judgments as `evalkit agree` reads a judgments file, taken
    from the results file (not the judge run on disk), with the answers they rate."""
    records = [JudgeRecord.model_validate(r) for r in data["judged"][LABELLED]["records"]]
    ratings = {
        r.id: Rating(
            id=r.id, scores=r.scores, failure_modes=frozenset(r.failure_modes), language=r.language
        )
        for r in records
        if r.ok
    }
    skipped = [r.id for r in records if not r.ok]
    texts = responses(data["generated"], LABELLED)
    return Ratings(f"{LABELLED} judgments", ratings, skipped, texts=texts)


async def measure(host: str) -> dict[str, Any]:
    suite = load_suite(SUITE)
    shutil.rmtree(WORK, ignore_errors=True)
    gpu = gpu_info()
    async with httpx.AsyncClient(base_url=host, timeout=10) as client:
        ollama = (await client.get("/api/version")).json().get("version")
    generated = await generate(suite, host)
    judged = await judge_all(suite, generated, host)
    return {
        "meta": {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "evalkit_version": __version__,
            "models": MODELS,
            "judge": JUDGE,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "params": PARAMS.explicit(),
            "ollama_version": ollama,
            "gpu_at_start": gpu,
            "python": platform.python_version(),
            "suite": SUITE.as_posix(),
        },
        "generated": generated,
        "judged": judged,
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
