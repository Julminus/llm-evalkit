"""Measure the EN/ID language detector on the hand-reviewed held-out corpus.

Reports how often the detector makes a call at all (coverage: enough function
words to decide) and how often that call is right (accuracy on decided rows),
broken down by true language and response length. Rows labelled ``null``
(language-neutral answers such as "Tokyo.") are reported separately.

If the third-party ``langdetect`` package is importable, it is scored on the
same rows as a baseline:
    uv run --with langdetect==1.0.9 python scripts/measure_langdetect.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evalkit.checks import LanguageCheck, Status
from evalkit.langdetect import detect_language
from evalkit.text import words

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "tests/fixtures/langdetect_heldout.jsonl"


def _bucket(n_words: int) -> str:
    return "<10 words" if n_words < 10 else "10-39 words" if n_words < 40 else "40+ words"


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labelled = [r for r in rows if r["label"] is not None]
    decided = correct = 0
    by_lang: dict[str, Counter[str]] = {"en": Counter(), "id": Counter()}
    by_len: dict[str, Counter[str]] = {}
    errors = []
    check_outcomes: Counter[str] = Counter()
    for r in labelled:
        guess = detect_language(r["text"])
        n_words = len(words(r["text"]))
        bucket = by_len.setdefault(_bucket(n_words), Counter())
        lang = by_lang[r["label"]]
        for c in (bucket, lang):
            c["rows"] += 1
        # The check's own semantics: skip below 2 signal words, then compare.
        result = LanguageCheck(expected=r["label"]).check(r["text"])
        check_outcomes[result.status.value] += 1
        if result.status is Status.SKIP:
            continue
        decided += 1
        for c in (bucket, lang):
            c["decided"] += 1
        if guess.language == r["label"]:
            correct += 1
            for c in (bucket, lang):
                c["correct"] += 1
        if result.status is Status.FAIL:
            errors.append((r["id"], r["label"], guess, result.message, r["text"][:120]))
    return {
        "rows": len(rows),
        "labelled": len(labelled),
        "decided": decided,
        "correct": correct,
        "by_lang": by_lang,
        "by_len": by_len,
        "check_outcomes": check_outcomes,
        "errors": errors,
        "neutral": [
            (r["id"], detect_language(r["text"]).signal_words) for r in rows if r["label"] is None
        ],
    }


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({num / den:.1%})" if den else "n/a"


def baseline_langdetect(rows: list[dict[str, Any]]) -> list[str] | None:
    """Score seeded ``langdetect`` on the same rows, split the way the README reports it:
    the rows evalkit decided, the rows it abstained on, and every wrong guess."""
    try:
        from langdetect import DetectorFactory, detect
    except ImportError:
        return None
    DetectorFactory.seed = 0  # langdetect is randomized unless seeded
    labelled = [r for r in rows if r["label"] is not None]
    hits: dict[str, Counter[str]] = {"decided": Counter(), "abstained": Counter()}
    wrong: list[str] = []
    for r in labelled:
        guess = "error"
        with contextlib.suppress(Exception):  # raises on text with no letters
            guess = detect(r["text"])
        ours = LanguageCheck(expected=r["label"]).check(r["text"]).status
        group = hits["abstained" if ours is Status.SKIP else "decided"]
        group["rows"] += 1
        group["correct"] += guess == r["label"]
        if guess != r["label"]:
            wrong.append(f"{r['id']} (true {r['label']}): {guess!r} for {r['text'][:30]!r}")
    total = sum(g["correct"] for g in hits.values())
    lines = [
        f"all labelled rows: {_pct(total, len(labelled))}",
        "  on rows evalkit decided:   " + _pct(hits["decided"]["correct"], hits["decided"]["rows"]),
        "  on rows evalkit abstained: "
        + _pct(hits["abstained"]["correct"], hits["abstained"]["rows"]),
    ]
    lines += [f"  wrong: {w}" for w in wrong]
    neutral = [r for r in rows if r["label"] is None]
    for r in neutral:
        guess = "error"
        with contextlib.suppress(Exception):
            guess = detect(r["text"])
        lines.append(f"  neutral row {r['id']}: {guess!r} for {r['text'][:30]!r}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    res = evaluate(rows)

    print(f"corpus: {args.corpus.name}, {res['rows']} rows, {res['labelled']} with a language")
    print(f"coverage (decided / labelled): {_pct(res['decided'], res['labelled'])}")
    print(f"accuracy on decided rows:      {_pct(res['correct'], res['decided'])}")
    print(f"LanguageCheck outcomes:        {dict(res['check_outcomes'])}")
    groups = [(lang, c) for lang, c in res["by_lang"].items()]
    groups += [
        (b, res["by_len"].get(b, Counter())) for b in ("<10 words", "10-39 words", "40+ words")
    ]
    for name, c in groups:
        decided = _pct(c["decided"], c["rows"])
        print(f"  {name:>11}: decided {decided}, correct {_pct(c['correct'], c['decided'])}")
    print(f"language-neutral rows (label null): {res['neutral']}")
    if res["errors"]:
        print("misclassified or flagged as mixed:")
        for rid, label, guess, message, snippet in res["errors"]:
            print(f"  {rid} (true {label}): {message}; hits={guess.hits} :: {snippet!r}")
    baseline = baseline_langdetect(rows)
    if baseline:
        print("baseline langdetect 1.0.9 (seeded):")
        for line in baseline:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
