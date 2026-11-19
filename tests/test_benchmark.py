"""Regression guard: the benchmark numbers the README publishes, from the saved answers.

``examples/responses/benchmark-*.jsonl`` are qwen2.5:7b's and llama3.1:8b's answers to
``examples/suites/benchmark.yaml`` (``scripts/measure_benchmark.py``). If a change to
the checks, the suite or the answer parser moves these numbers, this test says so;
re-run ``measure_benchmark.py --summary`` and update the README with it.
"""

import sys
from pathlib import Path

import pytest

from evalkit.checks import Status
from evalkit.evaluate import check_suite, load_responses
from evalkit.schema import load_suite

ROOT = Path(__file__).resolve().parents[1]
SUITE = load_suite(ROOT / "examples" / "suites" / "benchmark.yaml")
sys.path.insert(0, str(ROOT / "scripts"))  # the script imports its sibling measure_report
try:
    from measure_benchmark import final_answer, is_correct
finally:
    sys.path.remove(str(ROOT / "scripts"))


def answers(model: str) -> dict[str, str]:
    return load_responses(ROOT / "examples" / "responses" / f"benchmark-{model}.jsonl")


@pytest.mark.parametrize(
    ("model", "passed", "failed", "skipped"),
    [("qwen2.5-7b", 36, 3, 1), ("llama3.1-8b", 31, 7, 2)],
)
def test_check_results(model: str, passed: int, failed: int, skipped: int) -> None:
    counts = check_suite(SUITE, answers(model)).counts
    assert (counts[Status.PASS], counts[Status.FAIL], counts[Status.SKIP]) == (
        passed,
        failed,
        skipped,
    )


@pytest.mark.parametrize(
    ("model", "right"),
    [
        ("qwen2.5-7b", {"en-price-change", "en-weekday", "en-letter-count", "en-mangoes",
                        "id-price-change", "id-mangoes"}),
        ("llama3.1-8b", {"en-weekday", "en-mangoes", "id-price-change", "id-mangoes"}),
    ],
)  # fmt: skip
def test_reasoning_trap_answers(model: str, right: set[str]) -> None:
    texts = answers(model)
    traps = [c for c in SUITE.cases if "reasoning-trap" in c.tags]
    assert {c.id for c in traps if is_correct(c, final_answer(texts[c.id]))} == right


def test_answer_parser() -> None:
    price, clock, day = (SUITE.case(i) for i in ("en-price-change", "id-time-zone", "en-weekday"))
    assert final_answer("x\n**Answer:** Rp 100.000\n") == "Rp 100.000"
    assert final_answer("Jawaban: 1\nJawaban: 2") == "2"  # the last label counts
    assert final_answer("The final answer is: $\\boxed{100000}$") is None
    assert is_correct(price, "Rp 100,000")
    assert is_correct(price, "100000 rupiah")
    assert not is_correct(price, "Rp 96.000")
    assert not is_correct(price, None)
    assert is_correct(clock, "11.00 WIT")
    assert is_correct(clock, "11:00")
    assert not is_correct(clock, "10.00")
    assert is_correct(day, "**Monday**")
    assert not is_correct(day, "Mondays")
