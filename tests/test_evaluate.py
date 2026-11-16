import json
from pathlib import Path

import pytest

from evalkit.checks import Status
from evalkit.evaluate import (
    CheckSummary,
    ResponsesError,
    check_case,
    check_suite,
    iter_failures,
    load_responses,
)
from evalkit.schema import EvalCase, Suite
from evalkit.taxonomy import FailureMode


@pytest.fixture
def suite() -> Suite:
    return Suite.model_validate(
        {
            "name": "t",
            "cases": [
                {
                    "id": "en-1",
                    "language": "en",
                    "prompt": "Two bullets about tests.",
                    "constraints": [{"type": "bullet_list", "min_items": 2, "max_items": 2}],
                },
                {
                    "id": "id-1",
                    "language": "id",
                    "prompt": "Jelaskan singkat.",
                    "constraints": [{"type": "word_count", "max": 12}],
                },
                {"id": "num-1", "language": "en", "response_language": "any", "prompt": "2+2?"},
            ],
        }
    )


GOOD_EN = "- Tests catch bugs early in the cycle\n- Tests make the refactor safer"
GOOD_ID = "Tes membantu kita menemukan bug lebih awal."
BAD_ID_ENGLISH_AND_LONG = (
    "Tests help us find bugs early, and they also make the code a lot easier to change later on."
)


def test_check_case_passes(suite: Suite) -> None:
    report = check_case(suite.case("en-1"), GOOD_EN)
    assert report.status is Status.PASS
    assert [r.check for r in report.results] == ["bullet_list", "language"]
    assert report.failures == []
    assert report.failure_modes == []


def test_check_case_collects_distinct_failure_modes(suite: Suite) -> None:
    report = check_case(suite.case("id-1"), BAD_ID_ENGLISH_AND_LONG)
    assert report.status is Status.FAIL
    assert {r.check for r in report.failures} == {"word_count", "language"}
    assert report.failure_modes == [
        FailureMode.CONSTRAINT_VIOLATION,
        FailureMode.LANGUAGE_MISMATCH,
    ]


def test_case_with_no_applicable_checks_is_skip(suite: Suite) -> None:
    report = check_case(suite.case("num-1"), "4")
    assert report.results == []
    assert report.status is Status.SKIP


def test_all_skipped_is_skip() -> None:
    case = EvalCase.model_validate({"id": "x", "language": "id", "prompt": "1+1?"})
    report = check_case(case, "2")
    assert [r.status for r in report.results] == [Status.SKIP]
    assert report.status is Status.SKIP


def test_status_is_serialized(suite: Suite) -> None:
    dumped = check_case(suite.case("en-1"), GOOD_EN).model_dump(mode="json")
    assert dumped["status"] == "pass"
    assert dumped["results"][0]["check"] == "bullet_list"


def test_check_suite_summary(suite: Suite) -> None:
    summary = check_suite(suite, {"en-1": GOOD_EN, "id-1": BAD_ID_ENGLISH_AND_LONG, "extra": "?"})
    assert summary.missing == ["num-1"]
    assert summary.unknown == ["extra"]
    assert summary.counts[Status.PASS] == 1
    assert summary.counts[Status.FAIL] == 1
    assert summary.pass_rate == 0.5
    assert summary.pass_rate_by_language() == {"en": 1.0, "id": 0.0}
    assert summary.by_check()["language"][Status.PASS] == 1
    assert summary.by_check()["language"][Status.FAIL] == 1
    assert summary.failure_modes()[FailureMode.LANGUAGE_MISMATCH] == 1
    assert [(cid, r.check) for cid, r in iter_failures(summary)] == [
        ("id-1", "word_count"),
        ("id-1", "language"),
    ]


def test_skips_do_not_count_toward_pass_rate(suite: Suite) -> None:
    summary = check_suite(suite, {"en-1": GOOD_EN, "id-1": GOOD_ID, "num-1": "4"})
    assert summary.counts[Status.SKIP] == 1
    assert summary.pass_rate == 1.0


def test_empty_summary_pass_rate() -> None:
    assert CheckSummary(reports=[]).pass_rate == 0.0


class TestLoadResponses:
    def _write(self, tmp_path: Path, *lines: str) -> Path:
        path = tmp_path / "r.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_loads_and_skips_blank_lines(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            json.dumps({"id": "a", "response": "satu", "model": "ignored"}),
            "",
            json.dumps({"id": "b", "response": "dua"}),
        )
        assert load_responses(path) == {"a": "satu", "b": "dua"}

    def test_keeps_unicode(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path, json.dumps({"id": "a", "response": "café – ok"}, ensure_ascii=False)
        )
        assert load_responses(path)["a"] == "café – ok"

    @pytest.mark.parametrize(
        ("line", "message"),
        [
            ("{not json", "invalid JSON"),
            ('["a", "b"]', "string 'id'"),
            ('{"id": 1, "response": "x"}', "string 'id'"),
            ('{"id": "a"}', "string 'response'"),
        ],
    )
    def test_bad_lines(self, tmp_path: Path, line: str, message: str) -> None:
        with pytest.raises(ResponsesError, match=message):
            load_responses(self._write(tmp_path, line))

    def test_duplicate_id(self, tmp_path: Path) -> None:
        row = json.dumps({"id": "a", "response": "x"})
        with pytest.raises(ResponsesError, match=r"r.jsonl:2: duplicate id 'a'"):
            load_responses(self._write(tmp_path, row, row))

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ResponsesError, match="cannot read file"):
            load_responses(tmp_path / "nope.jsonl")
