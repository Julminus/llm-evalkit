import json
from pathlib import Path

import pytest

from evalkit.agreement import (
    LabelsError,
    Rating,
    Ratings,
    compare,
    consistency,
    load_judgments,
    load_labels,
    load_ratings,
    write_label_sheet,
)
from evalkit.checks import Status
from evalkit.judge import JudgeRecord
from evalkit.schema import Dimension
from evalkit.taxonomy import FailureMode

ROOT = Path(__file__).resolve().parents[1]
SMOKE_LABELS = ROOT / "examples" / "labels" / "smoke.csv"

IF = Dimension.INSTRUCTION_FOLLOWING
RQ = Dimension.REASONING_QUALITY
FA = Dimension.FAITHFULNESS
HE = Dimension.HELPFULNESS


def write(tmp_path: Path, text: str, name: str = "labels.csv", encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


def record(
    case_id: str,
    scores: dict[Dimension, int] | None = None,
    *,
    sample: int = 0,
    language: str = "en",
    judge_modes: list[FailureMode] | None = None,
    check_modes: list[FailureMode] | None = None,
    error: str | None = None,
) -> JudgeRecord:
    return JudgeRecord(
        id=case_id,
        language=language,  # type: ignore[arg-type]
        judge="mock:j",
        scores={} if error else (scores if scores is not None else {IF: 4}),
        rationale=None if error else "r",
        judge_failure_modes=judge_modes or [],
        check_status=Status.PASS,
        check_failure_modes=check_modes or [],
        error=error,
        sample=sample,
    )


def write_judgments(tmp_path: Path, records: list[JudgeRecord], name: str = "j.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(
        "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8", newline="\n"
    )
    return path


def ratings(scores: dict[str, dict[Dimension, int]], source: str = "x", **modes: object) -> Ratings:
    return Ratings(
        source,
        {
            i: Rating(id=i, scores=s, failure_modes=modes.get(i))  # type: ignore[arg-type]
            for i, s in scores.items()
        },
    )


class TestLoadLabels:
    def test_the_example_sheet(self) -> None:
        labels = load_labels(SMOKE_LABELS)
        assert len(labels.ratings) == 10
        assert labels.skipped == []
        bullets = labels.ratings["id-bullets-001"]
        assert bullets.scores == {IF: 3, RQ: 5, FA: 5, HE: 4}
        assert bullets.failure_modes == {FailureMode.FORMAT_ERROR}
        # n/a is "not rated", not a score
        assert labels.ratings["en-json-001"].scores == {IF: 4, FA: 5}

    def test_byte_order_mark_and_extra_columns(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            "﻿id,notes,instruction_following,prompt\nc1,fine,5,hello\n",
        )
        labels = load_labels(path)
        assert labels.ratings["c1"].scores == {IF: 5}
        assert labels.ratings["c1"].failure_modes is None  # column absent: not labelled

    def test_semicolon_delimited_sheet(self, tmp_path: Path) -> None:
        # What Excel saves in an Indonesian locale.
        path = write(
            tmp_path, "id;instruction_following;failure_modes\nc1;2;format_error, hallucination\n"
        )
        rating = load_labels(path).ratings["c1"]
        assert rating.scores == {IF: 2}
        assert rating.failure_modes == {FailureMode.FORMAT_ERROR, FailureMode.HALLUCINATION}

    def test_tab_delimited_sheet(self, tmp_path: Path) -> None:
        path = write(tmp_path, "id\tinstruction_following\nc1\t3\n")
        assert load_labels(path).ratings["c1"].scores == {IF: 3}

    def test_crlf_and_header_case(self, tmp_path: Path) -> None:
        path = write(tmp_path, "ID,Instruction_Following\r\nc1,4\r\n")
        assert load_labels(path).ratings["c1"].scores == {IF: 4}

    @pytest.mark.parametrize("written", ["Format Error", "format-error", " FORMAT_ERROR "])
    def test_failure_mode_spellings(self, tmp_path: Path, written: str) -> None:
        path = write(tmp_path, f'id,failure_modes\nc1,"{written}"\n')
        assert load_labels(path).ratings["c1"].failure_modes == {FailureMode.FORMAT_ERROR}

    @pytest.mark.parametrize("written", ["none", "[]"])
    def test_explicitly_no_failure_mode(self, tmp_path: Path, written: str) -> None:
        path = write(tmp_path, f"id,instruction_following,failure_modes\nc1,5,{written}\n")
        assert load_labels(path).ratings["c1"].failure_modes == frozenset()

    def test_blank_failure_modes_means_none_applies(self, tmp_path: Path) -> None:
        path = write(tmp_path, "id,instruction_following,failure_modes\nc1,5,\n")
        assert load_labels(path).ratings["c1"].failure_modes == frozenset()

    def test_unlabelled_rows_are_skipped(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            "id,instruction_following,helpfulness,failure_modes,notes\n"
            "c1,,,,still to do\n"
            "c2,n/a,N/A,-,\n"
            "c3,4,,,\n"
            ",,,,\n"
            "\n",
        )
        labels = load_labels(path)
        assert list(labels.ratings) == ["c3"]
        assert labels.skipped == ["c1", "c2"]

    def test_short_rows_are_blank_cells(self, tmp_path: Path) -> None:
        path = write(tmp_path, "id,instruction_following,helpfulness\nc1,4\n")
        assert load_labels(path).ratings["c1"].scores == {IF: 4}

    @pytest.mark.parametrize("bad", ["0", "6", "4.5", "four", "+3", "٣"])
    def test_score_must_be_a_whole_number_1_to_5(self, tmp_path: Path, bad: str) -> None:
        path = write(tmp_path, f"id,instruction_following\nc1,{bad}\n")
        with pytest.raises(LabelsError, match=r"labels\.csv:2: instruction_following must be"):
            load_labels(path)

    def test_unknown_failure_mode(self, tmp_path: Path) -> None:
        path = write(tmp_path, "id,failure_modes\nc1,toxicity\n")
        with pytest.raises(
            LabelsError, match="unknown failure mode 'toxicity'; use: constraint_violation"
        ):
            load_labels(path)

    def test_labelled_twice(self, tmp_path: Path) -> None:
        path = write(tmp_path, "id,instruction_following\nc1,4\nc2,3\nc1,5\n")
        with pytest.raises(LabelsError, match=r":4: id 'c1' is labelled twice \(first on line 2\)"):
            load_labels(path)

    def test_labels_without_an_id(self, tmp_path: Path) -> None:
        path = write(tmp_path, "id,instruction_following\n,4\n")
        with pytest.raises(LabelsError, match="has labels but no id"):
            load_labels(path)

    def test_line_numbers_count_multiline_cells(self, tmp_path: Path) -> None:
        path = write(tmp_path, 'id,instruction_following,response\nc1,4,"two\nlines"\nc2,9,x\n')
        with pytest.raises(LabelsError, match=r":4: instruction_following"):
            load_labels(path)

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("", "the file is empty"),
            ("case,instruction_following\nc1,4\n", "no 'id' column"),
            ("id,notes\nc1,x\n", "no label columns"),
            ("id,helpfulness,Helpfulness\nc1,4,4\n", "the header repeats helpfulness"),
        ],
    )
    def test_bad_header(self, tmp_path: Path, text: str, message: str) -> None:
        with pytest.raises(LabelsError, match=message):
            load_labels(write(tmp_path, text))

    def test_not_utf8(self, tmp_path: Path) -> None:
        path = write(tmp_path, "id,instruction_following,notes\nc1,4,café\n", encoding="cp1252")
        with pytest.raises(LabelsError, match="not UTF-8"):
            load_labels(path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(LabelsError, match=r"nope\.csv"):
            load_labels(tmp_path / "nope.csv")

    def test_an_unclosed_quote_does_not_swallow_the_rows_after_it(self, tmp_path: Path) -> None:
        # Lenient CSV reading made the rest of the file one notes cell, and
        # c2, c3 vanished without an error.
        path = write(
            tmp_path, 'id,instruction_following,notes\nc1,4,"forgot to close\nc2,3,\nc3,5,\n'
        )
        with pytest.raises(LabelsError, match=r"not valid CSV: unexpected end of data"):
            load_labels(path)

    def test_quotes_a_person_would_type_still_load(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            'id,instruction_following,notes\nc1,4,said "hi" twice\nc2,3,"a, b and ""c"""\n',
        )
        labels = load_labels(path)
        assert set(labels.ratings) == {"c1", "c2"}

    def test_text_after_a_closing_quote(self, tmp_path: Path) -> None:
        path = write(tmp_path, 'id,instruction_following,notes\nc1,4,"ab"x\n')
        with pytest.raises(LabelsError, match=r":2: not valid CSV"):
            load_labels(path)


class TestLoadJudgments:
    def test_errors_are_skipped(self, tmp_path: Path) -> None:
        path = write_judgments(tmp_path, [record("c1"), record("c2", error="timeout")])
        loaded = load_judgments(path)
        assert list(loaded.ratings) == ["c1"]
        assert loaded.skipped == ["c2"]
        assert loaded.note is None

    def test_counted_failure_modes_and_language(self, tmp_path: Path) -> None:
        path = write_judgments(
            tmp_path,
            [
                record(
                    "c1",
                    language="id",
                    # the judge's format_error doesn't count: a check decides that mode
                    judge_modes=[FailureMode.FORMAT_ERROR, FailureMode.HALLUCINATION],
                    check_modes=[FailureMode.LANGUAGE_MISMATCH],
                )
            ],
        )
        rating = load_judgments(path).ratings["c1"]
        assert rating.failure_modes == {FailureMode.LANGUAGE_MISMATCH, FailureMode.HALLUCINATION}
        assert rating.language == "id"

    def test_picks_a_sample(self, tmp_path: Path) -> None:
        path = write_judgments(
            tmp_path,
            [
                record("c1", {IF: 1}, sample=0),
                record("c1", {IF: 2}, sample=1),
                record("c1", {IF: 3}, sample=2),
            ],
        )
        loaded = load_judgments(path, sample=1)
        assert loaded.ratings["c1"].scores == {IF: 2}
        assert loaded.note == "used sample 1 of 3"
        with pytest.raises(LabelsError, match=r"no sample 5 \(the file has \[0, 1, 2\]\)"):
            load_judgments(path, sample=5)

    def test_one_case_twice_is_an_error(self, tmp_path: Path) -> None:
        path = write_judgments(tmp_path, [record("c1"), record("c1")])
        with pytest.raises(LabelsError, match=r":2: case 'c1' \(sample 0\) appears twice"):
            load_judgments(path)

    def test_not_a_judge_record(self, tmp_path: Path) -> None:
        path = write(tmp_path, '{"id": "c1", "response": "an answer"}\n', name="responses.jsonl")
        with pytest.raises(LabelsError, match=r"responses\.jsonl:1: not a judge record"):
            load_judgments(path)

    def test_empty_file(self, tmp_path: Path) -> None:
        with pytest.raises(LabelsError, match="no judge records"):
            load_judgments(write(tmp_path, "\n\n", name="j.jsonl"))

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(LabelsError, match=r"j\.jsonl"):
            load_judgments(tmp_path / "j.jsonl")

    def test_load_ratings_goes_by_suffix(self, tmp_path: Path) -> None:
        judged = write_judgments(tmp_path, [record("c1")])
        assert load_ratings(judged).ratings["c1"].scores == {IF: 4}
        labels = write(tmp_path, "id,instruction_following\nc1,2\n", name="L.CSV")
        assert load_ratings(labels).ratings["c1"].scores == {IF: 2}


class TestCompare:
    def test_perfect_agreement(self) -> None:
        scores = {f"c{i}": {IF: 1 + i % 5} for i in range(10)}
        report = compare(ratings(scores), ratings(scores), resamples=200)
        d = report.dimensions[IF]
        assert (d.n, d.exact, d.within_1) == (10, 10, 10)
        assert d.kappa == d.kappa_linear == d.kappa_quadratic == d.spearman == 1.0
        assert d.kappa_quadratic_ci == [1.0, 1.0]
        assert d.confusion == [
            [2, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 2],
        ]

    def test_numbers_worked_out_by_hand(self) -> None:
        # reference 1 2 3 4 5, judge 2 2 3 5 5: exact 3/5, within one 5/5.
        ref = ratings({f"c{i}": {IF: s} for i, s in enumerate([1, 2, 3, 4, 5])}, "ref")
        jud = ratings({f"c{i}": {IF: s} for i, s in enumerate([2, 2, 3, 5, 5])}, "judge")
        report = compare(jud, ref, resamples=0)
        d = report.dimensions[IF]
        assert (d.n, d.exact, d.within_1) == (5, 3, 5)
        # Unweighted: p_o = 3/5; p_e = (1*0 + 1*2 + 1*1 + 1*0 + 1*2) / 25 = 5/25
        assert d.kappa == round((3 / 5 - 5 / 25) / (1 - 5 / 25), 3)
        assert d.mean_reference == 3.0
        assert d.mean_judge == 3.4
        assert d.kappa_quadratic_ci is None  # resamples=0
        assert d.confusion[0] == [0, 1, 0, 0, 0]  # the reference's 1 was judged 2
        assert report.judge == "judge"
        assert report.reference == "ref"

    def test_undefined_statistics_are_none(self) -> None:
        same = ratings({f"c{i}": {IF: 5} for i in range(4)})
        d = compare(same, same, resamples=50).dimensions[IF]
        assert d.exact == 4
        assert d.kappa is None
        assert d.spearman is None
        assert d.spearman_ci is None

    def test_only_cases_and_dimensions_both_rated(self) -> None:
        jud = ratings({"c1": {IF: 4, HE: 3}, "c2": {IF: 5}, "j-only": {IF: 1}})
        ref = Ratings(
            "ref",
            ratings({"c1": {IF: 4}, "c2": {IF: 5, HE: 2}, "r-only": {IF: 1}}).ratings,
            skipped=["todo"],
        )
        report = compare(jud, ref, resamples=0)
        assert report.matched == 2
        assert report.dimensions[IF].n == 2
        assert (
            HE not in report.dimensions
        )  # c1 has HE only from the judge, c2 only from the reference
        assert report.only_judge == ["j-only"]
        assert report.only_reference == ["r-only"]
        assert report.skipped_reference == ["todo"]

    def test_by_language_uses_the_judges_language(self) -> None:
        jud = Ratings(
            "j",
            {
                "a": Rating(id="a", scores={IF: 4}, language="en"),
                "b": Rating(id="b", scores={IF: 2}, language="id"),
                "c": Rating(id="c", scores={IF: 3}, language="id"),
            },
        )
        ref = ratings({"a": {IF: 4}, "b": {IF: 3}, "c": {IF: 3}})
        report = compare(jud, ref, resamples=100)
        assert set(report.by_language) == {"en", "id"}
        assert report.by_language["id"][IF].n == 2
        assert report.by_language["id"][IF].exact == 1
        assert report.by_language["id"][IF].kappa_quadratic_ci is None  # no intervals per language

    def test_failure_modes(self) -> None:
        H, C = FailureMode.HALLUCINATION, FailureMode.CONSTRAINT_VIOLATION
        jud = ratings(
            {i: {IF: 3} for i in "abcd"},
            a=frozenset({H}),
            b=frozenset({H}),
            c=frozenset(),
            d=frozenset({C}),
        )
        ref = ratings(
            {i: {IF: 3} for i in "abcd"},
            a=frozenset({H}),
            b=frozenset(),
            c=frozenset({H}),
            d=frozenset({C}),
        )
        report = compare(jud, ref, resamples=0)
        h = report.failure_modes[H]
        assert (h.n, h.both, h.judge_only, h.reference_only, h.neither) == (4, 1, 1, 1, 1)
        assert (h.precision, h.recall) == (0.5, 0.5)
        assert h.decided_by == "judge"
        c = report.failure_modes[C]
        assert c.decided_by == "checks"
        assert (c.both, c.kappa, c.precision, c.recall) == (1, 1.0, 1.0, 1.0)
        over = report.failure_modes[FailureMode.OVER_REFUSAL]
        assert (over.neither, over.kappa, over.precision, over.recall) == (4, None, None, None)

    def test_failure_modes_need_both_raters_to_label_them(self) -> None:
        jud = ratings({"a": {IF: 3}}, a=frozenset())
        ref = ratings({"a": {IF: 3}})  # scores only
        assert compare(jud, ref, resamples=0).failure_modes == {}

    def test_nothing_in_common(self) -> None:
        report = compare(ratings({"a": {IF: 3}}), ratings({"b": {IF: 3}}), resamples=10)
        assert report.matched == 0
        assert report.dimensions == {}
        assert report.by_language == {}

    def test_report_is_json_ready_and_seeded(self) -> None:
        jud = ratings({f"c{i}": {IF: 1 + i % 5} for i in range(20)})
        ref = ratings({f"c{i}": {IF: 1 + (i * 3) % 5} for i in range(20)})
        first = compare(jud, ref, resamples=300, seed=1).to_dict()
        assert json.loads(json.dumps(first)) == first
        assert compare(jud, ref, resamples=300, seed=1).to_dict() == first
        assert first["dimensions"]["instruction_following"]["confusion"][0][0] == 4

    def test_notes_are_passed_on(self) -> None:
        jud = Ratings("j", {}, note="used sample 1 of 3")
        assert compare(jud, ratings({}), resamples=0).notes == ["used sample 1 of 3"]


class TestConsistency:
    def test_stable_and_unstable_cases(self) -> None:
        records = [
            record("a", {IF: 4, HE: 4}, sample=0),
            record("a", {IF: 4, HE: 2}, sample=1),
            record("a", {IF: 4, HE: 3}, sample=2),
            record("b", {IF: 5}, sample=2),
            record("b", {IF: 3}, sample=0),
            record("b", {IF: 5}, sample=1),
        ]
        report = consistency(records)
        assert (report.samples, report.cases, report.errors) == (3, 2, 0)
        d = report.dimensions[IF]
        assert (d.cases, d.all_same, d.max_range, d.range_2_plus) == (2, 1, 2, 1)
        assert d.mode_share == round(5 / 6, 3)
        assert d.mean_range == 1.0
        he = report.dimensions[HE]
        assert (he.cases, he.all_same, he.max_range) == (1, 0, 2)
        assert he.mean_sd == 1.0
        # in sample order, whatever order the records came in
        assert report.unstable == {"a": {HE: [4, 2, 3]}, "b": {IF: [3, 5, 5]}}

    def test_errors_and_single_samples_are_left_out(self) -> None:
        records = [
            record("a", {IF: 4}, sample=0),
            record("a", sample=1, error="timeout"),
            record("b", {IF: 2}, sample=0),
            record("b", {IF: 2}, sample=1),
        ]
        report = consistency(records)
        assert report.errors == 1
        assert report.dimensions[IF].cases == 1
        assert report.dimensions[RQ].cases == 0
        assert report.dimensions[RQ].mode_share is None
        assert report.dimensions[RQ].max_range is None

    def test_judge_owned_failure_modes_only(self) -> None:
        H = FailureMode.HALLUCINATION
        records = [
            record("a", sample=0, judge_modes=[H, FailureMode.FORMAT_ERROR]),
            record("a", sample=1, judge_modes=[H]),
            record("b", sample=0, judge_modes=[H]),
            record("b", sample=1),
            record("c", sample=0),
            record("c", sample=1),
        ]
        report = consistency(records)
        assert FailureMode.FORMAT_ERROR not in report.failure_modes
        h = report.failure_modes[H]
        assert (h.always, h.sometimes, h.never) == (1, 1, 1)

    def test_empty(self) -> None:
        report = consistency([])
        assert (report.samples, report.cases) == (0, 0)
        assert report.to_dict()["unstable"] == {}


class TestLabelSheet:
    def test_round_trip_through_load_labels(self, tmp_path: Path) -> None:
        path = tmp_path / "sheet" / "labels.csv"
        rows = [
            {
                "id": "c1",
                "language": "id",
                "prompt": 'Tulis, "tiga" poin',
                "response": "- satu\n- dua",
            },
            {"id": "c2", "language": "en", "prompt": "p", "response": "r"},
        ]
        n = write_label_sheet(
            path, rows, scored=lambda i: [IF, HE] if i == "c1" else list(Dimension)
        )
        assert n == 2
        data = path.read_bytes()
        assert data.startswith(b"\xef\xbb\xbf")  # Excel needs the BOM to read UTF-8
        assert b"\r\n" not in data
        loaded = load_labels(path)
        assert loaded.ratings == {}  # nothing labelled yet
        assert loaded.skipped == ["c1", "c2"]
        header, first = path.read_text(encoding="utf-8-sig").splitlines()[:2]
        assert header == (
            "id,language,instruction_following,reasoning_quality,faithfulness,helpfulness,"
            "failure_modes,notes,prompt,response"
        )
        assert first.startswith("c1,id,,n/a,n/a,,,,")

    def test_filled_in_sheet_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.csv"
        write_label_sheet(
            path,
            [{"id": "c1", "language": "en", "prompt": "p", "response": "r"}],
            scored=lambda i: [IF],
        )
        text = path.read_text(encoding="utf-8-sig").replace("c1,en,,", "c1,en,2,", 1)
        path.write_text(text, encoding="utf-8-sig")
        assert load_labels(path).ratings["c1"].scores == {IF: 2}
