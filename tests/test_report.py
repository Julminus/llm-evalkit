import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalkit.checks import Status
from evalkit.cli import app
from evalkit.judge import JudgeRecord
from evalkit.report import (
    CaseView,
    RunError,
    RunView,
    build_report,
    default_name,
    diff_runs,
    load_run,
    load_runs,
    render_html,
    render_markdown,
)
from evalkit.schema import Dimension, load_suite
from evalkit.taxonomy import FailureMode

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SUITE = ROOT / "examples" / "suites" / "smoke.yaml"
SMOKE_RESPONSES = ROOT / "examples" / "responses" / "smoke.jsonl"
SMOKE_B = ROOT / "examples" / "responses" / "smoke-b.jsonl"
SMOKE_JUDGMENTS = ROOT / "examples" / "judgments" / "smoke.jsonl"
SUITE = load_suite(SMOKE_SUITE)
IF, HE = Dimension.INSTRUCTION_FOLLOWING, Dimension.HELPFULNESS
CV, FE, HA = FailureMode.CONSTRAINT_VIOLATION, FailureMode.FORMAT_ERROR, FailureMode.HALLUCINATION
NOW = datetime(2026, 11, 19, tzinfo=UTC)

cli = CliRunner()


def case(
    case_id: str,
    status: str = "pass",
    modes: tuple[FailureMode, ...] = (),
    *,
    scores: dict[Dimension, int] | None = None,
    judge_modes: tuple[FailureMode, ...] = (),
    error: str | None = None,
    response: str | None = None,
) -> CaseView:
    record = None
    if scores is not None or error is not None:
        record = JudgeRecord(
            id=case_id,
            language="en",
            judge="mock:default",
            scores=scores or {},
            rationale=None if error else "fine",
            error=error,
            judge_failure_modes=list(judge_modes),
            check_status=Status(status),
            check_failure_modes=list(modes),
        )
    return CaseView(case_id, "en", Status(status), modes, None, response, record)


def run(name: str, *cases: CaseView) -> RunView:
    judged = any(c.record is not None for c in cases)
    return RunView(name, f"{name}.jsonl", "judgments" if judged else "responses", list(cases))


@pytest.fixture
def judged_run(tmp_path: Path) -> Path:
    """A generation run of the smoke responses and a mock-judge run of it."""
    dirs = ["--out", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "cache")]
    gen = ["run", str(SMOKE_SUITE), "-p", "mock", "--mock-responses", str(SMOKE_RESPONSES)]
    assert cli.invoke(app, [*gen, *dirs, "--run-id", "gen"]).exit_code == 0
    judge = [
        "judge",
        str(SMOKE_SUITE),
        str(tmp_path / "runs" / "gen" / "responses.jsonl"),
        "-j",
        "mock",
        "--mock-judgments",
        str(SMOKE_JUDGMENTS),
    ]
    assert cli.invoke(app, [*judge, *dirs, "--run-id", "judged"]).exit_code == 0
    return tmp_path / "runs" / "judged"


# ---------------------------------------------------------------- reading runs


def test_responses_file_gives_the_check_results_of_evalkit_check() -> None:
    view = load_run(SMOKE_RESPONSES, SUITE)
    assert (view.name, view.kind, view.model, view.judge) == ("smoke", "responses", None, None)
    # The same numbers `evalkit check` prints for this file.
    assert view.pass_rate() == (6, 10)
    assert view.pass_rate("en") == (4, 5)
    assert view.pass_rate("id") == (2, 5)
    assert view.mode_counts() == {FE: 3, FailureMode.LANGUAGE_MISMATCH: 1}
    assert view.summary() is None
    failing = [c for c in view.cases if c.status is Status.FAIL]
    assert all(c.failed_checks for c in failing)
    assert all(c.response is not None for c in view.cases)


def test_judge_run_reads_the_judged_responses_and_their_model(judged_run: Path) -> None:
    view = load_run(judged_run, SUITE)
    assert view.kind == "judgments"
    assert (view.name, view.model, view.judge) == ("mock:default", "mock:default", "mock:default")
    assert view.judge_prompt_version == "1"
    assert view.notes == []
    summary = view.summary()
    assert summary is not None
    # The numbers `evalkit judge` prints for the same judgments.
    assert summary.mean_scores()[IF] == 4.1
    assert summary.mean_scores()[HE] == 4.429
    assert view.pass_rate() == (6, 10)
    assert all(c.response is not None and c.failed_checks is not None for c in view.cases)
    assert load_run(judged_run / "judgments.jsonl", SUITE).name == "mock:default"


def test_judge_run_without_its_responses_still_loads(judged_run: Path, tmp_path: Path) -> None:
    (tmp_path / "runs" / "gen" / "responses.jsonl").unlink()
    view = load_run(judged_run, SUITE)
    assert view.pass_rate() == (6, 10)
    assert all(c.response is None and c.failed_checks is None for c in view.cases)
    assert "can't be read from here" in view.notes[0]
    assert "no response text" in view.notes[1]


def test_judge_run_whose_responses_changed(judged_run: Path, tmp_path: Path) -> None:
    log = tmp_path / "runs" / "gen" / "responses.jsonl"
    first, *rest = log.read_text(encoding="utf-8").splitlines()
    edited = json.dumps({**json.loads(first), "response": "edited"})
    log.write_text("\n".join([edited, *rest]) + "\n", encoding="utf-8")
    view = load_run(judged_run, SUITE)
    assert "changed after they were judged" in view.notes[0]
    assert all(c.response is None for c in view.cases)


def test_judgments_file_on_its_own(judged_run: Path, tmp_path: Path) -> None:
    alone = tmp_path / "alone" / "qwen.jsonl"
    alone.parent.mkdir()
    shutil.copy(judged_run / "judgments.jsonl", alone)
    view = load_run(alone, SUITE)
    assert (view.name, view.model, view.judge, view.notes) == ("qwen", None, "mock:default", [])


def test_several_samples_use_the_first(judged_run: Path, tmp_path: Path) -> None:
    lines = (judged_run / "judgments.jsonl").read_text(encoding="utf-8").splitlines()
    second = [json.dumps({**json.loads(line), "sample": 1, "scores": {}}) for line in lines]
    path = tmp_path / "samples.jsonl"
    path.write_text("\n".join(second + lines) + "\n", encoding="utf-8")
    view = load_run(path, SUITE)
    assert view.notes == ["2 samples per case; this report uses sample 0"]
    summary = view.summary()
    assert summary is not None
    assert summary.mean_scores()[IF] == 4.1


def test_suite_digest_mismatch_is_noted(tmp_path: Path) -> None:
    shutil.copy(SMOKE_RESPONSES, tmp_path / "responses.jsonl")
    manifest = {"suite": {"digest": "0000"}, "provider": {"spec": "ollama:x"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    view = load_run(tmp_path, SUITE)
    assert view.name == "ollama:x"
    assert view.notes == ["made with a different version of the suite (its digest differs)"]


def test_unreadable_manifest_is_ignored(tmp_path: Path) -> None:
    shutil.copy(SMOKE_RESPONSES, tmp_path / "responses.jsonl")
    (tmp_path / "manifest.json").write_text("[1, 2", encoding="utf-8")
    assert load_run(tmp_path, SUITE).name == tmp_path.name
    (tmp_path / "manifest.json").write_text("[1, 2]", encoding="utf-8")
    assert load_run(tmp_path, SUITE).name == tmp_path.name


def test_missing_and_unknown_cases(tmp_path: Path) -> None:
    path = tmp_path / "part.jsonl"
    rows = [{"id": "en-math-001", "response": "42"}, {"id": "nope", "response": "x"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    view = load_run(path, SUITE)
    assert [c.id for c in view.cases] == ["en-math-001"]
    assert len(view.missing) == 9
    assert view.unknown == ["nope"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b'{"id": "a", "response": "x"}\n{oops\n', "invalid JSON"),
        (b"\xff\xfe{\x00", "UTF-16"),
        (b'{"id": "a", "check_status": "pass"}\n', "not a judge record"),
    ],
)
def test_unreadable_files(tmp_path: Path, content: bytes, message: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_bytes(content)
    with pytest.raises(RunError, match=message):
        load_run(path, SUITE)


def test_directory_without_a_log(tmp_path: Path) -> None:
    with pytest.raises(RunError, match=r"no judgments\.jsonl or responses\.jsonl"):
        load_run(tmp_path, SUITE)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RunError, match=r"nope\.jsonl"):
        load_run(tmp_path / "nope.jsonl", SUITE)


def test_names_and_duplicates() -> None:
    runs = load_runs([SMOKE_RESPONSES, SMOKE_RESPONSES, SMOKE_B], SUITE, ["base"])
    assert [r.name for r in runs] == ["base", "smoke", "smoke-b"]
    runs = load_runs([SMOKE_RESPONSES, SMOKE_RESPONSES, SMOKE_RESPONSES], SUITE)
    assert [r.name for r in runs] == ["smoke", "smoke (2)", "smoke (3)"]
    with pytest.raises(RunError, match="2 names for 1 runs"):
        load_runs([SMOKE_RESPONSES], SUITE, ["a", "b"])


def test_default_name() -> None:
    assert default_name(Path("runs/r1/responses.jsonl")) == "r1"
    assert default_name(Path("runs/r1/judgments.jsonl")) == "r1"
    assert default_name(Path("out/qwen.jsonl")) == "qwen"


# ---------------------------------------------------------------- diffs


def test_check_regressions_and_fixes() -> None:
    base = run("base", case("a"), case("b", "fail", (FE,)), case("c"), case("d", "skip"))
    new = run("new", case("a", "fail", (CV,)), case("b"), case("c"), case("d", "fail", (CV,)))
    diff = diff_runs(base, new)
    assert diff.shared == ["a", "b", "c", "d"]
    assert [c.id for c in diff.changes] == ["a", "b", "d"]
    assert [c.id for c in diff.worse] == ["a", "d"]  # skip -> fail is worse too
    assert [c.id for c in diff.better] == ["b"]
    a = diff.changes[0]
    assert (a.status, a.added, a.removed) == ((Status.PASS, Status.FAIL), (CV,), ())
    assert diff.pass_rates() == ((2, 3), (2, 4))
    assert diff.mode_counts() == {CV: (0, 2), FE: (1, 0)}
    assert diff.text_changed is None


def test_status_moves_that_are_neither_worse_nor_better() -> None:
    diff = diff_runs(run("b", case("a")), run("n", case("a", "skip")))
    [change] = diff.changes
    assert not change.worse
    assert not change.better


def test_score_changes_and_min_change() -> None:
    base = run("b", case("a", scores={IF: 5, HE: 3}), case("b", scores={IF: 4}))
    new = run("n", case("a", scores={IF: 4, HE: 4}), case("b", scores={IF: 2}))
    diff = diff_runs(base, new)
    a, b = diff.changes
    assert a.scores == {IF: (5, 4), HE: (3, 4)}
    assert (a.worse, a.better) == (True, True)
    assert (b.worse, b.better) == (True, False)
    assert diff.mean_scores() == {IF: (4.5, 3.0, 2), HE: (3.0, 4.0, 1)}
    strict = diff_runs(base, new, min_change=2)
    assert [c.id for c in strict.changes] == ["b"]
    assert strict.changes[0].scores == {IF: (4, 2)}
    with pytest.raises(ValueError, match="min_change"):
        diff_runs(base, new, min_change=0)


def test_judge_modes_count_only_where_both_runs_judged() -> None:
    base = run("b", case("a", scores={IF: 4}), case("b", scores={IF: 4}, judge_modes=(HA,)))
    new = run("n", case("a", scores={IF: 4}, judge_modes=(HA, CV)), case("b", error="bad reply"))
    diff = diff_runs(base, new)
    # a: the judge's hallucination tag counts, its constraint_violation tag doesn't.
    # b: judged in the base run only, so its lost hallucination tag isn't a change.
    assert [(c.id, c.added) for c in diff.changes] == [("a", (HA,))]
    assert diff.unjudged == ["b"]


def test_judge_run_against_a_responses_run_compares_the_checks() -> None:
    base = run("b", case("a", scores={IF: 4}, judge_modes=(HA,)))
    new = run("n", case("a"))
    diff = diff_runs(base, new)
    assert diff.changes == []
    assert diff.unjudged == []
    assert diff.mean_scores() == {}


def test_lost_gained_and_text() -> None:
    base = run("b", case("a", response="x"), case("b", response="y"), case("c", response="z"))
    new = run("n", case("a", response="x"), case("b", response="Y"), case("d", response="w"))
    diff = diff_runs(base, new)
    assert (diff.lost, diff.gained, diff.text_changed) == (["c"], ["d"], 1)
    payload = diff.to_dict()
    assert payload["lost"] == ["c"]
    assert payload["text_changed"] == 1
    assert payload["pass_rate"] == {"base": [2, 2], "new": [2, 2]}


def test_to_dict() -> None:
    base = run("b", case("a", scores={IF: 5}))
    new = run("n", case("a", "fail", (CV,), scores={IF: 3}))
    payload = diff_runs(base, new).to_dict()
    assert payload["worse"] == ["a"]
    assert payload["mean_scores"] == {"instruction_following": {"base": 5, "new": 3, "cases": 1}}
    assert payload["failure_modes"] == {"constraint_violation": {"base": 0, "new": 1}}
    assert payload["changes"] == [
        {
            "id": "a",
            "language": "en",
            "checks": ["pass", "fail"],
            "scores": {"instruction_following": [5, 3]},
            "added": ["constraint_violation"],
            "removed": [],
        }
    ]


# ---------------------------------------------------------------- the report


def test_markdown_report_of_two_runs() -> None:
    runs = load_runs([SMOKE_RESPONSES, SMOKE_B], SUITE)
    md = render_markdown(build_report(SUITE, runs, now=NOW))
    assert md.startswith("# Evaluation report: smoke\n\n- Suite smoke v1: 10 cases (5 en, 5 id)")
    assert "on 2026-11-19 00:00 UTC." in md
    assert "| smoke | 60.0% (6/10) | 80.0% (4/5) | 40.0% (2/5) |\n" in md
    assert "| smoke-b | 80.0% (8/10) | 60.0% (3/5) | 100.0% (5/5) |\n" in md
    assert "## Changes: smoke → smoke-b\n\n10 cases in both runs: 2 worse, 4 better;" in md
    assert "| en-bullets-001 | en | pass → fail | +format_error |\n" in md
    assert "Judge scores" not in md  # no judge run, no judge tables
    assert md.endswith("\n")
    assert "\n\n\n" not in md


def test_markdown_report_with_a_judge_run(judged_run: Path) -> None:
    md = render_markdown(build_report(SUITE, load_runs([judged_run], SUITE), now=NOW))
    assert (
        "| mock:default | 60.0% (6/10) | 80.0% (4/5) | 40.0% (2/5) | 10/10 | 4.10 | 5.00 | 5.00 | 4.43 |"
        in md
    )
    assert "| mock:default | instruction_following | 0 | 1 | 3 | 0 | 6 | 4.10 |" in md
    assert "### Judge tags that don't count" in md
    assert "Changes" not in md


def test_markdown_cells_are_escaped() -> None:
    runs = load_runs([SMOKE_RESPONSES], SUITE, ["a|b\nc<d>\\"])
    md = render_markdown(build_report(SUITE, runs, now=NOW))
    assert "| a\\|b c&lt;d>\\\\ |" in md


def test_html_escapes_everything_from_runs(tmp_path: Path) -> None:
    path = tmp_path / "evil.jsonl"
    rows = [{"id": "en-math-001", "response": "<script>alert(1)</script>"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    runs = load_runs([path], SUITE, ['<img src=x onerror="y">'])
    html = render_html(build_report(SUITE, runs, now=NOW))
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<img" not in html
    # Self-contained: nothing loaded from anywhere.
    assert "<link" not in html
    assert "src=" not in html.replace("&lt;img src=x", "")


def test_html_report_shows_judge_details(judged_run: Path) -> None:
    runs = load_runs([judged_run, SMOKE_B], SUITE, ["judged", "b"])
    html = render_html(build_report(SUITE, runs, now=NOW))
    assert '<section class="case" id="case-en-bullets-001">' in html
    assert '<a href="#case-en-bullets-001">en-bullets-001</a>' in html
    assert "<b>Judge:</b>" in html
    assert "Only one of the two runs was judged" in html


def test_html_report_shows_judge_errors_and_overruled_tags(tmp_path: Path) -> None:
    good = case("en-math-001", scores={IF: 4}, judge_modes=(CV,))
    bad = case("id-json-001", error="unusable judge reply")
    html = render_html(build_report(SUITE, [run("j", good, bad)], now=NOW))
    assert "No judgment: unusable judge reply" in html
    assert "The judge also named constraint_violation, which no check found" in html


def test_diff_of_two_judge_runs_in_the_report() -> None:
    base = run(
        "base",
        case("en-math-001", scores={IF: 5}),
        case("id-json-001", scores={IF: 4}),
        case("en-json-001", scores={IF: 3}),
        case("id-summary-001", scores={IF: 3}),
    )
    new = run(
        "new",
        case("en-math-001", scores={IF: 2}),
        case("id-json-001", error="unusable"),
        case("en-json-001", scores={IF: 4}),
        case("en-bullets-001", scores={IF: 4}),
    )
    md = render_markdown(build_report(SUITE, [base, new], min_change=2, now=NOW))
    assert "| instruction_following (mean of 2) | 4.00 | 3.00 | -1.00 |" in md
    assert "- In base only: id-summary-001\n- In new only: en-bullets-001\n" in md
    assert "- Judged in one run only (scores and judge tags not compared): id-json-001" in md
    assert "Score moves under 2 points are ignored." in md
    # en-json-001 moved 1 point, under --min-change 2.
    assert "| Case | Language | Checks | IF | RQ | FA | HE | Failure modes |" in md
    assert "| en-math-001 | en | pass | 5 → 2 |  |  |  |  |" in md
    assert "en-json-001 | en" not in md


def test_first_line_that_is_not_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{oops\n", encoding="utf-8")
    with pytest.raises(RunError, match="invalid JSON"):
        load_run(path, SUITE)


def test_judgments_for_some_cases_only(judged_run: Path, tmp_path: Path) -> None:
    first = (judged_run / "judgments.jsonl").read_text(encoding="utf-8").splitlines()[0]
    path = tmp_path / "one.jsonl"
    path.write_text(first + "\n", encoding="utf-8")
    view = load_run(path, SUITE)
    assert [c.id for c in view.cases] == [json.loads(first)["id"]]
    assert len(view.missing) == 9


def test_report_needs_a_run() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        build_report(SUITE, [])
