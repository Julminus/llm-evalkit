"""The S4 commands: agree, labels, consistency, pairwise."""

import csv
import json
import re
from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from evalkit import cli
from evalkit.cli import app
from evalkit.prompts import Prompt
from evalkit.providers import Completion, GenerationParams, Provider, ProviderError

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "examples" / "suites" / "smoke.yaml"
RESPONSES = ROOT / "examples" / "responses" / "smoke.jsonl"
RESPONSES_B = ROOT / "examples" / "responses" / "smoke-b.jsonl"
JUDGMENTS = ROOT / "examples" / "judgments" / "smoke.jsonl"
PAIRWISE = ROOT / "examples" / "judgments" / "smoke-pairwise.jsonl"
LABELS = ROOT / "examples" / "labels" / "smoke.csv"

runner = CliRunner()


@pytest.fixture
def dirs(tmp_path: Path) -> list[str]:
    return ["--out", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "cache")]


def invoke(*args: object):  # type: ignore[no-untyped-def]
    return runner.invoke(app, [str(a) for a in args])


def mock_judgments(tmp_path: Path, dirs: list[str]) -> Path:
    """judgments.jsonl of the mock judge on the example responses."""
    result = invoke(
        "judge",
        SUITE,
        RESPONSES,
        "-j",
        "mock",
        "--mock-judgments",
        JUDGMENTS,
        *dirs,
        "--run-id",
        "j",
    )
    assert result.exit_code == 0, result.output
    return tmp_path / "runs" / "j" / "judgments.jsonl"


class _Seeded(Provider):
    """A judge whose instruction_following score is 1 + seed % 3, so samples disagree."""

    name = "seeded"
    supported_params: ClassVar[frozenset[str]] = frozenset({"temperature", "seed", "max_tokens"})

    def __init__(self, model: str, fail_on: str | None = None) -> None:
        super().__init__(model)
        self.fail_on = fail_on
        self.params: list[GenerationParams] = []

    async def generate(self, prompt: Prompt, params: GenerationParams) -> Completion:
        self.params.append(params)
        if self.fail_on and self.fail_on in prompt.user:
            raise ProviderError("server down", retryable=False)
        score = 1 + (params.seed or 0) % 3
        text = json.dumps(
            {
                "rationale": "r",
                "scores": {
                    "instruction_following": score,
                    "reasoning_quality": 5,
                    "faithfulness": 5,
                    "helpfulness": 4,
                },
                "failure_modes": ["hallucination"] if score == 1 else [],
            }
        )
        return Completion(text=text, model=self.model, finish_reason="stop")


class TestAgree:
    def test_mock_judge_vs_the_example_labels(self, tmp_path: Path, dirs: list[str]) -> None:
        result = invoke("agree", mock_judgments(tmp_path, dirs), LABELS)
        assert result.exit_code == 0, result.output
        out = result.output
        # Table cells wrap at the terminal width; their values are pinned in test_json.
        assert "10 cases rated by both" in out
        assert "Scores: judge vs reference" in out
        assert "Failure modes: does each apply?" in out
        assert "percentile bootstrap over 2000 resamples of cases (seed 0)" in out
        assert "  id: IF n 5, exact 5, quadratic kappa 1.00" in out
        assert "neither tagged constraint_violation, intent_misread" in out

    def test_json(self, tmp_path: Path, dirs: list[str]) -> None:
        result = invoke(
            "agree", mock_judgments(tmp_path, dirs), LABELS, "--json", "--resamples", "0"
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        d = data["dimensions"]["instruction_following"]
        assert (d["n"], d["exact"], d["kappa_quadratic"], d["kappa_quadratic_ci"]) == (
            10,
            9,
            0.96,
            None,
        )
        assert data["failure_modes"]["format_error"]["both"] == 3
        assert data["matched"] == 10

    def test_verbose_prints_confusion_matrices(self, tmp_path: Path, dirs: list[str]) -> None:
        result = invoke("agree", mock_judgments(tmp_path, dirs), LABELS, "-v", "--resamples", "0")
        assert result.exit_code == 0, result.output
        assert "instruction_following: rows = reference, columns = judge" in result.output
        assert "[95% CI]" not in result.output

    def test_judge_vs_judge_and_a_half_filled_sheet(self, tmp_path: Path, dirs: list[str]) -> None:
        judged = mock_judgments(tmp_path, dirs)
        sheet = tmp_path / "half.csv"
        rows = LABELS.read_text(encoding="utf-8").splitlines()
        # keep the first 3 labelled; blank the label cells of the rest
        blanked = [
            re.sub(r"^([^,]+,[^,]+),[^,]*,[^,]*,[^,]*,[^,]*,[^,]*", r"\1,,,,,", r) for r in rows[4:]
        ]
        sheet.write_text("\n".join(rows[:4] + blanked) + "\n", encoding="utf-8")
        result = invoke("agree", judged, sheet)
        assert result.exit_code == 0, result.output
        assert "3 cases rated by both" in result.output
        assert "not labelled yet (skipped) (7):" in result.output
        same = invoke("agree", judged, judged, "--resamples", "0")
        assert same.exit_code == 0
        assert "10 cases rated by both" in same.output

    def test_nothing_in_common_fails(self, tmp_path: Path, dirs: list[str]) -> None:
        other = tmp_path / "other.csv"
        other.write_text("id,instruction_following\nnot-a-case,3\n", encoding="utf-8")
        result = invoke("agree", mock_judgments(tmp_path, dirs), other)
        assert result.exit_code == 1
        assert "no case is rated in both files" in result.output
        assert "only the reference rated (1): not-a-case" in result.output
        assert "only the judge rated (10):" in result.output

    def test_bad_labels_file(self, tmp_path: Path, dirs: list[str]) -> None:
        bad = tmp_path / "bad.csv"
        bad.write_text("id,instruction_following\nen-json-001,7\n", encoding="utf-8")
        result = invoke("agree", mock_judgments(tmp_path, dirs), bad)
        assert result.exit_code == 1
        assert "must be a whole number from 1 to 5" in result.output

    def test_a_responses_file_is_not_judgments(self) -> None:
        result = invoke("agree", RESPONSES, LABELS)
        assert result.exit_code == 1
        assert "not a judge record" in result.output


class TestLabels:
    def test_writes_a_sheet_that_agree_reads(self, tmp_path: Path) -> None:
        out = tmp_path / "labels.csv"
        result = invoke("labels", SUITE, RESPONSES, "-o", out)
        assert result.exit_code == 0, result.output
        assert "wrote 10 rows" in result.output
        lines = out.read_text(encoding="utf-8-sig").splitlines()
        assert lines[0].startswith("id,language,instruction_following,")
        # en-json-001's rubric doesn't score reasoning_quality or helpfulness
        assert any(line.startswith("en-json-001,en,,n/a,,n/a,") for line in lines)
        # The text to summarise is the case's context: without it faithfulness
        # can't be labelled (found in the post-S4 review).
        with out.open(encoding="utf-8-sig", newline="") as f:
            rows = {r["id"]: r for r in csv.DictReader(f)}
        assert "Python" in rows["en-summary-001"]["context"]
        assert rows["en-bullets-001"]["response"].startswith("'- ")  # not a formula
        assert set(rows["en-bullets-001"]) >= {"criteria", "reference_answer", "system"}

    def test_refuses_to_overwrite_without_force(self, tmp_path: Path) -> None:
        out = tmp_path / "labels.csv"
        out.write_text("hours of work", encoding="utf-8")
        result = invoke("labels", SUITE, RESPONSES, "-o", out)
        assert result.exit_code == 1
        assert "pass --force to overwrite it" in result.output
        assert out.read_text(encoding="utf-8") == "hours of work"
        forced = invoke("labels", SUITE, RESPONSES, "-o", out, "--force", "--case", "en-math-001")
        assert forced.exit_code == 0, forced.output
        assert len(out.read_text(encoding="utf-8-sig").splitlines()) == 2

    def test_no_responses_for_the_selected_cases(self, tmp_path: Path) -> None:
        empty = tmp_path / "r.jsonl"
        empty.write_text('{"id": "other", "response": "x"}\n', encoding="utf-8")
        result = invoke("labels", SUITE, empty, "-o", tmp_path / "l.csv")
        assert result.exit_code == 1
        assert "no selected case has a response" in result.output

    def test_unwritable_path(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        result = invoke("labels", SUITE, RESPONSES, "-o", blocker / "labels.csv")
        assert result.exit_code == 1
        assert "could not write" in result.output


class TestConsistency:
    def test_mock_is_perfectly_consistent(self, dirs: list[str]) -> None:
        result = invoke(
            "consistency",
            SUITE,
            RESPONSES,
            "-j",
            "mock",
            "--mock-judgments",
            JUDGMENTS,
            "-n",
            "3",
            *dirs,
        )
        assert result.exit_code == 0, result.output
        out = result.output
        assert "Self-consistency: 3 samples per case" in out
        assert "0 of 10 cases changed a score between samples" in out
        assert "temperature 0.7, seeds 0-2): 30 judgments" in out

    def test_scores_that_move(self, dirs: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _Seeded("m")
        monkeypatch.setattr(cli, "make_provider", lambda spec, **kw: provider)
        result = invoke(
            "consistency",
            SUITE,
            RESPONSES,
            "-j",
            "seeded:m",
            "-n",
            "3",
            "--seed",
            "10",
            "-v",
            *dirs,
        )
        assert result.exit_code == 0, result.output
        out = result.output
        # seeds 10, 11, 12 give IF 2, 3, 1 for every case
        assert "10 of 10 cases changed a score between samples" in out
        assert "  en-bullets-001: IF 2 3 1" in out
        assert "hallucination: always 0, sometimes 10, never 0" in out
        assert sorted({p.seed for p in provider.params}) == [10, 11, 12]
        assert {p.temperature for p in provider.params} == {0.7}

    def test_json(self, dirs: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "make_provider", lambda spec, **kw: _Seeded("m"))
        result = invoke(
            "consistency",
            SUITE,
            RESPONSES,
            "-j",
            "seeded:m",
            "-n",
            "2",
            "--json",
            "--case",
            "en-math-001",
            *dirs,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["run"]["params"] == {"temperature": 0.7, "seed": 0}
        assert data["consistency"]["samples"] == 2
        assert data["consistency"]["unstable"] == {"en-math-001": {"instruction_following": [1, 2]}}

    def test_temperature_zero_note(self, dirs: list[str]) -> None:
        result = invoke(
            "consistency",
            SUITE,
            RESPONSES,
            "-j",
            "mock",
            "--mock-judgments",
            JUDGMENTS,
            "--temperature",
            "0",
            *dirs,
        )
        assert result.exit_code == 0, result.output
        assert "this measures determinism" in result.output

    def test_failed_samples_fail_the_exit_code(
        self, dirs: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cli, "make_provider", lambda spec, **kw: _Seeded("m", fail_on="three benefits")
        )
        result = invoke(
            "consistency", SUITE, RESPONSES, "-j", "seeded:m", "-n", "2", "--retries", "0", *dirs
        )
        assert result.exit_code == 1
        assert "no judgment en-bullets-001 (sample 1): server down" in result.output

    def test_needs_two_samples(self, dirs: list[str]) -> None:
        result = invoke("consistency", SUITE, RESPONSES, "-j", "mock", "-n", "1", *dirs)
        assert result.exit_code == 2

    def test_nothing_to_judge(self, dirs: list[str], tmp_path: Path) -> None:
        other = tmp_path / "r.jsonl"
        other.write_text('{"id": "nope", "response": "x"}\n', encoding="utf-8")
        result = invoke("consistency", SUITE, other, "-j", "mock", *dirs)
        assert result.exit_code == 1
        assert "nothing to judge" in result.output


class TestPairwise:
    def args(self, dirs: list[str], *extra: object) -> list[object]:
        return [
            "pairwise",
            SUITE,
            RESPONSES,
            RESPONSES_B,
            "-j",
            "mock",
            "--mock-judgments",
            PAIRWISE,
            *dirs,
            *extra,
        ]

    def test_the_example_files(self, dirs: list[str], tmp_path: Path) -> None:
        result = invoke(*self.args(dirs))
        assert result.exit_code == 0, result.output
        out = result.output
        assert "A = smoke, B = smoke-b" in out
        assert "10/10 pairs judged in both orders: A wins 3, B wins 4, ties 2, flips 1" in out
        assert "win rate A: 0.450 (ties and flips count half)" in out
        assert "  id: A 1, B 3, ties 1, flips 0; win rate A 0.300" in out
        assert "flip rate: 1/10 = 0.100 (position 1 won both times 1" in out
        assert (
            "over 20 judgments the judge chose position 1 9 times, position 2 7 times, a tie 4"
            in out
        )
        [run_dir] = (tmp_path / "runs").iterdir()
        assert run_dir.name.endswith("-pairwise-mock-default")
        assert len((run_dir / "pairs.jsonl").read_text(encoding="utf-8").splitlines()) == 10

    def test_json_and_names(self, dirs: list[str]) -> None:
        result = invoke(*self.args(dirs, "--json", "--name-a", "qwen", "--name-b", "llama"))
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert (data["summary"]["a"], data["summary"]["b"]) == ("qwen", "llama")
        assert data["summary"]["flip_rate"] == 0.1
        verdicts = {p["id"]: p["verdict"] for p in data["pairs"]}
        assert verdicts["en-summary-001"] == "inconsistent"
        assert verdicts["en-json-001"] == "tie"

    def test_a_run_log_is_named_after_its_run(self, tmp_path: Path, dirs: list[str]) -> None:
        log = tmp_path / "20261118-ollama-qwen" / "responses.jsonl"
        log.parent.mkdir()
        log.write_text(RESPONSES.read_text(encoding="utf-8"), encoding="utf-8")
        args = ["pairwise", SUITE, log, log, "-j", "mock", *dirs, "--case", "en-math-001"]
        result = invoke(*args)
        assert "A = 20261118-ollama-qwen, B = 20261118-ollama-qwen" in result.output
        data = json.loads(invoke(*args, "--json").output)
        assert data["summary"]["identical_pairs"] == 1
        assert data["pairs"][0]["identical"] is True

    def test_only_one_order_answered_fails_the_pair(self, dirs: list[str], tmp_path: Path) -> None:
        canned = tmp_path / "pairs.jsonl"
        reply = json.dumps({"rationale": "r", "winner": "1"})
        canned.write_text(
            json.dumps({"id": "en-math-001", "response": reply}) + "\n", encoding="utf-8"
        )
        # Without an "en-math-001/swapped" entry the B-first prompt gets the same reply...
        both = invoke(
            "pairwise",
            SUITE,
            RESPONSES,
            RESPONSES_B,
            "-j",
            "mock",
            "--mock-judgments",
            canned,
            *dirs,
            "--case",
            "en-math-001",
        )
        assert both.exit_code == 0, both.output
        assert "flips 1" in both.output
        # ...and a case with no entry at all has no answer in either order.
        result = invoke(
            "pairwise",
            SUITE,
            RESPONSES,
            RESPONSES_B,
            "-j",
            "mock",
            "--mock-judgments",
            canned,
            *dirs,
            "--case",
            "en-json-001",
        )
        assert result.exit_code == 1
        assert "no judgment en-json-001 (A shown first)" in result.output
        assert "0/1 pairs judged in both orders" in result.output

    def test_missing_pairs_are_listed(self, dirs: list[str], tmp_path: Path) -> None:
        b = tmp_path / "b.jsonl"
        b.write_text(
            "\n".join(RESPONSES_B.read_text(encoding="utf-8").splitlines()[:2]) + "\n",
            encoding="utf-8",
        )
        result = invoke(
            "pairwise", SUITE, RESPONSES, b, "-j", "mock", "--mock-judgments", PAIRWISE, *dirs
        )
        assert result.exit_code == 0, result.output
        assert "2/2 pairs judged" in result.output
        assert "not in both files: en-json-001, id-json-001" in result.output

    def test_nothing_to_compare(self, dirs: list[str], tmp_path: Path) -> None:
        other = tmp_path / "r.jsonl"
        other.write_text('{"id": "nope", "response": "x"}\n', encoding="utf-8")
        result = invoke("pairwise", SUITE, RESPONSES, other, "-j", "mock", *dirs)
        assert result.exit_code == 1
        assert "nothing to compare" in result.output

    def test_mock_judgments_need_the_mock_judge(self, dirs: list[str]) -> None:
        result = invoke(
            "pairwise",
            SUITE,
            RESPONSES,
            RESPONSES_B,
            "-j",
            "ollama:x",
            "--mock-judgments",
            PAIRWISE,
            *dirs,
        )
        assert result.exit_code == 1
        assert "--mock-judgments only works with --judge mock" in result.output


class TestReviewFindings:
    """CLI behaviour the post-S4 review found misleading."""

    def consistency_run(self, dirs: list[str], tmp_path: Path) -> Path:
        result = invoke(
            "consistency",
            SUITE,
            RESPONSES,
            "-j",
            "mock",
            "--mock-judgments",
            JUDGMENTS,
            "-n",
            "3",
            *dirs,
            "--run-id",
            "cons",
            "--json",
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        # 10 cases judged 3 times: the counts said "cases": 30.
        assert (data["run"]["counts"]["cases"], data["run"]["counts"]["judgments"]) == (10, 30)
        return tmp_path / "runs" / "cons"

    def test_the_consistency_report_is_saved_with_the_run(
        self, dirs: list[str], tmp_path: Path
    ) -> None:
        run_dir = self.consistency_run(dirs, tmp_path)
        saved = json.loads((run_dir / "consistency.json").read_text(encoding="utf-8"))
        assert saved["samples"] == 3
        assert saved["cases"] == 10
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["counts"]["cases"] == 10

    def test_a_sample_of_a_consistency_run_against_a_plain_judge_run(
        self, dirs: list[str], tmp_path: Path
    ) -> None:
        cons = self.consistency_run(dirs, tmp_path) / "judgments.jsonl"
        plain = mock_judgments(tmp_path, dirs)
        # The plain run has no sample 2: say how to pick the reference's.
        missing = invoke("agree", cons, plain, "--sample", "2", "--resamples", "0")
        assert missing.exit_code == 1
        assert "no sample 2 (the file has [0]); pick the reference's with --reference-sample" in (
            missing.output
        )
        result = invoke(
            "agree", cons, plain, "--sample", "2", "--reference-sample", "0", "--resamples", "0"
        )
        assert result.exit_code == 0, result.output
        assert "10 cases rated by both" in result.output
        assert "used sample 2 of 3" in result.output
        # By default --sample applies to a judgments reference too, as before the fix.
        same = invoke("agree", cons, cons, "--sample", "2", "--resamples", "0", "--json")
        assert json.loads(same.output)["notes"] == [f"{cons}: used sample 2 of 3"] * 2
        both = invoke(
            "agree", cons, cons, "--sample", "1", "--reference-sample", "2", "--resamples", "0"
        )
        assert both.exit_code == 0, both.output

    def test_an_unfilled_sheet_says_so(self, dirs: list[str], tmp_path: Path) -> None:
        sheet = tmp_path / "sheet.csv"
        assert invoke("labels", SUITE, RESPONSES, "-o", sheet).exit_code == 0
        result = invoke("agree", mock_judgments(tmp_path, dirs), sheet)
        assert result.exit_code == 1
        assert "no row is labelled yet" in result.output
        # each id listed once, as not labelled, not also as "only the judge rated"
        assert "reference: not labelled yet (skipped) (10):" in result.output
        assert "only the judge rated" not in result.output

    def test_messages_follow_the_kind_of_file(self, dirs: list[str], tmp_path: Path) -> None:
        judged = mock_judgments(tmp_path, dirs)
        failed = tmp_path / "failed.jsonl"
        rows = [json.loads(x) for x in judged.read_text(encoding="utf-8").splitlines()]
        for r in rows:
            r.update(error="timeout", scores={}, rationale=None)
        failed.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        # an all-error judgments file as the reference is not "not labelled"
        result = invoke("agree", judged, failed)
        assert result.exit_code == 1
        assert "every judgment failed" in result.output
        assert "reference: no judgment (judge error, skipped) (10):" in result.output
        # and as the judge, against the example labels
        result = invoke("agree", failed, LABELS)
        assert "every judgment failed" in result.output
        assert "judge: no judgment (judge error, skipped) (10):" in result.output
        # a note about the answer texts is not a note about samples (round-3 review)
        run = tmp_path / "run"
        run.mkdir()
        (run / "judgments.jsonl").write_bytes(failed.read_bytes())
        (run / "manifest.json").write_text('{"responses": {"path": "gone.jsonl"}}', "utf-8")
        result = invoke("agree", run / "judgments.jsonl", LABELS)
        assert "every judgment failed" in result.output


class TestRoundThreeFindings:
    """From the review of the second round of fixes."""

    def sheet(self, tmp_path: Path, rows: dict[str, str], name: str = "s.csv") -> Path:
        """The example sheet with only `rows` filled in (id -> failure_modes)."""
        path = tmp_path / name
        lines = ["id,instruction_following,failure_modes"]
        lines += [f"{i},,{modes}" for i, modes in rows.items()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_a_modes_only_pass_warns_about_untouched_rows(
        self, dirs: list[str], tmp_path: Path
    ) -> None:
        # Tag only the row with a mode, leave the rest untouched: the judge's
        # tags on the untouched rows can't count, and agree must say so.
        # (The mock judge tags format_error on id-json-001, not on en-bullets-001.)
        rows = {"id-bullets-001": "format_error", "id-json-001": "", "en-bullets-001": ""}
        judged = mock_judgments(tmp_path, dirs)
        result = invoke("agree", judged, self.sheet(tmp_path, rows), "--json")
        assert result.exit_code == 0, result.output
        [warning] = json.loads(result.output)["warnings"]
        assert "tagged failure modes on 1 row that" in warning
        assert "leaves blank (id-json-001); those tags don't count" in warning
        # and the sheet as the first argument is checked the same way
        swapped = invoke("agree", self.sheet(tmp_path, rows), judged, "--json")
        assert len(json.loads(swapped.output)["warnings"]) == 1

    def test_no_warning_for_blank_rows_the_judge_left_untagged(
        self, dirs: list[str], tmp_path: Path
    ) -> None:
        rows = {"id-bullets-001": "format_error", "en-bullets-001": ""}
        result = invoke("agree", mock_judgments(tmp_path, dirs), self.sheet(tmp_path, rows))
        assert result.exit_code == 0, result.output
        assert "warning:" not in result.output
        assert "1 case rated by both" in result.output

    def test_no_warning_when_every_row_says_something(
        self, dirs: list[str], tmp_path: Path
    ) -> None:
        rows = {"id-bullets-001": "format_error", "en-bullets-001": "none"}
        result = invoke("agree", mock_judgments(tmp_path, dirs), self.sheet(tmp_path, rows))
        assert result.exit_code == 0, result.output
        assert "warning:" not in result.output

    def test_a_tab_separated_sheet_saved_as_txt(self, dirs: list[str], tmp_path: Path) -> None:
        sheet = tmp_path / "labels.txt"
        sheet.write_text(LABELS.read_text(encoding="utf-8").replace(",", "\t"), encoding="utf-8")
        # the notes column has commas inside quotes; drop it to keep the test simple
        rows = [r.split("\t")[:7] for r in sheet.read_text(encoding="utf-8").splitlines()]
        sheet.write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")
        result = invoke("agree", mock_judgments(tmp_path, dirs), sheet, "--resamples", "0")
        assert result.exit_code == 0, result.output
        assert "10 cases rated by both" in result.output

    def test_a_judgments_file_with_another_name(self, dirs: list[str], tmp_path: Path) -> None:
        renamed = tmp_path / "judgments.txt"
        renamed.write_bytes(mock_judgments(tmp_path, dirs).read_bytes())
        result = invoke("agree", renamed, LABELS, "--resamples", "0")
        assert result.exit_code == 0, result.output

    def test_the_suffix_decides_when_there_is_one(self, dirs: list[str], tmp_path: Path) -> None:
        # Round 4: a broken .jsonl got a CSV error, and a sheet whose first cell
        # starts with { got a JSON one.
        broken = tmp_path / "g.jsonl"
        broken.write_text("warning: something\n", encoding="utf-8")
        result = invoke("agree", broken, LABELS)
        assert "not a judge record" in result.output
        brace = tmp_path / "brace.csv"
        brace.write_text("{id},instruction_following\na,4\n", encoding="utf-8")
        result = invoke("agree", mock_judgments(tmp_path, dirs), brace)
        assert "the header has no 'id' column" in result.output

    def test_scores_not_compared_on_a_modes_only_sheet(
        self, dirs: list[str], tmp_path: Path
    ) -> None:
        sheet = tmp_path / "modes.csv"
        sheet.write_text("id,failure_modes\nid-json-001,format_error\n", encoding="utf-8")
        result = invoke("agree", mock_judgments(tmp_path, dirs), sheet)
        assert result.exit_code == 0, result.output
        assert "scores: not compared" in result.output
        assert "Scores: judge vs reference" not in result.output

    def test_a_sheet_with_only_a_header(self, dirs: list[str], tmp_path: Path) -> None:
        empty = tmp_path / "empty.csv"
        empty.write_text("id,instruction_following\n", encoding="utf-8")
        result = invoke("agree", mock_judgments(tmp_path, dirs), empty)
        assert result.exit_code == 1
        assert "the sheet has no rows" in result.output

    def test_shared_cases_skipped_on_one_side(self, dirs: list[str], tmp_path: Path) -> None:
        judged = mock_judgments(tmp_path, dirs)
        rows = [json.loads(x) for x in judged.read_text(encoding="utf-8").splitlines()]
        rows[0].update(error="timeout", scores={}, rationale=None)  # en-bullets-001
        broken = tmp_path / "one-error.jsonl"
        broken.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        one = tmp_path / "one.csv"
        one.write_text("id,instruction_following\nen-bullets-001,5\n", encoding="utf-8")
        result = invoke("agree", broken, one)
        assert result.exit_code == 1
        assert "the cases both files have were skipped on one side" in result.output

    def test_every_judgment_of_one_sample_failed(self, dirs: list[str], tmp_path: Path) -> None:
        result = invoke(
            "consistency",
            SUITE,
            RESPONSES,
            "-j",
            "mock",
            "--mock-judgments",
            JUDGMENTS,
            "-n",
            "2",
            *dirs,
            "--run-id",
            "c2",
        )
        assert result.exit_code == 0, result.output
        path = tmp_path / "runs" / "c2" / "judgments.jsonl"
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        for r in rows:
            if r["sample"] == 1:
                r.update(error="timeout", scores={}, rationale=None)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        result = invoke("agree", path, LABELS, "--sample", "1")
        assert result.exit_code == 1
        assert "every judgment of sample 1 failed" in result.output

    def test_scores_only_says_failure_modes_were_not_compared(
        self, dirs: list[str], tmp_path: Path
    ) -> None:
        sheet = tmp_path / "scores.csv"
        sheet.write_text("id,instruction_following\nen-bullets-001,5\n", encoding="utf-8")
        result = invoke("agree", mock_judgments(tmp_path, dirs), sheet, "--resamples", "0")
        assert result.exit_code == 0, result.output
        assert "failure modes: not compared" in result.output

    def test_nothing_shared_to_compare_fails(self, dirs: list[str], tmp_path: Path) -> None:
        # en-json-001's rubric doesn't score reasoning_quality, so the judge didn't.
        sheet = tmp_path / "rq.csv"
        sheet.write_text("id,reasoning_quality\nen-json-001,4\n", encoding="utf-8")
        result = invoke("agree", mock_judgments(tmp_path, dirs), sheet)
        assert result.exit_code == 1
        assert "share no rated dimension or failure mode" in result.output


class TestRoundFiveFindings:
    """Low findings from the fifth review."""

    def test_a_judgments_file_named_csv_says_so(self, dirs: list[str], tmp_path: Path) -> None:
        named = tmp_path / "j.csv"
        named.write_bytes(mock_judgments(tmp_path, dirs).read_bytes())
        result = invoke("agree", named, LABELS)
        assert result.exit_code == 1
        assert "this looks like a judgments file, not a sheet: name it .jsonl" in result.output

    def test_one_case_and_one_row_are_singular(self, dirs: list[str], tmp_path: Path) -> None:
        sheet = tmp_path / "s.csv"
        sheet.write_text(
            "id,instruction_following,failure_modes\nen-bullets-001,5,none\nid-json-001,,\n",
            encoding="utf-8",
        )
        result = invoke("agree", mock_judgments(tmp_path, dirs), sheet, "--resamples", "0")
        assert result.exit_code == 0, result.output
        assert "failure modes over the 1 case both labelled" in result.output
        assert "If that row is clean, write 'none' in its failure_modes" in result.output

    def test_nothing_matched_prints_no_scores_line(self, dirs: list[str], tmp_path: Path) -> None:
        sheet = tmp_path / "other.csv"
        sheet.write_text("id,helpfulness,failure_modes\nx,4,none\n", encoding="utf-8")
        result = invoke("agree", mock_judgments(tmp_path, dirs), sheet)
        assert result.exit_code == 1
        assert "scores: not compared" not in result.output
        assert "no case is rated in both files" in result.output
