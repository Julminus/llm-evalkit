import json
from pathlib import Path

from typer.testing import CliRunner

from evalkit.cli import app

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SUITE = ROOT / "examples" / "suites" / "smoke.yaml"
SMOKE_RESPONSES = ROOT / "examples" / "responses" / "smoke.jsonl"
SMOKE_B = ROOT / "examples" / "responses" / "smoke-b.jsonl"
SMOKE_JUDGMENTS = ROOT / "examples" / "judgments" / "smoke.jsonl"

runner = CliRunner()


def invoke(*args: object):
    return runner.invoke(app, [str(a) for a in args])


def test_report_writes_html_and_markdown(tmp_path: Path) -> None:
    dirs = ["--out", tmp_path / "runs", "--cache-dir", tmp_path / "cache"]
    judged = invoke(
        "judge", SMOKE_SUITE, SMOKE_RESPONSES, "-j", "mock", "--mock-judgments", SMOKE_JUDGMENTS,
        *dirs, "--run-id", "j",
    )  # fmt: skip
    assert judged.exit_code == 0, judged.output
    out = tmp_path / "report"
    result = invoke("report", SMOKE_SUITE, tmp_path / "runs" / "j", SMOKE_B, "-o", out)
    assert result.exit_code == 0, result.output
    assert "Overview" in result.output
    assert f"wrote {out / 'report.html'} and {out / 'report.md'}" in result.output
    md = (out / "report.md").read_text(encoding="utf-8")
    # Named after the judged file (the example file has no run manifest).
    assert "| smoke | 60.0% (6/10) | 80.0% (4/5) | 40.0% (2/5) | 10/10 | 4.10 |" in md
    assert "## Changes: smoke → smoke-b" in md
    assert (out / "report.html").read_text(encoding="utf-8").startswith("<!doctype html>")


def test_report_names_and_errors(tmp_path: Path) -> None:
    result = invoke("report", SMOKE_SUITE, SMOKE_RESPONSES, "--name", "a", "--name", "b")
    assert result.exit_code == 1
    assert "2 names for 1 runs" in result.output

    taken = tmp_path / "file"
    taken.write_text("", encoding="utf-8")
    result = invoke("report", SMOKE_SUITE, SMOKE_RESPONSES, "-o", taken)
    assert result.exit_code == 1
    assert "cannot write the report to" in result.output

    result = invoke("report", SMOKE_SUITE, tmp_path)
    assert result.exit_code == 1
    assert "no judgments.jsonl or responses.jsonl" in result.output


def test_report_prints_run_notes(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "responses.jsonl").write_bytes(SMOKE_RESPONSES.read_bytes())
    manifest = {"suite": {"digest": "0000"}}
    (tmp_path / "run" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = invoke("report", SMOKE_SUITE, tmp_path / "run", "-o", tmp_path / "out")
    assert result.exit_code == 0, result.output
    assert "note: run: made with a different version of the suite" in result.output


def test_compare_two_runs() -> None:
    result = invoke("compare", SMOKE_SUITE, SMOKE_RESPONSES, SMOKE_B)
    assert result.exit_code == 0, result.output
    out = result.output
    assert "base: smoke (" in out
    assert "10 cases in both runs: 2 worse, 4 better; the response text differs on 10." in out
    assert "pass → fail" in out

    strict = invoke("compare", SMOKE_SUITE, SMOKE_RESPONSES, SMOKE_B, "--strict")
    assert strict.exit_code == 1
    same = invoke("compare", SMOKE_SUITE, SMOKE_RESPONSES, SMOKE_RESPONSES, "--strict")
    assert same.exit_code == 0, same.output
    assert "0 worse, 0 better; the response text differs on 0." in same.output


def test_compare_json() -> None:
    result = invoke("compare", SMOKE_SUITE, SMOKE_RESPONSES, SMOKE_B, "--json", "--name", "old")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert (payload["base"], payload["new"], payload["shared"]) == ("old", "smoke-b", 10)
    assert payload["worse"] == ["en-bullets-001", "en-explain-001"]
    assert payload["pass_rate"] == {"base": [6, 10], "new": [8, 10]}


def test_compare_prints_heading_then_intro_and_wraps_cells(monkeypatch) -> None:
    out = invoke("compare", SMOKE_SUITE, SMOKE_RESPONSES, SMOKE_B).output
    assert out.index("Changes: smoke → smoke-b") < out.index("10 cases in both runs")
    assert out.index("Cases that changed") < out.index("Worse: fails its checks")
    same = invoke("compare", SMOKE_SUITE, SMOKE_RESPONSES, SMOKE_RESPONSES).output
    assert "Cases that changed\nNone.\n" in same
    # Long names squeeze the columns; cells wrap instead of ending in "…".
    long = [
        "--name",
        "a-rather-long-name-for-the-base-run",
        "--name",
        "and-another-for-the-new-one",
    ]
    assert "…" not in invoke("compare", SMOKE_SUITE, SMOKE_RESPONSES, SMOKE_B, *long).output
    # A stream that can't hold "→" gets "->", not a → escape.
    monkeypatch.setattr("evalkit.cli._stdout_is_utf8", lambda: False)
    ascii_out = invoke("compare", SMOKE_SUITE, SMOKE_RESPONSES, SMOKE_B).output
    assert "Changes: smoke -> smoke-b" in ascii_out
    assert "pass -> fail" in ascii_out


def test_compare_json_keeps_notes_on_stderr(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "responses.jsonl").write_bytes(SMOKE_B.read_bytes())
    manifest = {"suite": {"digest": "0000"}}
    (tmp_path / "run" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = invoke("compare", SMOKE_SUITE, SMOKE_RESPONSES, tmp_path / "run", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["shared"] == 10
    assert "made with a different version of the suite" in result.stderr


def test_compare_strict_fails_on_a_lost_case(tmp_path: Path) -> None:
    fewer = tmp_path / "fewer.jsonl"
    fewer.write_text(
        "".join(SMOKE_RESPONSES.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]),
        encoding="utf-8",
    )
    result = invoke("compare", SMOKE_SUITE, SMOKE_RESPONSES, fewer, "--strict")
    assert result.exit_code == 1
    assert "In smoke only: id-explain-001" in result.output
    assert invoke("compare", SMOKE_SUITE, SMOKE_RESPONSES, fewer).exit_code == 0


def test_compare_with_nothing_in_common(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text('{"id": "en-math-001", "response": "42"}\n', encoding="utf-8")
    b.write_text('{"id": "id-json-001", "response": "{}"}\n', encoding="utf-8")
    result = invoke("compare", SMOKE_SUITE, a, b)
    assert result.exit_code == 1
    assert "the two runs have no case in common" in result.output


def test_a_run_of_another_suite_is_an_error(tmp_path: Path) -> None:
    other = tmp_path / "other.jsonl"
    other.write_text('{"id": "nope", "response": "x"}\n', encoding="utf-8")
    for command in (["report", "-o", tmp_path / "out"], ["compare", SMOKE_RESPONSES]):
        result = invoke(command[0], SMOKE_SUITE, *command[1:], other)
        assert result.exit_code == 1
        assert "no case of suite 'smoke' in it (it has nope)" in result.output


def test_compare_prints_notes(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "responses.jsonl").write_bytes(SMOKE_B.read_bytes())
    manifest = {"suite": {"digest": "0000"}, "provider": {"spec": "ollama:m"}}
    (tmp_path / "run" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = invoke("compare", SMOKE_SUITE, SMOKE_RESPONSES, tmp_path / "run")
    assert result.exit_code == 0, result.output
    assert "new:  ollama:m (" in result.output
    assert "note: ollama:m: made with a different version of the suite" in result.output
