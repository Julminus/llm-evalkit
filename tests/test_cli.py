import io
import json
import sys
from pathlib import Path

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from evalkit import __version__, cli
from evalkit.cli import app

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SUITE = ROOT / "examples" / "suites" / "smoke.yaml"
SMOKE_RESPONSES = ROOT / "examples" / "responses" / "smoke.jsonl"

runner = CliRunner()


@pytest.fixture
def bad_suite(tmp_path: Path) -> Path:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: bad\ncases:\n  - id: x\n    language: en\n    prompt: hi\n"
        "    constraints:\n      - {type: word_count}\n",
        encoding="utf-8",
    )
    return path


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "validate" in result.output
    assert "check" in result.output


def test_validate_ok() -> None:
    result = runner.invoke(app, ["validate", str(SMOKE_SUITE)])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output
    assert "10 cases (5 en, 5 id)" in result.output
    assert "language (7)" in result.output


def test_validate_reports_invalid_suite(bad_suite: Path) -> None:
    result = runner.invoke(app, ["validate", str(SMOKE_SUITE), str(bad_suite)])
    assert result.exit_code == 1
    assert "invalid" in result.output
    assert "at least one of 'min' or 'max'" in result.output


def test_validate_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2  # Typer usage error: path does not exist


def test_check_table() -> None:
    result = runner.invoke(app, ["check", str(SMOKE_SUITE), str(SMOKE_RESPONSES)])
    assert result.exit_code == 0, result.output
    assert "6 pass, 4 fail, 0 skip" in result.output
    assert "pass rate 60% (en 80%, id 40%)" in result.output
    assert "format_error (3)" in result.output
    assert "expected id, detected en" in result.output


def test_check_verbose_shows_passing_checks() -> None:
    result = runner.invoke(app, ["check", "-v", str(SMOKE_SUITE), str(SMOKE_RESPONSES)])
    assert result.exit_code == 0
    assert "top-level list items" in result.output


def test_check_json() -> None:
    result = runner.invoke(app, ["check", "--json", str(SMOKE_SUITE), str(SMOKE_RESPONSES)])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["pass_rate"] == 0.6
    assert payload["pass_rate_by_language"] == {"en": 0.8, "id": 0.4}
    assert payload["counts"] == {"pass": 6, "fail": 4, "skip": 0}
    assert payload["failure_modes"] == {"format_error": 3, "language_mismatch": 1}
    assert payload["by_check"]["language"] == {"pass": 6, "fail": 1, "skip": 0}
    by_id = {c["case_id"]: c for c in payload["cases"]}
    assert by_id["id-json-001"]["status"] == "fail"
    assert by_id["en-json-001"]["results"][0]["observed"]["found_in"] == "code fence"


def test_check_strict_exit_code() -> None:
    result = runner.invoke(app, ["check", "--strict", str(SMOKE_SUITE), str(SMOKE_RESPONSES)])
    assert result.exit_code == 1


def _one_passing_case(tmp_path: Path) -> tuple[Path, Path]:
    """A one-case suite (en-bullets-001) and a responses file that passes it."""
    raw = yaml.safe_load(SMOKE_SUITE.read_text(encoding="utf-8"))
    raw["cases"] = [c for c in raw["cases"] if c["id"] == "en-bullets-001"]
    suite = tmp_path / "one.yaml"
    suite.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    rows = [json.loads(line) for line in SMOKE_RESPONSES.read_text(encoding="utf-8").splitlines()]
    passing = tmp_path / "pass.jsonl"
    passing.write_text(
        json.dumps(next(r for r in rows if r["id"] == "en-bullets-001")) + "\n", encoding="utf-8"
    )
    return suite, passing


def test_check_strict_passes_when_all_pass(tmp_path: Path) -> None:
    suite, passing = _one_passing_case(tmp_path)
    result = runner.invoke(app, ["check", "--strict", str(suite), str(passing)])
    assert result.exit_code == 0, result.output


def test_check_strict_fails_on_missing_responses(tmp_path: Path) -> None:
    # One passing response out of ten cases must not pass a CI gate.
    _, passing = _one_passing_case(tmp_path)
    result = runner.invoke(app, ["check", "--strict", str(SMOKE_SUITE), str(passing)])
    assert result.exit_code == 1
    assert "no response for:" in result.output
    lenient = runner.invoke(app, ["check", str(SMOKE_SUITE), str(passing)])
    assert lenient.exit_code == 0


def test_check_strict_fails_on_unknown_responses(tmp_path: Path) -> None:
    suite, passing = _one_passing_case(tmp_path)
    with passing.open("a", encoding="utf-8") as f:
        f.write('{"id": "ghost", "response": "boo"}\n')
    result = runner.invoke(app, ["check", "--strict", str(suite), str(passing)])
    assert result.exit_code == 1
    assert "responses with no matching case: ghost" in result.output


def test_output_is_utf8_when_redirected(monkeypatch: pytest.MonkeyPatch) -> None:
    # A redirected stream on Windows defaults to cp1252, which cannot encode "→".
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    cli._use_utf8_output()
    stream.write("→ 好")
    stream.flush()
    assert raw.getvalue() == "→ 好".encode()


def test_check_reports_unknown_ids(tmp_path: Path) -> None:
    responses = tmp_path / "r.jsonl"
    responses.write_text('{"id": "ghost", "response": "boo"}\n', encoding="utf-8")
    result = runner.invoke(app, ["check", str(SMOKE_SUITE), str(responses)])
    assert result.exit_code == 0
    assert "responses with no matching case: ghost" in result.output


def test_check_invalid_suite(bad_suite: Path) -> None:
    result = runner.invoke(app, ["check", str(bad_suite), str(SMOKE_RESPONSES)])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_check_invalid_responses(tmp_path: Path) -> None:
    responses = tmp_path / "r.jsonl"
    responses.write_text("not json\n", encoding="utf-8")
    result = runner.invoke(app, ["check", str(SMOKE_SUITE), str(responses)])
    assert result.exit_code == 1
    assert "invalid JSON" in result.output


def test_list_checks() -> None:
    result = runner.invoke(app, ["list-checks"])
    assert result.exit_code == 0
    for name in ("word_count", "json_schema", "language", "bullet_list", "ends_with"):
        assert name in result.output


def test_list_checks_keeps_square_brackets(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: Rich parsed "[min, max]" in the summaries as a markup tag and dropped it.
    # A wide console keeps each summary on one line so the assertion sees it whole.
    wide = Console(width=200, record=True, file=io.StringIO())
    monkeypatch.setattr(cli, "console", wide)
    result = runner.invoke(app, ["list-checks"])
    assert result.exit_code == 0
    text = wide.export_text()
    assert "Word count within [min, max]" in text
    assert "[min_items, max_items]" in text


def test_bracketed_tags_and_ids_are_printed_literally(tmp_path: Path) -> None:
    suite = tmp_path / "s.yaml"
    suite.write_text(
        "name: s\ncases:\n  - id: a\n    language: en\n    prompt: hi\n    tags: ['[draft]']\n",
        encoding="utf-8",
    )
    validated = runner.invoke(app, ["validate", str(suite)])
    assert "[draft] (1)" in validated.output

    responses = tmp_path / "r.jsonl"
    responses.write_text('{"id": "[bold]x", "response": "hi"}\n', encoding="utf-8")
    checked = runner.invoke(app, ["check", str(suite), str(responses)])
    assert "responses with no matching case: [bold]x" in checked.output
