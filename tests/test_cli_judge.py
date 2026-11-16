import json
from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from evalkit import cli
from evalkit.cli import app
from evalkit.prompts import Prompt
from evalkit.providers import Completion, GenerationParams, Provider, ProviderError

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SUITE = ROOT / "examples" / "suites" / "smoke.yaml"
SMOKE_RESPONSES = ROOT / "examples" / "responses" / "smoke.jsonl"
SMOKE_JUDGMENTS = ROOT / "examples" / "judgments" / "smoke.jsonl"

runner = CliRunner()


@pytest.fixture
def dirs(tmp_path: Path) -> list[str]:
    return ["--out", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "cache")]


def mock_judge(dirs: list[str], *extra: str, judgments: Path = SMOKE_JUDGMENTS):
    args = [
        "judge",
        str(SMOKE_SUITE),
        str(SMOKE_RESPONSES),
        "-j",
        "mock",
        "--mock-judgments",
        str(judgments),
        *dirs,
    ]
    return runner.invoke(app, [*args, *extra])


def test_mock_judge_on_the_example_files(dirs: list[str], tmp_path: Path) -> None:
    result = mock_judge(dirs)
    assert result.exit_code == 0, result.output
    out = result.output
    assert "judged 10/10 cases: 8 clean, 2 repaired locally, 0 after a re-ask" in out
    assert "mean scores (1-5): IF 4.10, RQ 5.00, FA 5.00, HE 4.43" in out
    assert "  id: IF 3.60, RQ 5.00, FA 5.00, HE 4.00" in out
    assert "failure modes: format_error 3 (check 3); language_mismatch 1 (check 1)" in out
    assert (
        "judge vs checks on checkable modes (only the checks count):\n"
        "  format_error: both 3, judge only 0, checks only 0\n"
        "  language_mismatch: both 1, judge only 0, checks only 0\n"
    ) in out
    assert "10 requests, 0 cases from cache, 0 re-asks" in out
    [run_dir] = (tmp_path / "runs").iterdir()
    assert run_dir.name.endswith("-judge-mock-default")
    assert len((run_dir / "judgments.jsonl").read_text(encoding="utf-8").splitlines()) == 10

    again = mock_judge(dirs)
    assert again.exit_code == 0
    assert "0 requests, 10 cases from cache, 0 re-asks" in again.output


def test_json_output(dirs: list[str]) -> None:
    result = mock_judge(dirs, "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run"]["judge"] == "mock:default"
    assert payload["summary"]["parse"] == {
        "clean": 8,
        "repaired": 2,
        "reasked": 0,
        "unusable": 0,
        "request_failed": 0,
    }
    by_id = {c["id"]: c for c in payload["cases"]}
    assert by_id["id-json-001"]["repairs"] == ["read a Python-style object"]
    assert by_id["id-explain-001"]["failure_modes"] == ["language_mismatch"]
    assert by_id["en-math-001"]["scores"] == {"instruction_following": 3, "reasoning_quality": 5}


def test_verbose_shows_rationale_and_repairs(dirs: list[str]) -> None:
    result = mock_judge(dirs, "-v", "--case", "id-json-001")
    assert result.exit_code == 0, result.output
    assert "rationale" in result.output
    assert "repairs:" in result.output
    assert "The values are right" in result.output


def test_an_unusable_reply_fails_the_case_and_the_exit_code(
    dirs: list[str], tmp_path: Path
) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "en-math-001", "response": "Score: 4/5"}\n', encoding="utf-8")
    result = mock_judge(dirs, "--case", "en-math-001", "--case", "en-json-001", judgments=bad)
    assert result.exit_code == 1
    # The re-asks get the mock's placeholder, which is no better.
    assert (
        "no judgment en-math-001: unusable judge reply (after 2 re-asks): "
        "the reply contains no JSON object"
    ) in result.output
    assert "judged 0/2 cases" in result.output
    assert "2 unusable" in result.output


def test_no_checks_changes_the_prompt_and_so_the_mock_lookup(dirs: list[str]) -> None:
    # The canned replies are keyed by the prompt, which differs without check results.
    result = mock_judge(dirs, "--no-checks", "--max-reasks", "0", "--case", "en-math-001")
    assert result.exit_code == 0, result.output
    assert "judged 1/1 cases: 1 clean" in result.output


def test_missing_and_unknown_responses_are_reported(dirs: list[str], tmp_path: Path) -> None:
    responses = tmp_path / "r.jsonl"
    responses.write_text(
        '{"id": "en-math-001", "response": "40000"}\n{"id": "nope", "response": "x"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "judge",
            str(SMOKE_SUITE),
            str(responses),
            "-j",
            "mock",
            "--mock-judgments",
            str(SMOKE_JUDGMENTS),
            *dirs,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no response for: en-bullets-001" in result.output
    assert "responses with no matching case: nope" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            ["-j", "ollama:x", "--mock-judgments", str(SMOKE_JUDGMENTS)],
            "only works with --judge mock",
        ),
        (["-j", "nope:x"], "unknown provider 'nope'"),
        (["-j", "mock", "--case", "missing-id"], "no such case id(s)"),
    ],
)
def test_bad_options(dirs: list[str], args: list[str], message: str) -> None:
    result = runner.invoke(app, ["judge", str(SMOKE_SUITE), str(SMOKE_RESPONSES), *args, *dirs])
    assert result.exit_code == 1
    assert message in result.output


def test_bad_responses_file(dirs: list[str], tmp_path: Path) -> None:
    bad = tmp_path / "r.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    result = runner.invoke(app, ["judge", str(SMOKE_SUITE), str(bad), "-j", "mock", *dirs])
    assert result.exit_code == 1
    assert "invalid JSON" in result.output


class _DownJudge(Provider):
    name = "down"
    supported_params = frozenset({"temperature", "seed", "max_tokens"})

    async def preflight(self) -> None:
        raise ProviderError("down: no server", retryable=False)

    async def generate(self, prompt: Prompt, params: GenerationParams) -> Completion:
        raise AssertionError("never called")


class _Recorder(Provider):
    name = "rec"
    supported_params = frozenset({"temperature", "seed", "max_tokens"})
    params: ClassVar[list[GenerationParams]] = []

    async def generate(self, prompt: Prompt, params: GenerationParams) -> Completion:
        type(self).params.append(params)
        reply = json.dumps(
            {
                "rationale": "ok",
                "scores": {
                    "instruction_following": 5,
                    "reasoning_quality": 5,
                    "faithfulness": 5,
                    "helpfulness": 5,
                },
                "failure_modes": [],
            }
        )
        return Completion(text=reply, model="r", finish_reason="stop")


def test_preflight_failure_exits_1(dirs: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "make_provider", lambda spec, **kw: _DownJudge("m"))
    result = runner.invoke(
        app, ["judge", str(SMOKE_SUITE), str(SMOKE_RESPONSES), "-j", "ollama:m", *dirs]
    )
    assert result.exit_code == 1
    assert "down: no server" in result.output


def test_judge_defaults_are_deterministic_and_capped(
    dirs: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _Recorder.params = []
    monkeypatch.setattr(cli, "make_provider", lambda spec, **kw: _Recorder("m"))
    result = runner.invoke(
        app,
        [
            "judge",
            str(SMOKE_SUITE),
            str(SMOKE_RESPONSES),
            "-j",
            "ollama:m",
            "--case",
            "en-bullets-001",
            *dirs,
        ],
    )
    assert result.exit_code == 0, result.output
    [params] = _Recorder.params
    assert (params.temperature, params.seed, params.max_tokens) == (0.0, 0, 1024)


def test_taxonomy_lists_every_failure_mode() -> None:
    result = runner.invoke(app, ["taxonomy"])
    assert result.exit_code == 0
    for mode in (
        "constraint_violation",
        "intent_misread",
        "hallucination",
        "partial_completion",
        "format_error",
        "over_refusal",
        "language_mismatch",
    ):
        assert mode in result.output
    assert "checks (language)" in result.output
    [row] = [line for line in result.output.splitlines() if "intent_misread" in line]
    assert "judge" in row
