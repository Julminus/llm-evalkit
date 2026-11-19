import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from evalkit import cli
from evalkit.cache import ResponseCache
from evalkit.cli import app
from evalkit.prompts import Prompt
from evalkit.providers import Completion, GenerationParams, Provider, ProviderError

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SUITE = ROOT / "examples" / "suites" / "smoke.yaml"
SMOKE_RESPONSES = ROOT / "examples" / "responses" / "smoke.jsonl"

runner = CliRunner()


@pytest.fixture
def dirs(tmp_path: Path) -> list[str]:
    return ["--out", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "cache")]


def mock_run(dirs: list[str], *extra: str):
    args = ["run", str(SMOKE_SUITE), "-p", "mock", "--mock-responses", str(SMOKE_RESPONSES), *dirs]
    return runner.invoke(app, [*args, *extra])


def test_mock_run_reproduces_check_results(dirs: list[str], tmp_path: Path) -> None:
    result = mock_run(dirs)
    assert result.exit_code == 0, result.output
    # Same numbers as `evalkit check` on the hand-written responses.
    assert "6 pass, 4 fail, 0 skip; pass rate 60% (en 80%, id 40%)" in result.output
    assert "10 generated, 0 from cache, 0 failed" in result.output
    [run_dir] = (tmp_path / "runs").iterdir()
    lines = (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["suite"]["path"] == SMOKE_SUITE.as_posix()
    assert manifest["cache"] == (tmp_path / "cache").as_posix()

    check = runner.invoke(app, ["check", str(SMOKE_SUITE), str(run_dir / "responses.jsonl")])
    assert check.exit_code == 0
    assert "pass rate 60% (en 80%, id 40%)" in check.output


def test_model_names_print_as_written(dirs: list[str]) -> None:
    # Rich reads ":smile:" as an emoji code; escaping markup doesn't stop that.
    args = ["run", str(SMOKE_SUITE), "-p", "mock:smile:latest", "--mock-responses"]
    result = runner.invoke(app, [*args, str(SMOKE_RESPONSES), *dirs])
    assert result.exit_code == 0, result.output
    assert "(mock:smile:latest)" in result.output


def test_second_run_comes_from_cache(dirs: list[str]) -> None:
    mock_run(dirs)
    result = mock_run(dirs)
    assert "0 generated, 10 from cache" in result.output


def test_no_cache(dirs: list[str], tmp_path: Path) -> None:
    result = mock_run(dirs, "--no-cache")
    assert result.exit_code == 0
    assert not (tmp_path / "cache").exists()


def test_json_output(dirs: list[str]) -> None:
    result = mock_run(dirs, "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["run"]["counts"]["cases"] == 10
    assert payload["run"]["provider"] == "mock:default"
    assert payload["checks"]["pass_rate"] == 0.6


def test_strict_fails_on_check_failures(dirs: list[str]) -> None:
    assert mock_run(dirs, "--strict").exit_code == 1


def test_case_filter(dirs: list[str]) -> None:
    result = mock_run(dirs, "--case", "en-json-001", "--case", "id-json-001", "--run-id", "two")
    assert result.exit_code == 0, result.output
    # id-json-001's hand-written response fails on purpose ("umur": "34").
    assert "1 pass, 1 fail, 0 skip; pass rate 50%" in result.output
    assert "no response for" not in result.output
    assert "run two" in result.output


def test_unknown_case(dirs: list[str]) -> None:
    result = mock_run(dirs, "--case", "nope")
    assert result.exit_code == 1
    assert "no such case id(s)" in result.output


def test_truncation_is_reported(dirs: list[str]) -> None:
    result = mock_run(dirs, "--max-tokens", "3")
    assert "hit the token limit" in result.output


def test_placeholder_responses_without_mock_file(dirs: list[str]) -> None:
    result = runner.invoke(app, ["run", str(SMOKE_SUITE), "-p", "mock", *dirs])
    assert result.exit_code == 0, result.output
    assert "10 generated" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["-p", "gpt:4"], "unknown provider"),
        (["-p", "ollama"], "needs a model name"),
        (
            ["-p", "ollama:m", "--mock-responses", str(SMOKE_RESPONSES)],
            "only works with --provider mock",
        ),
    ],
)
def test_provider_setup_errors(dirs: list[str], args: list[str], message: str) -> None:
    result = runner.invoke(app, ["run", str(SMOKE_SUITE), *args, *dirs])
    assert result.exit_code == 1
    assert message in result.output


def test_bad_mock_responses_file(dirs: list[str], tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{nope\n", encoding="utf-8")
    result = runner.invoke(
        app, ["run", str(SMOKE_SUITE), "-p", "mock", "--mock-responses", str(bad), *dirs]
    )
    assert result.exit_code == 1
    assert "invalid JSON" in result.output


def test_provider_is_required() -> None:
    assert runner.invoke(app, ["run", str(SMOKE_SUITE)]).exit_code == 2


class Fake(Provider):
    """No seed support, one case fails, optional preflight failure."""

    name = "fake"
    supported_params: ClassVar[frozenset[str]] = frozenset({"temperature", "max_tokens"})

    def __init__(self, *, preflight_error: bool = False) -> None:
        super().__init__("m")
        self.preflight_error = preflight_error

    async def preflight(self) -> None:
        if self.preflight_error:
            raise ProviderError("fake: model 'm' is not pulled", retryable=False)

    async def generate(self, prompt: Prompt, params: GenerationParams) -> Completion:
        if "JSON" in prompt.user:
            raise ProviderError("fake: HTTP 400: bad request", retryable=False, status=400)
        return Completion(
            text="ok", model="m", finish_reason="refusal" if "Pasal" in prompt.user else "stop"
        )


def use_fake(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> None:
    monkeypatch.setattr(cli, "make_provider", lambda spec, **kw: Fake(**kwargs))


def test_errors_ignored_params_and_refusals_are_reported(
    dirs: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake(monkeypatch)
    result = runner.invoke(
        app, ["run", str(SMOKE_SUITE), "-p", "ollama:m", "--retries", "0", "--seed", "1", *dirs]
    )
    assert result.exit_code == 1  # some cases got no response
    assert "fake does not accept seed" in result.output
    assert "failed en-json-001: fake: HTTP 400: bad request" in result.output
    assert "refused: id-citation-001" in result.output
    assert "no response for:" in result.output


def test_default_seed_is_only_sent_where_supported(
    dirs: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Fake takes temperature but not seed: temperature defaults to 0, seed is
    # left unset, and no "does not accept" note is printed for a default.
    use_fake(monkeypatch)
    result = runner.invoke(
        app, ["run", str(SMOKE_SUITE), "-p", "ollama:m", "--retries", "0", *dirs]
    )
    assert "does not accept" not in result.output
    manifest = json.loads(next((tmp_path / "runs").glob("*/manifest.json")).read_text("utf-8"))
    assert manifest["params"]["requested"] == {"temperature": 0.0}


def test_preflight_failure_stops_before_running(
    dirs: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_fake(monkeypatch, preflight_error=True)
    result = runner.invoke(app, ["run", str(SMOKE_SUITE), "-p", "ollama:m", *dirs])
    assert result.exit_code == 1
    assert "is not pulled" in result.output
    assert not (tmp_path / "runs").exists()


def test_run_warnings_are_shown_without_failing_the_run(
    dirs: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def full_disk(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ResponseCache, "put", full_disk)
    result = mock_run(dirs, "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["run"]["warnings"][0].startswith("10 response(s) could not be written")
    table = mock_run(dirs, "--no-cache")  # no cache, no warning
    assert "warning:" not in table.output
    warned = mock_run(dirs)
    assert "warning: 10 response(s) could not be written to the cache" in warned.output


def test_interrupt_exits_130(dirs: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    async def interrupted(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_suite", interrupted)
    result = mock_run(dirs)
    assert result.exit_code == 130
    assert "run the same command again" in result.output
