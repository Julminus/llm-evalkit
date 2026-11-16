import asyncio
import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest

from evalkit import runner as runner_module
from evalkit.cache import ResponseCache
from evalkit.evaluate import load_responses
from evalkit.prompts import Prompt
from evalkit.providers import Completion, GenerationParams, MockProvider, Provider, ProviderError
from evalkit.runner import RetryPolicy, RunLog, Runner, RunRecord, RunResult, make_run_id, run_suite
from evalkit.schema import Suite, parse_suite


def make_suite(n: int = 5) -> Suite:
    return parse_suite(
        {
            "name": "t",
            "cases": [{"id": f"c{i}", "language": "en", "prompt": f"prompt {i}"} for i in range(n)],
        }
    )


class Scripted(Provider):
    """Per-prompt scripted outcomes; tracks calls and concurrency."""

    name = "fake"
    supported_params: ClassVar[frozenset[str]] = frozenset({"temperature", "max_tokens"})

    def __init__(
        self,
        script: dict[str, list[Exception | str]] | None = None,
        *,
        delay: dict[str, float] | None = None,
        finish: str = "stop",
    ) -> None:
        super().__init__("m")
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.delay = delay or {}
        self.finish = finish
        self.calls: defaultdict[str, int] = defaultdict(int)
        self.in_flight = 0
        self.max_in_flight = 0

    async def generate(self, prompt: Prompt, params: GenerationParams) -> Completion:
        self.calls[prompt.user] += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay.get(prompt.user, 0.001))
            steps = self.script.get(prompt.user)
            outcome: Exception | str = steps.pop(0) if steps else f"answer to {prompt.user}"
            if isinstance(outcome, Exception):
                raise outcome
            return Completion(
                text=outcome,
                model="m-served",
                finish_reason=self.finish,
                input_tokens=2,
                output_tokens=3,  # type: ignore[arg-type]
            )
        finally:
            self.in_flight -= 1


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def transient(retry_after: float | None = None) -> ProviderError:
    return ProviderError(
        "fake: HTTP 503: busy", retryable=True, status=503, retry_after=retry_after
    )


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class TestRetryPolicy:
    def test_full_jitter_bounds_and_cap(self) -> None:
        policy = RetryPolicy(max_retries=10, base_delay=1.0, max_delay=5.0)
        rng = random.Random(0)
        for retry, upper in [(0, 1.0), (1, 2.0), (2, 4.0), (3, 5.0), (8, 5.0)]:
            for _ in range(50):
                assert 0.0 <= policy.delay(retry, retry_after=None, rng=rng) <= upper

    def test_retry_after_wins_but_is_capped(self) -> None:
        policy = RetryPolicy(max_delay=10.0)
        rng = random.Random(0)
        assert policy.delay(0, retry_after=4.0, rng=rng) == 4.0
        assert policy.delay(0, retry_after=120.0, rng=rng) == 10.0

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            RetryPolicy(max_retries=-1)


class TestRunner:
    def test_concurrency_limit_is_respected(self) -> None:
        provider = Scripted(delay={f"prompt {i}": 0.02 for i in range(12)})
        records = run(Runner(provider, concurrency=3).run(make_suite(12).cases))
        assert len(records) == 12
        assert provider.max_in_flight == 3

    def test_records_come_back_in_case_order(self) -> None:
        # c0 is slowest, so it finishes last; the result order is still c0..c4.
        provider = Scripted(delay={"prompt 0": 0.05})
        finished: list[str] = []
        records = run(
            Runner(provider, concurrency=5).run(
                make_suite().cases, on_record=lambda r: finished.append(r.id)
            )
        )
        assert [r.id for r in records] == ["c0", "c1", "c2", "c3", "c4"]
        assert finished[-1] == "c0"

    def test_retries_transient_errors_then_succeeds(self) -> None:
        provider = Scripted({"prompt 0": [transient(), transient(), "finally"]})
        sleeps = Sleeps()
        runner = Runner(
            provider,
            retry=RetryPolicy(max_retries=3, base_delay=1.0),
            sleep=sleeps,
            rng=random.Random(1),
        )
        record = run(runner.run(make_suite(1).cases))[0]
        assert record.response == "finally"
        assert record.attempts == 3
        assert len(sleeps.calls) == 2
        assert 0 <= sleeps.calls[0] <= 1.0
        assert 0 <= sleeps.calls[1] <= 2.0

    def test_uses_retry_after(self) -> None:
        provider = Scripted({"prompt 0": [transient(retry_after=2.5), "ok"]})
        sleeps = Sleeps()
        record = run(Runner(provider, sleep=sleeps).run(make_suite(1).cases))[0]
        assert record.ok
        assert sleeps.calls == [2.5]

    def test_gives_up_after_max_retries(self) -> None:
        provider = Scripted({"prompt 0": [transient()] * 10})
        sleeps = Sleeps()
        record = run(
            Runner(provider, retry=RetryPolicy(max_retries=2), sleep=sleeps).run(
                make_suite(1).cases
            )
        )[0]
        assert not record.ok
        assert record.response is None
        assert record.error == "fake: HTTP 503: busy (after 3 attempts)"
        assert record.attempts == 3
        assert len(sleeps.calls) == 2

    def test_non_retryable_fails_fast(self) -> None:
        bad = ProviderError("fake: HTTP 400: bad request", retryable=False, status=400)
        provider = Scripted({"prompt 0": [bad, "never reached"]})
        sleeps = Sleeps()
        record = run(Runner(provider, sleep=sleeps).run(make_suite(1).cases))[0]
        assert record.error == "fake: HTTP 400: bad request"
        assert record.attempts == 1
        assert sleeps.calls == []

    def test_one_failure_does_not_stop_the_others(self) -> None:
        bad = ProviderError("nope", retryable=False)
        provider = Scripted({"prompt 2": [bad]})
        records = run(Runner(provider).run(make_suite().cases))
        assert [r.ok for r in records] == [True, True, False, True, True]

    def test_record_fields(self) -> None:
        record = run(Runner(Scripted()).run(make_suite(1).cases))[0]
        assert record.provider == "fake"
        assert record.model == "m"
        assert record.served_model == "m-served"
        assert record.finish_reason == "stop"
        assert (record.input_tokens, record.output_tokens) == (2, 3)
        assert record.latency_s is not None
        assert record.latency_s >= 0
        assert record.cached is False
        assert len(record.cache_key) == 64

    def test_unexpected_provider_exception_fails_only_that_case(self) -> None:
        # Before, a non-ProviderError (a decoding error, an SDK validation
        # error) cancelled every case in flight and crashed the run.
        provider = Scripted({"prompt 1": [KeyError("choices"), "never sent"]})
        records = run(Runner(provider, retry=RetryPolicy(max_retries=3)).run(make_suite(3).cases))
        assert [r.ok for r in records] == [True, False, True]
        assert records[1].error == "fake: unexpected KeyError: 'choices'"
        assert records[1].attempts == 1  # not retried
        assert provider.calls["prompt 1"] == 1

    def test_run_case_without_slots(self) -> None:
        record = run(Runner(Scripted()).run_case(make_suite(1).cases[0]))
        assert record.response == "answer to prompt 0"

    def test_rejects_zero_concurrency(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            Runner(Scripted(), concurrency=0)


class TestCaching:
    def test_second_run_is_served_from_cache(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        cases = make_suite(3).cases
        first = Scripted()
        r1 = run(Runner(first, cache=cache).run(cases))
        second = Scripted()
        r2 = run(Runner(second, cache=cache).run(cases))
        assert sum(first.calls.values()) == 3
        assert sum(second.calls.values()) == 0
        assert all(r.cached and r.attempts == 0 for r in r2)
        assert [r.response for r in r2] == [r.response for r in r1]
        assert [r.latency_s for r in r2] == [r.latency_s for r in r1]

    def test_identical_requests_share_one_generation(self, tmp_path: Path) -> None:
        # Two cases, same prompt, different ids (e.g. different constraints).
        suite = parse_suite(
            {
                "name": "t",
                "cases": [
                    {"id": "short", "language": "en", "prompt": "same"},
                    {"id": "long", "language": "en", "prompt": "same"},
                ],
            }
        )
        provider = Scripted({"same": ["sample 0", "sample 1"]})
        records = run(Runner(provider, cache=ResponseCache(tmp_path)).run(suite.cases))
        assert provider.calls["same"] == 1
        assert [r.response for r in records] == ["sample 0", "sample 0"]
        assert sorted(r.cached for r in records) == [False, True]
        # A re-run replays exactly what the first run recorded.
        again = run(Runner(Scripted(), cache=ResponseCache(tmp_path)).run(suite.cases))
        assert [r.response for r in again] == ["sample 0", "sample 0"]

    def test_without_a_cache_identical_requests_are_independent(self) -> None:
        suite = parse_suite(
            {
                "name": "t",
                "cases": [
                    {"id": "a", "language": "en", "prompt": "same"},
                    {"id": "b", "language": "en", "prompt": "same"},
                ],
            }
        )
        provider = Scripted({"same": ["sample 0", "sample 1"]})
        records = run(Runner(provider).run(suite.cases))
        assert provider.calls["same"] == 2
        assert sorted(r.response or "" for r in records) == ["sample 0", "sample 1"]

    def test_cache_write_failure_does_not_stop_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = ResponseCache(tmp_path)

        def full_disk(*args: Any, **kwargs: Any) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(cache, "put", full_disk)
        result = run(run_suite(make_suite(3), Scripted(), cache=cache, out_dir=tmp_path / "runs"))
        assert all(r.ok for r in result.records)
        assert len(result.warnings) == 1
        assert result.warnings[0].startswith("3 response(s) could not be written to the cache")
        manifest = json.loads(next((tmp_path / "runs").glob("*/manifest.json")).read_text("utf-8"))
        assert manifest["status"] == "complete"
        assert manifest["warnings"] == result.warnings

    def test_failures_are_not_cached(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        cases = make_suite(1).cases
        run(
            Runner(Scripted({"prompt 0": [ProviderError("x", retryable=False)]}), cache=cache).run(
                cases
            )
        )
        record = run(Runner(Scripted(), cache=cache).run(cases))[0]
        assert record.ok
        assert not record.cached

    def test_sent_params_are_in_the_key_ignored_ones_are_not(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        cases = make_suite(1).cases
        run(Runner(Scripted(), cache=cache, params=GenerationParams(max_tokens=10)).run(cases))
        # seed is not supported by Scripted, so it is never sent: cache hit.
        hit = run(
            Runner(Scripted(), cache=cache, params=GenerationParams(max_tokens=10, seed=5)).run(
                cases
            )
        )
        # max_tokens is sent, so changing it is a different request: cache miss.
        miss = run(
            Runner(Scripted(), cache=cache, params=GenerationParams(max_tokens=20)).run(cases)
        )
        assert hit[0].cached is True
        assert miss[0].cached is False


class TestRunSuite:
    def test_writes_run_directory(self, tmp_path: Path) -> None:
        suite = make_suite(4)
        # Completion order differs from suite order.
        provider = Scripted(delay={"prompt 0": 0.05, "prompt 1": 0.03})
        result = run(run_suite(suite, provider, out_dir=tmp_path, run_id="r1", concurrency=4))
        assert result.run_dir == tmp_path / "r1"
        lines = (tmp_path / "r1" / "responses.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["id"] for line in lines] == ["c0", "c1", "c2", "c3"]
        manifest = json.loads((tmp_path / "r1" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "complete"
        assert manifest["run_id"] == "r1"
        assert manifest["suite"]["cases_selected"] == 4
        assert manifest["provider"] == {"provider": "fake", "model": "m", "spec": "fake:m"}
        assert manifest["counts"]["generated"] == 4
        assert manifest["counts"]["output_tokens"] == 12
        assert not list((tmp_path / "r1").glob("*.tmp"))

    def test_log_is_valid_check_input(self, tmp_path: Path) -> None:
        provider = Scripted({"prompt 1": [ProviderError("x", retryable=False)]})
        result = run(run_suite(make_suite(3), provider, out_dir=tmp_path, run_id="r"))
        assert result.run_dir is not None
        responses = load_responses(result.run_dir / "responses.jsonl")
        assert set(responses) == {"c0", "c2"}

    def test_params_in_manifest(self, tmp_path: Path) -> None:
        params = GenerationParams(temperature=0, seed=1, max_tokens=8)
        result = run(run_suite(make_suite(1), Scripted(), params=params, out_dir=tmp_path))
        assert result.run_dir is not None
        manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["params"] == {
            "requested": {"temperature": 0, "seed": 1, "max_tokens": 8},
            "sent": {"temperature": 0, "max_tokens": 8},
            "ignored": ["seed"],
        }

    def test_case_selection(self) -> None:
        result = run(run_suite(make_suite(5), Scripted(), out_dir=None, case_ids=["c3", "c1"]))
        assert [r.id for r in result.records] == ["c1", "c3"]
        assert result.run_dir is None

    def test_unknown_case_id(self) -> None:
        with pytest.raises(ValueError, match="no such case id"):
            run(run_suite(make_suite(2), Scripted(), out_dir=None, case_ids=["c9"]))

    def test_existing_run_id_gets_a_suffix(self, tmp_path: Path) -> None:
        (tmp_path / "same").mkdir()
        result = run(run_suite(make_suite(1), Scripted(), out_dir=tmp_path, run_id="same"))
        assert result.run_id == "same-2"
        assert (tmp_path / "same-2" / "manifest.json").is_file()

    def test_interrupted_run_keeps_completed_records(self, tmp_path: Path) -> None:
        suite = make_suite(4)
        gate = asyncio.Event()

        class Hangs(Scripted):
            async def generate(self, prompt: Prompt, params: GenerationParams) -> Completion:
                if prompt.user == "prompt 3":
                    await gate.wait()  # never set
                return await super().generate(prompt, params)

        async def go() -> None:
            done: list[RunRecord] = []
            task = asyncio.create_task(
                run_suite(suite, Hangs(), out_dir=tmp_path, run_id="int", on_record=done.append)
            )
            while len(done) < 3:
                await asyncio.sleep(0.005)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        run(go())
        manifest = json.loads((tmp_path / "int" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "interrupted"
        assert manifest["counts"]["cases"] == 3
        lines = (tmp_path / "int" / "responses.jsonl").read_text(encoding="utf-8").splitlines()
        assert sorted(json.loads(line)["id"] for line in lines) == ["c0", "c1", "c2"]

    def test_log_rewrite_failure_still_completes_the_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # On Windows, `Get-Content -Wait responses.jsonl` holds the file open,
        # so the final suite-order rewrite fails. Before, that skipped the
        # manifest update: its status stayed "running" forever.
        real_replace = runner_module.replace_file

        def locked(src: Any, dst: Any) -> None:
            if Path(dst).name == "responses.jsonl":
                raise PermissionError(13, "The process cannot access the file")
            real_replace(src, dst)

        monkeypatch.setattr(runner_module, "replace_file", locked)
        result = run(run_suite(make_suite(3), Scripted(), out_dir=tmp_path, run_id="held"))
        manifest = json.loads((tmp_path / "held" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "complete"
        assert "completion order" in manifest["warnings"][0]
        assert result.warnings == manifest["warnings"]
        assert len(load_responses(tmp_path / "held" / "responses.jsonl")) == 3
        assert not (tmp_path / "held" / "responses.jsonl.tmp").exists()

    def test_a_crash_is_recorded_as_crashed(self, tmp_path: Path) -> None:
        def broken(record: RunRecord) -> None:
            raise RuntimeError("bug in a callback")

        with pytest.raises(ExceptionGroup):
            run(
                run_suite(make_suite(2), Scripted(), out_dir=tmp_path, run_id="x", on_record=broken)
            )
        manifest = json.loads((tmp_path / "x" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "crashed"

    def test_resume_from_cache(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "cache")
        suite = make_suite(3)
        run(run_suite(suite, MockProvider(), cache=cache, out_dir=None, case_ids=["c0"]))
        result = run(run_suite(suite, MockProvider(), cache=cache, out_dir=None))
        assert (result.n_cached, result.n_generated) == (1, 2)


class TestRunResult:
    def rec(self, **kw: Any) -> RunRecord:
        base: dict[str, Any] = {
            "id": "x",
            "response": "r",
            "provider": "p",
            "model": "m",
            "cache_key": "k",
        }
        return RunRecord(**{**base, **kw})

    def test_counts(self) -> None:
        records = [
            self.rec(id="a", input_tokens=3, output_tokens=4, finish_reason="length"),
            self.rec(id="b", cached=True, finish_reason="refusal", output_tokens=1),
            self.rec(id="c", response=None, error="boom"),
        ]
        result = RunResult("r", None, records, 1.0)
        assert result.truncated == ["a"]
        assert result.refused == ["b"]
        assert result.responses() == {"a": "r", "b": "r"}
        assert result.counts() == {
            "cases": 3,
            "generated": 1,
            "cached": 1,
            "errors": 1,
            "truncated": 1,
            "refused": 1,
            "input_tokens": 3,
            "output_tokens": 5,
        }


def test_make_run_id_is_filesystem_safe() -> None:
    from evalkit.providers import OllamaProvider

    now = datetime(2026, 11, 16, 10, 15, 0, tzinfo=UTC)
    assert make_run_id(OllamaProvider("qwen2.5:7b"), now) == "20261116T101500Z-ollama-qwen2.5-7b"
    assert make_run_id(MockProvider("a/b c"), now) == "20261116T101500Z-mock-a-b-c"


def test_runlog_close_is_idempotent(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "x.jsonl")
    log.close()
    log.close()
