"""Async runner: send a suite's cases to a provider and log every result.

The pieces, from the inside out:

* :class:`RetryPolicy` decides how long to wait before retrying a request
  that failed with a *retryable* :class:`ProviderError`.
* :class:`Runner` sends cases with at most ``concurrency`` requests in flight,
  checks the :class:`ResponseCache` first, and turns each case into one
  :class:`RunRecord`. A case that keeps failing becomes a record with an
  ``error``; it never stops the other cases. With a cache, cases that make the
  identical request share one generation instead of paying for it twice.
* :func:`run_suite` wraps a runner in a run directory::

      runs/<run_id>/manifest.json      what was run, how, and the totals
      runs/<run_id>/responses.jsonl    one RunRecord per line

  ``responses.jsonl`` is appended to as cases finish, so an interrupted run
  keeps everything completed so far. At the end it is rewritten in suite order,
  which makes two runs easy to diff. The file is valid input for
  ``evalkit check``. Re-running an interrupted run resumes it, because every
  completed response is already in the cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from evalkit import __version__
from evalkit.cache import ResponseCache, cache_key
from evalkit.files import write_text_atomic
from evalkit.prompts import Prompt, build_prompt
from evalkit.providers.base import (
    Completion,
    FinishReason,
    GenerationParams,
    Provider,
    ProviderError,
)
from evalkit.schema import EvalCase, Suite

__all__ = [
    "RetryPolicy",
    "RunLog",
    "RunRecord",
    "RunResult",
    "Runner",
    "make_run_id",
    "run_suite",
]

Sleep = Callable[[float], Awaitable[None]]
OnRecord = Callable[["RunRecord"], None]


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter, capped, honouring ``Retry-After``.

    ``max_retries=3`` means up to 4 attempts. Without a ``Retry-After`` from the
    server, retry ``n`` (0-based) waits a random time in
    ``[0, min(max_delay, base_delay * 2**n)]``. The randomness ("full jitter")
    spreads out retries from concurrent requests that failed together, so they
    don't all hit a struggling server again at the same instant.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0

    def __post_init__(self) -> None:
        if self.max_retries < 0 or self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("retry settings must not be negative")

    def delay(self, retry: int, *, retry_after: float | None, rng: random.Random) -> float:
        if retry_after is not None:
            return min(retry_after, self.max_delay)
        return rng.uniform(0.0, min(self.max_delay, self.base_delay * 2**retry))


class RunRecord(BaseModel):
    """The outcome for one case. Exactly one of ``response`` / ``error`` is set."""

    model_config = ConfigDict(frozen=True)

    id: str
    response: str | None
    error: str | None = None
    provider: str
    model: str
    served_model: str | None = None
    finish_reason: FinishReason | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float | None = None
    """Request time; for a cache hit, the time of the original request."""
    attempts: int = 0
    """Requests sent for this case in this run (0 for a cache hit)."""
    cached: bool = False
    cache_key: str

    @property
    def ok(self) -> bool:
        return self.response is not None


def _from_completion(
    base: dict[str, Any], c: Completion, *, latency_s: float, attempts: int, cached: bool
) -> RunRecord:
    return RunRecord(
        **base,
        response=c.text,
        served_model=c.model,
        finish_reason=c.finish_reason,
        input_tokens=c.input_tokens,
        output_tokens=c.output_tokens,
        latency_s=round(latency_s, 4),
        attempts=attempts,
        cached=cached,
    )


class Runner:
    """Runs cases against one provider with a concurrency limit and retries.

    ``sleep``, ``rng`` and ``clock`` are injectable so tests can check the
    backoff schedule without waiting for it.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        params: GenerationParams | None = None,
        concurrency: int = 4,
        retry: RetryPolicy | None = None,
        cache: ResponseCache | None = None,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.provider = provider
        self.params = params or GenerationParams()
        self.concurrency = concurrency
        self.retry = retry or RetryPolicy()
        self.cache = cache
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._clock = clock
        self._key_locks: dict[str, asyncio.Lock] = {}
        self.cache_write_errors: list[str] = []
        """Responses that could not be written to the cache (they are still logged)."""

    async def run(
        self, cases: Sequence[EvalCase], *, on_record: OnRecord | None = None
    ) -> list[RunRecord]:
        """Run every case; return records in the order of ``cases``."""
        slots = asyncio.Semaphore(self.concurrency)
        done: dict[str, RunRecord] = {}

        async def one(case: EvalCase) -> None:
            record = await self.run_case(case, slots)
            done[case.id] = record
            if on_record is not None:
                on_record(record)

        async with asyncio.TaskGroup() as tg:
            for case in cases:
                tg.create_task(one(case))
        return [done[c.id] for c in cases]

    async def run_case(self, case: EvalCase, slots: asyncio.Semaphore | None = None) -> RunRecord:
        prompt = build_prompt(case)
        params = self.provider.effective_params(self.params)
        identity = self.provider.identity()
        key = cache_key(identity, prompt, params)
        base: dict[str, Any] = {
            "id": case.id,
            "provider": self.provider.name,
            "model": self.provider.model,
            "cache_key": key,
        }
        if self.cache is None:
            return await self._fresh(prompt, base, slots)

        # One lock per request: a second case with the identical request waits
        # for the first, then finds its answer in the cache. Without this both
        # generate, and a re-run replays whichever was written last.
        async with self._key_locks.setdefault(key, asyncio.Lock()):
            entry = self.cache.get(key)
            if entry is not None:
                return _from_completion(
                    base, entry.completion, latency_s=entry.latency_s, attempts=0, cached=True
                )
            request = {"identity": identity, "prompt": prompt.model_dump(), "params": params}
            return await self._fresh(prompt, base, slots, cache_as=(key, request))

    async def _fresh(
        self,
        prompt: Prompt,
        base: dict[str, Any],
        slots: asyncio.Semaphore | None,
        *,
        cache_as: tuple[str, dict[str, Any]] | None = None,
    ) -> RunRecord:
        """Generate (with retries) and, on success, store under ``cache_as``."""
        async with slots or asyncio.Semaphore(1):
            outcome, latency, attempts = await self._generate(prompt)
        if isinstance(outcome, str):
            return RunRecord(**base, response=None, error=outcome, attempts=attempts)
        if cache_as is not None and self.cache is not None:
            key, request = cache_as
            try:
                self.cache.put(key, outcome, latency_s=latency, request=request)
            except OSError as exc:
                # The response is still in the run log; only a re-run will
                # have to generate it again.
                self.cache_write_errors.append(f"{base['id']}: {exc}")
        return _from_completion(base, outcome, latency_s=latency, attempts=attempts, cached=False)

    async def _generate(self, prompt: Prompt) -> tuple[Completion | str, float, int]:
        """Try until success, a non-retryable error, or no retries left.

        Returns ``(completion or error message, latency_s, attempts)``.
        The concurrency slot is held while backing off: the point of waiting is
        to send fewer requests, not to let another request take the slot.
        """
        attempts = 0
        while True:
            attempts += 1
            start = self._clock()
            try:
                completion = await self.provider.generate(prompt, self.params)
            except ProviderError as exc:
                retries_used = attempts - 1
                if not exc.retryable or retries_used >= self.retry.max_retries:
                    message = f"{exc} (after {attempts} attempts)" if attempts > 1 else str(exc)
                    return message, 0.0, attempts
                delay = self.retry.delay(retries_used, retry_after=exc.retry_after, rng=self._rng)
                await self._sleep(delay)
                continue
            except Exception as exc:
                # Anything else from a provider (a decoding error, an SDK
                # validation error, a bug) fails this case only, not retried,
                # so one bad response can't cancel the cases in flight.
                return (
                    f"{self.provider.name}: unexpected {type(exc).__name__}: {exc}",
                    0.0,
                    attempts,
                )
            return completion, self._clock() - start, attempts


class RunLog:
    """JSONL log of run records: appended while running, sorted at the end."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = path.open("w", encoding="utf-8", newline="\n")

    def write(self, record: RunRecord) -> None:
        self._fh.write(record.model_dump_json() + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def finalize(self, records: Sequence[RunRecord]) -> None:
        """Replace the log with ``records`` in the given (suite) order.

        If the replace fails, the log keeps its completion order, which is
        still complete and valid ``evalkit check`` input.
        """
        self.close()
        write_text_atomic(self.path, "".join(r.model_dump_json() + "\n" for r in records))


@dataclass
class RunResult:
    """Everything a caller needs after a run: records, timing and totals."""

    run_id: str
    run_dir: Path | None
    records: list[RunRecord]
    duration_s: float
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    warnings: list[str] = field(default_factory=list)
    """Problems that lost no responses, e.g. a cache write that failed."""

    @property
    def errors(self) -> list[RunRecord]:
        return [r for r in self.records if not r.ok]

    @property
    def n_cached(self) -> int:
        return sum(r.cached for r in self.records)

    @property
    def n_generated(self) -> int:
        return sum(r.ok and not r.cached for r in self.records)

    @property
    def truncated(self) -> list[str]:
        """Cases whose response hit the token limit."""
        return [r.id for r in self.records if r.finish_reason == "length"]

    @property
    def refused(self) -> list[str]:
        return [r.id for r in self.records if r.finish_reason == "refusal"]

    def tokens(self) -> tuple[int, int]:
        """(input, output) tokens over the records that report them."""
        return (
            sum(r.input_tokens or 0 for r in self.records),
            sum(r.output_tokens or 0 for r in self.records),
        )

    def responses(self) -> dict[str, str]:
        """``{case_id: response}`` for the cases that got one (for ``check_suite``)."""
        return {r.id: r.response for r in self.records if r.response is not None}

    def counts(self) -> dict[str, int]:
        inp, out = self.tokens()
        return {
            "cases": len(self.records),
            "generated": self.n_generated,
            "cached": self.n_cached,
            "errors": len(self.errors),
            "truncated": len(self.truncated),
            "refused": len(self.refused),
            "input_tokens": inp,
            "output_tokens": out,
        }


def make_run_id(provider: Provider, now: datetime | None = None) -> str:
    """``20261116T101500Z-ollama-qwen2.5-7b``: sortable, filesystem-safe."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", provider.spec).strip("-")
    return f"{stamp}-{slug}"


def _unique_dir(path: Path) -> Path:
    candidate, n = path, 1
    while True:
        try:
            candidate.mkdir(parents=True)
        except FileExistsError:
            n += 1
            candidate = path.with_name(f"{path.name}-{n}")
            continue
        return candidate


def _suite_digest(suite: Suite) -> str:
    return hashlib.sha256(suite.model_dump_json().encode("utf-8")).hexdigest()[:16]


def _select(suite: Suite, case_ids: Sequence[str] | None) -> list[EvalCase]:
    if not case_ids:
        return list(suite.cases)
    wanted = set(case_ids)
    unknown = sorted(wanted - {c.id for c in suite.cases})
    if unknown:
        raise ValueError(f"no such case id(s) in suite {suite.name!r}: {', '.join(unknown)}")
    return [c for c in suite.cases if c.id in wanted]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


async def run_suite(
    suite: Suite,
    provider: Provider,
    *,
    params: GenerationParams | None = None,
    concurrency: int = 4,
    retry: RetryPolicy | None = None,
    cache: ResponseCache | None = None,
    out_dir: Path | None = Path("runs"),
    run_id: str | None = None,
    case_ids: Sequence[str] | None = None,
    suite_path: Path | None = None,
    on_record: OnRecord | None = None,
    sleep: Sleep = asyncio.sleep,
) -> RunResult:
    """Run ``suite`` (or the ``case_ids`` subset) and write the run directory.

    With ``out_dir=None`` nothing is written to disk except the cache.
    """
    params = params or GenerationParams()
    retry = retry or RetryPolicy()
    cases = _select(suite, case_ids)
    started = datetime.now(UTC)
    run_id = run_id or make_run_id(provider, started)
    runner = Runner(
        provider, params=params, concurrency=concurrency, retry=retry, cache=cache, sleep=sleep
    )

    run_dir = _unique_dir(out_dir / run_id) if out_dir is not None else None
    if run_dir is not None:
        run_id = run_dir.name
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "evalkit_version": __version__,
        "suite": {
            "name": suite.name,
            "version": suite.version,
            "path": str(suite_path) if suite_path else None,
            "digest": _suite_digest(suite),
            "cases_selected": len(cases),
            "cases_total": len(suite.cases),
        },
        "provider": {**provider.identity(), "spec": provider.spec},
        "params": {
            "requested": params.explicit(),
            "sent": provider.effective_params(params),
            "ignored": provider.ignored_params(params),
        },
        "concurrency": concurrency,
        "retry": {
            "max_retries": retry.max_retries,
            "base_delay": retry.base_delay,
            "max_delay": retry.max_delay,
        },
        "cache": str(cache.root) if cache is not None else None,
        "started_at": started.isoformat(timespec="seconds"),
    }
    manifest_path = run_dir / "manifest.json" if run_dir is not None else None
    log = RunLog(run_dir / "responses.jsonl") if run_dir is not None else None
    if manifest_path is not None:
        _write_json(manifest_path, manifest)

    seen: list[RunRecord] = []

    def record_done(record: RunRecord) -> None:
        seen.append(record)
        if log is not None:
            log.write(record)
        if on_record is not None:
            on_record(record)

    t0 = time.perf_counter()
    # "interrupted" covers Ctrl-C and cancellation (BaseException); an
    # exception from evalkit itself is a crash, and the manifest says so.
    status: Literal["complete", "interrupted", "crashed"] = "interrupted"
    records: list[RunRecord] = seen
    try:
        records = await runner.run(cases, on_record=record_done)
        status = "complete"
    except Exception:
        status = "crashed"
        raise
    finally:
        duration = time.perf_counter() - t0
        result = RunResult(run_id, run_dir, list(records), round(duration, 3), started)
        if runner.cache_write_errors:
            n = len(runner.cache_write_errors)
            result.warnings.append(
                f"{n} response(s) could not be written to the cache "
                f"(first: {runner.cache_write_errors[0]}); they are in the run log, "
                "but a re-run will generate them again"
            )
        if log is not None:
            try:
                if status == "complete":
                    log.finalize(records)
                else:
                    log.close()
            except OSError as exc:
                # The appended log is complete; only the suite-order rewrite failed.
                log.close()
                result.warnings.append(
                    f"could not rewrite {log.path} in suite order ({exc}); "
                    "it is complete but in completion order"
                )
        if manifest_path is not None:
            manifest.update(
                status=status,
                finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
                duration_s=result.duration_s,
                counts=result.counts(),
            )
            if result.warnings:
                manifest["warnings"] = result.warnings
            _write_json(manifest_path, manifest)
    return result
