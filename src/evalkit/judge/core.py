"""The LLM judge: grade responses against each case's rubric, with repair and re-asks.

For each response the judge model gets one prompt (:func:`build_judge_prompt`)
and must reply with a JSON object. Its reply goes through
:func:`parse_judgment`; syntax slips are repaired locally, and anything else
goes back to the judge as a re-ask that shows the unusable reply and lists
what was wrong. After ``max_reasks`` re-asks the case gets an error instead
of a judgment. Every judge request goes through the same :class:`Runner`
as generation, so it shares the response cache, the retries and the
concurrency limit: re-judging a run replays the cached judgments exactly.

The deterministic checks run first. Their results go into the judge's prompt
(unless ``include_checks=False``), and their failure modes are kept apart
from the judge's, so a report can say who found what.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from evalkit import __version__
from evalkit.cache import ResponseCache
from evalkit.checks import Status
from evalkit.evaluate import check_case
from evalkit.judge.parse import JudgmentError, parse_judgment
from evalkit.judge.prompt import JUDGE_PROMPT_VERSION, build_judge_prompt, build_repair_prompt
from evalkit.langdetect import Language
from evalkit.providers.base import GenerationParams, Provider
from evalkit.runner import (
    RetryPolicy,
    RunDir,
    Runner,
    RunStatus,
    Sleep,
    cache_warnings,
    make_run_id,
    select_cases,
    suite_info,
)
from evalkit.schema import Dimension, EvalCase, Suite
from evalkit.taxonomy import FailureMode

__all__ = [
    "Judge",
    "JudgeRecord",
    "JudgeResult",
    "JudgeSummary",
    "judge_suite",
]

OnJudgment = Callable[["JudgeRecord"], None]
ParseOutcome = Literal["clean", "repaired", "reasked", "unusable", "request_failed"]


class JudgeRecord(BaseModel):
    """The judge's verdict on one response, or why there is none.

    Exactly one of ``rationale`` (with ``scores``) and ``error`` is set.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    language: Language
    tags: list[str] = Field(default_factory=list)
    judge: str
    """The judge's ``provider:model`` spec."""
    scores: dict[Dimension, int] = Field(default_factory=dict)
    rationale: str | None = None
    judge_failure_modes: list[FailureMode] = Field(default_factory=list)
    check_status: Status
    """The deterministic checks' verdict on the same response."""
    check_failure_modes: list[FailureMode] = Field(default_factory=list)
    error: str | None = None
    repairs: list[str] = Field(default_factory=list)
    """Syntax fixes applied locally to the reply that was accepted."""
    reasks: int = 0
    """Times the judge was asked again after an unusable reply."""
    requests: int = 0
    """Requests sent in this run, retries included (0 if all came from the cache)."""
    cached: bool = False
    """Every judge reply for this case came from the cache."""
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    replies: list[str] = Field(default_factory=list)
    """Every raw reply from the judge, in order; the last one was accepted (if ok)."""

    @property
    def ok(self) -> bool:
        return self.error is None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failure_modes(self) -> list[FailureMode]:
        """Check and judge failure modes together, checks first, without repeats."""
        return list(dict.fromkeys([*self.check_failure_modes, *self.judge_failure_modes]))

    @property
    def parse_outcome(self) -> ParseOutcome:
        if self.ok:
            if self.reasks:
                return "reasked"
            return "repaired" if self.repairs else "clean"
        return "unusable" if self.replies else "request_failed"


def _add(total: int | None, value: int | None) -> int | None:
    if value is None:
        return total
    return (total or 0) + value


class Judge:
    """Grades responses with a judge model.

    ``include_checks`` puts the deterministic check results in the judge's
    prompt as reliable facts. ``max_reasks`` caps the follow-up requests after
    an unusable reply (0 = never re-ask). The other settings are the
    :class:`Runner`'s and apply to every judge request, re-asks included.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        params: GenerationParams | None = None,
        concurrency: int = 4,
        retry: RetryPolicy | None = None,
        cache: ResponseCache | None = None,
        include_checks: bool = True,
        max_reasks: int = 2,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if max_reasks < 0:
            raise ValueError("max_reasks must not be negative")
        self.runner = Runner(
            provider,
            params=params,
            concurrency=concurrency,
            retry=retry,
            cache=cache,
            sleep=sleep,
            rng=rng,
        )
        self.include_checks = include_checks
        self.max_reasks = max_reasks

    @property
    def provider(self) -> Provider:
        return self.runner.provider

    async def judge_case(
        self, case: EvalCase, response: str, slots: asyncio.Semaphore | None = None
    ) -> JudgeRecord:
        """Judge one response. Never raises for a bad reply or a failed request."""
        report = check_case(case, response)
        dimensions = case.rubric.dimensions if case.rubric is not None else list(Dimension)
        original = build_judge_prompt(
            case, response, report.results if self.include_checks else None
        )
        base: dict[str, Any] = {
            "id": case.id,
            "language": case.language,
            "tags": case.tags,
            "judge": self.provider.spec,
            "check_status": report.status,
            "check_failure_modes": report.failure_modes,
        }
        replies: list[str] = []
        requests, latency = 0, 0.0
        tokens_in: int | None = None
        tokens_out: int | None = None
        all_cached = True
        prompt = original
        for reasks in range(self.max_reasks + 1):
            out = await self.runner.complete(prompt, slots, label=case.id)
            requests += out.attempts
            latency += out.latency_s
            all_cached = all_cached and out.cached
            usage = {
                "requests": requests,
                "cached": all_cached,
                "latency_s": round(latency, 4),
                "replies": replies,
            }
            if out.completion is None:
                return JudgeRecord(**base, **usage, error=out.error, reasks=reasks)
            completion = out.completion
            tokens_in = _add(tokens_in, completion.input_tokens)
            tokens_out = _add(tokens_out, completion.output_tokens)
            usage.update(input_tokens=tokens_in, output_tokens=tokens_out)
            replies.append(completion.text)
            if completion.finish_reason == "refusal":
                return JudgeRecord(**base, **usage, error="the judge refused", reasks=reasks)
            try:
                parsed = parse_judgment(completion.text, dimensions)
            except JudgmentError as exc:
                problems = exc.problems
                if completion.finish_reason == "length":
                    problems = [
                        *problems,
                        "the reply was cut off at the token limit: keep the rationale short",
                    ]
                if reasks == self.max_reasks:
                    tries = f" (after {reasks} re-asks)" if reasks else ""
                    return JudgeRecord(
                        **base,
                        **usage,
                        error=f"unusable judge reply{tries}: {'; '.join(problems)}",
                        reasks=reasks,
                    )
                prompt = build_repair_prompt(original, completion.text, problems)
                continue
            return JudgeRecord(
                **base,
                **usage,
                scores=parsed.scores,
                rationale=parsed.rationale,
                judge_failure_modes=parsed.failure_modes,
                repairs=parsed.repairs,
                reasks=reasks,
            )
        raise AssertionError("unreachable")  # pragma: no cover - the loop always returns

    async def judge_all(
        self,
        items: Sequence[tuple[EvalCase, str]],
        *,
        on_record: OnJudgment | None = None,
    ) -> list[JudgeRecord]:
        """Judge every ``(case, response)``; records come back in input order."""
        slots = asyncio.Semaphore(self.runner.concurrency)
        done: dict[str, JudgeRecord] = {}

        async def one(case: EvalCase, response: str) -> None:
            record = await self.judge_case(case, response, slots)
            done[case.id] = record
            if on_record is not None:
                on_record(record)

        async with asyncio.TaskGroup() as tg:
            for case, response in items:
                tg.create_task(one(case, response))
        return [done[case.id] for case, _ in items]


def _mean(values: Sequence[int]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


class JudgeSummary(BaseModel):
    """Aggregates over a batch of judge records."""

    records: list[JudgeRecord]
    missing: list[str] = Field(default_factory=list)
    """Case ids with no response to judge."""
    unknown: list[str] = Field(default_factory=list)
    """Response ids that match no case."""

    @property
    def judged(self) -> list[JudgeRecord]:
        return [r for r in self.records if r.ok]

    @property
    def errors(self) -> list[JudgeRecord]:
        return [r for r in self.records if not r.ok]

    def mean_scores(self, language: str | None = None) -> dict[Dimension, float | None]:
        """Mean score per dimension over judged records (``None`` if none scored it)."""
        records = [r for r in self.judged if language is None or r.language == language]
        return {d: _mean([r.scores[d] for r in records if d in r.scores]) for d in Dimension}

    def languages(self) -> list[str]:
        return sorted({r.language for r in self.judged})

    def score_counts(self) -> dict[Dimension, dict[int, int]]:
        """How often each score 1-5 was given, per dimension."""
        out: dict[Dimension, dict[int, int]] = {}
        for dim in Dimension:
            counts = Counter(r.scores[dim] for r in self.judged if dim in r.scores)
            out[dim] = {score: counts[score] for score in range(1, 6)}
        return out

    def failure_modes(self) -> dict[FailureMode, dict[str, int]]:
        """Per failure mode: cases where only the checks, only the judge, or both found it.

        Counts cover judged records only, so the three numbers are comparable.
        """
        out: dict[FailureMode, Counter[str]] = defaultdict(Counter)
        for r in self.judged:
            for mode in r.failure_modes:
                by_check = mode in r.check_failure_modes
                by_judge = mode in r.judge_failure_modes
                source = "both" if by_check and by_judge else "check" if by_check else "judge"
                out[mode][source] += 1
        order = list(FailureMode)
        return {
            mode: {k: c[k] for k in ("check", "judge", "both")}
            for mode, c in sorted(out.items(), key=lambda kv: order.index(kv[0]))
        }

    def parse_outcomes(self) -> dict[ParseOutcome, int]:
        counts = Counter(r.parse_outcome for r in self.records)
        keys: tuple[ParseOutcome, ...] = (
            "clean",
            "repaired",
            "reasked",
            "unusable",
            "request_failed",
        )
        return {k: counts[k] for k in keys}

    def to_dict(self) -> dict[str, Any]:
        """The aggregates as plain JSON-ready data (records not included)."""
        return {
            "judged": len(self.judged),
            "errors": len(self.errors),
            "parse": self.parse_outcomes(),
            "mean_scores": {d.value: v for d, v in self.mean_scores().items()},
            "mean_scores_by_language": {
                lang: {d.value: v for d, v in self.mean_scores(lang).items()}
                for lang in self.languages()
            },
            "score_counts": {
                d.value: {str(k): v for k, v in c.items()} for d, c in self.score_counts().items()
            },
            "failure_modes": {m.value: c for m, c in self.failure_modes().items()},
            "missing": self.missing,
            "unknown": self.unknown,
        }


@dataclass
class JudgeResult:
    """A finished (or interrupted) judge run."""

    run_id: str
    run_dir: Path | None
    summary: JudgeSummary
    duration_s: float
    warnings: list[str] = field(default_factory=list)

    @property
    def records(self) -> list[JudgeRecord]:
        return self.summary.records

    def counts(self) -> dict[str, int]:
        records = self.records
        return {
            "cases": len(records),
            "judged": sum(r.ok for r in records),
            "errors": sum(not r.ok for r in records),
            "cached": sum(r.cached for r in records),
            "requests": sum(r.requests for r in records),
            "reasks": sum(r.reasks for r in records),
            "input_tokens": sum(r.input_tokens or 0 for r in records),
            "output_tokens": sum(r.output_tokens or 0 for r in records),
        }


def responses_digest(responses: Mapping[str, str]) -> str:
    """A short hash of the responses being judged, independent of their order."""
    blob = json.dumps(sorted(responses.items()), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def make_judge_run_id(judge: Judge, now: datetime | None = None) -> str:
    """``20261117T010000Z-judge-ollama-qwen2.5-14b``."""
    stamp, _, slug = make_run_id(judge.provider, now).partition("-")
    return f"{stamp}-judge-{slug}"


async def judge_suite(
    suite: Suite,
    responses: Mapping[str, str],
    judge: Judge,
    *,
    out_dir: Path | None = Path("runs"),
    run_id: str | None = None,
    case_ids: Sequence[str] | None = None,
    suite_path: Path | None = None,
    responses_path: Path | None = None,
    on_record: OnJudgment | None = None,
) -> JudgeResult:
    """Judge every selected case that has a response, and write a run directory.

    The directory holds ``judgments.jsonl`` (one :class:`JudgeRecord` per
    line, in suite order) and ``manifest.json``. With ``out_dir=None``
    nothing is written except the cache.
    """
    cases = select_cases(suite, case_ids)
    items = [(c, responses[c.id]) for c in cases if c.id in responses]
    missing = [c.id for c in cases if c.id not in responses]
    unknown = sorted(set(responses) - {c.id for c in suite.cases})
    started = datetime.now(UTC)
    run_id = run_id or make_judge_run_id(judge, started)
    provider, runner = judge.provider, judge.runner

    rundir = RunDir(out_dir, run_id, "judgments.jsonl") if out_dir is not None else None
    if rundir is not None:
        run_id = rundir.run_id
        rundir.write_manifest(
            run_id=run_id,
            kind="judge",
            status="running",
            evalkit_version=__version__,
            suite=suite_info(suite, suite_path, len(cases)),
            responses={
                "path": str(responses_path) if responses_path else None,
                "digest": responses_digest(responses),
                "count": len(responses),
            },
            judge={**provider.identity(), "spec": provider.spec},
            judge_prompt_version=JUDGE_PROMPT_VERSION,
            include_checks=judge.include_checks,
            max_reasks=judge.max_reasks,
            params={
                "requested": runner.params.explicit(),
                "sent": provider.effective_params(runner.params),
                "ignored": provider.ignored_params(runner.params),
            },
            concurrency=runner.concurrency,
            cache=str(runner.cache.root) if runner.cache is not None else None,
            started_at=started.isoformat(timespec="seconds"),
        )

    seen: list[JudgeRecord] = []

    def record_done(record: JudgeRecord) -> None:
        seen.append(record)
        if rundir is not None:
            rundir.log.write(record)
        if on_record is not None:
            on_record(record)

    t0 = time.perf_counter()
    status: RunStatus = "interrupted"
    records = seen
    try:
        records = await judge.judge_all(items, on_record=record_done)
        status = "complete"
    except Exception:
        status = "crashed"
        raise
    finally:
        summary = JudgeSummary(records=list(records), missing=missing, unknown=unknown)
        result = JudgeResult(
            run_id,
            rundir.path if rundir is not None else None,
            summary,
            round(time.perf_counter() - t0, 3),
            warnings=cache_warnings(runner),
        )
        if rundir is not None:
            warning = rundir.close(status, records)
            if warning:
                result.warnings.append(warning)
            rundir.write_manifest(
                status=status,
                finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
                duration_s=result.duration_s,
                counts=result.counts(),
                summary=summary.to_dict(),
                **({"warnings": result.warnings} if result.warnings else {}),
            )
    return result
