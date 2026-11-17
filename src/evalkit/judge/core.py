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
from the judge's, so a report can say who found what. For the checkable modes
(:data:`evalkit.taxonomy.CHECKABLE`) only the checks count; the judge's tags
for them are kept and compared with the checks (:meth:`JudgeSummary.judge_vs_checks`).
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
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

from evalkit import __version__
from evalkit.cache import ResponseCache
from evalkit.checks import Status
from evalkit.evaluate import check_case
from evalkit.judge.parse import JudgmentError, parse_judgment
from evalkit.judge.prompt import (
    JUDGE_PROMPT_VERSION,
    build_judge_prompt,
    build_repair_prompt,
    cut_response,
)
from evalkit.langdetect import Language
from evalkit.prompts import Prompt
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
from evalkit.taxonomy import CHECKABLE, FailureMode

__all__ = [
    "Judge",
    "JudgeRecord",
    "JudgeResult",
    "JudgeSummary",
    "judge_suite",
]

OnJudgment = Callable[["JudgeRecord"], None]

DEFAULT_MAX_RESPONSE_CHARS = 6000
"""About 1,500-2,000 tokens of EN/ID text: with the ~1,200-token rubric and a
reply, the judge's prompt stays inside a 4,096-token local context."""
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
    """Every failure mode the judge named, including checkable ones that don't count."""
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
    response_cut: bool = False
    """The response was longer than ``max_response_chars``; the judge saw only its start."""
    sample: int = 0
    """Which sample this is, when each response is judged several times (self-consistency)."""

    @property
    def ok(self) -> bool:
        return self.error is None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failure_modes(self) -> list[FailureMode]:
        """The failure modes that count: the checks', then the judge's for the
        modes no check decides. Checks first, without repeats."""
        judged = [m for m in self.judge_failure_modes if m not in CHECKABLE]
        return list(dict.fromkeys([*self.check_failure_modes, *judged]))

    def sources(self) -> dict[FailureMode, Literal["check", "judge", "both"]]:
        """Each counted failure mode and who found it.

        "both" happens only when a check is re-tagged with ``fails_as`` as a
        mode the judge also named: the judge's tags for checkable modes
        don't count, so they never make a checkable mode "both".
        """
        out: dict[FailureMode, Literal["check", "judge", "both"]] = {}
        for mode in self.failure_modes:
            by_check = mode in self.check_failure_modes
            by_judge = mode in self.judge_failure_modes and mode not in CHECKABLE
            out[mode] = "both" if by_check and by_judge else "check" if by_check else "judge"
        return out

    @property
    def parse_outcome(self) -> ParseOutcome:
        if self.ok:
            if self.reasks:
                return "reasked"
            return "repaired" if self.repairs else "clean"
        # Each step sends one prompt; a step whose request failed adds no reply.
        return "unusable" if len(self.replies) > self.reasks else "request_failed"


def _add(total: int | None, value: int | None) -> int | None:
    if value is None:
        return total
    return (total or 0) + value


_P = TypeVar("_P")


@dataclass
class Asked(Generic[_P]):
    """The outcome of :meth:`Judge.ask`: a parsed reply or an error, plus what it cost."""

    value: _P | None
    error: str | None = None
    replies: list[str] = field(default_factory=list)
    reasks: int = 0
    requests: int = 0
    cached: bool = True
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None

    def usage(self) -> dict[str, Any]:
        """The fields :class:`JudgeRecord` and the pairwise records share."""
        return {
            "requests": self.requests,
            "cached": self.cached,
            "latency_s": round(self.latency_s, 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "replies": self.replies,
            "reasks": self.reasks,
        }


class Judge:
    """Grades responses with a judge model.

    ``include_checks`` puts the deterministic check results in the judge's
    prompt as reliable facts. ``max_reasks`` caps the follow-up requests after
    an unusable reply (0 = never re-ask). ``max_response_chars`` cuts longer
    responses in the judge's prompt (``None`` = never cut); the checks still
    see the whole response. The other settings are the :class:`Runner`'s and
    apply to every judge request, re-asks included.
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
        max_response_chars: int | None = DEFAULT_MAX_RESPONSE_CHARS,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if max_reasks < 0:
            raise ValueError("max_reasks must not be negative")
        if max_response_chars is not None and max_response_chars < 1:
            raise ValueError("max_response_chars must be at least 1 (or None for no cut)")
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
        self.max_response_chars = max_response_chars

    @property
    def provider(self) -> Provider:
        return self.runner.provider

    def sample_params(self, sample: int) -> GenerationParams | None:
        """Settings for sample ``sample``: the runner's, with the seed moved on by
        ``sample`` if one is set, so samples differ yet each can be repeated.
        ``None`` (the runner's own settings) for sample 0."""
        params = self.runner.params
        if sample == 0 or params.seed is None:
            return None
        return params.model_copy(update={"seed": params.seed + sample})

    async def ask(
        self,
        original: Prompt,
        parse: Callable[[str], _P],
        *,
        label: str,
        slots: asyncio.Semaphore | None = None,
        sample: int = 0,
    ) -> Asked[_P]:
        """Send ``original`` and parse the reply, re-asking after an unusable one.

        ``parse`` raises :class:`JudgmentError` with the problems to send back.
        Never raises for a bad reply or a failed request: those end up in
        :attr:`Asked.error`.
        """
        out_: Asked[_P] = Asked(None)
        params = self.sample_params(sample)
        prompt = original
        for reasks in range(self.max_reasks + 1):
            out = await self.runner.complete(
                prompt, slots, label=label, sample=sample, params=params
            )
            out_.reasks = reasks
            out_.requests += out.attempts
            out_.latency_s += out.latency_s
            out_.cached = out_.cached and out.cached
            completion = out.completion
            if completion is None:
                out_.error = out.error
                return out_
            out_.input_tokens = _add(out_.input_tokens, completion.input_tokens)
            out_.output_tokens = _add(out_.output_tokens, completion.output_tokens)
            out_.replies.append(completion.text)
            if completion.finish_reason == "refusal":
                out_.error = "the judge refused"
                return out_
            try:
                out_.value = parse(completion.text)
            except JudgmentError as exc:
                problems = exc.problems
                if completion.finish_reason == "length":
                    problems = [
                        *problems,
                        "the reply was cut off at the token limit: keep the rationale short",
                    ]
                if reasks == self.max_reasks:
                    tries = f" (after {reasks} re-asks)" if reasks else ""
                    out_.error = f"unusable judge reply{tries}: {'; '.join(problems)}"
                    return out_
                prompt = build_repair_prompt(original, completion.text, problems)
                continue
            return out_
        raise AssertionError("unreachable")  # pragma: no cover - the loop always returns

    async def judge_case(
        self,
        case: EvalCase,
        response: str,
        slots: asyncio.Semaphore | None = None,
        *,
        sample: int = 0,
    ) -> JudgeRecord:
        """Judge one response. Never raises for a bad reply or a failed request."""
        report = check_case(case, response)
        dimensions = case.rubric.dimensions if case.rubric is not None else list(Dimension)
        checks = report.results if self.include_checks else None
        original = build_judge_prompt(
            case, response, checks, max_response_chars=self.max_response_chars
        )
        _, was_cut = cut_response(response, self.max_response_chars)
        base: dict[str, Any] = {
            "id": case.id,
            "language": case.language,
            "tags": case.tags,
            "judge": self.provider.spec,
            "check_status": report.status,
            "check_failure_modes": report.failure_modes,
            "response_cut": was_cut,
            "sample": sample,
        }
        asked = await self.ask(
            original,
            lambda text: parse_judgment(text, dimensions),
            label=case.id,
            slots=slots,
            sample=sample,
        )
        parsed = asked.value
        if parsed is None:
            return JudgeRecord(**base, **asked.usage(), error=asked.error)
        return JudgeRecord(
            **base,
            **asked.usage(),
            scores=parsed.scores,
            rationale=parsed.rationale,
            judge_failure_modes=parsed.failure_modes,
            repairs=parsed.repairs,
        )

    async def judge_all(
        self,
        items: Sequence[tuple[EvalCase, str]],
        *,
        samples: int = 1,
        on_record: OnJudgment | None = None,
    ) -> list[JudgeRecord]:
        """Judge every ``(case, response)`` ``samples`` times.

        Records come back in input order, and within an item by sample.
        """
        if samples < 1:
            raise ValueError("samples must be at least 1")
        slots = asyncio.Semaphore(self.runner.concurrency)
        # Keyed by position, not case id: two answers to one case are two items.
        done: dict[tuple[int, int], JudgeRecord] = {}

        async def one(index: int, case: EvalCase, response: str, sample: int) -> None:
            record = await self.judge_case(case, response, slots, sample=sample)
            done[index, sample] = record
            if on_record is not None:
                on_record(record)

        async with asyncio.TaskGroup() as tg:
            for index, (case, response) in enumerate(items):
                for sample in range(samples):
                    tg.create_task(one(index, case, response, sample))
        return [done[index, s] for index in range(len(items)) for s in range(samples)]


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
        """Per counted failure mode: cases where the checks, the judge, or both found it.

        Counts cover judged records only, so the three numbers are comparable.
        A checkable mode is counted from the checks only, so its "judge" count
        is 0 (see :meth:`judge_vs_checks`); "both" happens when a check is
        re-tagged with ``fails_as`` as a mode the judge also named.
        """
        out: dict[FailureMode, Counter[str]] = defaultdict(Counter)
        for r in self.judged:
            for mode, source in r.sources().items():
                out[mode][source] += 1
        order = list(FailureMode)
        return {
            mode: {k: c[k] for k in ("check", "judge", "both")}
            for mode, c in sorted(out.items(), key=lambda kv: order.index(kv[0]))
        }

    def judge_vs_checks(self) -> dict[FailureMode, dict[str, int]]:
        """For each checkable mode either side named: how often the judge agreed.

        ``both``: the judge named it and a check failed with it; ``judge_only``:
        the judge named it but no check did (these tags don't count);
        ``checks_only``: a check failed with it and the judge didn't name it.
        """
        out: dict[FailureMode, dict[str, int]] = {}
        for mode in (m for m in FailureMode if m in CHECKABLE):
            pairs = [
                (mode in r.judge_failure_modes, mode in r.check_failure_modes) for r in self.judged
            ]
            if any(j or c for j, c in pairs):
                out[mode] = {
                    "both": sum(j and c for j, c in pairs),
                    "judge_only": sum(j and not c for j, c in pairs),
                    "checks_only": sum(c and not j for j, c in pairs),
                }
        return out

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
            "judge_vs_checks": {m.value: c for m, c in self.judge_vs_checks().items()},
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
            # One record per case and sample: with several samples, cases < judgments.
            "cases": sum(r.sample == 0 for r in records),
            "judgments": len(records),
            "judged": sum(r.ok for r in records),
            "errors": sum(not r.ok for r in records),
            "cached": sum(r.cached for r in records),
            "requests": sum(r.requests for r in records),
            "reasks": sum(r.reasks for r in records),
            "responses_cut": sum(r.response_cut for r in records),
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
    samples: int = 1,
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
            max_response_chars=judge.max_response_chars,
            samples=samples,
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
        records = await judge.judge_all(items, samples=samples, on_record=record_done)
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
