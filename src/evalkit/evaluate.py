"""Apply a suite's deterministic checks to a set of responses."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field

from evalkit.checks import CheckResult, Status
from evalkit.langdetect import Language
from evalkit.schema import EvalCase, Suite
from evalkit.taxonomy import FailureMode


class CaseReport(BaseModel):
    """All check results for one case's response."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    language: Language
    tags: list[str] = Field(default_factory=list)
    results: list[CheckResult]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> Status:
        """``fail`` if any check failed, ``skip`` if none applied, else ``pass``."""
        if any(r.failed for r in self.results):
            return Status.FAIL
        if not any(r.passed for r in self.results):
            return Status.SKIP
        return Status.PASS

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.failed]

    @property
    def failure_modes(self) -> list[FailureMode]:
        """Distinct failure-mode tags, in the order the checks raised them."""
        modes = (r.failure_mode for r in self.failures if r.failure_mode is not None)
        return list(dict.fromkeys(modes))


def check_case(case: EvalCase, response: str) -> CaseReport:
    """Run every effective constraint of ``case`` on ``response``."""
    return CaseReport(
        case_id=case.id,
        language=case.language,
        tags=case.tags,
        results=[c.check(response) for c in case.effective_constraints()],
    )


class ResponsesError(Exception):
    """A responses file could not be read. The message is user-facing."""


def load_responses(path: str | Path) -> dict[str, str]:
    """Read a JSONL file of ``{"id": ..., "response": ...}`` objects.

    Blank lines are skipped. A repeated id is an error: silently keeping the
    first or last response would hide a bug in whatever produced the file.
    """
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ResponsesError(f"{path}: cannot read file ({exc.strerror})") from exc
    out: dict[str, str] = {}
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResponsesError(f"{path}:{n}: invalid JSON ({exc.msg})") from exc
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str):
            raise ResponsesError(f"{path}:{n}: each line needs a string 'id'")
        if not isinstance(obj.get("response"), str):
            raise ResponsesError(f"{path}:{n}: each line needs a string 'response'")
        if obj["id"] in out:
            raise ResponsesError(f"{path}:{n}: duplicate id {obj['id']!r}")
        out[obj["id"]] = obj["response"]
    return out


class CheckSummary(BaseModel):
    """Aggregate view of a batch of case reports."""

    reports: list[CaseReport]
    missing: list[str] = Field(default_factory=list)
    """Case ids with no response."""
    unknown: list[str] = Field(default_factory=list)
    """Response ids that match no case."""

    @property
    def counts(self) -> Counter[Status]:
        return Counter(r.status for r in self.reports)

    @property
    def pass_rate(self) -> float:
        """Share of graded cases (pass + fail) that passed every check."""
        c = self.counts
        graded = c[Status.PASS] + c[Status.FAIL]
        return c[Status.PASS] / graded if graded else 0.0

    def pass_rate_by_language(self) -> dict[str, float]:
        groups: dict[str, list[CaseReport]] = defaultdict(list)
        for r in self.reports:
            groups[r.language].append(r)
        return {lang: CheckSummary(reports=rs).pass_rate for lang, rs in sorted(groups.items())}

    def by_check(self) -> dict[str, Counter[Status]]:
        """Per constraint type: how many results passed, failed, were skipped."""
        out: dict[str, Counter[Status]] = defaultdict(Counter)
        for report in self.reports:
            for result in report.results:
                out[result.check][result.status] += 1
        return dict(sorted(out.items()))

    def failure_modes(self) -> Counter[FailureMode]:
        return Counter(m for r in self.reports for m in r.failure_modes)


def check_suite(suite: Suite, responses: Mapping[str, str]) -> CheckSummary:
    """Check every case that has a response; report missing and unknown ids."""
    reports = [check_case(c, responses[c.id]) for c in suite.cases if c.id in responses]
    case_ids = {c.id for c in suite.cases}
    return CheckSummary(
        reports=reports,
        missing=[c.id for c in suite.cases if c.id not in responses],
        unknown=sorted(set(responses) - case_ids),
    )


def iter_failures(summary: CheckSummary) -> Iterable[tuple[str, CheckResult]]:
    """Yield ``(case_id, result)`` for every failed check, in suite order."""
    for report in summary.reports:
        for result in report.failures:
            yield report.case_id, result
