"""Apply a suite's deterministic checks to a set of responses."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field

from evalkit.checks import CheckResult, Status
from evalkit.files import NotUtf8Error, read_utf8
from evalkit.langdetect import Language
from evalkit.schema import EvalCase, Suite
from evalkit.taxonomy import FailureMode
from evalkit.text import scrub_surrogates


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
    Run logs (``runs/<id>/responses.jsonl``) are valid input: a row whose
    generation failed (``"response": null`` with an ``"error"``) is left out,
    so the case is reported as having no response. A lone surrogate escape
    (``"\\ud83d"``) becomes U+FFFD, as in every other UTF-8 tool.
    """
    path = Path(path)
    try:
        # Split on "\n" only. str.splitlines() also splits on U+2028, U+2029
        # and U+0085, which JSON allows unescaped inside strings and which
        # model_dump_json writes raw, so a response containing one would
        # break its own run log.
        lines = read_utf8(path).split("\n")
    except OSError as exc:
        raise ResponsesError(f"{path}: cannot read file ({exc.strerror})") from exc
    except NotUtf8Error as exc:
        raise ResponsesError(f"{path}: {exc}") from exc
    out: dict[str, str] = {}
    failed: set[str] = set()
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResponsesError(f"{path}:{n}: invalid JSON ({exc.msg})") from exc
        except RecursionError as exc:
            raise ResponsesError(f"{path}:{n}: invalid JSON (nested too deeply)") from exc
        except ValueError as exc:  # e.g. an integer over Python's 4,300-digit limit
            raise ResponsesError(f"{path}:{n}: invalid JSON ({exc})") from exc
        if isinstance(obj, dict):  # ids too: a lone surrogate can't be written out
            obj = {k: scrub_surrogates(v) if isinstance(v, str) else v for k, v in obj.items()}
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str):
            raise ResponsesError(f"{path}:{n}: each line needs a string 'id'")
        if obj["id"] in out or obj["id"] in failed:
            raise ResponsesError(f"{path}:{n}: duplicate id {obj['id']!r}")
        if obj.get("response") is None and isinstance(obj.get("error"), str):
            failed.add(obj["id"])
            continue
        if not isinstance(obj.get("response"), str):
            raise ResponsesError(f"{path}:{n}: each line needs a string 'response'")
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
