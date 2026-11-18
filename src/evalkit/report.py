"""Reports over one suite and one or more runs, and the diff between two runs.

A *run* is what ``evalkit run`` or ``evalkit judge`` wrote: a run directory,
or the ``responses.jsonl`` / ``judgments.jsonl`` in it (any responses JSONL
works too). :func:`load_run` reads one into a :class:`RunView`, one
:class:`CaseView` per case; :func:`diff_runs` sets two runs side by side,
case by case; :func:`build_report` turns runs into the tables that both
:func:`render_markdown` and :func:`render_html` print, so the two formats
can't disagree.

For a judge run the check verdicts are the ones the judge saw (from its
records). The responses it judged are read from the path in its manifest when
that file is still there and unchanged (same digest); they add the response
text and the failed checks' messages to the HTML report and let a diff say
how many responses changed.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from evalkit import __version__
from evalkit.agreement import LabelsError, read_judgments
from evalkit.checks import Status
from evalkit.evaluate import ResponsesError, check_case, load_responses
from evalkit.files import NotUtf8Error, read_utf8
from evalkit.judge.core import JudgeRecord, JudgeSummary, responses_digest
from evalkit.runner import suite_digest
from evalkit.schema import Dimension, EvalCase, Suite
from evalkit.taxonomy import CHECKABLE, FailureMode

__all__ = [
    "CaseChange",
    "CaseView",
    "Report",
    "RunDiff",
    "RunError",
    "RunView",
    "build_report",
    "default_name",
    "diff_runs",
    "diff_tables",
    "load_run",
    "load_runs",
    "render_html",
    "render_markdown",
]

DIM_SHORT = {
    Dimension.INSTRUCTION_FOLLOWING: "IF",
    Dimension.REASONING_QUALITY: "RQ",
    Dimension.FAITHFULNESS: "FA",
    Dimension.HELPFULNESS: "HE",
}
RunKind = Literal["responses", "judgments"]


class RunError(ValueError):
    """A run can't be read. The message is user-facing."""


def default_name(path: Path) -> str:
    """A run log is ``runs/<run_id>/<log>.jsonl``: name it after the run."""
    if path.name in {"responses.jsonl", "judgments.jsonl"}:
        return path.parent.name
    return path.stem


# ---------------------------------------------------------------- reading runs


@dataclass(frozen=True)
class CaseView:
    """One case in one run: the checks' verdict and, in a judge run, the judge's."""

    id: str
    language: str
    status: Status
    check_modes: tuple[FailureMode, ...]
    failed_checks: tuple[str, ...] | None = None
    """``check: message`` per failed check; ``None`` when the response isn't at hand."""
    response: str | None = None
    record: JudgeRecord | None = None

    @property
    def judged(self) -> bool:
        return self.record is not None and self.record.ok

    @property
    def scores(self) -> dict[Dimension, int]:
        return self.record.scores if self.record is not None else {}

    @property
    def failure_modes(self) -> list[FailureMode]:
        """The modes that count: the checks', then the judge's for modes no check decides."""
        return self.record.failure_modes if self.record is not None else list(self.check_modes)

    @property
    def judge_only(self) -> list[FailureMode]:
        """Checkable modes the judge named that no check raised. They don't count."""
        if self.record is None:
            return []
        named = self.record.judge_failure_modes
        return [m for m in named if m in CHECKABLE and m not in self.check_modes]


@dataclass
class RunView:
    """One run of a suite, case by case, in suite order."""

    name: str
    source: str
    kind: RunKind
    cases: list[CaseView]
    missing: list[str] = field(default_factory=list)
    """Suite cases the run has nothing for."""
    unknown: list[str] = field(default_factory=list)
    """Ids in the run that match no case of the suite."""
    model: str | None = None
    """The ``provider:model`` that wrote the responses, when a manifest says so."""
    judge: str | None = None
    judge_prompt_version: str | None = None
    notes: list[str] = field(default_factory=list)

    def get(self, case_id: str) -> CaseView | None:
        return next((c for c in self.cases if c.id == case_id), None)

    def pass_rate(self, language: str | None = None) -> tuple[int, int]:
        """(cases that passed every check, cases graded); skipped cases aren't graded."""
        return _pass_rate([c for c in self.cases if language in (None, c.language)])

    def summary(self) -> JudgeSummary | None:
        """The judge's aggregates (the numbers ``evalkit judge`` prints), for a judge run."""
        if self.kind != "judgments":
            return None
        return JudgeSummary(records=[c.record for c in self.cases if c.record is not None])

    def mode_counts(self) -> Counter[FailureMode]:
        return Counter(m for c in self.cases for m in c.failure_modes)

    def judge_only_counts(self) -> Counter[FailureMode]:
        return Counter(m for c in self.cases for m in c.judge_only)


def _pass_rate(cases: Sequence[CaseView]) -> tuple[int, int]:
    statuses = Counter(c.status for c in cases)
    return statuses[Status.PASS], statuses[Status.PASS] + statuses[Status.FAIL]


def _manifest(path: Path) -> dict[str, Any]:
    """A run's manifest.json, or ``{}`` if there is none (a plain JSONL file)."""
    try:
        data = json.loads(read_utf8(path))
    except (OSError, NotUtf8Error, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _get(data: Any, *keys: str) -> str | None:
    """``data[k1][k2]...`` if it is a string, else ``None``."""
    for key in keys:
        data = data.get(key) if isinstance(data, dict) else None
    return data if isinstance(data, str) else None


def _log_file(path: Path) -> Path:
    if not path.is_dir():
        return path
    for name in ("judgments.jsonl", "responses.jsonl"):
        if (path / name).is_file():
            return path / name
    raise RunError(f"{path}: no judgments.jsonl or responses.jsonl in this directory")


def _is_judgments(path: Path) -> bool:
    """A judge record has ``check_status``; a line of responses doesn't."""
    try:
        text = read_utf8(path)
    except (OSError, NotUtf8Error) as exc:
        raise RunError(f"{path}: {exc}") from exc
    first = next((line for line in text.split("\n") if line.strip()), "")
    try:
        obj = json.loads(first)
    except json.JSONDecodeError:
        return False  # load_responses reports the line
    return isinstance(obj, dict) and "check_status" in obj


def _failed(case: EvalCase, response: str | None) -> tuple[str, ...] | None:
    if response is None:
        return None
    return tuple(f"{r.check}: {r.message}" for r in check_case(case, response).failures)


def _judged_responses(
    manifest: dict[str, Any], notes: list[str]
) -> tuple[Path | None, dict[str, str]]:
    """The file a judge run judged and its responses, if it is still the same file."""
    where = _get(manifest, "responses", "path")
    if where is None:
        return None, {}
    path = Path(where)
    try:
        responses = load_responses(path)
    except ResponsesError:
        notes.append(f"the judged responses ({where}) can't be read from here")
        return path, {}
    if responses_digest(responses) != _get(manifest, "responses", "digest"):
        notes.append(f"the judged responses ({where}) changed after they were judged")
        return path, {}
    return path, responses


def load_run(path: Path, suite: Suite, *, name: str | None = None) -> RunView:
    """Read a run directory, or a responses or judgments file, as a run of ``suite``.

    Raises :class:`RunError` if it can't be read.
    """
    log = _log_file(path)
    manifest = _manifest(log.parent / "manifest.json")
    notes: list[str] = []
    if _get(manifest, "suite", "digest") not in (None, suite_digest(suite)):
        notes.append("made with a different version of the suite (its digest differs)")
    cases: list[CaseView] = []
    if _is_judgments(log):
        try:
            records = read_judgments(log)
        except LabelsError as exc:
            raise RunError(str(exc)) from exc
        samples = sorted({r.sample for r in records})
        if len(samples) > 1:
            notes.append(f"{len(samples)} samples per case; this report uses sample {samples[0]}")
        found = {r.id: r for r in records if r.sample == samples[0]}
        responses_path, responses = _judged_responses(manifest, notes)
        if responses_path is not None and not responses:
            notes.append("so the HTML report has no response text or failed-check messages")
        for case in suite.cases:
            if (record := found.get(case.id)) is not None:
                response = responses.get(case.id)
                failed = _failed(case, response)
                modes = tuple(record.check_failure_modes)
                cases.append(
                    CaseView(
                        case.id, case.language, record.check_status, modes, failed, response, record
                    )
                )
        ids = set(found)
        kind: RunKind = "judgments"
        judge = _get(manifest, "judge", "spec") or records[0].judge
        named_after = responses_path or log
        # The judged file's own run manifest names the model that wrote it.
        model = _get(_manifest(named_after.parent / "manifest.json"), "provider", "spec")
    else:
        try:
            responses = load_responses(log)
        except ResponsesError as exc:
            raise RunError(str(exc)) from exc
        for case in suite.cases:
            if case.id in responses:
                report = check_case(case, responses[case.id])
                failed = tuple(f"{r.check}: {r.message}" for r in report.failures)
                modes = tuple(report.failure_modes)
                cases.append(
                    CaseView(
                        case.id, case.language, report.status, modes, failed, responses[case.id]
                    )
                )
        ids = set(responses)
        kind = "responses"
        judge = None
        named_after = log
        model = _get(manifest, "provider", "spec")
    return RunView(
        name=name or model or default_name(named_after),
        source=path.as_posix(),
        kind=kind,
        cases=cases,
        missing=[c.id for c in suite.cases if c.id not in ids],
        unknown=sorted(ids - {c.id for c in suite.cases}),
        model=model,
        judge=judge,
        judge_prompt_version=_get(manifest, "judge_prompt_version"),
        notes=notes,
    )


def load_runs(paths: Sequence[Path], suite: Suite, names: Sequence[str] = ()) -> list[RunView]:
    """:func:`load_run` for each path; ``names`` name the first runs, in order.

    A name used twice gets `` (2)``, `` (3)``... so every run has its own column.
    """
    if len(names) > len(paths):
        raise RunError(f"{len(names)} names for {len(paths)} runs")
    runs = [
        load_run(p, suite, name=names[i] if i < len(names) else None) for i, p in enumerate(paths)
    ]
    seen: Counter[str] = Counter()
    for run in runs:
        seen[run.name] += 1
        if seen[run.name] > 1:
            run.name = f"{run.name} ({seen[run.name]})"
    return runs


# ---------------------------------------------------------------- diffs


@dataclass(frozen=True)
class CaseChange:
    """How one case's verdict changed from the base run to the new one."""

    id: str
    language: str
    status: tuple[Status, Status]
    scores: dict[Dimension, tuple[int, int]]
    """Dimensions scored in both runs whose score moved by at least ``min_change``."""
    added: tuple[FailureMode, ...]
    removed: tuple[FailureMode, ...]

    @property
    def worse(self) -> bool:
        """Fails its checks now and didn't, lost points, or gained a failure mode."""
        base, new = self.status
        failed_now = new is Status.FAIL and base is not Status.FAIL
        dropped = any(n < b for b, n in self.scores.values())
        return failed_now or dropped or bool(self.added)

    @property
    def better(self) -> bool:
        base, new = self.status
        fixed = base is Status.FAIL and new is not Status.FAIL
        rose = any(n > b for b, n in self.scores.values())
        return fixed or rose or bool(self.removed)


@dataclass
class RunDiff:
    """Two runs of a suite, case by case, over the cases both have."""

    base: RunView
    new: RunView
    min_change: int
    shared: list[str]
    """Case ids both runs have, in suite order."""
    changes: list[CaseChange]
    """Shared cases whose verdict changed (checks, a score, or a failure mode)."""
    lost: list[str]
    """Cases the base run has and the new one doesn't."""
    gained: list[str]
    text_changed: int | None
    """Shared cases whose response text differs; ``None`` if either run lacks the text."""
    unjudged: list[str]
    """In two judge runs, shared cases judged in one and not the other (a judge
    error): their scores and judge tags are not compared."""

    @property
    def worse(self) -> list[CaseChange]:
        return [c for c in self.changes if c.worse]

    @property
    def better(self) -> list[CaseChange]:
        return [c for c in self.changes if c.better]

    def _pairs(self) -> list[tuple[CaseView, CaseView]]:
        base, new = self.base, self.new
        return [(b, n) for i in self.shared if (b := base.get(i)) and (n := new.get(i))]

    def pass_rates(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Each run's (passed, graded) over the shared cases."""
        pairs = self._pairs()
        return _pass_rate([b for b, _ in pairs]), _pass_rate([n for _, n in pairs])

    def mean_scores(self) -> dict[Dimension, tuple[float, float, int]]:
        """Per dimension scored in both runs: (base mean, new mean, cases) over those cases."""
        out: dict[Dimension, tuple[float, float, int]] = {}
        for dim in Dimension:
            both = [
                (b.scores[dim], n.scores[dim])
                for b, n in self._pairs()
                if dim in b.scores and dim in n.scores
            ]
            if both:
                k = len(both)
                out[dim] = (sum(b for b, _ in both) / k, sum(n for _, n in both) / k, k)
        return out

    def mode_counts(self) -> dict[FailureMode, tuple[int, int]]:
        """Per failure mode either run counts on a shared case: (base, new)."""
        pairs = self._pairs()
        base = Counter(m for b, _ in pairs for m in b.failure_modes)
        new = Counter(m for _, n in pairs for m in n.failure_modes)
        return {m: (base[m], new[m]) for m in FailureMode if base[m] or new[m]}

    def to_dict(self) -> dict[str, Any]:
        (bp, bg), (np_, ng) = self.pass_rates()
        return {
            "base": self.base.name,
            "new": self.new.name,
            "min_change": self.min_change,
            "shared": len(self.shared),
            "lost": self.lost,
            "gained": self.gained,
            "unjudged": self.unjudged,
            "text_changed": self.text_changed,
            "pass_rate": {"base": [bp, bg], "new": [np_, ng]},
            "mean_scores": {
                d.value: {"base": round(b, 3), "new": round(n, 3), "cases": k}
                for d, (b, n, k) in self.mean_scores().items()
            },
            "failure_modes": {
                m.value: {"base": b, "new": n} for m, (b, n) in self.mode_counts().items()
            },
            "worse": [c.id for c in self.worse],
            "better": [c.id for c in self.better],
            "changes": [
                {
                    "id": c.id,
                    "language": c.language,
                    "checks": [s.value for s in c.status],
                    "scores": {d.value: list(v) for d, v in c.scores.items()},
                    "added": [m.value for m in c.added],
                    "removed": [m.value for m in c.removed],
                }
                for c in self.changes
            ],
        }


def diff_runs(base: RunView, new: RunView, *, min_change: int = 1) -> RunDiff:
    """Case by case, what changed from ``base`` to ``new``.

    Score moves smaller than ``min_change`` points are ignored. Judge scores
    and the judge's failure modes are compared only where both runs judged
    the case; otherwise only the checks' modes are.
    """
    if min_change < 1:
        raise ValueError("min_change must be at least 1")
    new_ids = {c.id for c in new.cases}
    base_ids = {c.id for c in base.cases}
    shared = [c.id for c in base.cases if c.id in new_ids]
    changes: list[CaseChange] = []
    unjudged: list[str] = []
    texts: list[bool] = []
    both_judged_runs = base.kind == new.kind == "judgments"
    for case_id in shared:
        b, n = base.get(case_id), new.get(case_id)
        assert b is not None
        assert n is not None
        if b.response is not None and n.response is not None:
            texts.append(b.response != n.response)
        if b.judged and n.judged:
            b_modes, n_modes = b.failure_modes, n.failure_modes
            scores = {
                d: (b.scores[d], n.scores[d])
                for d in Dimension
                if d in b.scores and d in n.scores and abs(n.scores[d] - b.scores[d]) >= min_change
            }
        else:
            if both_judged_runs and (b.judged or n.judged):
                unjudged.append(case_id)
            b_modes, n_modes = list(b.check_modes), list(n.check_modes)
            scores = {}
        added = tuple(m for m in n_modes if m not in b_modes)
        removed = tuple(m for m in b_modes if m not in n_modes)
        if b.status != n.status or scores or added or removed:
            changes.append(
                CaseChange(case_id, b.language, (b.status, n.status), scores, added, removed)
            )
    return RunDiff(
        base=base,
        new=new,
        min_change=min_change,
        shared=shared,
        changes=changes,
        lost=[c.id for c in base.cases if c.id not in new_ids],
        gained=[c.id for c in new.cases if c.id not in base_ids],
        text_changed=sum(texts) if len(texts) == len(shared) else None,
        unjudged=unjudged,
    )


# ---------------------------------------------------------------- report tables


@dataclass(frozen=True)
class Cell:
    text: str
    css: str = ""
    """A class for the HTML report: pass, fail, skip, worse, better, mixed, num."""
    anchor: str = ""
    """A case id: the HTML report links the cell to that case's details."""


@dataclass
class Table:
    heading: str
    header: list[str]
    rows: list[list[Cell]]
    level: int = 2
    intro: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class Report:
    title: str
    meta: list[str]
    tables: list[Table]
    suite: Suite
    runs: list[RunView]


def _pct(passed: int, graded: int) -> str:
    return f"{passed / graded:.1%} ({passed}/{graded})" if graded else "-"


def _mean(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _num(n: int) -> Cell:
    return Cell(str(n), "num")


def _runs_table(runs: Sequence[RunView]) -> Table:
    # A column only when some run fills it: plain JSONL files name no model.
    show_model, show_judge = any(r.model for r in runs), any(r.judge for r in runs)
    rows = []
    for r in runs:
        row = [Cell(r.name)]
        if show_model:
            row.append(Cell(r.model or "-"))
        if show_judge:
            version = f" (prompt v{r.judge_prompt_version})" if r.judge_prompt_version else ""
            row.append(Cell(f"{r.judge}{version}" if r.judge else "-"))
        rows.append([*row, Cell(r.source), _num(len(r.cases)), _num(len(r.missing))])
    notes = [f"{r.name}: {note}" for r in runs for note in r.notes]
    notes += [
        f"{r.name}: no case in the suite for {', '.join(r.unknown)}" for r in runs if r.unknown
    ]
    header = ["Run", *["Model"] * show_model, *["Judge"] * show_judge, "Source", "Cases", "Missing"]
    return Table("Runs", header, rows, notes=notes)


def _overview(runs: Sequence[RunView], languages: list[str]) -> Table:
    judged = any(r.kind == "judgments" for r in runs)
    header = ["Run", "Checks passed", *languages]
    if judged:
        header += ["Judged", *DIM_SHORT.values()]
    rows = []
    for r in runs:
        row = [Cell(r.name), Cell(_pct(*r.pass_rate()), "num")]
        row += [Cell(_pct(*r.pass_rate(lang)), "num") for lang in languages]
        summary = r.summary()
        if summary is not None:
            means = summary.mean_scores()
            row += [Cell(f"{len(summary.judged)}/{len(r.cases)}", "num")]
            row += [Cell(_mean(means[d]), "num") for d in DIM_SHORT]
        elif judged:
            row += [Cell("-")] * (1 + len(DIM_SHORT))
        rows.append(row)
    legend = ", ".join(f"{s} = {d.value}" for d, s in DIM_SHORT.items())
    intro = (
        "Checks passed: cases that passed every deterministic check, of those graded "
        "(a case where no check applied is not graded)."
    )
    if judged:
        intro += f" Judge scores are means on a 1-5 scale: {legend}."
    return Table("Overview", header, rows, intro=intro)


def _by_language(runs: Sequence[RunView], languages: list[str]) -> Table:
    rows = []
    for r in runs:
        summary = r.summary()
        if summary is None:
            continue
        for lang in languages:
            means = summary.mean_scores(lang)
            judged = sum(1 for c in summary.judged if c.language == lang)
            rows.append(
                [Cell(r.name), Cell(lang), _num(judged)]
                + [Cell(_mean(means[d]), "num") for d in DIM_SHORT]
            )
    return Table(
        "Judge scores by language", ["Run", "Language", "Judged", *DIM_SHORT.values()], rows
    )


def _distribution(runs: Sequence[RunView]) -> Table:
    rows = []
    for r in runs:
        summary = r.summary()
        if summary is None:
            continue
        means = summary.mean_scores()
        for dim, counts in summary.score_counts().items():
            if sum(counts.values()):
                rows.append(
                    [Cell(r.name), Cell(dim.value)]
                    + [_num(counts[s]) for s in range(1, 6)]
                    + [Cell(_mean(means[dim]), "num")]
                )
    return Table(
        "Score distribution",
        ["Run", "Dimension", "1", "2", "3", "4", "5", "Mean"],
        rows,
        intro="How many judged cases got each score.",
    )


def _modes(runs: Sequence[RunView]) -> list[Table]:
    counts = [r.mode_counts() for r in runs]
    rows = [
        [Cell(m.value), Cell("checks" if m in CHECKABLE else "judge")]
        + [_num(c[m]) for c in counts]
        for m in FailureMode
    ]
    tables = [
        Table(
            "Failure modes",
            ["Failure mode", "Decided by", *(r.name for r in runs)],
            rows,
            intro="Cases tagged with each mode. The checks decide the three checkable modes; "
            "the judge decides the rest, so a run without a judge has only the checkable ones.",
        )
    ]
    judged = [r for r in runs if r.kind == "judgments"]
    if judged:
        over = [r.judge_only_counts() for r in judged]
        tables.append(
            Table(
                "Judge tags that don't count",
                ["Failure mode", *(r.name for r in judged)],
                [
                    [Cell(m.value)] + [_num(c[m]) for c in over]
                    for m in FailureMode
                    if m in CHECKABLE
                ],
                level=3,
                intro="Checkable modes the judge named on cases where no check raised them. "
                "Only the checks decide these modes, so the counts above leave these tags out.",
            )
        )
    return tables


def _signed(value: float) -> str:
    return f"{value:+.2f}" if value else "0"


def _cases(n: int) -> str:
    return f"{n:+d} case{'' if abs(n) == 1 else 's'}" if n else "0"


_DIRECTION = {(True, False): "worse", (False, True): "better", (True, True): "mixed"}
"""CSS class of a changed case by (worse, better); neither (pass -> skip) gets none."""


def diff_tables(diff: RunDiff) -> list[Table]:
    (bp, bg), (np_, ng) = diff.pass_rates()
    rows = [
        [
            Cell("Checks passed"),
            Cell(_pct(bp, bg), "num"),
            Cell(_pct(np_, ng), "num"),
            Cell(_cases(np_ - bp), "num"),
        ]
    ]
    for dim, (b, n, k) in diff.mean_scores().items():
        rows.append(
            [
                Cell(f"{dim.value} (mean of {k})"),
                Cell(f"{b:.2f}", "num"),
                Cell(f"{n:.2f}", "num"),
                Cell(_signed(n - b), "num"),
            ]
        )
    for mode, (b, n) in diff.mode_counts().items():
        rows.append(
            [
                Cell(f"{mode.value} (cases)"),
                _num(b),
                _num(n),
                Cell(f"{n - b:+d}" if n != b else "0", "num"),
            ]
        )
    worse, better = len(diff.worse), len(diff.better)
    intro = f"{len(diff.shared)} cases in both runs: {worse} worse, {better} better"
    if diff.text_changed is not None:
        intro += f"; the response text differs on {diff.text_changed}"
    intro += "."
    notes = []
    if diff.lost:
        notes.append(f"In {diff.base.name} only: {', '.join(diff.lost)}")
    if diff.gained:
        notes.append(f"In {diff.new.name} only: {', '.join(diff.gained)}")
    if diff.unjudged:
        unjudged = ", ".join(diff.unjudged)
        notes.append(f"Judged in one run only (scores and judge tags not compared): {unjudged}")
    if diff.base.kind != diff.new.kind:
        notes.append("Only one of the two runs was judged, so only the checks are compared.")
    summary = Table(
        f"Changes: {diff.base.name} → {diff.new.name}",
        ["", diff.base.name, diff.new.name, "Change"],
        rows,
        intro=intro,
        notes=notes,
    )
    judged = any(c.scores for c in diff.changes) or bool(diff.mean_scores())
    header = (
        ["Case", "Language", "Checks"]
        + (list(DIM_SHORT.values()) if judged else [])
        + ["Failure modes"]
    )
    case_rows = []
    for c in diff.changes:
        was, now = c.status
        row = [
            Cell(c.id, _DIRECTION.get((c.worse, c.better), ""), anchor=c.id),
            Cell(c.language),
            Cell(f"{was.value} → {now.value}" if was != now else now.value, now.value),
        ]
        if judged:
            row += [
                Cell(f"{c.scores[d][0]} → {c.scores[d][1]}" if d in c.scores else "", "num")
                for d in DIM_SHORT
            ]
        modes = [f"+{m.value}" for m in c.added] + [f"-{m.value}" for m in c.removed]
        row.append(Cell(" ".join(modes)))
        case_rows.append(row)
    why = "Worse: fails its checks now and didn't, a judge score fell, or a failure mode appeared."
    if diff.min_change > 1:
        why += f" Score moves under {diff.min_change} points are ignored."
    cases = Table(
        "Cases that changed", header, case_rows, level=3, intro=why if case_rows else "None."
    )
    return [summary, cases]


def build_report(
    suite: Suite, runs: Sequence[RunView], *, min_change: int = 1, now: datetime | None = None
) -> Report:
    """The report's tables. With several runs, each later run is diffed against the first."""
    if not runs:
        raise ValueError("a report needs at least one run")
    langs = Counter[str](c.language for c in suite.cases)
    languages = sorted(langs)
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    meta = [
        f"Suite {suite.name} v{suite.version}: {len(suite.cases)} cases "
        f"({', '.join(f'{n} {lang}' for lang, n in sorted(langs.items()))}), "
        f"digest {suite_digest(suite)}.",
        f"Made with llm-evalkit {__version__} on {stamp}.",
    ]
    if len(runs) > 1:
        meta.append(f"Changes are against the first run, {runs[0].name}.")
    tables = [_runs_table(runs), _overview(runs, languages)]
    if any(r.kind == "judgments" for r in runs):
        tables += [_by_language(runs, languages), _distribution(runs)]
    tables += _modes(runs)
    for new in runs[1:]:
        tables += diff_tables(diff_runs(runs[0], new, min_change=min_change))
    return Report(f"Evaluation report: {suite.name}", meta, tables, suite, list(runs))


# ---------------------------------------------------------------- rendering


def _md(text: object) -> str:
    """Text for a Markdown table cell: no line breaks, no column breaks, no HTML."""
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("<", "&lt;")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _environment() -> Any:
    import jinja2  # here, not at the top: `evalkit --help` shouldn't pay for it

    env = jinja2.Environment(
        loader=jinja2.PackageLoader("evalkit", "templates"),
        autoescape=jinja2.select_autoescape(["html"]),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["md"] = _md
    return env


def render_markdown(report: Report) -> str:
    return str(_environment().get_template("report.md").render(report=report))


def render_html(report: Report) -> str:
    """A single self-contained HTML file: inline CSS, no scripts, nothing fetched."""
    return str(
        _environment()
        .get_template("report.html")
        .render(report=report, dims=list(DIM_SHORT.items()))
    )
