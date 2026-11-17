"""How far to trust the judge: agreement with a reference, and with itself.

* :func:`compare` sets the judge's scores and failure modes next to a
  reference rater's (human labels from a CSV, or another judge run) and
  reports, per dimension, exact and within-one agreement, Cohen's kappa
  (unweighted, linear and quadratic), Spearman's rho with bootstrap
  intervals, and the confusion matrix; per failure mode, how often the two
  agree that it applies.
* :func:`consistency` summarises a judge that rated each response several
  times (``evalkit consistency``): how much its scores move between samples.

Label files are CSV, one row per response, so they can be filled in a
spreadsheet. ``evalkit labels`` writes an empty sheet with the case and the
response next to each row. The columns:

``id``
    The case id (required).
``instruction_following``, ``reasoning_quality``, ``faithfulness``, ``helpfulness``
    A whole number 1-5, or blank / ``n/a`` for "not rated". Any subset of
    the four columns may be present.
``failure_modes``
    The failure modes that apply, separated by ``|``, ``;`` or ``,``, or
    ``none`` when no mode applies. All or nothing: either every labelled row
    fills it in (``none`` included), or every row leaves it blank and only
    the scores are compared. A mix is an error, because a blank beside a
    tagged row could mean "none" or "not done yet".
``language``
    Optional; used for the per-language split when the other side has none.

Every other column (``notes``, ``prompt``, ``response``, ``labeler``...) is
ignored; any that evalkit doesn't know are listed in a note, so a misspelt
label column shows up. Column names are matched ignoring case, spaces and
hyphens. A row whose label cells are all blank counts as not labelled yet
and is skipped, so a half-filled sheet can be compared as it stands; when
that leaves rows the judge rated out of the failure-mode comparison,
:func:`compare` warns (a blank row can't also mean "no mode applies"). The
delimiter is read from the header, since a spreadsheet in an Indonesian
locale saves CSV with ``;``. :func:`load_ratings` tells a judgments file
from a sheet by its first character (``{``), not its name.
"""

from __future__ import annotations

import csv
import io
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from evalkit.files import NotUtf8Error, read_utf8
from evalkit.judge.core import JudgeRecord
from evalkit.schema import Dimension
from evalkit.stats import SCALE, bootstrap_ci, cohen_kappa, confusion_matrix, spearman
from evalkit.taxonomy import CHECKABLE, FailureMode

__all__ = [
    "LABEL_COLUMNS",
    "AgreementReport",
    "ConsistencyReport",
    "DimensionAgreement",
    "LabelsError",
    "ModeAgreement",
    "NoSuchSampleError",
    "Rating",
    "Ratings",
    "compare",
    "consistency",
    "load_judgments",
    "load_labels",
    "load_ratings",
    "write_label_sheet",
]

LABEL_COLUMNS: tuple[str, ...] = (*(d.value for d in Dimension), "failure_modes")
_BLANK = {"", "n/a", "na", "-"}
_MODE_SPLIT = re.compile(r"[;,|\n]+")
_MODE_NAMES = {m.value: m for m in FailureMode}


class LabelsError(ValueError):
    """A labels or judgments file can't be read. The message is user-facing."""


class NoSuchSampleError(LabelsError):
    """A judgments file doesn't hold the sample asked for."""


class Rating(BaseModel):
    """One rater's verdict on one response."""

    model_config = ConfigDict(frozen=True)

    id: str
    scores: dict[Dimension, int]
    failure_modes: frozenset[FailureMode] | None = None
    """``None`` when the rater didn't label failure modes at all."""
    language: str | None = None


@dataclass
class Ratings:
    """Everything read from one file: the ratings by case id, and what was left out."""

    source: str
    ratings: dict[str, Rating]
    skipped: list[str] = field(default_factory=list)
    """Ids with no usable rating: unlabelled rows, or judge errors."""
    note: str | None = None
    kind: Literal["labels", "judgments"] = "judgments"
    """What the file is, so messages can say "not labelled yet" or "judge error"."""


def _mode(value: str, where: str) -> FailureMode:
    key = re.sub(r"[\s-]+", "_", value.strip().lower())
    if key not in _MODE_NAMES:
        raise LabelsError(
            f"{where}: unknown failure mode {value.strip()!r}; use: {', '.join(_MODE_NAMES)}"
        )
    return _MODE_NAMES[key]


def _delimiter(header: str) -> str:
    counts = {d: header.count(d) for d in (",", ";", "\t")}
    return max(counts, key=lambda d: counts[d]) if any(counts.values()) else ","


def _column(name: str) -> str:
    """ "Instruction Following" and "instruction-following" name the same column."""
    return re.sub(r"[\s-]+", "_", name.strip().lower())


def _check_header(path: Path, raw: list[str]) -> list[str]:
    header = [_column(h) for h in raw]
    if "id" not in header:
        raise LabelsError(f"{path}: the header has no 'id' column (found: {', '.join(raw)})")
    repeated = sorted({h for h in header if h and header.count(h) > 1})
    if repeated:
        raise LabelsError(f"{path}: the header repeats {', '.join(repeated)}")
    if not any(name in LABEL_COLUMNS for name in header):
        raise LabelsError(f"{path}: no label columns; expected some of: {', '.join(LABEL_COLUMNS)}")
    return header


def _unread_note(header: list[str], raw: list[str]) -> str | None:
    """Columns that are neither read nor written by `evalkit labels`, so that a
    misspelt label column ("helpfullness") shows up instead of vanishing."""
    expected = {"id", "language", *LABEL_COLUMNS, "notes", "labeler", *SHEET_TEXT_COLUMNS}
    unread = [
        shown.strip()
        for name, shown in zip(header, raw, strict=True)
        if name and name not in expected
    ]
    return f"columns not read: {', '.join(unread)}" if unread else None


def load_labels(path: Path) -> Ratings:
    """Read a labels CSV (see the module docstring for the columns)."""
    try:
        text = read_utf8(path)
    except (OSError, NotUtf8Error) as exc:
        raise LabelsError(f"{path}: {exc}") from exc
    first_line = text.split("\n", 1)[0]
    # strict: an unclosed quote is an error, not a cell that swallows every row after it.
    reader = csv.reader(
        io.StringIO(text, newline=""), delimiter=_delimiter(first_line), strict=True
    )
    try:
        raw_header = next(reader)
        header = _check_header(path, raw_header)
    except StopIteration:
        raise LabelsError(f"{path}: the file is empty") from None
    except csv.Error as exc:
        raise LabelsError(f"{path}:1: not valid CSV: {exc}") from exc
    col = {name: header.index(name) for name in header}
    dims = [d for d in Dimension if d.value in col]
    has_modes = "failure_modes" in col

    ratings: dict[str, Rating] = {}
    first_seen: dict[str, int] = {}
    skipped: list[str] = []
    line = reader.line_num + 1  # where the next row starts; a quoted cell can span lines
    try:
        for row in reader:
            row_line, line = line, reader.line_num + 1
            if not any(cell.strip() for cell in row):
                continue
            where = f"{path}:{row_line}"
            if any(cell.strip() for cell in row[len(header) :]):
                raise LabelsError(
                    f"{where}: the row has {len(row)} cells but the header has {len(header)} "
                    "columns; quote a cell that contains the delimiter, or separate failure "
                    "modes with '|'"
                )

            def cell(name: str, row: list[str] = row) -> str:
                i = col[name]
                return row[i].strip() if i < len(row) else ""

            case_id = cell("id")
            if not case_id:
                raise LabelsError(f"{where}: the row has labels but no id")
            if case_id in first_seen:
                raise LabelsError(
                    f"{where}: id {case_id!r} is labelled twice "
                    f"(first on line {first_seen[case_id]})"
                )
            first_seen[case_id] = row_line
            names = [d.value for d in dims] + (["failure_modes"] if has_modes else [])
            raw = {name: cell(name) for name in names}
            if all(value.lower() in _BLANK for value in raw.values()):
                skipped.append(case_id)
                continue
            scores: dict[Dimension, int] = {}
            for dim in dims:
                value = raw[dim.value]
                if value.lower() in _BLANK:
                    continue
                if not re.fullmatch(r"[1-5]", value):
                    raise LabelsError(
                        f"{where}: {dim.value} must be a whole number from 1 to 5 "
                        f"(or blank), not {value!r}"
                    )
                scores[dim] = int(value)
            modes: frozenset[FailureMode] | None = None  # blank: not labelled yet
            listed = raw.get("failure_modes", "")
            if listed.lower() in {"none", "[]"}:
                modes = frozenset()
            elif listed.lower() not in _BLANK:
                parts = [p for p in _MODE_SPLIT.split(listed) if p.strip()]
                modes = frozenset(_mode(p, where) for p in parts)
            language = cell("language") if "language" in col else ""
            ratings[case_id] = Rating(
                id=case_id, scores=scores, failure_modes=modes, language=language or None
            )
    except csv.Error as exc:
        # `line` is where the row being read started, e.g. where an unclosed quote opened.
        raise LabelsError(f"{path}:{line}: not valid CSV: {exc}") from exc
    # Failure modes are labelled on every labelled row or on none. A blank
    # beside a scored row could mean "none applies" or "not done yet", and
    # either guess skews precision, so it is an error rather than a guess.
    blank = [i for i, r in ratings.items() if r.failure_modes is None]
    if has_modes and blank and len(blank) < len(ratings):
        lines = ", ".join(str(first_seen[i]) for i in blank[:5])
        more = f" and {len(blank) - 5} more" if len(blank) > 5 else ""
        raise LabelsError(
            f"{path}: {len(blank)} labelled rows have no failure_modes (blank, n/a or -; lines "
            f"{lines}{more}) while others name modes; write 'none' where no mode applies, or "
            "leave the column blank on every row to compare scores only"
        )
    return Ratings(str(path), ratings, skipped, _unread_note(header, raw_header), kind="labels")


def load_judgments(path: Path, *, sample: int = 0) -> Ratings:
    """Read a ``judgments.jsonl`` from ``evalkit judge`` (or ``evalkit consistency``).

    Judge errors are skipped (and listed). With several samples per case,
    ``sample`` picks one; the note says how many there were.
    """
    try:
        text = read_utf8(path)
    except (OSError, NotUtf8Error) as exc:
        raise LabelsError(f"{path}: {exc}") from exc
    ratings: dict[str, Rating] = {}
    skipped: list[str] = []
    seen: set[tuple[str, int]] = set()
    samples: set[int] = set()
    # Split on "\n" only: str.splitlines() also splits on U+2028 and friends,
    # which JSON writes raw inside strings (a judge's rationale can hold one).
    for n, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            record = JudgeRecord.model_validate_json(line)
        except ValidationError as exc:
            first = exc.errors()[0]
            raise LabelsError(
                f"{path}:{n}: not a judge record ({first['msg']}); "
                "expected a judgments.jsonl written by `evalkit judge`"
            ) from exc
        if (record.id, record.sample) in seen:
            raise LabelsError(
                f"{path}:{n}: case {record.id!r} (sample {record.sample}) appears twice; "
                "compare one model's judgments at a time"
            )
        seen.add((record.id, record.sample))
        samples.add(record.sample)
        if record.sample != sample:
            continue
        if not record.ok:
            skipped.append(record.id)
            continue
        ratings[record.id] = Rating(
            id=record.id,
            scores=record.scores,
            failure_modes=frozenset(record.failure_modes),
            language=record.language,
        )
    if not seen:
        raise LabelsError(f"{path}: no judge records")
    if sample not in samples:
        raise NoSuchSampleError(f"{path}: no sample {sample} (the file has {sorted(samples)})")
    note = f"used sample {sample} of {len(samples)}" if len(samples) > 1 else None
    return Ratings(str(path), ratings, skipped, note)


def load_ratings(path: Path, *, sample: int = 0) -> Ratings:
    """A judgments file if its first non-blank character is ``{``, else a labels sheet.

    By content, not by suffix: a spreadsheet saves tab-separated text as
    ``.txt``, and a judgments file can be renamed to anything.
    """
    try:
        with path.open("rb") as f:
            start = f.read(4096).lstrip(b"\xef\xbb\xbf \t\r\n")
    except OSError as exc:
        raise LabelsError(f"{path}: {exc}") from exc
    if start.startswith(b"{"):
        return load_judgments(path, sample=sample)
    return load_labels(path)


# ---------------------------------------------------------------- comparison


def _r(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def _ci(ci: tuple[float, float] | None) -> list[float] | None:
    return None if ci is None else [round(ci[0], 3), round(ci[1], 3)]


class DimensionAgreement(BaseModel):
    """Judge vs reference on one scored dimension, over the items both scored."""

    n: int
    exact: int
    within_1: int
    kappa: float | None
    """Unweighted: every disagreement counts the same."""
    kappa_linear: float | None
    kappa_quadratic: float | None
    """The headline number: a 2-point miss costs four times a 1-point miss."""
    kappa_quadratic_ci: list[float] | None
    spearman: float | None
    spearman_ci: list[float] | None
    mean_reference: float | None
    mean_judge: float | None
    confusion: list[list[int]]
    """Rows: the reference's score 1-5; columns: the judge's."""


class ModeAgreement(BaseModel):
    """Do the two raters agree that a failure mode applies?"""

    decided_by: str
    """Who decides the mode in evalkit: ``checks`` or ``judge``."""
    n: int
    both: int
    judge_only: int
    reference_only: int
    neither: int
    kappa: float | None
    precision: float | None
    """Of the judge's tags, the share the reference also gave."""
    recall: float | None
    """Of the reference's tags, the share the judge also gave."""


class AgreementReport(BaseModel):
    judge: str
    reference: str
    judge_kind: Literal["labels", "judgments"] = "judgments"
    reference_kind: Literal["labels", "judgments"] = "labels"
    matched: int
    only_judge: list[str]
    only_reference: list[str]
    skipped_judge: list[str]
    skipped_reference: list[str]
    notes: list[str]
    warnings: list[str] = []
    """Ways the numbers cover less than they seem to."""
    dimensions: dict[Dimension, DimensionAgreement]
    by_language: dict[str, dict[Dimension, DimensionAgreement]]
    failure_modes: dict[FailureMode, ModeAgreement]
    resamples: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _dimension(
    pairs: Sequence[tuple[int, int]], *, resamples: int, seed: int
) -> DimensionAgreement:
    ref = [p for p, _ in pairs]
    jud = [q for _, q in pairs]

    def kq(a: Sequence[int], b: Sequence[int]) -> float | None:
        return cohen_kappa(a, b, SCALE, weights="quadratic")

    return DimensionAgreement(
        n=len(pairs),
        exact=sum(p == q for p, q in pairs),
        within_1=sum(abs(p - q) <= 1 for p, q in pairs),
        kappa=_r(cohen_kappa(ref, jud, SCALE)),
        kappa_linear=_r(cohen_kappa(ref, jud, SCALE, weights="linear")),
        kappa_quadratic=_r(kq(ref, jud)),
        kappa_quadratic_ci=_ci(bootstrap_ci(ref, jud, kq, resamples=resamples, seed=seed)),
        spearman=_r(spearman(ref, jud)),
        spearman_ci=_ci(bootstrap_ci(ref, jud, spearman, resamples=resamples, seed=seed)),
        mean_reference=_r(statistics.fmean(ref)) if ref else None,
        mean_judge=_r(statistics.fmean(jud)) if jud else None,
        confusion=confusion_matrix(ref, jud, SCALE),
    )


def _mode_agreement(mode: FailureMode, pairs: Sequence[tuple[bool, bool]]) -> ModeAgreement:
    both = sum(r and j for r, j in pairs)
    judge_only = sum(j and not r for r, j in pairs)
    ref_only = sum(r and not j for r, j in pairs)
    return ModeAgreement(
        decided_by="checks" if mode in CHECKABLE else "judge",
        n=len(pairs),
        both=both,
        judge_only=judge_only,
        reference_only=ref_only,
        neither=len(pairs) - both - judge_only - ref_only,
        kappa=_r(cohen_kappa([r for r, _ in pairs], [j for _, j in pairs], [False, True])),
        precision=_r(both / (both + judge_only)) if both + judge_only else None,
        recall=_r(both / (both + ref_only)) if both + ref_only else None,
    )


def compare(
    judge: Ratings,
    reference: Ratings,
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> AgreementReport:
    """Agreement between ``judge`` and ``reference`` on the cases both rated.

    Statistics for a dimension use the cases where both gave that dimension
    a score; failure modes use the cases where both labelled failure modes.
    Bootstrap intervals are 95% percentile intervals over ``resamples``
    resamples of cases (0 = no intervals). The per-language breakdown takes
    the language from the judge's records, else from the reference.
    """
    common = [i for i in judge.ratings if i in reference.ratings]
    pairs = [(reference.ratings[i], judge.ratings[i]) for i in common]

    def dims_for(
        group: Sequence[tuple[Rating, Rating]], n_resamples: int
    ) -> dict[Dimension, DimensionAgreement]:
        out: dict[Dimension, DimensionAgreement] = {}
        for dim in Dimension:
            scored = [
                (r.scores[dim], j.scores[dim])
                for r, j in group
                if dim in r.scores and dim in j.scores
            ]
            if scored:
                out[dim] = _dimension(scored, resamples=n_resamples, seed=seed)
        return out

    by_lang: dict[str, list[tuple[Rating, Rating]]] = defaultdict(list)
    for r, j in pairs:
        lang = j.language or r.language
        if lang:
            by_lang[lang].append((r, j))

    modes: dict[FailureMode, ModeAgreement] = {}
    labelled = [
        (r.failure_modes, j.failure_modes)
        for r, j in pairs
        if r.failure_modes is not None and j.failure_modes is not None
    ]
    if labelled:
        for mode in FailureMode:
            modes[mode] = _mode_agreement(mode, [(mode in r, mode in j) for r, j in labelled])

    notes = [f"{side.source}: {side.note}" for side in (judge, reference) if side.note]
    warnings: list[str] = []
    unlabelled = [i for i in reference.skipped if i in judge.ratings]
    if labelled and reference.kind == "labels" and unlabelled:
        # A wholly blank row is "not labelled yet": it can't also mean "clean".
        warnings.append(
            f"failure modes are compared on {len(labelled)} cases; {len(unlabelled)} rows the "
            "judge rated have nothing filled in, so its tags on them don't count. If those "
            "rows are clean, write 'none' in their failure_modes."
        )
    return AgreementReport(
        judge=judge.source,
        reference=reference.source,
        matched=len(common),
        # An id the other side skipped (unlabelled, or a judge error) is listed there instead.
        only_judge=sorted(set(judge.ratings) - set(reference.ratings) - set(reference.skipped)),
        only_reference=sorted(set(reference.ratings) - set(judge.ratings) - set(judge.skipped)),
        skipped_judge=judge.skipped,
        skipped_reference=reference.skipped,
        judge_kind=judge.kind,
        reference_kind=reference.kind,
        notes=notes,
        warnings=warnings,
        dimensions=dims_for(pairs, resamples),
        by_language={lang: dims_for(group, 0) for lang, group in sorted(by_lang.items())},
        failure_modes=modes,
        resamples=resamples,
        seed=seed,
    )


# ---------------------------------------------------------------- self-consistency


class DimensionConsistency(BaseModel):
    """How one dimension's score moves across samples of the same judgment."""

    cases: int
    """Cases with at least two scored samples."""
    all_same: int
    """Cases where every sample gave the same score."""
    mode_share: float | None
    """Share of all samples that equal their case's most common score."""
    mean_sd: float | None
    """Mean, over cases, of the standard deviation of that case's samples."""
    mean_range: float | None
    max_range: int | None
    range_2_plus: int
    """Cases whose samples are two or more points apart."""


class ModeConsistency(BaseModel):
    always: int
    sometimes: int
    """Tagged in some samples of a case but not all: an unstable tag."""
    never: int


class ConsistencyReport(BaseModel):
    samples: int
    """Samples requested per case."""
    cases: int
    errors: int
    """Samples that ended without a usable judgment."""
    dimensions: dict[Dimension, DimensionConsistency]
    failure_modes: dict[FailureMode, ModeConsistency]
    """The judge's own tags for the judge-owned modes."""
    unstable: dict[str, dict[Dimension, list[int]]]
    """Per case, the dimensions whose samples disagree, with the scores in sample order."""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def consistency(records: Sequence[JudgeRecord]) -> ConsistencyReport:
    """Summarise records that judge each case several times (one per ``sample``).

    Two answers to the same case id are not told apart: pass one model's
    judgments at a time.
    """
    by_case: dict[str, list[JudgeRecord]] = defaultdict(list)
    for r in records:
        by_case[r.id].append(r)
    samples = max((r.sample for r in records), default=-1) + 1
    dims: dict[Dimension, DimensionConsistency] = {}
    unstable: dict[str, dict[Dimension, list[int]]] = defaultdict(dict)
    for dim in Dimension:
        per_case: list[list[int]] = []
        for case_id, group in by_case.items():
            ordered = sorted((r for r in group if r.ok and dim in r.scores), key=lambda r: r.sample)
            scores = [r.scores[dim] for r in ordered]
            if len(scores) < 2:
                continue
            per_case.append(scores)
            if len(set(scores)) > 1:
                unstable[case_id][dim] = scores
        ranges = [max(s) - min(s) for s in per_case]
        total = sum(len(s) for s in per_case)
        dims[dim] = DimensionConsistency(
            cases=len(per_case),
            all_same=sum(r == 0 for r in ranges),
            mode_share=_r(sum(Counter(s).most_common(1)[0][1] for s in per_case) / total)
            if total
            else None,
            mean_sd=_r(statistics.fmean(statistics.stdev(s) for s in per_case))
            if per_case
            else None,
            mean_range=_r(statistics.fmean(ranges)) if ranges else None,
            max_range=max(ranges) if ranges else None,
            range_2_plus=sum(r >= 2 for r in ranges),
        )
    modes: dict[FailureMode, ModeConsistency] = {}
    for mode in (m for m in FailureMode if m not in CHECKABLE):
        counts: Counter[str] = Counter()
        for group in by_case.values():
            ok = [r for r in group if r.ok]
            if len(ok) < 2:
                continue
            tagged = sum(mode in r.judge_failure_modes for r in ok)
            counts["always" if tagged == len(ok) else "never" if tagged == 0 else "sometimes"] += 1
        modes[mode] = ModeConsistency(**{k: counts[k] for k in ("always", "sometimes", "never")})
    return ConsistencyReport(
        samples=samples,
        cases=len(by_case),
        errors=sum(not r.ok for r in records),
        dimensions=dims,
        failure_modes=modes,
        unstable=dict(sorted(unstable.items())),
    )


# ---------------------------------------------------------------- label sheets


SHEET_TEXT_COLUMNS: tuple[str, ...] = (
    "prompt",
    "response",
    "context",
    "criteria",
    "reference_answer",
    "system",
)
"""What the judge saw, written beside each row of a label sheet (and ignored on reading)."""


def _as_text(value: str) -> str:
    """A spreadsheet reads a cell that starts with = + - @ as a formula (a
    bulleted answer would show as #NAME?); a leading ' keeps it text."""
    return "'" + value if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def write_label_sheet(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    *,
    scored: Callable[[str], Sequence[Dimension]],
) -> int:
    """Write an empty labels CSV with what the judge saw beside each row.

    ``rows`` are ``{"id", "language", "prompt", "response"}`` plus, where the
    case has them, ``context``, ``criteria``, ``reference_answer`` and
    ``system``. ``scored(id)`` gives the dimensions the case's rubric scores;
    the others are pre-filled with ``n/a``. Written as UTF-8 with a
    byte-order mark, which is what makes Excel read Indonesian text correctly.
    """
    columns = ["id", "language", *LABEL_COLUMNS, "notes", *SHEET_TEXT_COLUMNS]
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        dims = set(scored(row["id"]))
        writer.writerow(
            [
                row["id"],
                row["language"],
                *("" if d in dims else "n/a" for d in Dimension),
                "",
                "",
                *(_as_text(row.get(name) or "") for name in SHEET_TEXT_COLUMNS),
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8-sig", newline="")
    return len(rows)
