"""Read the judge's reply: find the JSON, repair its syntax, validate its structure.

One rule decides what is fixed here and what goes back to the judge:
**syntax is repaired locally, structure is re-asked.**

Repaired locally (free, deterministic, and recorded in ``repairs``):

* JSON wrapped in a Markdown code fence, or in prose ("Here is my evaluation: {...}");
* Python-style objects: single quotes, ``True`` / ``None``, a trailing comma
  (read with :func:`ast.literal_eval`, which evaluates literals only, never code);
* a score written as ``"4"`` or ``4.0``;
* a failure mode written as ``"Constraint Violation"`` or ``"over-refusal"``.

Re-asked (the judge must decide): a missing key or score, a score outside
1-5 or between two whole numbers (``4.5``), an unknown failure mode, a
nested shape such as ``{"score": 4, "reason": ...}`` in place of a number,
a key given twice, a reply over 30,000 characters. Guessing at those would
put words in the judge's mouth.

When a reply holds more than one complete judgment, the first one counts and
a note in ``repairs`` says how many there were.

No reply raises anything but :class:`JudgmentError`: whatever goes wrong
while reading one (deep nesting, a 5,000-digit number, a literal Python
can't evaluate) becomes a problem to re-ask about.
"""

from __future__ import annotations

import ast
import json
import re
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from evalkit.schema import Dimension
from evalkit.taxonomy import FailureMode
from evalkit.text import scrub_surrogates

__all__ = ["JudgmentError", "ParsedJudgment", "parse_judgment"]

_FENCE_RE = re.compile(r"```[ \t]*[A-Za-z]*[ \t]*\r?\n(.*?)```", re.DOTALL)
_INT_TEXT_RE = re.compile(r"\d{1,3}")
_MODE_NAMES = {m.value: m for m in FailureMode}
_KEYS = ("rationale", "scores", "failure_modes")
_SHOWN_CHARS = 60
_MAX_REPLY_CHARS = 30_000
"""A judgment is a few hundred characters; a longer reply is a runaway, not a verdict."""
_OBJECT_START_RE = re.compile(r'\{\s*["}]')
"""Only a "{" followed by a quote or "}" can start a JSON object, so LaTeX,
code and prose braces are never tried."""
_MAX_FAILED_STARTS = 1000
"""A safety bound: each failed decode costs time in proportion to its position."""


@dataclass(frozen=True)
class ParsedJudgment:
    rationale: str
    scores: dict[Dimension, int]
    failure_modes: list[FailureMode]
    repairs: list[str] = field(default_factory=list)
    """What had to be fixed locally, in plain words. Empty for a clean reply."""


class JudgmentError(ValueError):
    """The reply can't be used. ``problems`` go back to the judge verbatim."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


class _DuplicateKey(ValueError):
    pass


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKey(key)
        out[key] = value
    return out


_DECODER = json.JSONDecoder(object_pairs_hook=_unique_pairs)


def _decode(text: str) -> dict[str, Any] | str:
    """A JSON object, or why ``text`` is not one."""
    try:
        data = _DECODER.decode(text)
    except _DuplicateKey as exc:
        return _duplicate(exc)
    except json.JSONDecodeError as exc:
        return f"not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
    except RecursionError:
        return "not valid JSON: nested too deeply"
    except ValueError:  # e.g. a number with more digits than Python converts
        return "not valid JSON: a number is too long"
    if not isinstance(data, dict):
        return f"the reply must be a JSON object, not a JSON {type(data).__name__}"
    return data


def _duplicate(exc: _DuplicateKey) -> str:
    return f"the key {_show(exc.args[0])} appears twice; give each key once"


def _has_duplicate_keys(node: ast.AST) -> bool:
    if not isinstance(node, ast.Dict):
        return False
    keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
    try:
        return len(set(keys)) != len(keys)
    except TypeError:  # an unhashable constant can't be a dict key anyway
        return True


def _python_literal(text: str) -> dict[str, Any] | None:
    """``text`` as a Python dict literal (no code runs), or None."""
    try:
        tree = ast.parse(text, mode="eval")
        if any(_has_duplicate_keys(node) for node in ast.walk(tree)):
            return None
        data = ast.literal_eval(tree)
    except Exception:  # any failure (an OverflowError, say) just means "not a literal"
        return None
    return data if isinstance(data, dict) else None


def _object(text: str) -> tuple[dict[str, Any], list[str]] | str:
    """``text`` as a JSON object, else as a Python-style one; or why neither works."""
    data = _decode(text)
    if isinstance(data, dict):
        return data, []
    literal = _python_literal(text)
    if literal is not None:
        return literal, ["read a Python-style object"]
    return data


def _embedded(
    text: str,
) -> tuple[list[tuple[int, dict[str, Any]]], str | None, str | None]:
    """The top-level JSON objects in ``text`` as ``(position, object)``, plus the
    problem with the first object that repeats a key and with the first one
    that doesn't parse, if any.

    Scanning resumes after each decoded object, so an object nested inside
    another is never a candidate of its own.
    """
    found: list[tuple[int, dict[str, Any]]] = []
    duplicate: str | None = None
    broken: str | None = None
    failures = 0
    match = _OBJECT_START_RE.search(text)
    while match is not None and failures < _MAX_FAILED_STARTS:
        pos = match.start()
        try:
            data, end = _DECODER.raw_decode(text, pos)
        except _DuplicateKey as exc:
            duplicate = duplicate or _duplicate(exc)
        except json.JSONDecodeError as exc:
            broken = broken or f"not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
        except (ValueError, RecursionError):
            broken = broken or "not valid JSON"
        else:
            if isinstance(data, dict):
                found.append((pos, data))
            match = _OBJECT_START_RE.search(text, end)
            continue
        failures += 1
        match = _OBJECT_START_RE.search(text, pos + 1)
    return found, duplicate, broken


def _rank(data: dict[str, Any]) -> int:
    return sum(k in data for k in _KEYS)


def _extract(text: str) -> tuple[dict[str, Any], list[str]]:
    """The judge's object and the syntax repairs it needed.

    The whole reply, or a reply that is one code fence, is read directly
    (as JSON, else as a Python literal). Otherwise the candidates are the
    top-level JSON objects in the reply, each fenced block that isn't JSON
    read as a Python literal, and, with no JSON object at all, the span from
    the first ``{`` to the last ``}`` read as one. The first candidate with
    all three expected keys is the judgment. Without one, the candidate with
    the most keys is checked, so the problems name what is missing; an
    object with none of them is never used.
    """
    stripped = text.strip()
    if len(stripped) > _MAX_REPLY_CHARS:
        raise JudgmentError(
            [f"the reply is {len(stripped):,} characters long; reply with only the JSON object"]
        )
    found = _object(stripped)
    if not isinstance(found, str):
        return found
    error = found

    fences = list(_FENCE_RE.finditer(stripped))
    if len(fences) == 1 and fences[0].group(0) == stripped:
        found = _object(fences[0].group(1).strip())
        if not isinstance(found, str):
            return found[0], ["removed a code fence", *found[1]]
        error = found

    start = stripped.find("{")
    if start == -1:
        raise JudgmentError(["the reply contains no JSON object"])

    spans = [fence.span(1) for fence in fences]
    span_starts = [a for a, _ in spans]

    def note_for(pos: int) -> str:
        i = bisect_right(span_starts, pos) - 1
        fenced = i >= 0 and pos < spans[i][1]
        return (
            "took the JSON from a code fence"
            if fenced
            else "took the JSON object out of surrounding text"
        )

    embedded, duplicate, broken = _embedded(stripped)
    candidates = [(pos, data, [note_for(pos)]) for pos, data in embedded if _rank(data)]
    for fence in fences:
        body = fence.group(1).strip()
        if isinstance(_decode(body), str):  # JSON fences are already candidates
            literal = _python_literal(body)
            if literal is not None and _rank(literal):
                notes = ["took the JSON from a code fence", "read a Python-style object"]
                candidates.append((fence.start(1), literal, notes))
    if not embedded:
        literal = _python_literal(stripped[start : stripped.rfind("}") + 1])
        if literal is not None and _rank(literal):
            notes = ["took the object out of surrounding text", "read a Python-style object"]
            candidates.append((start, literal, notes))

    candidates.sort(key=lambda c: c[0])
    complete = [c for c in candidates if _rank(c[1]) == len(_KEYS)]
    if complete:
        _, data, notes = complete[0]
        if len(complete) > 1:
            notes = [*notes, f"the reply has {len(complete)} judgments; used the first"]
        return data, notes
    if candidates:
        _, data, notes = max(candidates, key=lambda c: (_rank(c[1]), -c[0]))
        return data, notes
    if duplicate is not None:
        raise JudgmentError([duplicate])
    if stripped.startswith("{"):  # one attempted object: why it doesn't parse
        raise JudgmentError([error])
    if broken is not None:
        raise JudgmentError([broken])
    if embedded:
        raise JudgmentError(
            [
                'the reply has JSON, but no object with the keys "rationale", "scores" and '
                '"failure_modes"'
            ]
        )
    raise JudgmentError([error])


def _show(value: Any) -> str:
    """A short, printable rendering of a value from the reply, for a problem message."""
    try:
        shown = json.dumps(value, default=str)
    except (TypeError, ValueError, RecursionError):
        shown = f"a {type(value).__name__} that can't be shown"
    return shown if len(shown) <= _SHOWN_CHARS else shown[: _SHOWN_CHARS - 3] + "..."


def _score(value: Any) -> tuple[int | None, str | None]:
    """(score, repair note) or (None, None) if ``value`` is not a whole number 1-5."""
    if isinstance(value, bool):
        return None, None
    if isinstance(value, int):
        return (value, None) if 1 <= value <= 5 else (None, None)
    if isinstance(value, float) and value.is_integer() and 1 <= value <= 5:
        return int(value), f"read score {value!r} as {int(value)}"
    if isinstance(value, str) and _INT_TEXT_RE.fullmatch(value.strip()):
        number = int(value.strip())
        if 1 <= number <= 5:
            return number, f"read score {value!r} as {number}"
    return None, None


def _mode(value: Any) -> tuple[FailureMode | None, str | None]:
    if not isinstance(value, str):
        return None, None
    if value in _MODE_NAMES:
        return _MODE_NAMES[value], None
    key = re.sub(r"[\s-]+", "_", value.strip().lower())
    if key in _MODE_NAMES:
        return _MODE_NAMES[key], f"read failure mode {value!r} as {key}"
    return None, None


def parse_judgment(text: str, dimensions: Sequence[Dimension]) -> ParsedJudgment:
    """Validate a judge reply that must score exactly ``dimensions``.

    Raises :class:`JudgmentError` listing every problem at once, so a single
    re-ask can fix them all. Scores for dimensions that were not asked for,
    and keys other than the three expected, are ignored.
    """
    try:
        return _parse(text, dimensions)
    except JudgmentError:
        raise
    except Exception as exc:  # a reply is untrusted input: never crash the run on one
        raise JudgmentError([f"the reply could not be read ({type(exc).__name__})"]) from exc


def _parse(text: str, dimensions: Sequence[Dimension]) -> ParsedJudgment:
    data, repairs = _extract(text)
    problems: list[str] = []

    rationale = data.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        problems.append('"rationale" must be a non-empty string')
        rationale = ""

    wanted = ", ".join(f'"{d.value}"' for d in dimensions)
    scores: dict[Dimension, int] = {}
    raw_scores = data.get("scores")
    if not isinstance(raw_scores, dict):
        problems.append(f'"scores" must be an object with a whole number from 1 to 5 for {wanted}')
    else:
        for dim in dimensions:
            if dim.value not in raw_scores:
                problems.append(f'"scores" has no score for "{dim.value}"')
                continue
            score, note = _score(raw_scores[dim.value])
            if score is None:
                problems.append(
                    f'the score for "{dim.value}" must be a whole number from 1 to 5, '
                    f"not {_show(raw_scores[dim.value])}"
                )
                continue
            scores[dim] = score
            if note:
                repairs.append(note)

    modes: list[FailureMode] = []
    raw_modes = data.get("failure_modes")
    if not isinstance(raw_modes, list):
        problems.append('"failure_modes" must be a list of failure modes ([] if none applies)')
    else:
        for item in raw_modes:
            mode, note = _mode(item)
            if mode is None:
                names = ", ".join(_MODE_NAMES)
                problems.append(f"unknown failure mode {_show(item)}; use only: {names}")
                continue
            if mode not in modes:
                modes.append(mode)
            if note:
                repairs.append(note)

    if problems:
        raise JudgmentError(problems)
    return ParsedJudgment(scrub_surrogates(rationale.strip()), scores, modes, repairs)
