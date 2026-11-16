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
a key given twice. Guessing at those would put words in the judge's mouth.

No reply, however odd (deep nesting, a 5,000-digit number, a lone surrogate
escape), raises anything but :class:`JudgmentError`.
"""

from __future__ import annotations

import ast
import json
import re
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
        return f'the key "{exc}" appears twice; give each key once'
    except json.JSONDecodeError as exc:
        return f"not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
    except RecursionError:
        return "not valid JSON: nested too deeply"
    except ValueError:  # e.g. a number with more digits than Python converts
        return "not valid JSON: a number is too long"
    if not isinstance(data, dict):
        return f"the reply must be a JSON object, not a JSON {type(data).__name__}"
    return data


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
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
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


def _embedded(text: str) -> list[tuple[int, dict[str, Any]]]:
    """``(position, object)`` for every top-level JSON object in ``text``.

    Scanning resumes after each decoded object, so an object nested inside
    another is never a candidate of its own.
    """
    found: list[tuple[int, dict[str, Any]]] = []
    pos = text.find("{")
    while pos != -1:
        try:
            data, end = _DECODER.raw_decode(text, pos)
        except (ValueError, RecursionError):
            pos = text.find("{", pos + 1)
            continue
        if isinstance(data, dict):
            found.append((pos, data))
        pos = text.find("{", end)
    return found


def _extract(text: str) -> tuple[dict[str, Any], list[str]]:
    """The judge's object and the syntax repairs it needed.

    The whole reply, or a reply that is one code fence, is read directly
    (as JSON, else as a Python literal). Otherwise every object in the reply
    is a candidate: each top-level JSON object, each fenced block read as a
    Python literal, and the span from the first ``{`` to the last ``}`` read
    as one. The candidate with the most of the three expected keys wins, and
    on a tie the last one, since a judge that drafts or quotes something
    gives its verdict last. A template echoed back (``<1-5>``, not JSON) or
    a quoted JSON response has fewer keys, so it loses; an object with none
    of them is never chosen.
    """
    stripped = text.strip()
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

    in_fence = [fence.span(1) for fence in fences]
    candidates: list[tuple[int, dict[str, Any], list[str]]] = []
    for pos, data in _embedded(stripped):
        fenced = any(a <= pos < b for a, b in in_fence)
        note = (
            "took the JSON from a code fence"
            if fenced
            else "took the JSON object out of surrounding text"
        )
        candidates.append((pos, data, [note]))
    for fence in fences:
        literal = _python_literal(fence.group(1).strip())
        if literal is not None:
            notes = ["took the JSON from a code fence", "read a Python-style object"]
            candidates.append((fence.start(1), literal, notes))
    literal = _python_literal(stripped[start : stripped.rfind("}") + 1])
    if literal is not None:
        notes = ["took the object out of surrounding text", "read a Python-style object"]
        candidates.append((start, literal, notes))
    # An object with none of the keys is noise, or a fragment of a broken
    # reply; the decode error tells the judge more than its missing keys would.
    ranked = [(sum(k in c[1] for k in _KEYS), c[0], c) for c in candidates]
    ranked = [r for r in ranked if r[0] > 0]
    if not ranked:
        raise JudgmentError([error])
    _, _, (_, data, notes) = max(ranked, key=lambda r: (r[0], r[1]))
    return data, notes


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
