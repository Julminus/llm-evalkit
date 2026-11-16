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
nested shape such as ``{"score": 4, "reason": ...}`` in place of a number.
Guessing at those would put words in the judge's mouth.
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

__all__ = ["JudgmentError", "ParsedJudgment", "parse_judgment"]

_FENCE_RE = re.compile(r"```[ \t]*[A-Za-z]*[ \t]*\r?\n(.*?)```", re.DOTALL)
_INT_TEXT_RE = re.compile(r"\d+")
_MODE_NAMES = {m.value: m for m in FailureMode}


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


def _decode(text: str) -> dict[str, Any] | str:
    """A JSON object, or why ``text`` is not one."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
    if not isinstance(data, dict):
        return f"the reply must be a JSON object, not a JSON {type(data).__name__}"
    return data


def _python_literal(text: str) -> dict[str, Any] | None:
    try:
        data = ast.literal_eval(text)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _object(text: str, where: str | None) -> tuple[dict[str, Any], list[str]] | str:
    """``text`` as a JSON object, else as a Python-style one; or why neither works."""
    repairs = [where] if where else []
    data = _decode(text)
    if isinstance(data, dict):
        return data, repairs
    literal = _python_literal(text)
    if literal is not None:
        return literal, [*repairs, "read a Python-style object"]
    return data


def _extract(text: str) -> tuple[dict[str, Any], list[str]]:
    """The judge's object and the syntax repairs it needed.

    Tried in order, first success wins: the whole reply; the first code
    fence; the first JSON object in the reply, with any text before and
    after it ignored; the span from the first ``{`` to the last ``}``.
    The first two and the last are also read as Python literals.
    """
    stripped = text.strip()
    found = _object(stripped, None)
    if not isinstance(found, str):
        return found
    error = found

    fence = _FENCE_RE.search(stripped)
    if fence is not None:
        whole_fence = fence.group(0) == stripped
        found = _object(
            fence.group(1).strip(),
            "removed a code fence" if whole_fence else "took the JSON from a code fence",
        )
        if not isinstance(found, str):
            return found
        error = found

    start = stripped.find("{")
    if start == -1:
        raise JudgmentError(["the reply contains no JSON object"])
    try:
        data, _ = json.JSONDecoder().raw_decode(stripped, start)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(data, dict):
            return data, ["took the JSON object out of surrounding text"]

    end = stripped.rfind("}")
    if end > start:
        literal = _python_literal(stripped[start : end + 1])
        if literal is not None:
            return literal, [
                "took the object out of surrounding text",
                "read a Python-style object",
            ]
    raise JudgmentError([error])


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
                    f"not {json.dumps(raw_scores[dim.value], default=str)}"
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
                shown = json.dumps(item, default=str)
                problems.append(f"unknown failure mode {shown}; use only: {names}")
                continue
            if mode not in modes:
                modes.append(mode)
            if note:
                repairs.append(note)

    if problems:
        raise JudgmentError(problems)
    return ParsedJudgment(rationale.strip(), scores, modes, repairs)
