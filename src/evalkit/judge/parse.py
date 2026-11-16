"""Read the judge's reply: find the JSON, repair its syntax, validate its structure.

One rule decides what is fixed here and what goes back to the judge:
**syntax is repaired locally, structure is re-asked.**

Repaired locally (free, deterministic, and recorded in ``repairs``):

* text around the object: one Markdown code fence, or prose before the first
  ``{`` and after the last ``}`` ("Here is my evaluation: {...}");
* Python-style objects: single quotes, ``True`` / ``None``, a trailing comma
  (read with :func:`ast.literal_eval`, which evaluates literals only, never code);
* a score written as ``"4"`` or ``4.0``;
* a failure mode written as ``"Constraint Violation"`` or ``"over-refusal"``.

Re-asked (the judge must decide): a missing key or score, a score outside
1-5 or between two whole numbers (``4.5``), an unknown failure mode, a
nested shape such as ``{"score": 4, "reason": ...}`` in place of a number,
a key given twice, a reply over 30,000 characters, and a reply that holds
anything but one object (two judgments, a draft and then a verdict, an
echoed template). Guessing at those would put words in the judge's mouth.

That last rule is deliberately simple. Three rounds of smarter selection
(first object, most complete object, first complete object) each broke a
realistic reply, while all 981 real judge replies measured so far were a
single object, and a re-ask costs one short request.

No reply raises anything but :class:`JudgmentError`: whatever goes wrong
while reading one (deep nesting, a 5,000-digit number, a literal Python
can't evaluate) becomes a problem to re-ask about.
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
_SHOWN_CHARS = 60
_MAX_REPLY_CHARS = 30_000
"""A judgment is a few hundred characters; a longer reply is a runaway, not a verdict."""


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


_ONLY_ONE = "reply with only the JSON object, with no other text or objects around it"


def _extract(text: str) -> tuple[dict[str, Any], list[str]]:
    """The judge's object and the syntax repairs it needed.

    Tried in order, each as JSON and then as a Python literal: the whole
    reply; the inside of its one code fence, if it has exactly one and no
    ``{`` outside it; the span from the first ``{`` to the last ``}``.
    Anything else is re-asked.
    """
    stripped = text.strip()
    if len(stripped) > _MAX_REPLY_CHARS:
        raise JudgmentError([f"the reply is {len(stripped):,} characters long; {_ONLY_ONE}"])
    found = _object(stripped)
    if not isinstance(found, str):
        return found

    fences = list(_FENCE_RE.finditer(stripped))
    if len(fences) == 1:
        fence = fences[0]
        outside = stripped[: fence.start()] + stripped[fence.end() :]
        if "{" not in outside:
            fenced = _object(fence.group(1).strip())
            if not isinstance(fenced, str):
                note = (
                    "removed a code fence"
                    if not outside.strip()
                    else "took the JSON from a code fence"
                )
                return fenced[0], [note, *fenced[1]]
            raise JudgmentError([fenced] if not outside.strip() else [fenced, _ONLY_ONE])

    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1:
        raise JudgmentError(["the reply contains no JSON object"])
    if end < start:  # an object that is never closed: cut off, say
        raise JudgmentError([found])
    span = _object(stripped[start : end + 1])
    if not isinstance(span, str):
        return span[0], ["took the JSON object out of surrounding text", *span[1]]
    raise JudgmentError([span, _ONLY_ONE])


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
