"""Failure-mode taxonomy shared by the deterministic checks and the LLM judge.

Three modes are checkable, and code decides them: ``constraint_violation``,
``format_error`` and ``language_mismatch`` (:data:`CHECKABLE`). The other four
need judgment about meaning, and the judge decides them.

The judge is still offered all seven, and its tags for the checkable modes
are kept and compared with the checks, but they don't count. Measured on 80
real responses (README, Results): qwen2.5:14b tagged ``constraint_violation``
on 31 of the 69 whose checks all passed, with the passing results in its
prompt. Telling it not to contradict the checks changed nothing (33 of 68),
and taking the three modes off its list just moved the same complaints to
``partial_completion`` (false on 39 of 70 complete responses, up from 5 of 71).
"""

from enum import StrEnum


class FailureMode(StrEnum):
    """Why a response failed. One response can carry several tags."""

    CONSTRAINT_VIOLATION = "constraint_violation"
    INTENT_MISREAD = "intent_misread"
    HALLUCINATION = "hallucination"
    PARTIAL_COMPLETION = "partial_completion"
    FORMAT_ERROR = "format_error"
    OVER_REFUSAL = "over_refusal"
    LANGUAGE_MISMATCH = "language_mismatch"


DESCRIPTIONS: dict[FailureMode, str] = {
    FailureMode.CONSTRAINT_VIOLATION: (
        "A stated, checkable constraint is broken: a length or count limit, a required or "
        "forbidden word or phrase, a required pattern."
    ),
    FailureMode.INTENT_MISREAD: (
        "The response answers a different question, or does a different task, than the one asked."
    ),
    FailureMode.HALLUCINATION: (
        "The response states something false as fact, or something the given context does not "
        "support, including invented names, numbers, quotes or sources."
    ),
    FailureMode.PARTIAL_COMPLETION: (
        "Part of the request is left undone: a part of a multi-part request is skipped, or the "
        "response stops before it is finished."
    ),
    FailureMode.FORMAT_ERROR: (
        "The requested structure is wrong: invalid JSON, the wrong list shape, a missing "
        "required prefix or suffix."
    ),
    FailureMode.OVER_REFUSAL: (
        "A safe, answerable request is refused, or hedged so much that it is not actually answered."
    ),
    FailureMode.LANGUAGE_MISMATCH: (
        "The response, or part of it, is not in the language it should be in."
    ),
}
"""One line per failure mode. The judge's prompt and ``evalkit taxonomy`` use these."""

CHECKABLE: frozenset[FailureMode] = frozenset(
    {FailureMode.CONSTRAINT_VIOLATION, FailureMode.FORMAT_ERROR, FailureMode.LANGUAGE_MISMATCH}
)
"""Modes a deterministic check decides. The judge's own tags for them don't count."""
