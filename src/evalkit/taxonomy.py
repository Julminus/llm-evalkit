"""Failure-mode taxonomy shared by the deterministic checks and the LLM judge.

Deterministic checks can only ever produce three of these tags:
``constraint_violation``, ``format_error`` and ``language_mismatch``. The rest
need judgment about meaning, so only the judge assigns them. The judge may
assign any of the seven, including the three a check can raise: a suite rarely
encodes every constraint as a check.
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
