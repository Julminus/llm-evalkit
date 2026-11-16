"""Failure-mode taxonomy shared by deterministic checks and (from S3) the LLM judge.

Deterministic checks can only ever produce three of these tags:
``constraint_violation``, ``format_error`` and ``language_mismatch``. The rest
need judgment about meaning, so only the judge assigns them.
"""

from enum import StrEnum


class FailureMode(StrEnum):
    """Why a response failed. One response can carry several tags."""

    CONSTRAINT_VIOLATION = "constraint_violation"
    """A stated, checkable constraint was broken (length, required/forbidden phrase, regex)."""

    INTENT_MISREAD = "intent_misread"
    """The response answers a different question than the one asked."""

    HALLUCINATION = "hallucination"
    """The response states facts that are false or unsupported by the given context."""

    PARTIAL_COMPLETION = "partial_completion"
    """Some parts of a multi-part request were skipped."""

    FORMAT_ERROR = "format_error"
    """The output structure is wrong: invalid JSON, wrong list shape, missing prefix or suffix."""

    OVER_REFUSAL = "over_refusal"
    """A safe, answerable request was refused or hedged into uselessness."""

    LANGUAGE_MISMATCH = "language_mismatch"
    """The response is in a different language from the one requested."""
