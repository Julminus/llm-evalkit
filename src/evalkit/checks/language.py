"""Response-language check (English vs. Indonesian)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from evalkit.checks.base import BaseCheck, CheckResult
from evalkit.langdetect import Language, detect_language
from evalkit.taxonomy import FailureMode


class LanguageCheck(BaseCheck):
    """The response must be written in the expected language.

    Skipped (not failed) when the response has fewer than ``min_signal_words``
    function words, e.g. a bare number or a one-word answer: there is nothing
    to judge. Fails when the expected language wins but with less than
    ``min_confidence`` of the evidence, i.e. the response is heavily mixed.
    """

    type: Literal["language"] = "language"
    expected: Language
    min_confidence: float = Field(default=0.7, ge=0.5, le=1.0)
    min_signal_words: int = Field(default=2, ge=1)
    default_failure_mode: ClassVar[FailureMode] = FailureMode.LANGUAGE_MISMATCH
    summary: ClassVar[str] = "Response language must be 'en' or 'id' (function-word detector)."

    def check(self, response: str) -> CheckResult:
        guess = detect_language(response)
        observed = {
            "detected": guess.language,
            "confidence": round(guess.confidence, 3),
            "hits": dict(guess.hits),
        }
        if guess.signal_words < self.min_signal_words:
            return self._skip(
                f"too little text to tell ({guess.signal_words} signal words)", **observed
            )
        if guess.language != self.expected:
            detected = guess.language or "an even EN/ID mix"
            return self._fail(f"expected {self.expected}, detected {detected}", **observed)
        if guess.confidence < self.min_confidence:
            return self._fail(
                f"mostly {self.expected} but mixed "
                f"({guess.confidence:.0%} < {self.min_confidence:.0%} confidence)",
                **observed,
            )
        return self._pass(f"{self.expected} ({guess.confidence:.0%} confidence)", **observed)
