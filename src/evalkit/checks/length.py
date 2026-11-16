"""Length limits: words, sentences, paragraphs."""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Literal, Self

from pydantic import Field, model_validator

from evalkit import text
from evalkit.checks.base import BaseCheck, CheckResult


class _BoundedCount(BaseCheck):
    """Shared logic: count something in the response and compare to [min, max]."""

    min: int | None = Field(default=None, ge=0)
    max: int | None = Field(default=None, ge=0)
    unit: ClassVar[str]

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if self.min is None and self.max is None:
            raise ValueError("set at least one of 'min' or 'max'")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"'min' ({self.min}) is greater than 'max' ({self.max})")
        return self

    @abstractmethod
    def count(self, response: str) -> int:
        """Return the measured quantity."""

    def _bounds_text(self) -> str:
        if self.min is not None and self.max is not None:
            return f"exactly {self.min}" if self.min == self.max else f"{self.min}-{self.max}"
        return f"at least {self.min}" if self.min is not None else f"at most {self.max}"

    def check(self, response: str) -> CheckResult:
        n = self.count(response)
        observed = {self.unit: n}
        expected = self._bounds_text()
        too_few = self.min is not None and n < self.min
        too_many = self.max is not None and n > self.max
        if too_few or too_many:
            return self._fail(f"{n} {self.unit}, expected {expected}", **observed)
        return self._pass(f"{n} {self.unit} (expected {expected})", **observed)


class WordCount(_BoundedCount):
    """Limit the number of words. Hyphenated words ("anak-anak") count once."""

    type: Literal["word_count"] = "word_count"
    unit: ClassVar[str] = "words"
    summary: ClassVar[str] = "Word count within [min, max]; 'anak-anak' and 'don't' count once."

    def count(self, response: str) -> int:
        return len(text.words(response))


class SentenceCount(_BoundedCount):
    """Limit the number of sentences. Each line (list item, heading) is a sentence."""

    type: Literal["sentence_count"] = "sentence_count"
    unit: ClassVar[str] = "sentences"
    summary: ClassVar[str] = (
        "Sentence count within [min, max]; abbreviation-aware (e.g., dll., Dr.)."
    )

    def count(self, response: str) -> int:
        return len(text.sentences(response))


class ParagraphCount(_BoundedCount):
    """Limit the number of paragraphs (blocks separated by blank lines)."""

    type: Literal["paragraph_count"] = "paragraph_count"
    unit: ClassVar[str] = "paragraphs"
    summary: ClassVar[str] = "Paragraph count within [min, max]; paragraphs split on blank lines."

    def count(self, response: str) -> int:
        return len(text.paragraphs(response))
