"""Required and forbidden phrases."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import ClassVar, Literal

from pydantic import Field, field_validator

from evalkit.checks.base import BaseCheck, CheckResult
from evalkit.text import fold_quotes


@lru_cache(maxsize=1024)
def _phrase_pattern(phrase: str, case_sensitive: bool, whole_word: bool) -> re.Pattern[str]:
    # Any run of whitespace in the phrase matches any run of whitespace in the
    # response, so a line break inside a phrase does not hide it.
    body = r"\s+".join(re.escape(part) for part in phrase.split())
    if whole_word:
        # Lookarounds instead of \b so phrases that start or end with
        # punctuation ("C++", "$5") still work.
        body = rf"(?<!\w){body}(?!\w)"
    return re.compile(body, 0 if case_sensitive else re.IGNORECASE)


class _PhraseCheck(BaseCheck):
    phrases: list[str] = Field(min_length=1)
    case_sensitive: bool = False
    whole_word: bool = True
    """Match on word boundaries, so 'AI' does not match inside 'said'."""

    @field_validator("phrases")
    @classmethod
    def _non_blank(cls, phrases: list[str]) -> list[str]:
        cleaned = [p.strip() for p in phrases]
        if any(not p for p in cleaned):
            raise ValueError("phrases must not be blank")
        return cleaned

    def _present(self, response: str) -> list[str]:
        # Typographic quotes are folded on both sides, so "can't" matches "can’t".
        haystack = fold_quotes(unicodedata.normalize("NFKC", response))
        return [
            p
            for p in self.phrases
            if _phrase_pattern(
                fold_quotes(unicodedata.normalize("NFKC", p)), self.case_sensitive, self.whole_word
            ).search(haystack)
        ]


class RequiredPhrases(_PhraseCheck):
    """The response must contain all (or any) of the phrases."""

    type: Literal["required_phrases"] = "required_phrases"
    match: Literal["all", "any"] = "all"
    summary: ClassVar[str] = "Must contain all (or any) of the listed phrases."

    def check(self, response: str) -> CheckResult:
        present = self._present(response)
        missing = [p for p in self.phrases if p not in present]
        ok = not missing if self.match == "all" else bool(present)
        if ok:
            return self._pass(f"found {len(present)}/{len(self.phrases)} phrases", missing=missing)
        if self.match == "any":
            return self._fail(f"none of {self.phrases} found", missing=missing)
        return self._fail(f"missing {missing}", missing=missing)


class ForbiddenPhrases(_PhraseCheck):
    """The response must not contain any of the phrases."""

    type: Literal["forbidden_phrases"] = "forbidden_phrases"
    summary: ClassVar[str] = "Must not contain any of the listed phrases."

    def check(self, response: str) -> CheckResult:
        found = self._present(response)
        if found:
            return self._fail(f"contains forbidden {found}", found=found)
        return self._pass(f"none of {len(self.phrases)} forbidden phrases found", found=[])
