"""Regular-expression checks."""

from __future__ import annotations

import re
from functools import lru_cache, reduce
from operator import or_
from typing import ClassVar, Literal, Self

from pydantic import Field, field_validator, model_validator

from evalkit.checks.base import BaseCheck, CheckResult

RegexFlag = Literal["ignorecase", "multiline", "dotall"]
_FLAGS: dict[str, re.RegexFlag] = {
    "ignorecase": re.IGNORECASE,
    "multiline": re.MULTILINE,
    "dotall": re.DOTALL,
}


@lru_cache(maxsize=1024)
def _compile(pattern: str, flags: tuple[RegexFlag, ...]) -> re.Pattern[str]:
    return re.compile(pattern, reduce(or_, (_FLAGS[f] for f in flags), re.NOFLAG))


class RegexCheck(BaseCheck):
    """The response must (or must not) match a regular expression.

    ``fullmatch: true`` requires the whole response (stripped) to match, e.g.
    ``pattern: '-?\\d+'`` for "answer with a single integer". Otherwise the
    number of non-overlapping matches must fall within [min_matches, max_matches].
    """

    type: Literal["regex"] = "regex"
    pattern: str = Field(min_length=1)
    flags: tuple[RegexFlag, ...] = ()
    expect: Literal["match", "no_match"] = "match"
    fullmatch: bool = False
    min_matches: int = Field(default=1, ge=1)
    max_matches: int | None = Field(default=None, ge=1)
    summary: ClassVar[str] = "Must (or must not) match a regex; optional full match or count range."

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, pattern: str) -> str:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        return pattern

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.max_matches is not None and self.max_matches < self.min_matches:
            raise ValueError("'max_matches' is less than 'min_matches'")
        counted = "min_matches" in self.model_fields_set or self.max_matches is not None
        if counted and (self.fullmatch or self.expect == "no_match"):
            raise ValueError("match counts only apply to expect=match without fullmatch")
        return self

    def check(self, response: str) -> CheckResult:
        rx = _compile(self.pattern, self.flags)
        if self.fullmatch:
            matched = rx.fullmatch(response.strip()) is not None
            if matched == (self.expect == "match"):
                return self._pass(f"response {'fully matches' if matched else 'does not match'}")
            verb = "does not fully match" if not matched else "fully matches"
            return self._fail(f"response {verb} /{self.pattern}/")

        n = sum(1 for _ in rx.finditer(response))
        if self.expect == "no_match":
            if n:
                return self._fail(f"/{self.pattern}/ matched {n} time(s), expected none", matches=n)
            return self._pass(f"/{self.pattern}/ not found", matches=0)
        if n < self.min_matches:
            return self._fail(
                f"/{self.pattern}/ matched {n} time(s), expected at least {self.min_matches}",
                matches=n,
            )
        if self.max_matches is not None and n > self.max_matches:
            return self._fail(
                f"/{self.pattern}/ matched {n} time(s), expected at most {self.max_matches}",
                matches=n,
            )
        return self._pass(f"/{self.pattern}/ matched {n} time(s)", matches=n)
