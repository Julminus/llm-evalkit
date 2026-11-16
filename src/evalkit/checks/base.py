"""Base types for deterministic checks.

A *constraint* is a small Pydantic model that lives inside a test case in the
YAML suite (``{type: word_count, max: 80}``) and knows how to check a response
against itself. Keeping the config and the logic in one class means a new
check is one class plus its tests, and the YAML schema can never drift from
what the code accepts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from evalkit.taxonomy import FailureMode


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    """The check could not be applied (e.g. too little text to detect a language)."""


class CheckResult(BaseModel):
    """Outcome of one constraint on one response."""

    model_config = ConfigDict(frozen=True)

    check: str
    status: Status
    message: str
    failure_mode: FailureMode | None = None
    observed: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is Status.PASS

    @property
    def failed(self) -> bool:
        return self.status is Status.FAIL


class BaseCheck(BaseModel, ABC):
    """Common behaviour for every constraint type."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    type: str
    fails_as: FailureMode | None = None
    """Override the failure-mode tag for this constraint, e.g. tag a regex that
    enforces "answer with the number only" as ``format_error``."""

    default_failure_mode: ClassVar[FailureMode] = FailureMode.CONSTRAINT_VIOLATION
    summary: ClassVar[str] = ""
    """One line for ``evalkit list-checks``."""

    @property
    def failure_mode(self) -> FailureMode:
        """The tag a failure of this constraint carries."""
        return self.fails_as or self.default_failure_mode

    @abstractmethod
    def check(self, response: str) -> CheckResult:
        """Check ``response`` against this constraint."""

    # Small constructors so every check reports results the same way.
    def _pass(self, message: str, **observed: Any) -> CheckResult:
        return CheckResult(check=self.type, status=Status.PASS, message=message, observed=observed)

    def _fail(self, message: str, **observed: Any) -> CheckResult:
        return CheckResult(
            check=self.type,
            status=Status.FAIL,
            message=message,
            failure_mode=self.failure_mode,
            observed=observed,
        )

    def _skip(self, message: str, **observed: Any) -> CheckResult:
        return CheckResult(check=self.type, status=Status.SKIP, message=message, observed=observed)
