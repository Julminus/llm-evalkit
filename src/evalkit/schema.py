"""Test-suite schema and YAML loader.

A suite is a YAML file of evaluation cases. Each case carries the prompt, its
language, the deterministic constraints a response must satisfy, and a rubric
for the LLM judge (used from S3 on). Unknown keys are rejected everywhere, so
a typo such as ``max_word: 50`` fails loudly instead of being ignored.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from evalkit.checks import Constraint, LanguageCheck
from evalkit.langdetect import Language

__all__ = [
    "Dimension",
    "EvalCase",
    "Language",
    "Rubric",
    "Suite",
    "SuiteError",
    "load_suite",
]


class Dimension(StrEnum):
    """Judge scoring dimensions (each scored 1-5 from S3 on)."""

    INSTRUCTION_FOLLOWING = "instruction_following"
    REASONING_QUALITY = "reasoning_quality"
    FAITHFULNESS = "faithfulness"
    HELPFULNESS = "helpfulness"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Rubric(_Strict):
    """What the judge should look for in this specific case."""

    criteria: list[str] = Field(default_factory=list)
    """Case-specific requirements, e.g. "states that the deadline is unknown"."""

    reference_answer: str | None = None
    """An example of a good answer. Guidance for the judge, not an exact target."""

    dimensions: list[Dimension] = Field(default_factory=lambda: list(Dimension))
    """Which dimensions apply. Faithfulness only makes sense when there is a context."""


class EvalCase(_Strict):
    """One prompt plus everything needed to grade a response to it."""

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", max_length=80)
    prompt: str = Field(min_length=1)
    language: Language
    """Language of the prompt."""

    response_language: Language | Literal["any"] | None = None
    """Language the response must be in. ``None`` (default) means "same as the
    prompt"; ``any`` turns the automatic language check off."""

    system: str | None = None
    context: str | None = None
    """Source material the response must stay faithful to (RAG-style cases)."""

    constraints: list[Constraint] = Field(default_factory=list)
    rubric: Rubric | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, tags: list[str]) -> list[str]:
        cleaned = [t.strip().lower() for t in tags]
        if any(not t for t in cleaned):
            raise ValueError("tags must not be blank")
        return list(dict.fromkeys(cleaned))  # dedupe, keep order

    @property
    def expected_language(self) -> Language | None:
        """The language the response must be in, or ``None`` if any is fine."""
        if self.response_language == "any":
            return None
        return self.response_language or self.language

    def effective_constraints(self) -> list[Constraint]:
        """The constraints to run, including the automatic language check.

        Every case checks that the response is in :attr:`expected_language`,
        unless the case lists its own ``language`` constraint or sets
        ``response_language: any``. Language mismatch is the most common
        failure in bilingual evaluation, so it is opt-out, not opt-in.
        """
        constraints = list(self.constraints)
        explicit = any(isinstance(c, LanguageCheck) for c in constraints)
        if self.expected_language is not None and not explicit:
            constraints.append(LanguageCheck(expected=self.expected_language))
        return constraints


class Suite(_Strict):
    """A named collection of evaluation cases."""

    name: str = Field(min_length=1)
    version: str = "1"
    description: str = ""
    cases: list[EvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        dupes = sorted(i for i, n in Counter(c.id for c in self.cases).items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate case ids: {', '.join(dupes)}")
        return self

    def case(self, case_id: str) -> EvalCase:
        for c in self.cases:
            if c.id == case_id:
                return c
        raise KeyError(case_id)


class SuiteError(Exception):
    """A suite file could not be read or is invalid. The message is user-facing."""


def _format_validation_error(exc: ValidationError, raw: object) -> str:
    """Turn Pydantic's error list into lines like ``cases[2] (id=en-003).constraints[0]: ...``."""
    lines = []
    for err in exc.errors():
        loc = err["loc"]
        parts: list[str] = []
        for i, part in enumerate(loc):
            if isinstance(part, int):
                label = f"[{part}]"
                # Name the case by its id: easier to find in the YAML than an index.
                if i == 1 and loc[0] == "cases" and isinstance(raw, dict):
                    cases = raw.get("cases")
                    if isinstance(cases, list) and part < len(cases):
                        case = cases[part]
                        if isinstance(case, dict) and "id" in case:
                            label += f" (id={case['id']})"
                parts.append(label)
            else:
                parts.append(("." if parts else "") + str(part))
        where = "".join(parts) or "<root>"
        msg = err["msg"].removeprefix("Value error, ")
        lines.append(f"  {where}: {msg}")
    return "\n".join(lines)


def parse_suite(raw: object, source: str = "<suite>") -> Suite:
    """Validate an already-parsed YAML/JSON object as a :class:`Suite`."""
    if not isinstance(raw, dict):
        raise SuiteError(f"{source}: expected a mapping at the top level")
    try:
        return Suite.model_validate(raw)
    except ValidationError as exc:
        n = exc.error_count()
        detail = _format_validation_error(exc, raw)
        raise SuiteError(f"{source}: {n} validation error(s)\n{detail}") from exc


def load_suite(path: str | Path) -> Suite:
    """Load and validate a suite from a YAML file."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SuiteError(f"{path}: cannot read file ({exc.strerror})") from exc
    except yaml.YAMLError as exc:
        raise SuiteError(f"{path}: invalid YAML\n  {exc}") from exc
    return parse_suite(raw, source=str(path))
