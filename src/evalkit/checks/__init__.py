"""Deterministic checks.

Each constraint type is a Pydantic model tagged by its ``type`` field.
:data:`Constraint` is the discriminated union the suite schema uses, so YAML
like ``{type: word_count, max: 80}`` is parsed straight into a
:class:`~evalkit.checks.length.WordCount` and checked with ``.check(response)``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from evalkit.checks.base import BaseCheck, CheckResult, Status
from evalkit.checks.language import LanguageCheck
from evalkit.checks.length import ParagraphCount, SentenceCount, WordCount
from evalkit.checks.pattern import RegexCheck
from evalkit.checks.phrases import ForbiddenPhrases, RequiredPhrases
from evalkit.checks.structure import BulletListCheck, EndsWith, JsonSchemaCheck, StartsWith

Constraint = Annotated[
    WordCount
    | SentenceCount
    | ParagraphCount
    | RequiredPhrases
    | ForbiddenPhrases
    | RegexCheck
    | JsonSchemaCheck
    | BulletListCheck
    | StartsWith
    | EndsWith
    | LanguageCheck,
    Field(discriminator="type"),
]

#: Every constraint class, in the order ``evalkit list-checks`` shows them.
CHECK_TYPES: tuple[type[BaseCheck], ...] = (
    WordCount,
    SentenceCount,
    ParagraphCount,
    RequiredPhrases,
    ForbiddenPhrases,
    RegexCheck,
    JsonSchemaCheck,
    BulletListCheck,
    StartsWith,
    EndsWith,
    LanguageCheck,
)

__all__ = [
    "CHECK_TYPES",
    "BaseCheck",
    "BulletListCheck",
    "CheckResult",
    "Constraint",
    "EndsWith",
    "ForbiddenPhrases",
    "JsonSchemaCheck",
    "LanguageCheck",
    "ParagraphCount",
    "RegexCheck",
    "RequiredPhrases",
    "SentenceCount",
    "StartsWith",
    "Status",
    "WordCount",
]
