"""Structural checks: JSON schema, bullet lists, required prefix/suffix."""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar, Literal, Self

from jsonschema import FormatChecker
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic import Field, field_validator, model_validator

from evalkit.checks.base import BaseCheck, CheckResult
from evalkit.taxonomy import FailureMode
from evalkit.text import normalize

# ---------------------------------------------------------------- JSON

_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\n(.*?)\n[ \t]*```", re.DOTALL)
_MAX_REPORTED_ERRORS = 3


def _json_path(path: Any) -> str:
    out = "$"
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


class JsonSchemaCheck(BaseCheck):
    """The response must be JSON that validates against a JSON Schema.

    By default the whole response must be JSON, optionally wrapped in a single
    Markdown code fence. ``allow_surrounding_text: true`` accepts JSON embedded
    in prose (the first fenced block, else the first JSON object or array).
    """

    type: Literal["json_schema"] = "json_schema"
    json_schema: dict[str, Any] = Field(alias="schema")
    allow_code_fence: bool = True
    allow_surrounding_text: bool = False
    default_failure_mode: ClassVar[FailureMode] = FailureMode.FORMAT_ERROR
    summary: ClassVar[str] = "Must be valid JSON matching a JSON Schema (draft 2020-12 default)."

    @field_validator("json_schema")
    @classmethod
    def _valid_schema(cls, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            validator_for(schema).check_schema(schema)
        except SchemaError as exc:
            raise ValueError(f"invalid JSON Schema: {exc.message}") from exc
        return schema

    def _extract(self, response: str) -> tuple[Any, str] | str:
        """Return (data, where-it-was-found), or an error message."""
        stripped = response.strip()
        try:
            return json.loads(stripped), "whole response"
        except json.JSONDecodeError as exc:
            first_error = f"not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"

        fence = _FENCE_RE.search(stripped)
        whole_fence = fence is not None and fence.span() == (0, len(stripped))
        if fence and ((whole_fence and self.allow_code_fence) or self.allow_surrounding_text):
            try:
                return json.loads(fence.group(1)), "code fence"
            except json.JSONDecodeError as exc:
                return f"fenced block is not valid JSON: {exc.msg} (line {exc.lineno})"

        if whole_fence and not self.allow_code_fence:
            return "JSON is wrapped in a code fence, which this case does not allow"
        if self.allow_surrounding_text:
            decoder = json.JSONDecoder()
            for i, ch in enumerate(stripped):
                if ch in "{[":
                    try:
                        data, _ = decoder.raw_decode(stripped, i)
                    except json.JSONDecodeError:
                        continue
                    return data, "embedded in text"
            return "no JSON object or array found in the response"
        return first_error

    def check(self, response: str) -> CheckResult:
        extracted = self._extract(response)
        if isinstance(extracted, str):
            return self._fail(extracted)
        data, where = extracted
        validator_cls = validator_for(self.json_schema)
        validator = validator_cls(self.json_schema, format_checker=FormatChecker())
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
        if errors:
            shown = "; ".join(
                f"{_json_path(e.absolute_path)}: {e.message}" for e in errors[:_MAX_REPORTED_ERRORS]
            )
            more = (
                f" (+{len(errors) - _MAX_REPORTED_ERRORS} more)"
                if len(errors) > _MAX_REPORTED_ERRORS
                else ""
            )
            return self._fail(
                f"schema violation: {shown}{more}", errors=len(errors), found_in=where
            )
        return self._pass(f"valid JSON ({where}) matching the schema", errors=0, found_in=where)


# ---------------------------------------------------------------- lists

_UNORDERED_RE = re.compile(r"^(?P<indent>[ \t]*)[-*+•][ \t]+\S")
_ORDERED_RE = re.compile(r"^(?P<indent>[ \t]*)\d{1,3}[.)][ \t]+\S")


def _indent_width(indent: str) -> int:
    return len(indent.expandtabs(4))


class BulletListCheck(BaseCheck):
    """The response must contain a Markdown list with a given number of items.

    Only top-level items are counted (the least-indented markers), so nested
    sub-points do not inflate the count. ``style`` restricts the marker type;
    ``only_list: true`` forbids prose outside the list, such as a preamble.
    """

    type: Literal["bullet_list"] = "bullet_list"
    min_items: int | None = Field(default=None, ge=1)
    max_items: int | None = Field(default=None, ge=1)
    style: Literal["any", "unordered", "ordered"] = "any"
    only_list: bool = False
    default_failure_mode: ClassVar[FailureMode] = FailureMode.FORMAT_ERROR
    summary: ClassVar[str] = "Markdown list with [min_items, max_items] top-level items."

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if (
            self.min_items is not None
            and self.max_items is not None
            and self.min_items > self.max_items
        ):
            raise ValueError("'min_items' is greater than 'max_items'")
        return self

    def check(self, response: str) -> CheckResult:
        items: list[tuple[int, int, str]] = []  # (line index, indent, kind)
        lines = response.splitlines()
        for i, line in enumerate(lines):
            for kind, rx in (("unordered", _UNORDERED_RE), ("ordered", _ORDERED_RE)):
                m = rx.match(line)
                if m:
                    items.append((i, _indent_width(m.group("indent")), kind))
                    break
        if not items:
            return self._fail("no list items found", items=0)

        top_indent = min(indent for _, indent, _ in items)
        top = [(i, kind) for i, indent, kind in items if indent == top_indent]
        n = len(top)
        observed: dict[str, Any] = {"items": n}

        if self.style != "any":
            wrong = sum(1 for _, kind in top if kind != self.style)
            if wrong:
                return self._fail(f"{wrong} of {n} items are not {self.style}", **observed)

        if self.only_list:
            item_lines = {i for i, _, _ in items}
            for i, line in enumerate(lines):
                if not line.strip() or i in item_lines:
                    continue
                continuation = _indent_width(line[: len(line) - len(line.lstrip())]) > top_indent
                if not continuation:
                    snippet = line.strip()[:60]
                    return self._fail(f"text outside the list: {snippet!r}", **observed)

        if self.min_items is not None and n < self.min_items:
            return self._fail(f"{n} items, expected at least {self.min_items}", **observed)
        if self.max_items is not None and n > self.max_items:
            return self._fail(f"{n} items, expected at most {self.max_items}", **observed)
        return self._pass(f"{n} top-level list items", **observed)


# ---------------------------------------------------------------- prefix/suffix

# Markdown wrappers that a reader does not see as text: emphasis, headings,
# blockquote markers, inline-code ticks. Loose mode ignores them at the edges.
_EDGE_MARKUP = "*_`#> \t\r\n"


class _AffixCheck(BaseCheck):
    text: str = Field(min_length=1)
    case_sensitive: bool = False
    loose: bool = True
    """Ignore Markdown emphasis/heading markers at the edge (``**Done.**`` ends with "Done.")."""

    def _prepare(self, value: str) -> str:
        value = normalize(value)
        return value if self.case_sensitive else value.casefold()


class StartsWith(_AffixCheck):
    """The response must start with the given text."""

    type: Literal["starts_with"] = "starts_with"
    default_failure_mode: ClassVar[FailureMode] = FailureMode.FORMAT_ERROR
    summary: ClassVar[str] = "Must start with the given text (Markdown-tolerant by default)."

    def check(self, response: str) -> CheckResult:
        body = response.lstrip(_EDGE_MARKUP) if self.loose else response.lstrip()
        if self._prepare(body).startswith(self._prepare(self.text)):
            return self._pass(f"starts with {self.text!r}")
        return self._fail(f"does not start with {self.text!r}; starts {body.strip()[:40]!r}")


class EndsWith(_AffixCheck):
    """The response must end with the given text."""

    type: Literal["ends_with"] = "ends_with"
    default_failure_mode: ClassVar[FailureMode] = FailureMode.FORMAT_ERROR
    summary: ClassVar[str] = "Must end with the given text (Markdown-tolerant by default)."

    def check(self, response: str) -> CheckResult:
        body = response.rstrip(_EDGE_MARKUP) if self.loose else response.rstrip()
        expected = self.text.rstrip(_EDGE_MARKUP) if self.loose else self.text
        if self._prepare(body).endswith(self._prepare(expected)):
            return self._pass(f"ends with {self.text!r}")
        return self._fail(f"does not end with {self.text!r}; ends {body.strip()[-40:]!r}")
