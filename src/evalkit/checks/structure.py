"""Structural checks: JSON schema, bullet lists, required prefix/suffix."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any, ClassVar, Literal, Self

import referencing.jsonschema
from jsonschema import FormatChecker
from jsonschema.exceptions import SchemaError
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for

# The standard metaschemas, bundled with jsonschema (a dependency of it).
from jsonschema_specifications import REGISTRY as _METASCHEMAS  # type: ignore[import-untyped]
from pydantic import Field, field_validator, model_validator
from referencing import Resource
from referencing.exceptions import Unresolvable

from evalkit.checks.base import BaseCheck, CheckResult
from evalkit.taxonomy import FailureMode
from evalkit.text import fold_quotes, normalize

# ---------------------------------------------------------------- JSON

_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```", re.DOTALL)
_JSON_START_RE = re.compile(r"[{\[]")
_MAX_REPORTED_ERRORS = 3
_MEMO_LIMIT = 256


def _json_path(path: Any) -> str:
    out = "$"
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


# Python's json module accepts NaN and Infinity; the JSON standard does not.
_DECODER = json.JSONDecoder(parse_constant=_reject_constant)


def _loads(text: str) -> Any:
    return _DECODER.decode(text)


def _decode_error(exc: ValueError) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"{exc.msg} (line {exc.lineno}, column {exc.colno})"
    return str(exc)


def _unresolvable_refs(schema: dict[str, Any], validator_cls: type[Validator]) -> list[str]:
    """Every ``$ref`` in ``schema`` that can't be resolved offline.

    jsonschema resolves references lazily, so a typo such as
    ``#/$defs/Persn`` would otherwise surface only when a response reaches
    that part of the schema, as a crash in the middle of a run. A ref may
    point inside the schema or at a standard metaschema (bundled, as
    jsonschema itself resolves them).

    The walk covers every real subschema, as the schema's draft defines them
    (draft-04 ``id`` scoping included), and also follows each ref into its
    target, because a ref can point anywhere (``#/x-defs/A``, or ``$defs``
    under draft-07), and jsonschema then uses that target as a schema. A
    "$ref" inside ``enum`` data or an unreferenced vendor keyword such as
    ``x-docs`` is data and is not checked. ``$dynamicRef`` is checked like
    ``$ref``: its first resolution is static, so a typo shows there too.
    """
    spec = referencing.jsonschema.specification_with(
        validator_cls.META_SCHEMA.get("$schema", ""), default=referencing.jsonschema.DRAFT202012
    )
    ours = _node_ids(schema)  # follow refs only into this schema, not the metaschemas
    seen: set[int] = set()
    bad: list[str] = []
    root = spec.create_resource(schema)
    # An explicit stack, not recursion: a long chain of $defs must not hit
    # Python's recursion limit.
    stack: list[tuple[Resource[Any], Any]] = [(root, _METASCHEMAS.resolver_with_root(root))]
    while stack:
        resource, resolver = stack.pop()
        contents = resource.contents
        if not isinstance(contents, dict) or id(contents) in seen:
            continue
        seen.add(id(contents))
        for keyword in ("$ref", "$dynamicRef"):
            ref = contents.get(keyword)
            if not isinstance(ref, str):
                continue
            try:
                resolved = resolver.lookup(ref)
            except Unresolvable:
                bad.append(ref)
                continue
            if id(resolved.contents) in ours:
                stack.append((spec.create_resource(resolved.contents), resolved.resolver))
        # Only object schemas can hold refs (booleans can't, and draft-07
        # "dependencies" values may be arrays of property names).
        subresources = [s for s in resource.subresources() if isinstance(s.contents, dict)]
        # referencing yields "dependencies" schemas only when the first value
        # is a schema; walk them all on the drafts where the keyword exists.
        deps = contents.get("dependencies")
        if spec in _DEPENDENCIES_DRAFTS and isinstance(deps, dict):
            subresources += [spec.create_resource(v) for v in deps.values() if isinstance(v, dict)]
        stack.extend((sub, resolver.in_subresource(sub)) for sub in subresources)
    return bad


_DEPENDENCIES_DRAFTS = (
    referencing.jsonschema.DRAFT3,
    referencing.jsonschema.DRAFT4,
    referencing.jsonschema.DRAFT6,
    referencing.jsonschema.DRAFT7,
)


def _node_ids(node: Any) -> set[int]:
    """The ids of every dict and list in a JSON-like tree."""
    ids: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            ids.add(id(current))
            stack.extend(current.values())
        elif isinstance(current, list):
            ids.add(id(current))
            stack.extend(current)
    return ids


def _schema_problem(schema: dict[str, Any]) -> str | None:
    """Why a schema is unusable, or None."""
    validator_cls = validator_for(schema)
    try:
        validator_cls.check_schema(schema)
    except SchemaError as exc:
        return f"invalid JSON Schema: {exc.message}"
    bad = _unresolvable_refs(schema, validator_cls)
    if bad:
        return (
            f"JSON Schema has unresolvable $ref {bad}; point it at a definition in this "
            "schema or a standard metaschema (other remote refs are not fetched)"
        )
    return None


def _new_validator(schema: dict[str, Any]) -> Validator:
    validator: Validator = validator_for(schema)(schema, format_checker=FormatChecker())
    return validator


# Memoised per distinct schema: a suite often repeats one schema across many
# cases, and the metaschema check costs ~0.8 ms. The key is the schema as
# sorted JSON, and the value is built from that JSON form too, because that is
# what a JSON Schema means: YAML's `properties: {1: ...}` has an int key that
# would never match a JSON object's "1". A schema that JSON can't represent
# at all (a YAML date, mixed key types) is used as loaded, without memoising.
_PROBLEMS: dict[str, str | None] = {}
_VALIDATORS: dict[str, Validator] = {}


def _schema_key(schema: dict[str, Any]) -> str | None:
    try:
        return json.dumps(schema, sort_keys=True)
    except (TypeError, ValueError):
        return None


def _memo(cache: dict[str, Any], schema: dict[str, Any], build: Any) -> Any:
    key = _schema_key(schema)
    if key is None:
        return build(schema)
    if key not in cache:
        if len(cache) >= _MEMO_LIMIT:
            cache.clear()
        cache[key] = build(json.loads(key))
    return cache[key]


class JsonSchemaCheck(BaseCheck):
    """The response must be JSON that validates against a JSON Schema.

    By default the whole response must be JSON, optionally wrapped in a single
    Markdown code fence. ``allow_surrounding_text: true`` accepts JSON embedded
    in prose: the first fenced block, else the first JSON object or array that
    matches the schema (so a citation like "[1]" before the real object does
    not hide it), else the first object. NaN and Infinity are rejected, as the
    JSON standard requires.
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
        error = _memo(_PROBLEMS, schema, _schema_problem)
        if error is not None:
            raise ValueError(error)
        return schema

    @property
    def _validator(self) -> Validator:
        validator: Validator = _memo(_VALIDATORS, self.json_schema, _new_validator)
        return validator

    def _embedded(self, text: str) -> Iterator[Any]:
        """Every top-level JSON object or array in ``text``, left to right.

        Scanning resumes after each decoded value, so a value nested inside
        another is never a candidate of its own: ``{"person": {...}}`` must
        not pass a schema for the inner object.
        """
        resume = 0
        for m in _JSON_START_RE.finditer(text):
            if m.start() < resume:
                continue
            try:
                data, resume = _DECODER.raw_decode(text, m.start())
            except ValueError:
                continue
            yield data

    def _extract(self, response: str) -> tuple[Any, str] | str:
        """Return (data, where-it-was-found), or an error message."""
        stripped = response.strip()
        try:
            return _loads(stripped), "whole response"
        except ValueError as exc:
            first_error = f"not valid JSON: {_decode_error(exc)}"

        fence = _FENCE_RE.search(stripped)
        whole_fence = fence is not None and fence.span() == (0, len(stripped))
        if fence and ((whole_fence and self.allow_code_fence) or self.allow_surrounding_text):
            try:
                return _loads(fence.group(1)), "code fence"
            except ValueError as exc:
                return f"fenced block is not valid JSON: {_decode_error(exc)}"

        if whole_fence and not self.allow_code_fence:
            return "JSON is wrapped in a code fence, which this case does not allow"
        if self.allow_surrounding_text:
            candidates = list(self._embedded(stripped))
            if not candidates:
                return "no JSON object or array found in the response"
            # The first candidate that matches; failing that, the first object,
            # since a bracketed citation like "[1]" is rarely the answer.
            fallback = next((c for c in candidates if isinstance(c, dict)), candidates[0])
            matching = (c for c in candidates if self._validator.is_valid(c))
            return next(matching, fallback), "embedded in text"
        return first_error

    def check(self, response: str) -> CheckResult:
        try:
            extracted = self._extract(response)
            if isinstance(extracted, str):
                return self._fail(extracted)
            data, where = extracted
            errors = sorted(self._validator.iter_errors(data), key=lambda e: list(e.absolute_path))
        except Unresolvable as exc:
            # A safety net behind the load-time $ref check: a schema problem is
            # not the model's failure, and it must not crash a run after the
            # responses have been paid for.
            return self._skip(f"cannot apply the schema: unresolvable reference ({exc})")
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
# A thematic break ("---", "* * *", "___") looks like a marker but is not an item.
_RULE_RE = re.compile(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_FENCE_LINE_RE = re.compile(r"^[ \t]*(```|~~~)")


def _prose_lines(lines: list[str]) -> Iterator[tuple[int, str]]:
    """(index, line) for lines outside fenced code blocks and thematic breaks."""
    fence: str | None = None
    for i, line in enumerate(lines):
        m = _FENCE_LINE_RE.match(line)
        if m:
            if fence is None:
                fence = m.group(1)
            elif m.group(1) == fence:
                fence = None
            continue
        if fence is None and not _RULE_RE.match(line):
            yield i, line


def _indent_width(indent: str) -> int:
    return len(indent.expandtabs(4))


class BulletListCheck(BaseCheck):
    """The response must contain a Markdown list with a given number of items.

    Only top-level items are counted (the least-indented markers), so nested
    sub-points do not inflate the count. Lines inside fenced code blocks and
    thematic breaks (``* * *``) are never items. ``style`` restricts the
    marker type; ``only_list: true`` forbids text outside the list, such as a
    preamble or a code block.
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
        for i, line in _prose_lines(lines):
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
                if not line.strip() or i in item_lines or _RULE_RE.match(line):
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
    """Ignore Markdown emphasis/heading markers at the edge (``**Done.**`` ends with "Done."),
    in the response and in ``text`` alike."""

    @model_validator(mode="after")
    def _not_only_markup(self) -> Self:
        if self.loose and not self.text.strip(_EDGE_MARKUP):
            raise ValueError(
                f"text {self.text!r} is only Markdown markup, which loose mode ignores; "
                "set 'loose: false' to match it literally"
            )
        return self

    def _prepare(self, value: str) -> str:
        value = fold_quotes(normalize(value))
        return value if self.case_sensitive else value.casefold()


class StartsWith(_AffixCheck):
    """The response must start with the given text."""

    type: Literal["starts_with"] = "starts_with"
    default_failure_mode: ClassVar[FailureMode] = FailureMode.FORMAT_ERROR
    summary: ClassVar[str] = "Must start with the given text (Markdown-tolerant by default)."

    def check(self, response: str) -> CheckResult:
        body = response.lstrip(_EDGE_MARKUP) if self.loose else response.lstrip()
        expected = self.text.lstrip(_EDGE_MARKUP) if self.loose else self.text
        if self._prepare(body).startswith(self._prepare(expected)):
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
