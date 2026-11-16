import pytest
from pydantic import ValidationError

from evalkit.checks import JsonSchemaCheck
from evalkit.taxonomy import FailureMode

SCHEMA = {
    "type": "object",
    "properties": {
        "nama": {"type": "string"},
        "umur": {"type": "integer", "minimum": 0},
        "hobi": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["nama", "umur"],
    "additionalProperties": False,
}


@pytest.fixture
def check() -> JsonSchemaCheck:
    return JsonSchemaCheck.model_validate({"type": "json_schema", "schema": SCHEMA})


def test_valid_json(check: JsonSchemaCheck) -> None:
    result = check.check('{"nama": "Ayu", "umur": 21, "hobi": ["catur"]}')
    assert result.passed
    assert result.observed == {"errors": 0, "found_in": "whole response"}


def test_whitespace_around_json_is_fine(check: JsonSchemaCheck) -> None:
    assert check.check('\n  {"nama": "Ayu", "umur": 21}  \n').passed


def test_not_json(check: JsonSchemaCheck) -> None:
    result = check.check("Nama: Ayu, umur 21")
    assert result.failed
    assert result.failure_mode is FailureMode.FORMAT_ERROR
    assert result.message.startswith("not valid JSON")
    assert "line 1" in result.message


def test_schema_violation_reports_path(check: JsonSchemaCheck) -> None:
    result = check.check('{"nama": "Ayu", "umur": "21", "hobi": [1]}')
    assert result.failed
    assert result.observed["errors"] == 2
    assert "$.hobi[0]" in result.message
    assert "$.umur" in result.message


def test_missing_required_and_extra_key(check: JsonSchemaCheck) -> None:
    result = check.check('{"nama": "Ayu", "kota": "Yogyakarta"}')
    assert result.failed
    assert "'umur' is a required property" in result.message
    assert "Additional properties" in result.message


def test_error_list_is_truncated() -> None:
    schema = {"type": "array", "items": {"type": "integer"}}
    check = JsonSchemaCheck.model_validate({"type": "json_schema", "schema": schema})
    result = check.check('["a", "b", "c", "d", "e"]')
    assert result.observed["errors"] == 5
    assert "(+2 more)" in result.message


def test_code_fence_allowed_by_default(check: JsonSchemaCheck) -> None:
    result = check.check('```json\n{"nama": "Ayu", "umur": 21}\n```')
    assert result.passed
    assert result.observed["found_in"] == "code fence"


def test_code_fence_can_be_forbidden() -> None:
    check = JsonSchemaCheck.model_validate(
        {"type": "json_schema", "schema": SCHEMA, "allow_code_fence": False}
    )
    result = check.check('```json\n{"nama": "Ayu", "umur": 21}\n```')
    assert result.failed
    assert "code fence" in result.message


def test_invalid_json_inside_fence(check: JsonSchemaCheck) -> None:
    result = check.check('```json\n{"nama": "Ayu", umur: 21}\n```')
    assert result.failed
    assert result.message.startswith("fenced block is not valid JSON")


def test_surrounding_text_rejected_by_default(check: JsonSchemaCheck) -> None:
    result = check.check('Berikut datanya:\n```json\n{"nama": "Ayu", "umur": 21}\n```')
    assert result.failed


def test_surrounding_text_allowed() -> None:
    check = JsonSchemaCheck.model_validate(
        {"type": "json_schema", "schema": SCHEMA, "allow_surrounding_text": True}
    )
    fenced = check.check(
        'Berikut datanya:\n```json\n{"nama": "Ayu", "umur": 21}\n```\nSemoga membantu.'
    )
    assert fenced.passed
    assert fenced.observed["found_in"] == "code fence"
    embedded = check.check('Hasil: {"nama": "Ayu", "umur": 21} selesai.')
    assert embedded.passed
    assert embedded.observed["found_in"] == "embedded in text"
    # A brace that does not start valid JSON is skipped; the next candidate is tried.
    assert check.check('Pakai {kurung} lalu {"nama": "Ayu", "umur": 21}').passed
    missing = check.check("Tidak ada JSON di sini.")
    assert missing.failed
    assert "no JSON object or array" in missing.message


def test_format_keyword_is_checked() -> None:
    schema = {"type": "object", "properties": {"tanggal": {"type": "string", "format": "date"}}}
    check = JsonSchemaCheck.model_validate({"type": "json_schema", "schema": schema})
    assert check.check('{"tanggal": "2026-11-16"}').passed
    assert check.check('{"tanggal": "16/11/2026"}').failed


def test_invalid_schema_rejected_at_load() -> None:
    with pytest.raises(ValidationError, match="invalid JSON Schema"):
        JsonSchemaCheck.model_validate({"type": "json_schema", "schema": {"type": "strng"}})


def test_parse_uses_schema_alias(parse) -> None:
    check = parse({"type": "json_schema", "schema": {"type": "array"}})
    assert isinstance(check, JsonSchemaCheck)
    assert check.json_schema == {"type": "array"}
    assert check.check("[1, 2]").passed
