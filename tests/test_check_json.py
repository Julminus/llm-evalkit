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


PERSON = {
    "$defs": {"Person": {"type": "object", "required": ["nama"]}},
    "type": "object",
    "properties": {"ketua": {"$ref": "#/$defs/Person"}},
}


def test_local_refs_resolve() -> None:
    check = JsonSchemaCheck.model_validate({"type": "json_schema", "schema": PERSON})
    assert check.check('{"ketua": {"nama": "Ayu"}}').passed
    assert check.check('{"ketua": {}}').failed


@pytest.mark.parametrize("ref", ["#/$defs/Persn", "https://example.com/person.json"])
def test_unresolvable_ref_rejected_at_load(ref: str) -> None:
    # jsonschema resolves refs lazily, so without this the typo would only
    # surface as a crash when a response reached that property.
    schema = {**PERSON, "properties": {"ketua": {"$ref": ref}}}
    with pytest.raises(ValidationError, match="unresolvable"):
        JsonSchemaCheck.model_validate({"type": "json_schema", "schema": schema})


def test_ref_shaped_data_in_enum_is_not_a_ref() -> None:
    schema = {"enum": [{"$ref": "not-a-reference"}]}
    check = JsonSchemaCheck.model_validate({"type": "json_schema", "schema": schema})
    assert check.check('{"$ref": "not-a-reference"}').passed


def test_ref_relative_to_a_nested_id() -> None:
    schema = {
        "type": "object",
        "properties": {
            "alamat": {
                "$id": "https://example.com/alamat",
                "$defs": {"Kota": {"type": "string"}},
                "type": "object",
                "properties": {"kota": {"$ref": "#/$defs/Kota"}},
            }
        },
    }
    check = JsonSchemaCheck.model_validate({"type": "json_schema", "schema": schema})
    assert check.check('{"alamat": {"kota": "Yogyakarta"}}').passed
    assert check.check('{"alamat": {"kota": 1}}').failed


def test_crlf_code_fence(check: JsonSchemaCheck) -> None:
    assert check.check('```json\r\n{"nama": "Ayu", "umur": 21}\r\n```').passed


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_are_not_json(constant: str) -> None:
    schema = {"type": "object", "properties": {"x": {"type": "number"}}}
    check = JsonSchemaCheck.model_validate({"type": "json_schema", "schema": schema})
    result = check.check(f'{{"x": {constant}}}')
    assert result.failed
    assert f"{constant.lstrip('-')} is not valid JSON" in result.message


def test_embedded_json_prefers_a_candidate_that_matches() -> None:
    check = JsonSchemaCheck.model_validate(
        {"type": "json_schema", "schema": SCHEMA, "allow_surrounding_text": True}
    )
    # "[1]" is a citation; the object after it is the answer.
    assert check.check('Menurut [1], datanya: {"nama": "Ayu", "umur": 21}').passed
    # With no matching candidate, the first object is reported, not the citation.
    result = check.check('Menurut [1], datanya: {"nama": "Ayu"}')
    assert result.failed
    assert "'umur' is a required property" in result.message
    only_arrays = check.check("Menurut [1] dan [2].")
    assert only_arrays.failed
    assert "is not of type 'object'" in only_arrays.message


def test_parse_uses_schema_alias(parse) -> None:
    check = parse({"type": "json_schema", "schema": {"type": "array"}})
    assert isinstance(check, JsonSchemaCheck)
    assert check.json_schema == {"type": "array"}
    assert check.check("[1, 2]").passed
