from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evalkit.checks import LanguageCheck, WordCount
from evalkit.schema import Dimension, EvalCase, Suite, SuiteError, load_suite, parse_suite
from evalkit.taxonomy import FailureMode


def _case(**overrides: Any) -> dict[str, Any]:
    case = {"id": "en-001", "prompt": "Say hi.", "language": "en"}
    case.update(overrides)
    return case


def _suite(*cases: dict[str, Any]) -> dict[str, Any]:
    return {"name": "t", "cases": list(cases) or [_case()]}


class TestEvalCase:
    def test_minimal_case(self) -> None:
        case = EvalCase.model_validate(_case())
        assert case.constraints == []
        assert case.rubric is None
        assert case.expected_language == "en"

    def test_constraints_are_parsed_by_type(self) -> None:
        case = EvalCase.model_validate(_case(constraints=[{"type": "word_count", "max": 10}]))
        assert isinstance(case.constraints[0], WordCount)

    def test_automatic_language_check(self) -> None:
        case = EvalCase.model_validate(_case(language="id"))
        [auto] = case.effective_constraints()
        assert isinstance(auto, LanguageCheck)
        assert auto.expected == "id"

    def test_response_language_overrides_prompt_language(self) -> None:
        # An Indonesian prompt asking for an English translation.
        case = EvalCase.model_validate(_case(language="id", response_language="en"))
        [auto] = case.effective_constraints()
        assert isinstance(auto, LanguageCheck)
        assert auto.expected == "en"

    def test_response_language_any_disables_check(self) -> None:
        case = EvalCase.model_validate(_case(response_language="any"))
        assert case.expected_language is None
        assert case.effective_constraints() == []

    def test_explicit_language_constraint_is_not_duplicated(self) -> None:
        case = EvalCase.model_validate(
            _case(constraints=[{"type": "language", "expected": "en", "min_confidence": 0.9}])
        )
        [only] = case.effective_constraints()
        assert isinstance(only, LanguageCheck)
        assert only.min_confidence == 0.9

    def test_effective_constraints_does_not_mutate_case(self) -> None:
        case = EvalCase.model_validate(_case())
        case.effective_constraints()
        assert case.constraints == []

    def test_tags_are_normalized(self) -> None:
        case = EvalCase.model_validate(_case(tags=[" Format", "format", "JSON "]))
        assert case.tags == ["format", "json"]

    def test_blank_tag_rejected(self) -> None:
        with pytest.raises(ValueError, match="blank"):
            EvalCase.model_validate(_case(tags=["ok", " "]))

    @pytest.mark.parametrize("bad_id", ["", "has space", "-leading-dash", "x" * 81])
    def test_bad_ids_rejected(self, bad_id: str) -> None:
        with pytest.raises(ValidationError, match=r"String should (match pattern|have at most 80)"):
            EvalCase.model_validate(_case(id=bad_id))

    def test_unsupported_language_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Input should be 'en' or 'id'"):
            EvalCase.model_validate(_case(language="ms"))

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValueError, match="Extra inputs"):
            EvalCase.model_validate(_case(promt="typo"))

    def test_rubric_defaults_to_all_dimensions(self) -> None:
        case = EvalCase.model_validate(_case(rubric={"criteria": ["Greets the user."]}))
        assert case.rubric is not None
        assert case.rubric.dimensions == list(Dimension)

    def test_fails_as_override(self) -> None:
        case = EvalCase.model_validate(
            _case(
                constraints=[
                    {
                        "type": "regex",
                        "pattern": r"\d+",
                        "fullmatch": True,
                        "fails_as": "format_error",
                    }
                ]
            )
        )
        result = case.constraints[0].check("about 42")
        assert result.failure_mode is FailureMode.FORMAT_ERROR


class TestSuite:
    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(SuiteError, match="duplicate case ids: a"):
            parse_suite(_suite(_case(id="a"), _case(id="b"), _case(id="a")))

    def test_needs_at_least_one_case(self) -> None:
        with pytest.raises(SuiteError):
            parse_suite({"name": "t", "cases": []})

    def test_lookup_by_id(self) -> None:
        suite = Suite.model_validate(_suite(_case(id="a"), _case(id="b")))
        assert suite.case("b").id == "b"
        with pytest.raises(KeyError):
            suite.case("zzz")

    def test_non_mapping_rejected(self) -> None:
        with pytest.raises(SuiteError, match="mapping"):
            parse_suite(["not", "a", "suite"])

    def test_error_message_names_the_case(self) -> None:
        bad = _case(id="id-007", constraints=[{"type": "word_count"}])
        with pytest.raises(SuiteError) as info:
            parse_suite(_suite(_case(), bad), source="suite.yaml")
        message = str(info.value)
        assert message.startswith("suite.yaml: 1 validation error(s)")
        assert "cases[1] (id=id-007).constraints[0]" in message
        assert "at least one of 'min' or 'max'" in message

    def test_error_message_without_case_id(self) -> None:
        bad = {"prompt": "no id here", "language": "en"}
        with pytest.raises(SuiteError) as info:
            parse_suite(_suite(bad))
        assert "cases[0].id: Field required" in str(info.value)

    def test_error_at_root(self) -> None:
        with pytest.raises(SuiteError, match="name: Field required"):
            parse_suite({"cases": [_case()]})


class TestLoadSuite:
    def test_loads_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "s.yaml"
        path.write_text(
            "name: demo\ncases:\n  - id: id-001\n    language: id\n    prompt: Halo!\n",
            encoding="utf-8",
        )
        suite = load_suite(path)
        assert suite.name == "demo"
        assert suite.cases[0].language == "id"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteError, match="cannot read file"):
            load_suite(tmp_path / "nope.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("name: [unclosed\n", encoding="utf-8")
        with pytest.raises(SuiteError, match="invalid YAML"):
            load_suite(path)

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(SuiteError, match="mapping"):
            load_suite(path)

    def test_duplicate_key_rejected(self, tmp_path: Path) -> None:
        # Plain YAML keeps the last value, which would silently drop the first list.
        path = tmp_path / "dup.yaml"
        path.write_text(
            "name: demo\ncases:\n  - id: id-001\n    language: id\n    prompt: Halo!\n"
            "    constraints: [{type: word_count, max: 5}]\n"
            "    constraints: [{type: word_count, min: 1}]\n",
            encoding="utf-8",
        )
        with pytest.raises(
            SuiteError, match=r"duplicate key 'constraints' \(first defined on line 6\)"
        ):
            load_suite(path)

    def test_duplicate_key_in_a_flow_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "dup.yaml"
        path.write_text("{name: a, name: b, cases: []}\n", encoding="utf-8")
        with pytest.raises(SuiteError, match="duplicate key 'name'"):
            load_suite(path)

    def test_merge_keys_may_be_overridden(self, tmp_path: Path) -> None:
        path = tmp_path / "merge.yaml"
        path.write_text(
            "base: &base {language: id, prompt: Halo!}\n"
            "name: demo\ncases:\n  - <<: *base\n    id: id-001\n    prompt: Hai!\n",
            encoding="utf-8",
        )
        with pytest.raises(SuiteError, match="base"):  # "base" is an unknown top-level key
            load_suite(path)
        path.write_text(
            "name: demo\ncases:\n  - &c {id: id-001, language: id, prompt: Halo!}\n"
            "  - <<: *c\n    id: id-002\n    prompt: Hai!\n",
            encoding="utf-8",
        )
        suite = load_suite(path)
        assert [c.prompt for c in suite.cases] == ["Halo!", "Hai!"]

    def test_utf8_bom_is_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "s.yaml"
        path.write_bytes(
            b"\xef\xbb\xbfname: demo\ncases:\n  - {id: id-001, language: id, prompt: Halo!}\n"
        )
        assert load_suite(path).name == "demo"

    def test_utf16_file_gets_a_clear_error(self, tmp_path: Path) -> None:
        path = tmp_path / "s.yaml"
        path.write_text("name: demo\n", encoding="utf-16")
        with pytest.raises(SuiteError, match="UTF-16, not UTF-8"):
            load_suite(path)
