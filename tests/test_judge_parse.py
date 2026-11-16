import json

import pytest

from evalkit.judge.parse import JudgmentError, parse_judgment
from evalkit.schema import Dimension
from evalkit.taxonomy import FailureMode

ALL = list(Dimension)
IF_FA = [Dimension.INSTRUCTION_FOLLOWING, Dimension.FAITHFULNESS]


def reply(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "rationale": "Does what was asked.",
        "scores": {"instruction_following": 4, "faithfulness": 5},
        "failure_modes": [],
    }
    data.update(overrides)
    return data


def text(**overrides: object) -> str:
    return json.dumps(reply(**overrides))


def problems(raw: str, dims: list[Dimension] = IF_FA) -> list[str]:
    with pytest.raises(JudgmentError) as info:
        parse_judgment(raw, dims)
    return info.value.problems


class TestClean:
    def test_plain_json(self) -> None:
        parsed = parse_judgment(text(), IF_FA)
        assert parsed.rationale == "Does what was asked."
        assert parsed.scores == {Dimension.INSTRUCTION_FOLLOWING: 4, Dimension.FAITHFULNESS: 5}
        assert parsed.failure_modes == []
        assert parsed.repairs == []

    def test_surrounding_whitespace_is_not_a_repair(self) -> None:
        assert parse_judgment(f"\n  {text()}  \n", IF_FA).repairs == []

    def test_failure_modes_keep_order_and_drop_repeats(self) -> None:
        raw = text(failure_modes=["hallucination", "format_error", "hallucination"])
        parsed = parse_judgment(raw, IF_FA)
        assert parsed.failure_modes == [FailureMode.HALLUCINATION, FailureMode.FORMAT_ERROR]

    def test_rationale_is_stripped(self) -> None:
        assert parse_judgment(text(rationale="  ok \n"), IF_FA).rationale == "ok"

    def test_scores_for_unrequested_dimensions_and_extra_keys_are_ignored(self) -> None:
        raw = text(
            scores={"instruction_following": 4, "faithfulness": 5, "helpfulness": 2},
            confidence="high",
        )
        parsed = parse_judgment(raw, IF_FA)
        assert set(parsed.scores) == set(IF_FA)
        assert parsed.repairs == []

    def test_non_ascii_rationale(self) -> None:
        raw = text(rationale="Jawaban memakai “kutipan” dan → panah.")
        assert parse_judgment(raw, IF_FA).rationale == "Jawaban memakai “kutipan” dan → panah."


class TestSyntaxRepairs:
    def test_whole_reply_in_a_code_fence(self) -> None:
        parsed = parse_judgment(f"```json\n{text()}\n```", IF_FA)
        assert parsed.scores[Dimension.FAITHFULNESS] == 5
        assert parsed.repairs == ["removed a code fence"]

    def test_fence_without_language_tag(self) -> None:
        assert parse_judgment(f"```\n{text()}\n```", IF_FA).repairs == ["removed a code fence"]

    def test_fence_inside_prose(self) -> None:
        parsed = parse_judgment(f"Here is my evaluation:\n```json\n{text()}\n```\nThanks.", IF_FA)
        assert parsed.repairs == ["took the JSON from a code fence"]

    def test_object_inside_prose(self) -> None:
        parsed = parse_judgment(f"My evaluation: {text()} I hope this helps.", IF_FA)
        assert parsed.repairs == ["took the JSON object out of surrounding text"]

    def test_first_object_wins_and_nested_objects_are_never_candidates(self) -> None:
        # The scores object alone must not be graded as the reply.
        raw = text() + ' and also {"scores": {"instruction_following": 1}}'
        assert parse_judgment(raw, IF_FA).scores[Dimension.INSTRUCTION_FOLLOWING] == 4

    def test_python_style_object(self) -> None:
        raw = (
            "{'rationale': 'It\\'s fine.', 'scores': {'instruction_following': 4, "
            "'faithfulness': 5}, 'failure_modes': [],}"
        )
        parsed = parse_judgment(raw, IF_FA)
        assert parsed.rationale == "It's fine."
        assert parsed.repairs == ["read a Python-style object"]

    def test_json_with_a_trailing_comma(self) -> None:
        raw = text()[:-1] + ",}"
        parsed = parse_judgment(raw, IF_FA)
        assert parsed.repairs == ["read a Python-style object"]
        assert parsed.scores[Dimension.INSTRUCTION_FOLLOWING] == 4

    def test_trailing_comma_does_not_promote_the_nested_object(self) -> None:
        raw = text()[:-1] + ",}"
        assert parse_judgment(raw, IF_FA).rationale == "Does what was asked."

    def test_python_style_object_in_a_fence(self) -> None:
        raw = "```python\n{'rationale': 'ok', 'scores': {'instruction_following': 4, 'faithfulness': 5}, 'failure_modes': []}\n```"
        parsed = parse_judgment(raw, IF_FA)
        assert parsed.repairs == ["removed a code fence", "read a Python-style object"]

    def test_python_style_object_in_prose(self) -> None:
        raw = "Result: {'rationale': 'ok', 'scores': {'instruction_following': 4, 'faithfulness': 5}, 'failure_modes': [],} done"
        parsed = parse_judgment(raw, IF_FA)
        assert parsed.repairs == [
            "took the object out of surrounding text",
            "read a Python-style object",
        ]

    @pytest.mark.parametrize(("value", "expected"), [("4", 4), (" 3 ", 3), (5.0, 5), ("05", 5)])
    def test_scores_written_as_text_or_float(self, value: object, expected: int) -> None:
        raw = text(scores={"instruction_following": value, "faithfulness": 5})
        parsed = parse_judgment(raw, IF_FA)
        assert parsed.scores[Dimension.INSTRUCTION_FOLLOWING] == expected
        assert parsed.repairs == [f"read score {value!r} as {expected}"]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("Constraint Violation", FailureMode.CONSTRAINT_VIOLATION),
            ("over-refusal", FailureMode.OVER_REFUSAL),
            ("  LANGUAGE_MISMATCH ", FailureMode.LANGUAGE_MISMATCH),
        ],
    )
    def test_failure_mode_spelling(self, value: str, expected: FailureMode) -> None:
        parsed = parse_judgment(text(failure_modes=[value]), IF_FA)
        assert parsed.failure_modes == [expected]
        assert parsed.repairs == [f"read failure mode {value!r} as {expected.value}"]


class TestReasked:
    """Structure problems go back to the judge, all of them at once."""

    def test_no_object_at_all(self) -> None:
        assert problems("I would rate this a 4 out of 5.") == ["the reply contains no JSON object"]

    def test_empty_reply(self) -> None:
        assert problems("") == ["the reply contains no JSON object"]

    def test_truncated_object_reports_the_decode_error(self) -> None:
        [problem] = problems(text()[:40])
        assert problem.startswith("not valid JSON:")
        assert "line 1" in problem

    def test_a_list_is_not_an_object(self) -> None:
        assert problems("[1, 2]") == ["the reply contains no JSON object"]

    def test_a_list_holding_a_broken_object(self) -> None:
        [problem] = problems('["x", {"rationale": ]')
        assert problem.startswith("not valid JSON")

    def test_object_inside_a_list_is_taken(self) -> None:
        parsed = parse_judgment(f"[{text()}]", IF_FA)
        assert parsed.repairs == ["took the JSON object out of surrounding text"]

    def test_every_problem_is_listed(self) -> None:
        raw = json.dumps({"scores": {"instruction_following": 9}, "failure_modes": ["typo"]})
        assert problems(raw) == [
            '"rationale" must be a non-empty string',
            'the score for "instruction_following" must be a whole number from 1 to 5, not 9',
            '"scores" has no score for "faithfulness"',
            'unknown failure mode "typo"; use only: constraint_violation, intent_misread, '
            "hallucination, partial_completion, format_error, over_refusal, language_mismatch",
        ]

    @pytest.mark.parametrize("rationale", ["", "   ", None, 3])
    def test_rationale_must_be_text(self, rationale: object) -> None:
        assert problems(text(rationale=rationale)) == ['"rationale" must be a non-empty string']

    @pytest.mark.parametrize(
        "value", [0, 6, 4.5, "4/5", "four", True, None, [4], {"score": 4}, "", -1, float("nan")]
    )
    def test_bad_scores(self, value: object) -> None:
        raw = text(scores={"instruction_following": value, "faithfulness": 5})
        [problem] = problems(raw)
        assert problem.startswith('the score for "instruction_following" must be a whole number')

    def test_nan_score_is_reported_readably(self) -> None:
        raw = '{"rationale": "x", "scores": {"instruction_following": NaN, "faithfulness": 5}, "failure_modes": []}'
        assert problems(raw) == [
            'the score for "instruction_following" must be a whole number from 1 to 5, not NaN'
        ]

    @pytest.mark.parametrize("scores", [None, [4, 5], "4, 5", 4])
    def test_scores_must_be_an_object(self, scores: object) -> None:
        [problem] = problems(text(scores=scores))
        assert problem == (
            '"scores" must be an object with a whole number from 1 to 5 for '
            '"instruction_following", "faithfulness"'
        )

    def test_flat_scores_are_not_guessed(self) -> None:
        raw = json.dumps(
            {"rationale": "x", "instruction_following": 4, "faithfulness": 5, "failure_modes": []}
        )
        assert problems(raw)[0].startswith('"scores" must be an object')

    @pytest.mark.parametrize("modes", [None, "none", "format_error", {"format_error": True}])
    def test_failure_modes_must_be_a_list(self, modes: object) -> None:
        assert problems(text(failure_modes=modes)) == [
            '"failure_modes" must be a list of failure modes ([] if none applies)'
        ]

    @pytest.mark.parametrize("mode", ["none", "bad_grammar", 3, None, ""])
    def test_unknown_failure_modes(self, mode: object) -> None:
        [problem] = problems(text(failure_modes=[mode]))
        assert problem.startswith("unknown failure mode ")

    def test_missing_failure_modes_key(self) -> None:
        data = reply()
        del data["failure_modes"]
        assert problems(json.dumps(data)) == [
            '"failure_modes" must be a list of failure modes ([] if none applies)'
        ]

    def test_all_four_dimensions_required_by_default(self) -> None:
        assert problems(text(), ALL) == [
            '"scores" has no score for "reasoning_quality"',
            '"scores" has no score for "helpfulness"',
        ]

    def test_python_literal_with_code_is_never_evaluated(self) -> None:
        raw = "{'rationale': __import__('os').getcwd(), 'scores': {}, 'failure_modes': []}"
        [problem] = problems(raw)
        assert problem.startswith("not valid JSON")

    def test_deeply_nested_input_does_not_crash(self) -> None:
        raw = "{" * 5000 + "}" * 5000
        assert problems(raw)[0].startswith("not valid JSON")
