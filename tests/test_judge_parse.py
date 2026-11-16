import json
import time

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

    def test_a_dimension_name_is_not_a_failure_mode(self) -> None:
        # Measured: both local judges sometimes listed "reasoning_quality" here.
        [problem] = problems(text(failure_modes=["reasoning_quality"]))
        assert problem.startswith('unknown failure mode "reasoning_quality"')

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


class TestOddReplies:
    """Found in the post-S3 review: none of these may raise anything but JudgmentError."""

    def test_deep_array_nesting(self) -> None:
        raw = '{"rationale": "x", "scores": ' + "[" * 5000
        [problem] = problems(raw)
        assert problem.startswith("not valid JSON")

    def test_number_too_long_for_python(self) -> None:
        raw = '{"rationale": "x", "scores": {"instruction_following": ' + "1" * 4400 + "}}"
        [problem] = problems(raw)
        assert problem == "not valid JSON: a number is too long"

    def test_score_text_too_long_for_python(self) -> None:
        [problem] = problems(text(scores={"instruction_following": "1" * 4400, "faithfulness": 5}))
        assert problem.startswith('the score for "instruction_following" must be a whole number')
        assert len(problem) < 160  # the value is shortened in the message

    def test_lone_surrogate_escape_becomes_a_replacement_character(self) -> None:
        raw = text(rationale="placeholder").replace("placeholder", "Nice \\ud83d answer")
        parsed = parse_judgment(raw, IF_FA)
        assert parsed.rationale == "Nice \ufffd answer"
        parsed.rationale.encode("utf-8")  # writable

    def test_unprintable_value_in_a_python_literal(self) -> None:
        raw = (
            "{'rationale': 'x', 'scores': {'instruction_following': {(1, 2): 4}, "
            "'faithfulness': 5}, 'failure_modes': []}"
        )
        [problem] = problems(raw)
        assert problem.endswith("not a dict that can't be shown")

    def test_duplicate_json_key_is_reasked(self) -> None:
        low = '"scores": {"instruction_following": 1, "faithfulness": 1}'
        high = '"scores": {"instruction_following": 4, "faithfulness": 5}'
        raw = '{"rationale": "x", ' + low + ", " + high + ', "failure_modes": []}'
        assert problems(raw) == ['the key "scores" appears twice; give each key once']

    def test_duplicate_python_key_is_not_read(self) -> None:
        raw = (
            "{'rationale': 'x', 'scores': {'instruction_following': 1, 'faithfulness': 1}, "
            "'scores': {'instruction_following': 4, 'faithfulness': 5}, 'failure_modes': []}"
        )
        [problem] = problems(raw)
        assert problem.startswith("not valid JSON")


class TestChoosingTheObject:
    """The candidate with the most expected keys wins; on a tie, the last one."""

    def test_an_invalid_brace_before_the_verdict(self) -> None:
        parsed = parse_judgment("Format: {rationale, scores}.\n" + text(), IF_FA)
        assert parsed.rationale == "Does what was asked."

    def test_echoed_template_before_the_verdict(self) -> None:
        template = '{"rationale": "<2 to 4 sentences>", "scores": {"instruction_following": <1-5>}}'
        parsed = parse_judgment(template + "\n" + text(), IF_FA)
        assert parsed.scores[Dimension.INSTRUCTION_FOLLOWING] == 4

    def test_quoted_json_response_before_the_verdict(self) -> None:
        quoted = '```json\n{"name": "Ayu", "age": 21}\n```'
        parsed = parse_judgment(f"The response was:\n{quoted}\nMy verdict: {text()}", IF_FA)
        assert parsed.rationale == "Does what was asked."
        assert parsed.repairs == ["took the JSON object out of surrounding text"]

    def test_two_verdicts_use_the_first_and_say_so(self) -> None:
        # One rule, visible in the repairs (the round-2 rule compared the two
        # objects and broke on reworded duplicates; round 3 replaced it).
        first = text(rationale="first", scores={"instruction_following": 2, "faithfulness": 2})
        parsed = parse_judgment(f"<think>{first}</think>\n{text()}", IF_FA)
        assert parsed.rationale == "first"
        assert parsed.repairs == [
            "took the JSON object out of surrounding text",
            "the reply has 2 judgments; used the first",
        ]

    def test_a_corrected_python_style_verdict_still_counts_as_a_second_one(self) -> None:
        # Found in the round-3 review: a later Python-style dict was never looked at.
        fixed = "{'rationale': 'Weak.', 'scores': {'instruction_following': 2, 'faithfulness': 2}, 'failure_modes': [],}"
        parsed = parse_judgment(
            f"```json\n{text()}\n```\nCorrection:\n```json\n{fixed}\n```", IF_FA
        )
        assert parsed.rationale == "Does what was asked."
        assert parsed.repairs[-1] == "the reply has 2 judgments; used the first"

    def test_the_same_verdict_twice_is_fine(self) -> None:
        parsed = parse_judgment(f"Verdict: {text()}\nAgain: {text()}", IF_FA)
        assert parsed.rationale == "Does what was asked."

    def test_a_partial_object_after_the_verdict_loses(self) -> None:
        raw = text() + ' and also {"scores": {"instruction_following": 1}}'
        assert parse_judgment(raw, IF_FA).scores[Dimension.INSTRUCTION_FOLLOWING] == 4

    def test_objects_with_none_of_the_keys_are_never_chosen(self) -> None:
        assert problems('Example: {"a": 1}. That is all.') == [
            'the reply has JSON, but no object with the keys "rationale", "scores" and '
            '"failure_modes"'
        ]

    def test_capitalised_keys_are_named_in_the_problem(self) -> None:
        raw = 'My evaluation: {"Rationale": "x", "Scores": {}, "Failure_modes": []}'
        [problem] = problems(raw)
        assert '"rationale", "scores" and "failure_modes"' in problem

    def test_a_broken_whole_object_reports_why_it_does_not_parse(self) -> None:
        # Its inner fragments ({} here) must not hide the decode error.
        [problem] = problems('{"rationale": "x" "scores": {}, "failure_modes": []}')
        assert problem.startswith("not valid JSON")

    def test_duplicate_key_inside_prose_is_named(self) -> None:
        raw = 'Verdict: {"rationale": "x", "rationale": "y", "scores": {}, "failure_modes": []}'
        assert problems(raw) == ['the key "rationale" appears twice; give each key once']


class TestBounds:
    """Found in the fix review: the scan was quadratic on replies full of "{"."""

    def test_overlong_reply_is_reasked(self) -> None:
        [problem] = problems("x" * 30_001)
        assert problem == "the reply is 30,001 characters long; reply with only the JSON object"

    @pytest.mark.parametrize("filler", ["{" * 29_000, "{x} " * 7_000, "```\n{}\n```\n" * 2_000])
    def test_adversarial_replies_parse_quickly(self, filler: str) -> None:
        start = time.perf_counter()
        with pytest.raises(JudgmentError):
            parse_judgment(filler, IF_FA)
        assert time.perf_counter() - start < 1.0

    @pytest.mark.parametrize(
        "before",
        [
            "Check: $\\frac{1}{2} + x^{2} = \\sqrt{y}$. " * 20,
            "```js\n" + "if (a) { b(); } " * 75 + "\n```\n",
            "{x} " * 500,
        ],
    )
    def test_a_verdict_after_brace_heavy_text_is_found(self, before: str) -> None:
        # Found in the round-3 review: a cap of 64 failed "{" hid these. Only a
        # "{" followed by a quote or "}" can start JSON, so these aren't tried.
        parsed = parse_judgment(before + text(), IF_FA)
        assert parsed.rationale == "Does what was asked."

    def test_a_literal_python_cannot_evaluate_does_not_crash(self) -> None:
        # Found in the round-3 review: OverflowError escaped ast.literal_eval.
        raw = "{'rationale': 1" + "0" * 400 + "+1j, 'scores': {}, 'failure_modes': []}"
        [problem] = problems(raw)
        assert problem.startswith("not valid JSON")

    def test_a_broken_verdict_after_prose_reports_its_own_error(self) -> None:
        raw = 'Here is my evaluation:\n{"rationale": "It says "hi" well.", "scores": {}, "failure_modes": []}'
        [problem] = problems(raw)
        assert problem.startswith("not valid JSON: Expecting")

    def test_any_unexpected_error_becomes_a_problem(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from evalkit.judge import parse

        def boom(text: str) -> None:
            raise RuntimeError("bug")

        monkeypatch.setattr(parse, "_extract", boom)
        assert problems(text()) == ["the reply could not be read (RuntimeError)"]

    def test_duplicate_key_spelled_as_a_lone_surrogate(self) -> None:
        # Found in the fix review: the raw key went into the re-ask and crashed it.
        raw = (
            '{"rationale": "ok", "scores": {"instruction_following": 4, "faithfulness": 5}, '
            '"failure_modes": [], "\\ud83d": 1, "\\ud83d": 2}'
        )
        [problem] = problems(raw)
        assert problem.endswith("appears twice; give each key once")
        problem.encode("utf-8")
