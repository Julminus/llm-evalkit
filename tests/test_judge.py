import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import ClassVar

import pytest

from evalkit.cache import ResponseCache
from evalkit.checks import Status
from evalkit.judge import (
    JUDGE_PROMPT_VERSION,
    Judge,
    JudgeRecord,
    JudgeSummary,
    judge_suite,
    responses_digest,
)
from evalkit.judge.core import make_judge_run_id
from evalkit.prompts import Prompt
from evalkit.providers import Completion, GenerationParams, Provider, ProviderError
from evalkit.runner import RetryPolicy
from evalkit.schema import Dimension, Suite, parse_suite
from evalkit.taxonomy import FailureMode

_REQUEST_RE = re.compile(r"<request>\n(.*?)\n</request>", re.DOTALL)


def good(**scores: int) -> str:
    return json.dumps(
        {
            "rationale": "ok",
            "scores": scores or {d.value: 4 for d in Dimension},
            "failure_modes": [],
        }
    )


def make_suite(n: int = 3, **case_fields: object) -> Suite:
    return parse_suite(
        {
            "name": "t",
            "cases": [
                {"id": f"c{i}", "language": "en", "prompt": f"prompt {i}", **case_fields}
                for i in range(n)
            ],
        }
    )


class FakeJudge(Provider):
    """Answers each judge prompt from a per-request script (keyed by the request text)."""

    name = "fakejudge"
    supported_params: ClassVar[frozenset[str]] = frozenset({"temperature", "max_tokens"})

    def __init__(
        self,
        script: dict[str, list[str | Exception]] | None = None,
        *,
        finish: str = "stop",
        default: str | None = None,
    ) -> None:
        super().__init__("j")
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.finish = finish
        self.default = default if default is not None else good()
        self.prompts: defaultdict[str, list[Prompt]] = defaultdict(list)
        self.in_flight = 0
        self.max_in_flight = 0

    async def generate(self, prompt: Prompt, params: GenerationParams) -> Completion:
        match = _REQUEST_RE.search(prompt.user)
        assert match is not None
        key = match.group(1)
        self.prompts[key].append(prompt)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.001)
            steps = self.script.get(key)
            outcome = steps.pop(0) if steps else self.default
            if isinstance(outcome, Exception):
                raise outcome
            return Completion(
                text=outcome,
                model="j",
                finish_reason=self.finish,  # type: ignore[arg-type]
                input_tokens=10,
                output_tokens=5,
            )
        finally:
            self.in_flight -= 1


def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def judge_one(provider: Provider, response: str = "An answer.", **kwargs: object) -> JudgeRecord:
    suite = make_suite(1)
    judge = Judge(provider, **kwargs)  # type: ignore[arg-type]
    return run(judge.judge_case(suite.cases[0], response))


class TestJudgeCase:
    def test_clean_reply(self) -> None:
        record = judge_one(FakeJudge())
        assert record.ok
        assert record.error is None
        assert record.scores == dict.fromkeys(Dimension, 4)
        assert record.rationale == "ok"
        assert record.parse_outcome == "clean"
        assert (record.requests, record.reasks, record.cached) == (1, 0, False)
        assert (record.input_tokens, record.output_tokens) == (10, 5)
        assert record.replies == [good()]
        assert record.judge == "fakejudge:j"
        assert record.check_status is Status.SKIP  # no constraints; language check skips 2 words

    def test_local_repair_needs_no_second_request(self) -> None:
        provider = FakeJudge({"prompt 0": [f"```json\n{good()}\n```"]})
        record = judge_one(provider)
        assert record.ok
        assert record.parse_outcome == "repaired"
        assert record.repairs == ["removed a code fence"]
        assert record.requests == 1

    def test_reask_shows_the_bad_reply_and_the_problems(self) -> None:
        provider = FakeJudge({"prompt 0": ["I'd give it a 4.", good()]})
        record = judge_one(provider)
        assert record.ok
        assert record.parse_outcome == "reasked"
        assert (record.reasks, record.requests) == (1, 2)
        assert record.replies == ["I'd give it a 4.", good()]
        assert (record.input_tokens, record.output_tokens) == (20, 10)
        first, second = provider.prompts["prompt 0"]
        assert second.user.startswith(first.user)
        assert "<previous_reply>\nI'd give it a 4.\n</previous_reply>" in second.user
        assert "- the reply contains no JSON object" in second.user

    def test_each_reask_starts_from_the_original_prompt(self) -> None:
        provider = FakeJudge({"prompt 0": ["bad one", "bad two", good()]})
        record = judge_one(provider)
        assert record.reasks == 2
        assert record.ok
        original, _, third = provider.prompts["prompt 0"]
        assert third.user.startswith(original.user)
        assert "bad two" in third.user
        assert "bad one" not in third.user

    def test_gives_up_after_max_reasks(self) -> None:
        provider = FakeJudge(default="no json")
        record = judge_one(provider, max_reasks=2)
        assert not record.ok
        assert record.error == (
            "unusable judge reply (after 2 re-asks): the reply contains no JSON object"
        )
        assert record.parse_outcome == "unusable"
        assert (record.requests, record.reasks, len(record.replies)) == (3, 2, 3)
        assert record.scores == {}
        assert record.rationale is None

    def test_max_reasks_zero_never_reasks(self) -> None:
        record = judge_one(FakeJudge(default="no json"), max_reasks=0)
        assert record.error == "unusable judge reply: the reply contains no JSON object"
        assert record.requests == 1

    def test_long_response_is_cut_for_the_judge_but_not_for_the_checks(self) -> None:
        suite = make_suite(1, constraints=[{"type": "word_count", "max": 100}])
        response = "kata " * 500  # 2,500 characters, 500 words
        provider = FakeJudge()
        judge = Judge(provider, max_response_chars=50)
        record = run(judge.judge_case(suite.cases[0], response))
        assert record.response_cut
        assert record.check_status is Status.FAIL  # the checks saw all 500 words
        [prompt] = provider.prompts["prompt 0"]
        assert "The response is 2,499 characters long. Only the first 50 are shown" in prompt.user

    def test_short_response_is_not_cut(self) -> None:
        record = judge_one(FakeJudge(), "short", max_response_chars=50)
        assert not record.response_cut

    @pytest.mark.parametrize("bad", [0, -5])
    def test_max_response_chars_must_be_positive(self, bad: int) -> None:
        with pytest.raises(ValueError, match="max_response_chars"):
            Judge(FakeJudge(), max_response_chars=bad)

    def test_negative_max_reasks_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_reasks"):
            Judge(FakeJudge(), max_reasks=-1)

    def test_truncated_reply_asks_for_a_shorter_rationale(self) -> None:
        provider = FakeJudge(default='{"rationale": "very long', finish="length")
        record = judge_one(provider, max_reasks=1)
        assert not record.ok
        assert "cut off at the token limit" in (record.error or "")
        assert "cut off at the token limit" in provider.prompts["prompt 0"][1].user

    def test_refusal_is_an_error_without_reasks(self) -> None:
        record = judge_one(FakeJudge(finish="refusal", default="I can't help."))
        assert record.error == "the judge refused"
        assert record.requests == 1
        assert record.reasks == 0

    def test_request_failure_after_retries(self) -> None:
        fail = ProviderError("fakejudge: HTTP 503", retryable=True, status=503)

        async def no_sleep(_: float) -> None:
            return None

        provider = FakeJudge({"prompt 0": [fail, fail]})
        record = judge_one(provider, retry=RetryPolicy(max_retries=1), sleep=no_sleep)
        assert record.error == "fakejudge: HTTP 503 (after 2 attempts)"
        assert record.parse_outcome == "request_failed"
        assert record.requests == 2
        assert record.replies == []

    def test_request_failure_on_a_reask_keeps_earlier_replies(self) -> None:
        provider = FakeJudge(
            {"prompt 0": ["garbage", ProviderError("fakejudge: HTTP 400", retryable=False)]}
        )
        record = judge_one(provider)
        assert record.error == "fakejudge: HTTP 400"
        assert record.replies == ["garbage"]
        assert record.reasks == 1
        # The case ended on a failed request, and the first reply's tokens still count.
        assert record.parse_outcome == "request_failed"
        assert (record.input_tokens, record.output_tokens) == (10, 5)

    def test_rubric_dimensions_are_the_ones_required(self) -> None:
        suite = make_suite(1, rubric={"dimensions": ["faithfulness"]})
        provider = FakeJudge(default=good(faithfulness=2))
        record = run(Judge(provider).judge_case(suite.cases[0], "x"))
        assert record.scores == {Dimension.FAITHFULNESS: 2}

    def test_check_failures_are_kept_apart_from_the_judges(self) -> None:
        suite = make_suite(1, constraints=[{"type": "word_count", "max": 2}])
        reply = json.dumps(
            {
                "rationale": "too long and invented",
                "scores": {d.value: 2 for d in Dimension},
                "failure_modes": ["hallucination", "constraint_violation"],
            }
        )
        record = run(Judge(FakeJudge(default=reply)).judge_case(suite.cases[0], "one two three"))
        assert record.check_status is Status.FAIL
        assert record.check_failure_modes == [FailureMode.CONSTRAINT_VIOLATION]
        assert record.judge_failure_modes == [
            FailureMode.HALLUCINATION,
            FailureMode.CONSTRAINT_VIOLATION,
        ]
        assert record.failure_modes == [FailureMode.CONSTRAINT_VIOLATION, FailureMode.HALLUCINATION]
        assert record.model_dump(mode="json")["failure_modes"] == [
            "constraint_violation",
            "hallucination",
        ]

    def test_judge_tags_for_checkable_modes_do_not_count(self) -> None:
        # Code owns constraint_violation, format_error and language_mismatch.
        suite = make_suite(1, constraints=[{"type": "word_count", "max": 5}])
        reply = json.dumps(
            {
                "rationale": "too long",
                "scores": {d.value: 3 for d in Dimension},
                "failure_modes": ["constraint_violation", "language_mismatch", "intent_misread"],
            }
        )
        record = run(Judge(FakeJudge(default=reply)).judge_case(suite.cases[0], "one two"))
        assert record.check_failure_modes == []
        assert record.judge_failure_modes == [
            FailureMode.CONSTRAINT_VIOLATION,
            FailureMode.LANGUAGE_MISMATCH,
            FailureMode.INTENT_MISREAD,
        ]
        assert record.failure_modes == [FailureMode.INTENT_MISREAD]
        assert record.parse_outcome == "clean"

    def test_a_check_retagged_as_a_judge_mode_can_agree_with_the_judge(self) -> None:
        suite = make_suite(
            1,
            constraints=[
                {"type": "required_phrases", "phrases": ["Jakarta"], "fails_as": "hallucination"}
            ],
        )
        reply = json.dumps(
            {
                "rationale": "wrong city",
                "scores": {d.value: 2 for d in Dimension},
                "failure_modes": ["hallucination"],
            }
        )
        record = run(Judge(FakeJudge(default=reply)).judge_case(suite.cases[0], "Bandung."))
        assert record.failure_modes == [FailureMode.HALLUCINATION]
        summary = JudgeSummary(records=[record])
        assert summary.failure_modes() == {
            FailureMode.HALLUCINATION: {"check": 0, "judge": 0, "both": 1}
        }

    @pytest.mark.parametrize("include", [True, False])
    def test_include_checks_controls_the_prompt(self, include: bool) -> None:
        suite = make_suite(1, constraints=[{"type": "word_count", "max": 2}])
        provider = FakeJudge()
        run(Judge(provider, include_checks=include).judge_case(suite.cases[0], "a b c"))
        [prompt] = provider.prompts["prompt 0"]
        assert ("<automated_checks>" in prompt.user) is include


class TestCache:
    def test_rejudging_replays_every_reply_including_reasks(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        provider = FakeJudge({"prompt 0": ["garbage", good()]})
        first = judge_one(provider, cache=cache)
        again = FakeJudge(default="never used")
        second = judge_one(again, cache=cache)
        assert first.ok
        assert second.ok
        assert second.cached
        assert second.requests == 0
        assert second.reasks == 1
        assert second.replies == first.replies
        assert again.prompts == {}

    def test_partly_cached(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        judge_one(FakeJudge(default="garbage"), cache=cache, max_reasks=0)
        record = judge_one(FakeJudge({"prompt 0": [good()]}), cache=cache, max_reasks=1)
        assert record.ok
        assert not record.cached
        assert record.requests == 1


class TestJudgeAll:
    def test_order_and_concurrency_limit(self) -> None:
        suite = make_suite(6)
        provider = FakeJudge()
        records = run(
            Judge(provider, concurrency=2).judge_all([(c, "x") for c in reversed(suite.cases)])
        )
        assert [r.id for r in records] == [f"c{i}" for i in reversed(range(6))]
        assert provider.max_in_flight == 2

    def test_two_answers_to_one_case_stay_apart(self) -> None:
        # Found in the post-S3 review: records were keyed by case id.
        suite = make_suite(1)
        provider = FakeJudge()
        case = suite.cases[0]

        async def go() -> list[JudgeRecord]:
            judge = Judge(provider)
            return await judge.judge_all([(case, "answer A"), (case, "answer B")])

        records = run(go())
        assert len(records) == 2
        [first, second] = provider.prompts["prompt 0"]
        assert "answer A" in first.user
        assert "answer B" in second.user
        assert records[0] is not records[1]

    def test_reasks_share_the_concurrency_limit(self) -> None:
        suite = make_suite(4)
        provider = FakeJudge({f"prompt {i}": ["bad", "bad", good()] for i in range(4)})
        records = run(Judge(provider, concurrency=1).judge_all([(c, "x") for c in suite.cases]))
        assert all(r.ok and r.reasks == 2 for r in records)
        assert provider.max_in_flight == 1


def records_for_summary() -> list[JudgeRecord]:
    base = {"judge": "fake:j", "tags": []}
    return [
        JudgeRecord(
            id="a",
            language="en",
            check_status=Status.FAIL,
            check_failure_modes=[FailureMode.CONSTRAINT_VIOLATION],
            judge_failure_modes=[FailureMode.CONSTRAINT_VIOLATION, FailureMode.HALLUCINATION],
            scores={Dimension.INSTRUCTION_FOLLOWING: 2, Dimension.FAITHFULNESS: 3},
            rationale="r",
            **base,
        ),
        JudgeRecord(
            id="b",
            language="id",
            check_status=Status.FAIL,
            check_failure_modes=[FailureMode.FORMAT_ERROR],
            scores={Dimension.INSTRUCTION_FOLLOWING: 4, Dimension.FAITHFULNESS: 5},
            rationale="r",
            repairs=["removed a code fence"],
            **base,
        ),
        JudgeRecord(
            id="c",
            language="id",
            check_status=Status.PASS,
            scores={Dimension.INSTRUCTION_FOLLOWING: 5},
            rationale="r",
            reasks=1,
            **base,
        ),
        JudgeRecord(
            id="d",
            language="en",
            check_status=Status.FAIL,
            check_failure_modes=[FailureMode.LANGUAGE_MISMATCH],
            error="unusable",
            replies=["x"],
            **base,
        ),
        JudgeRecord(id="e", language="en", check_status=Status.PASS, error="503", **base),
    ]


class TestSummary:
    def test_aggregates(self) -> None:
        summary = JudgeSummary(records=records_for_summary(), missing=["z"], unknown=["q"])
        assert [r.id for r in summary.judged] == ["a", "b", "c"]
        assert [r.id for r in summary.errors] == ["d", "e"]
        assert summary.mean_scores() == {
            Dimension.INSTRUCTION_FOLLOWING: 3.667,
            Dimension.REASONING_QUALITY: None,
            Dimension.FAITHFULNESS: 4.0,
            Dimension.HELPFULNESS: None,
        }
        assert summary.mean_scores("id")[Dimension.INSTRUCTION_FOLLOWING] == 4.5
        assert summary.languages() == ["en", "id"]
        assert summary.score_counts()[Dimension.INSTRUCTION_FOLLOWING] == {
            1: 0,
            2: 1,
            3: 0,
            4: 1,
            5: 1,
        }
        # Errors don't count: "d"'s language mismatch has no judge verdict to compare.
        assert summary.failure_modes() == {
            FailureMode.CONSTRAINT_VIOLATION: {"check": 1, "judge": 0, "both": 0},
            FailureMode.HALLUCINATION: {"check": 0, "judge": 1, "both": 0},
            FailureMode.FORMAT_ERROR: {"check": 1, "judge": 0, "both": 0},
        }
        # "a": judge and check agree on constraint_violation; "b": only the check.
        assert summary.judge_vs_checks() == {
            FailureMode.CONSTRAINT_VIOLATION: {"both": 1, "judge_only": 0, "checks_only": 0},
            FailureMode.FORMAT_ERROR: {"both": 0, "judge_only": 0, "checks_only": 1},
        }
        assert summary.parse_outcomes() == {
            "clean": 1,
            "repaired": 1,
            "reasked": 1,
            "unusable": 1,
            "request_failed": 1,
        }
        data = summary.to_dict()
        assert data["judged"] == 3
        assert data["errors"] == 2
        assert data["mean_scores"]["instruction_following"] == 3.667
        assert data["score_counts"]["faithfulness"] == {"1": 0, "2": 0, "3": 1, "4": 0, "5": 1}
        assert data["failure_modes"]["format_error"] == {"check": 1, "judge": 0, "both": 0}
        assert data["judge_vs_checks"]["format_error"]["checks_only"] == 1
        assert data["missing"] == ["z"]
        assert data["unknown"] == ["q"]
        json.dumps(data)  # JSON-ready

    def test_empty(self) -> None:
        summary = JudgeSummary(records=[])
        assert all(v is None for v in summary.mean_scores().values())
        assert summary.failure_modes() == {}
        assert summary.to_dict()["judged"] == 0


class TestJudgeSuite:
    def test_writes_run_dir_and_manifest(self, tmp_path: Path) -> None:
        suite = make_suite(3)
        responses = {"c0": "a", "c2": "b", "zz": "stray"}
        provider = FakeJudge({"prompt 2": ["garbage", good()]})
        seen: list[str] = []
        result = run(
            judge_suite(
                suite,
                responses,
                Judge(provider, params=GenerationParams(temperature=0, max_tokens=99)),
                out_dir=tmp_path,
                run_id="j1",
                suite_path=Path("s.yaml"),
                responses_path=Path("r.jsonl"),
                on_record=lambda r: seen.append(r.id),
            )
        )
        assert result.run_id == "j1"
        assert result.run_dir == tmp_path / "j1"
        assert [r.id for r in result.records] == ["c0", "c2"]
        assert sorted(seen) == ["c0", "c2"]
        assert result.summary.missing == ["c1"]
        assert result.summary.unknown == ["zz"]
        lines = (tmp_path / "j1" / "judgments.jsonl").read_text(encoding="utf-8").splitlines()
        assert [JudgeRecord.model_validate_json(line).id for line in lines] == ["c0", "c2"]
        manifest = json.loads((tmp_path / "j1" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["kind"] == "judge"
        assert manifest["status"] == "complete"
        assert manifest["judge"] == {"provider": "fakejudge", "model": "j", "spec": "fakejudge:j"}
        assert manifest["include_checks"] is True
        assert manifest["max_reasks"] == 2
        assert manifest["params"]["sent"] == {"temperature": 0, "max_tokens": 99}
        assert manifest["responses"] == {
            "path": "r.jsonl",
            "digest": responses_digest(responses),
            "count": 3,
        }
        assert manifest["suite"]["cases_selected"] == 3
        assert manifest["counts"] == {
            "cases": 2,
            "judgments": 2,
            "judged": 2,
            "errors": 0,
            "cached": 0,
            "requests": 3,
            "reasks": 1,
            "responses_cut": 0,
            "input_tokens": 30,
            "output_tokens": 15,
        }
        assert manifest["max_response_chars"] == 6000
        assert manifest["summary"]["parse"]["reasked"] == 1
        assert manifest["judge_prompt_version"] == JUDGE_PROMPT_VERSION

    def test_case_ids_and_no_out_dir(self, tmp_path: Path) -> None:
        suite = make_suite(3)
        result = run(
            judge_suite(
                suite, {"c0": "a", "c1": "b"}, Judge(FakeJudge()), out_dir=None, case_ids=["c1"]
            )
        )
        assert [r.id for r in result.records] == ["c1"]
        assert result.run_dir is None
        assert result.summary.missing == []

    def test_unknown_case_id(self) -> None:
        with pytest.raises(ValueError, match="no such case"):
            run(judge_suite(make_suite(1), {}, Judge(FakeJudge()), out_dir=None, case_ids=["x"]))

    def test_crash_is_recorded(self, tmp_path: Path) -> None:
        class Broken(Judge):
            async def judge_all(self, items, *, samples=1, on_record=None):  # type: ignore[no-untyped-def]
                raise RuntimeError("bug")

        with pytest.raises(RuntimeError):
            run(judge_suite(make_suite(1), {"c0": "a"}, Broken(FakeJudge()), out_dir=tmp_path))
        [run_dir] = tmp_path.iterdir()
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "crashed"

    def test_responses_digest_ignores_order(self) -> None:
        assert responses_digest({"a": "1", "b": "2"}) == responses_digest({"b": "2", "a": "1"})
        assert responses_digest({"a": "1"}) != responses_digest({"a": "2"})

    def test_run_id(self) -> None:
        from datetime import UTC, datetime

        judge = Judge(FakeJudge())
        when = datetime(2026, 11, 17, 1, 0, tzinfo=UTC)
        assert make_judge_run_id(judge, when) == "20261117T010000Z-judge-fakejudge-j"


class SeededJudge(FakeJudge):
    """A FakeJudge that takes a seed and answers each seed differently."""

    supported_params: ClassVar[frozenset[str]] = frozenset({"temperature", "seed", "max_tokens"})

    def __init__(self) -> None:
        super().__init__()
        self.seeds: list[int | None] = []

    async def generate(self, prompt: Prompt, params: GenerationParams) -> Completion:
        self.seeds.append(params.seed)
        await asyncio.sleep(0.001)
        score = 1 + (params.seed or 0) % 5
        text = good(**{d.value: score for d in Dimension})
        return Completion(text=text, model="j", finish_reason="stop")


class TestSamples:
    def test_each_sample_gets_the_next_seed(self) -> None:
        provider = SeededJudge()
        judge = Judge(provider, params=GenerationParams(temperature=0.7, seed=10))
        [case] = make_suite(1).cases
        records = run(judge.judge_all([(case, "x")], samples=3))
        assert [r.sample for r in records] == [0, 1, 2]
        assert sorted(provider.seeds) == [10, 11, 12]
        scores = [r.scores[Dimension.INSTRUCTION_FOLLOWING] for r in records]
        assert scores == [1 + s % 5 for s in (10, 11, 12)]

    def test_without_a_seed_every_sample_uses_the_same_settings(self) -> None:
        judge = Judge(FakeJudge(), params=GenerationParams(temperature=0.7))
        assert judge.sample_params(0) is None
        assert judge.sample_params(3) is None

    def test_sample_zero_keeps_the_runners_settings(self) -> None:
        judge = Judge(SeededJudge(), params=GenerationParams(seed=5))
        assert judge.sample_params(0) is None
        params = judge.sample_params(2)
        assert params is not None
        assert params.seed == 7

    def test_samples_are_cached_side_by_side(self, tmp_path: Path) -> None:
        # Sample 0 must hit the same cache entry a plain `evalkit judge` made,
        # and samples 1, 2 must not replay it.
        cache = ResponseCache(tmp_path)
        [case] = make_suite(1).cases
        params = GenerationParams(temperature=0.7, seed=0)
        first = SeededJudge()
        run(Judge(first, params=params, cache=cache).judge_all([(case, "x")]))
        again = SeededJudge()
        records = run(Judge(again, params=params, cache=cache).judge_all([(case, "x")], samples=3))
        assert [r.cached for r in records] == [True, False, False]
        assert sorted(again.seeds) == [1, 2]
        replay = SeededJudge()
        records = run(Judge(replay, params=params, cache=cache).judge_all([(case, "x")], samples=3))
        assert all(r.cached for r in records)
        assert replay.seeds == []

    def test_unseeded_samples_still_get_their_own_cache_entries(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        [case] = make_suite(1).cases
        provider = FakeJudge()
        records = run(Judge(provider, cache=cache).judge_all([(case, "x")], samples=3))
        assert len(provider.prompts["prompt 0"]) == 3
        assert not any(r.cached for r in records)

    def test_records_are_ordered_by_item_then_sample(self) -> None:
        suite = make_suite(2)
        records = run(Judge(FakeJudge()).judge_all([(c, "x") for c in suite.cases], samples=2))
        assert [(r.id, r.sample) for r in records] == [("c0", 0), ("c0", 1), ("c1", 0), ("c1", 1)]

    def test_samples_must_be_positive(self) -> None:
        [case] = make_suite(1).cases
        with pytest.raises(ValueError, match="at least 1"):
            run(Judge(FakeJudge()).judge_all([(case, "x")], samples=0))

    def test_judge_suite_records_samples_in_the_manifest(self, tmp_path: Path) -> None:
        result = run(
            judge_suite(
                make_suite(2),
                {"c0": "a", "c1": "b"},
                Judge(FakeJudge()),
                out_dir=tmp_path,
                samples=2,
            )
        )
        assert len(result.records) == 4
        assert result.run_dir is not None
        manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["samples"] == 2
        lines = (result.run_dir / "judgments.jsonl").read_text(encoding="utf-8").splitlines()
        assert sorted((json.loads(x)["id"], json.loads(x)["sample"]) for x in lines) == [
            ("c0", 0),
            ("c0", 1),
            ("c1", 0),
            ("c1", 1),
        ]

    def test_old_records_without_a_sample_field_still_load(self) -> None:
        record = judge_one(FakeJudge())
        data = record.model_dump(mode="json")
        del data["sample"]
        assert JudgeRecord.model_validate(data).sample == 0
