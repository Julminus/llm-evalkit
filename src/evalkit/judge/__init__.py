"""LLM-as-judge: rubric scores (1-5), failure-mode tags, and repair of bad judge replies.

* :mod:`evalkit.judge.prompt` builds the judge's prompt from a case, a
  response and the deterministic check results.
* :mod:`evalkit.judge.parse` turns the judge's reply into a validated
  :class:`ParsedJudgment`, repairing syntax slips locally.
* :mod:`evalkit.judge.core` runs the judge over a suite, re-asking when a
  reply can't be used, and aggregates the verdicts. With ``samples > 1``
  it judges each response several times (self-consistency).
* :mod:`evalkit.judge.pairwise` compares two responses to each case, in
  both orders, and measures how often the order changes the verdict.
"""

from evalkit.judge.core import (
    DEFAULT_MAX_RESPONSE_CHARS,
    Judge,
    JudgeRecord,
    JudgeResult,
    JudgeSummary,
    judge_suite,
    make_judge_run_id,
    responses_digest,
)
from evalkit.judge.parse import JudgmentError, ParsedJudgment, parse_judgment
from evalkit.judge.prompt import JUDGE_PROMPT_VERSION, build_judge_prompt, build_repair_prompt

__all__ = [
    "DEFAULT_MAX_RESPONSE_CHARS",
    "JUDGE_PROMPT_VERSION",
    "Judge",
    "JudgeRecord",
    "JudgeResult",
    "JudgeSummary",
    "JudgmentError",
    "ParsedJudgment",
    "build_judge_prompt",
    "build_repair_prompt",
    "judge_suite",
    "make_judge_run_id",
    "parse_judgment",
    "responses_digest",
]
