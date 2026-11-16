"""LLM-as-judge: rubric scores (1-5), failure-mode tags, and repair of bad judge replies.

* :mod:`evalkit.judge.prompt` builds the judge's prompt from a case, a
  response and the deterministic check results.
* :mod:`evalkit.judge.parse` turns the judge's reply into a validated
  :class:`ParsedJudgment`, repairing syntax slips locally.
* :mod:`evalkit.judge.core` runs the judge over a suite, re-asking when a
  reply can't be used, and aggregates the verdicts.
"""

from evalkit.judge.core import (
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
