"""LLM-as-judge: rubric scores (1-5), failure-mode tags, and repair of bad judge replies.

* :mod:`evalkit.judge.prompt` builds the judge's prompt from a case, a
  response and the deterministic check results.
* :mod:`evalkit.judge.parse` turns the judge's reply into a validated
  :class:`ParsedJudgment`, repairing syntax slips locally.
"""

from evalkit.judge.parse import JudgmentError, ParsedJudgment, parse_judgment
from evalkit.judge.prompt import JUDGE_PROMPT_VERSION, build_judge_prompt, build_repair_prompt

__all__ = [
    "JUDGE_PROMPT_VERSION",
    "JudgmentError",
    "ParsedJudgment",
    "build_judge_prompt",
    "build_repair_prompt",
    "parse_judgment",
]
