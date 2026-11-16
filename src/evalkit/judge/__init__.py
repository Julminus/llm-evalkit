"""LLM-as-judge: rubric scores (1-5), failure-mode tags, and repair of bad judge replies.

* :mod:`evalkit.judge.parse` turns the judge's reply into a validated
  :class:`ParsedJudgment`, repairing syntax slips locally.
"""

from evalkit.judge.parse import JudgmentError, ParsedJudgment, parse_judgment

__all__ = ["JudgmentError", "ParsedJudgment", "parse_judgment"]
