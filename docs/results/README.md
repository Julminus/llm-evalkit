# Measured results

Raw output of the measurement scripts. The repository README quotes these files; every number
there can be recomputed from them.

| File | Made by | What it holds |
|---|---|---|
| `runner-s2.json` | `scripts/measure_runner.py` | Providers and runner on `qwen2.5:7b` and `llama3.1:8b`: fidelity, repeatability, concurrency, cache, check pass rates, and every response. |
| `judge-s3-prompt-v1.json` | `scripts/measure_judge.py` | The LLM judge (prompt v1, the shipped one) on the 80 `responses.fresh` of `runner-s2.json`, four conditions: `qwen14b`, `qwen14b_nochk` (check results left out), `qwen14b_again` (second fresh load), `llama8b`. Every judge reply is kept. |
| `judge-s3-prompt-v2.json` | `scripts/measure_judge.py` | The same with prompt v2 (reverted). |
| `judge-s3-prompt-v3.json` | `scripts/measure_judge.py` | The same with prompt v3 (reverted). |
| `judge-s3-audit.json` | by hand | Claude's reading of the 8 `hallucination` tags `qwen2.5:14b` gave in `judge-s3-prompt-v1.json`; not yet checked by a person. |

`uv run python scripts/measure_judge.py --summary FILE` prints a judge file's table, and
`--regrade FILE` re-parses every saved reply with the current parser and checks that each
judgment's outcome is unchanged.

## The judge prompt versions

v1 is `src/evalkit/judge/prompt.py` as shipped (`JUDGE_PROMPT_VERSION = "1"`). The other two
differ from it only as follows.

**v2** (commit `9b9fc9f`) added a fourth rule to the system prompt:

> - Grade against what the request actually asks. Do not invent requirements it does not state,
>   and do not treat a matter of style as a broken instruction.

replaced the sentence introducing the check results with:

> Results of deterministic checks that code ran on this response. They are exact: use them instead
> of counting yourself, and never contradict one. A constraint marked PASS is met (a long sentence
> is still one sentence).

and replaced the line introducing the failure modes with:

> Failure modes. List a mode only for a clear failure you can point to in the response, not for
> something that could merely be better: a weakness that only costs a point is not a failure mode.
> Use [] if none applies.

**v3** (never committed as code) is the v1 text, except that the failure-mode list shows only
`intent_misread`, `hallucination`, `partial_completion` and `over_refusal`, with the same
descriptions as v1, and is followed by one more line:

> Broken length, format or language rules are not on this list: code checks those. They lower
> instruction_following instead.

The reply parser for v3 dropped any of the three checkable modes the judge still named; it never
had to (0 in all four conditions).
