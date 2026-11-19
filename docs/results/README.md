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
| `reliability-s4.json` | `scripts/measure_reliability.py` | Judge reliability on the same 80 responses, judges `qwen2.5:14b` and `llama3.1:8b`: `pairwise_*` (qwen2.5:7b vs llama3.1:8b in both orders, plus each model against itself as a control; every `PairRecord`) and `consistency_*` (5 samples per response at temperature 0.7; every `JudgeRecord`). Judge-vs-judge agreement is computed from `judge-s3-prompt-v1.json`. |
| `report-s5.json` | `scripts/measure_report.py` | The report tools on real runs: `qwen2.5:14b` judging the 80 `responses.fresh` and the 80 `responses.warm` of `runner-s2.json` through one cache (a warm answer identical to its fresh one replays the fresh judgment), every `JudgeRecord`, and the fresh-vs-warm diffs on the checks and on the judged runs. |
| `report-s5/` | `evalkit report` | The report of the two fresh judged runs, qwen2.5:7b vs llama3.1:8b: `report.md` and `report.html` (download it to open; GitHub shows HTML as source). |
| `benchmark-s6.json` | `scripts/measure_benchmark.py` | The benchmark (`examples/suites/benchmark.yaml`): qwen2.5:7b's and llama3.1:8b's answers with every `RunRecord` (tokens, latency, finish reason), and qwen2.5:7b's and llama3.1:8b's judgments by `qwen2.5:14b`, every `JudgeRecord` with its replies. |
| `benchmark-s6/` | `scripts/measure_benchmark.py` | The two judge runs as `evalkit judge` writes them (`judged-*/judgments.jsonl` + `manifest.json`, which points at `examples/responses/benchmark-*.jsonl`) and their `report.md` / `report.html`. The pre-commit whitespace hook trimmed trailing spaces at the ends of a few answer lines in `report.html`; nothing else differs from what the tool wrote. |

`uv run python scripts/measure_judge.py --summary FILE` prints a judge file's table, and
`--regrade FILE` re-parses every saved reply with the current parser and checks that each
judgment's outcome is unchanged. `uv run python scripts/measure_reliability.py --summary
docs/results/reliability-s4.json` recomputes the reliability table from the saved records with
the library's `PairwiseSummary`, `consistency` and `compare`. `uv run python
scripts/measure_report.py --summary docs/results/report-s5.json` rebuilds the fresh-vs-warm diffs
from the saved records with `diff_runs`, checks them against the saved ones and prints the table.
`report-s5/` was written by `uv run evalkit report examples/suites/everyday.yaml
runs/s5/judged-qwen2.5-7b-fresh runs/s5/judged-llama3.1-8b-fresh --name qwen2.5:7b --name
llama3.1:8b -o docs/results/report-s5` after the measurement, which leaves those run directories.

`uv run python scripts/measure_benchmark.py --summary docs/results/benchmark-s6.json` prints the
benchmark's tables from the saved records: checks by group and language, the reasoning-trap
answers against the suite's reference answers, the judge's means, failure modes, and the judge
against `examples/human_labels.csv` (Claude's labels of qwen2.5:7b's answers). The file is from
the third of three runs on 2026-11-20 (local time; the file records UTC). The first was a pilot:
after reading its answers, two tasks' judge criteria were corrected before the suite was
committed. `*-five-words` said the two constraints can't both be met, which is wrong if the 6
attractions come as a separate list; `*-tea-table` said "(caffeine is the one that matters
here)", which the judge read as a rule against other rows. That changed the judge's replies on 7
records and nothing else. All three runs gave byte-identical answers, and the last two byte-identical judge replies.

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
