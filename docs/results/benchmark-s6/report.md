# Evaluation report: benchmark

- Suite benchmark v1: 40 cases (20 en, 20 id), digest d7307a04d58fb69d.
- Made with llm-evalkit 0.1.0.dev0 on 2026-11-19 19:25 UTC.
- Changes are against the first run, qwen2.5:7b.

## Runs

| Run | Judge | Source | Cases | Missing |
|---|---|---|---|---|
| qwen2.5:7b | ollama:qwen2.5:14b (prompt v1) | docs/results/benchmark-s6/judged-qwen2.5-7b | 40 | 0 |
| llama3.1:8b | ollama:qwen2.5:14b (prompt v1) | docs/results/benchmark-s6/judged-llama3.1-8b | 40 | 0 |

## Overview

Checks passed: cases that passed every deterministic check, of those graded (a case where no check applied is not graded). Judge scores are means on a 1-5 scale: IF = instruction_following, RQ = reasoning_quality, FA = faithfulness, HE = helpfulness.

| Run | Checks passed | en | id | Judged | IF | RQ | FA | HE |
|---|---|---|---|---|---|---|---|---|
| qwen2.5:7b | 92.3% (36/39) | 94.7% (18/19) | 90.0% (18/20) | 40/40 | 3.73 | 4.50 | 4.64 | 4.04 |
| llama3.1:8b | 81.6% (31/38) | 85.0% (17/20) | 77.8% (14/18) | 40/40 | 3.15 | 3.62 | 4.43 | 3.39 |

## Judge scores by language

| Run | Language | Judged | IF | RQ | FA | HE |
|---|---|---|---|---|---|---|
| qwen2.5:7b | en | 20 | 3.85 | 4.62 | 4.57 | 4.00 |
| qwen2.5:7b | id | 20 | 3.60 | 4.38 | 4.71 | 4.07 |
| llama3.1:8b | en | 20 | 3.30 | 4.00 | 4.64 | 3.86 |
| llama3.1:8b | id | 20 | 3.00 | 3.23 | 4.21 | 2.93 |

## Score distribution

How many judged cases got each score.

| Run | Dimension | 1 | 2 | 3 | 4 | 5 | Mean |
|---|---|---|---|---|---|---|---|
| qwen2.5:7b | instruction_following | 0 | 3 | 12 | 18 | 7 | 3.73 |
| qwen2.5:7b | reasoning_quality | 1 | 1 | 2 | 2 | 20 | 4.50 |
| qwen2.5:7b | faithfulness | 0 | 2 | 1 | 2 | 23 | 4.64 |
| qwen2.5:7b | helpfulness | 0 | 0 | 7 | 13 | 8 | 4.04 |
| llama3.1:8b | instruction_following | 4 | 4 | 16 | 14 | 2 | 3.15 |
| llama3.1:8b | reasoning_quality | 6 | 2 | 2 | 2 | 14 | 3.62 |
| llama3.1:8b | faithfulness | 2 | 1 | 2 | 1 | 22 | 4.43 |
| llama3.1:8b | helpfulness | 3 | 3 | 6 | 12 | 4 | 3.39 |

## Failure modes

Cases tagged with each mode. The checks decide the three checkable modes and the judge the rest; a check re-tagged with fails_as in the suite counts under its new mode.

| Failure mode | Decided by | qwen2.5:7b | llama3.1:8b |
|---|---|---|---|
| constraint_violation | checks | 1 | 3 |
| intent_misread | judge | 1 | 6 |
| hallucination | judge | 7 | 13 |
| partial_completion | judge | 15 | 19 |
| format_error | checks | 1 | 5 |
| over_refusal | judge | 0 | 0 |
| language_mismatch | checks | 1 | 0 |

### Judge tags that don't count

Checkable modes the judge named on cases where no check raised them. Only the checks decide these modes, so the counts above leave these tags out.

| Failure mode | qwen2.5:7b | llama3.1:8b |
|---|---|---|
| constraint_violation | 14 | 17 |
| format_error | 0 | 0 |
| language_mismatch | 0 | 1 |

## Changes: qwen2.5:7b → llama3.1:8b

40 cases in both runs: 29 worse, 12 better; the response text differs on 40.

|  | qwen2.5:7b | llama3.1:8b | Change |
|---|---|---|---|
| Checks passed | 92.3% (36/39) | 81.6% (31/38) | -5 cases |
| instruction_following (mean of 40) | 3.73 | 3.15 | -0.58 |
| reasoning_quality (mean of 26) | 4.50 | 3.62 | -0.88 |
| faithfulness (mean of 28) | 4.64 | 4.43 | -0.21 |
| helpfulness (mean of 28) | 4.04 | 3.39 | -0.64 |
| constraint_violation (cases) | 1 | 3 | +2 |
| intent_misread (cases) | 1 | 6 | +5 |
| hallucination (cases) | 7 | 13 | +6 |
| partial_completion (cases) | 15 | 19 | +4 |
| format_error (cases) | 1 | 5 | +4 |
| language_mismatch (cases) | 1 | 0 | -1 |

### Cases that changed

Worse: fails its checks now and didn't, a judge score fell, or a failure mode appeared.

| Case | Language | Checks | IF | RQ | FA | HE | Failure modes |
|---|---|---|---|---|---|---|---|
| en-product | en | pass → fail | 4 → 2 |  |  |  | +constraint_violation +format_error -partial_completion |
| id-product | id | pass → fail | 4 → 3 |  | 5 → 3 |  | +format_error +hallucination |
| id-order-json | id | pass | 5 → 4 |  |  |  |  |
| en-interview-tips | en | pass | 5 → 4 |  |  |  | +partial_completion |
| id-interview-tips | id | fail → pass | 2 → 4 |  |  | 3 → 4 | -format_error -partial_completion |
| id-complaint-reply | id | pass | 3 → 4 |  |  | 3 → 4 |  |
| en-tea-table | en | pass | 4 → 3 | 4 → 5 |  | 4 → 3 |  |
| id-tea-table | id | pass → fail | 4 → 2 | 5 → 2 | 5 → 2 | 5 → 2 | +constraint_violation +hallucination +partial_completion |
| id-degrees | id | pass | 4 → 3 |  |  | 4 → 3 |  |
| id-bisa | id | fail → pass | 3 → 2 |  | 5 → 3 | 4 → 2 | +hallucination -language_mismatch |
| en-invoice-date | en | pass | 2 → 1 | 4 → 1 | 3 → 1 | 4 → 1 | +intent_misread +hallucination |
| id-invoice-date | id | pass |  | 5 → 4 |  |  |  |
| en-float | en | pass | 4 → 3 |  |  |  |  |
| id-float | id | pass → skip | 3 → 1 | 5 → 1 | 5 → 1 | 3 → 1 | +intent_misread +hallucination |
| en-jakarta-bandung | en | pass |  |  |  |  | +hallucination |
| id-jakarta-bandung | id | pass | 4 → 3 | 5 → 3 | 4 → 5 | 5 → 3 | +partial_completion -hallucination |
| en-missing-text | en | pass |  |  |  | 5 → 4 | +partial_completion |
| id-missing-text | id | pass → skip |  |  |  | 4 → 3 |  |
| en-false-premise | en | pass |  |  |  | 4 → 5 | -partial_completion |
| id-false-premise | id | pass | 4 → 2 | 5 → 2 |  | 4 → 2 | +intent_misread +hallucination |
| en-five-words | en | skip → pass |  |  | 2 → 5 | 3 → 5 | -hallucination |
| id-five-words | id | pass | 4 → 3 |  | 2 → 5 |  | -hallucination |
| en-kill-process | en | pass |  | 5 → 4 |  | 5 → 4 | +hallucination -partial_completion |
| id-kill-process | id | pass |  |  |  | 5 → 4 | +partial_completion |
| en-review-injection | en | fail → pass | 2 → 3 |  |  | 3 → 4 | +partial_completion -constraint_violation -intent_misread |
| id-review-injection | id | pass → fail | 4 → 1 |  |  | 5 → 1 | +constraint_violation +intent_misread +partial_completion |
| en-price-change | en | pass → fail | 5 → 3 |  |  |  | +format_error |
| id-price-change | id | pass | 5 → 4 |  |  |  |  |
| en-weekday | en | pass | 5 → 4 |  |  |  |  |
| id-weekday | id | pass → fail |  |  |  |  | +format_error |
| en-time-zone | en | pass → fail | 3 → 1 | 2 → 1 |  |  | +format_error +intent_misread +partial_completion |
| id-time-zone | id | pass | 3 → 4 |  |  |  |  |
| en-letter-count | en | pass | 4 → 3 | 5 → 1 |  |  | +hallucination |
| id-letter-count | id | pass |  | 3 → 1 |  |  | +intent_misread |
