# Evaluation report: everyday

- Suite everyday v1: 40 cases (20 en, 20 id), digest d7a367cb0df179bc.
- Made with llm-evalkit 0.1.0.dev0 on 2026-11-18 18:53 UTC.
- Changes are against the first run, qwen2.5:7b.

## Runs

| Run | Judge | Source | Cases | Missing |
|---|---|---|---|---|
| qwen2.5:7b | ollama:qwen2.5:14b (prompt v1) | runs/s5/judged-qwen2.5-7b-fresh | 40 | 0 |
| llama3.1:8b | ollama:qwen2.5:14b (prompt v1) | runs/s5/judged-llama3.1-8b-fresh | 40 | 0 |

## Overview

Checks passed: cases that passed every deterministic check, of those graded (a case where no check applied is not graded). Judge scores are means on a 1-5 scale: IF = instruction_following, RQ = reasoning_quality, FA = faithfulness, HE = helpfulness.

| Run | Checks passed | en | id | Judged | IF | RQ | FA | HE |
|---|---|---|---|---|---|---|---|---|
| qwen2.5:7b | 87.5% (35/40) | 90.0% (18/20) | 85.0% (17/20) | 40/40 | 4.00 | 4.72 | 4.65 | 4.25 |
| llama3.1:8b | 87.2% (34/39) | 85.0% (17/20) | 89.5% (17/19) | 40/40 | 3.80 | 4.60 | 4.92 | 4.10 |

## Judge scores by language

| Run | Language | Judged | IF | RQ | FA | HE |
|---|---|---|---|---|---|---|
| qwen2.5:7b | en | 20 | 3.95 | 4.95 | 4.95 | 4.40 |
| qwen2.5:7b | id | 20 | 4.05 | 4.50 | 4.35 | 4.10 |
| llama3.1:8b | en | 20 | 3.75 | 4.85 | 5.00 | 4.25 |
| llama3.1:8b | id | 20 | 3.85 | 4.35 | 4.85 | 3.95 |

## Score distribution

How many judged cases got each score.

| Run | Dimension | 1 | 2 | 3 | 4 | 5 | Mean |
|---|---|---|---|---|---|---|---|
| qwen2.5:7b | instruction_following | 0 | 0 | 11 | 18 | 11 | 4.00 |
| qwen2.5:7b | reasoning_quality | 0 | 0 | 2 | 7 | 31 | 4.72 |
| qwen2.5:7b | faithfulness | 0 | 1 | 4 | 3 | 32 | 4.65 |
| qwen2.5:7b | helpfulness | 0 | 0 | 4 | 22 | 14 | 4.25 |
| llama3.1:8b | instruction_following | 0 | 3 | 12 | 15 | 10 | 3.80 |
| llama3.1:8b | reasoning_quality | 0 | 0 | 4 | 8 | 28 | 4.60 |
| llama3.1:8b | faithfulness | 0 | 0 | 1 | 1 | 38 | 4.92 |
| llama3.1:8b | helpfulness | 0 | 0 | 8 | 20 | 12 | 4.10 |

## Failure modes

Cases tagged with each mode. The checks decide the three checkable modes and the judge the rest; a check re-tagged with fails_as in the suite counts under its new mode.

| Failure mode | Decided by | qwen2.5:7b | llama3.1:8b |
|---|---|---|---|
| constraint_violation | checks | 5 | 5 |
| intent_misread | judge | 0 | 0 |
| hallucination | judge | 6 | 2 |
| partial_completion | judge | 6 | 6 |
| format_error | checks | 0 | 0 |
| over_refusal | judge | 0 | 0 |
| language_mismatch | checks | 0 | 0 |

### Judge tags that don't count

Checkable modes the judge named on cases where no check raised them. Only the checks decide these modes, so the counts above leave these tags out.

| Failure mode | qwen2.5:7b | llama3.1:8b |
|---|---|---|
| constraint_violation | 14 | 17 |
| format_error | 0 | 0 |
| language_mismatch | 0 | 0 |

## Changes: qwen2.5:7b → llama3.1:8b

40 cases in both runs: 23 worse, 20 better; the response text differs on 40.

|  | qwen2.5:7b | llama3.1:8b | Change |
|---|---|---|---|
| Checks passed | 87.5% (35/40) | 87.2% (34/39) | -1 case |
| instruction_following (mean of 40) | 4.00 | 3.80 | -0.20 |
| reasoning_quality (mean of 40) | 4.72 | 4.60 | -0.12 |
| faithfulness (mean of 40) | 4.65 | 4.92 | +0.27 |
| helpfulness (mean of 40) | 4.25 | 4.10 | -0.15 |
| constraint_violation (cases) | 5 | 5 | 0 |
| hallucination (cases) | 6 | 2 | -4 |
| partial_completion (cases) | 6 | 6 | 0 |

### Cases that changed

Worse: fails its checks now and didn't, a judge score fell, or a failure mode appeared.

| Case | Language | Checks | IF | RQ | FA | HE | Failure modes |
|---|---|---|---|---|---|---|---|
| en-01 | en | fail → pass | 3 → 4 |  |  | 4 → 5 | -constraint_violation -partial_completion |
| id-01 | id | fail → pass | 3 → 4 | 5 → 3 | 5 → 3 | 4 → 3 | +hallucination -constraint_violation |
| en-02 | en | pass | 4 → 5 | 5 → 4 |  | 5 → 4 |  |
| id-02 | id | pass | 4 → 3 | 5 → 4 |  | 5 → 3 |  |
| en-03 | en | pass | 4 → 3 |  |  | 5 → 4 |  |
| id-03 | id | pass |  |  |  | 4 → 5 |  |
| id-04 | id | pass → skip |  |  |  |  |  |
| id-05 | id | pass | 4 → 3 | 5 → 3 |  | 4 → 3 |  |
| id-06 | id | pass | 5 → 3 |  |  | 5 → 4 |  |
| en-07 | en | fail → pass | 3 → 2 |  |  | 4 → 3 | -constraint_violation |
| id-07 | id | pass | 5 → 4 | 4 → 5 | 4 → 5 |  |  |
| en-08 | en | pass |  |  |  | 4 → 5 |  |
| en-09 | en | pass → fail | 5 → 3 | 4 → 5 |  |  | +constraint_violation |
| id-09 | id | fail → pass | 3 → 4 |  |  | 4 → 5 | -constraint_violation |
| en-10 | en | pass | 4 → 5 |  |  |  |  |
| id-10 | id | pass | 4 → 3 | 5 → 3 | 5 → 4 | 4 → 3 | +hallucination |
| en-11 | en | pass → fail | 5 → 3 |  | 4 → 5 | 5 → 4 | +constraint_violation -hallucination |
| id-11 | id | pass → fail | 4 → 2 |  | 3 → 5 | 4 → 3 | +constraint_violation +partial_completion -hallucination |
| id-12 | id | pass | 4 → 5 | 3 → 4 | 4 → 5 |  | -hallucination |
| en-13 | en | pass | 4 → 5 | 5 → 4 |  |  | -partial_completion |
| id-13 | id | pass | 5 → 4 | 4 → 5 |  |  |  |
| en-14 | en | pass | 3 → 2 |  |  | 4 → 3 |  |
| id-14 | id | pass |  |  |  |  | +partial_completion |
| en-15 | en | pass |  | 5 → 4 |  | 5 → 4 |  |
| en-17 | en | pass → fail | 4 → 3 |  |  |  | +constraint_violation |
| id-17 | id | pass → fail | 5 → 3 |  | 3 → 5 | 5 → 4 | +constraint_violation -hallucination |
| en-18 | en | pass | 3 → 4 |  |  | 3 → 5 | -partial_completion |
| id-18 | id | pass | 3 → 4 | 3 → 5 | 3 → 5 | 3 → 4 | +partial_completion |
| en-19 | en | pass | 5 → 4 |  |  |  |  |
| id-19 | id | fail → pass | 3 → 5 | 5 → 3 | 3 → 5 | 4 → 5 | -constraint_violation -hallucination |
| en-20 | en | pass |  |  |  | 5 → 4 |  |
| id-20 | id | pass | 4 → 5 | 4 → 5 | 2 → 5 | 3 → 4 | -hallucination |
