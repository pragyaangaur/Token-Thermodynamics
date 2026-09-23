# Experiment A: does T_melt/1.141 replace a temperature sweep

Written 23 September 2026. It is the experiment that decides whether the melting law has a practical use. The harness is `src/experiment_a.py`, and `notebooks/experiment_a_kaggle.ipynb` runs it on a free Kaggle GPU. Nothing in this file is a result yet, apart from the pre-flight check near the end.

## The question

Du, Yang and Welleck (arXiv 2502.05234) choose a sampling temperature for majority voting and best-of-N with an algorithm they call TURN. It generates samples at every temperature from 0.1 to 1.4, averages the token entropy, and takes the first point where `log H` against `T` turns convex. Section 7m of `FINDINGS.md` shows that the single-distribution analogue of that turning point sits below the melting temperature, with a ratio of 1.141. The melting temperature needs no sampling at all. If sampling at `T_melt/1.141` is as accurate as sampling at TURN's temperature, one cheap pass replaces the sweep.

## What the harness does

It uses TURN's own benchmark files, prompts, answer parser and grader, taken from a clone of `github.com/StigLidu/TURN` pinned to commit `64b42e2`. That repository is MIT licensed. The MATH set is their 200 problems, 40 per difficulty level, with their four-shot prompt. MBPP is the first 100 test problems with their zero-shot docstring prompt.

It runs four phases, and each one saves to disk so a Kaggle session can stop and resume.

1. **melt.** One greedy pass per question, float32, full vocabulary. T_melt is read at each of the first 32 generated positions and the median is taken.
2. **turn.** TURN's entropy sweep exactly as their `predict.py` does it: 32 samples per temperature, top 1000 log probabilities per step, entropy averaged over positions, then their turning-point code copied verbatim. For majority voting the adaptor is 0.0, and for best-of-N it is 0.1.
3. **grid.** `k` samples per question at every grid temperature from 0.1 to 1.4, plus two melting-law settings. One uses a single temperature per task, which is the median T_melt divided by 1.141. The other uses a separate temperature for each question.
4. **score.** MATH uses majority voting with TURN's answer clustering. MBPP uses unbiased pass@N from unit tests run in a subprocess. Accuracy is reported at N = 1, 2, 4 and so on up to `k`, averaged over disjoint blocks of samples.

The summary reports the best grid temperature, the best of TURN's six fixed baselines, TURN, and both melting-law settings, each with its accuracy drop from the best and whether it lands within TURN's epsilon of 0.02. It also reports the cost of each method in generated tokens.

## Three decisions that were made, and why

**TURN's entropy curve is computed their way.** The 1.141 ratio belongs to the `S` against `T` convention. TURN uses `log H` against linear `T`, and on single distributions that convention gave ratios between 2.5 and 27 across the seven cached models (`data/turning_point.json`). So nothing guarantees that `T_melt/1.141` lands near TURN's temperature. The harness measures TURN's turning point directly and reports `T_melt / t*`, which is the number that says whether the single-distribution result transfers.

**T_melt is read along the greedy continuation, and not at the end of the prompt.** The first plan was to read it at the first answer token. TURN's prompts end directly after the question text with no newline, and on Qwen2.5-0.5B the most likely next token is ` (` at about 0.3, because the model carries on writing the question. So the end of the prompt is not an answer position in this format. The prompt-end value is still saved and printed next to the main one.

**vLLM's log probabilities are checked.** TURN ran on a vLLM version that returned log probabilities after temperature scaling. Newer versions return raw ones by default. The harness asks for the tempered ones where the option exists. It then measures, at `T = 0.5`, the slope between vLLM's first-position log-probability differences and the float32 forward pass. A slope of 1 means raw and a slope of 2 means tempered. Both entropy curves are saved either way, and on the laptop test the slope came out at 0.99999 for raw log probabilities.

## Decision rule, to be fixed before the run

This is a proposal. It should be confirmed or changed before any GPU time is spent, because choosing it after seeing the numbers would defeat its purpose.

- The primary endpoint is MATH majority-voting accuracy at N = 16, averaged over the two disjoint blocks of `k = 32`.
- The prediction **works** on a model if the single-temperature melting setting lands within 0.02 of the best grid accuracy, and its drop is no more than 0.01 larger than TURN's drop.
- The prediction **fails** on a model if its drop from the best grid accuracy is more than 0.05.
- Anything in between is inconclusive and needs more samples.
- The paper claim needs the prediction to work on at least two of the first three models and fail on none.

A pass means the melting law predicts a useful sampling temperature. A fail means the ordering theorem and the ratio measurements stand as a measurement note, and the sampling claim is dropped.

## Models and budget

Start with `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct`. Both are on TURN's MATH list, so their reported temperatures are a check on this reimplementation. Both are gated, so the Hugging Face licence has to be accepted first. A third model can come from the same list once the first two have been read.

The run size is 16 temperature settings × 200 questions × `k` samples. At `k = 32` that is 102,400 generations of up to 1024 tokens each. The real throughput on a Kaggle T4 is not known until the first run, and each setting's wall time is saved in its output file. Run the 1B model first at `k = 32` and read the timings before choosing `k` for the 3B model. Kaggle gives about 30 GPU hours a week, and one session stops after 12 hours, which the checkpointing is there for.

## Pre-flight check, run locally on 23 September 2026

The melt phase needs no GPU, so it was run on the two Qwen models already cached on the laptop, over all 200 MATH problems and 32 positions each. The data is in `data/experiment_a_preflight.json`.

| Model | median T_melt | predicted temperature, T_melt/1.141 | ratio T_melt/T_turn on these prompts, S against T | same ratio on 20 generic prompts |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | 2.08 | 1.81 | 1.025 | 1.113 |
| Qwen2.5-1.5B-Instruct | 2.05 | 1.80 | 1.028 | 1.142 |

Two things follow, and both count against a clean pass. The predicted temperature is about 1.8, which is above TURN's whole grid and well above the 0.6 to 0.9 they report as good for general models. The ratio itself also depends on the text, so the constant 1.141 from generic prompts does not carry over to MATH. A second laptop check sampled 8 answers for each of the first 40 problems on Qwen2.5-0.5B, at four temperatures, with a 400-token cap. It is `src/experiment_a_preflight.py` and the data is `data/experiment_a_preflight_acc.json`.

| Temperature | majority of 8 | single sample | no parseable answer |
| --- | --- | --- | --- |
| 0.6 | 0.125 | 0.047 | 71% |
| 1.0 | 0.000 | 0.000 | 88% |
| 1.4 | 0.000 | 0.000 | 100% |
| 1.8 | 0.000 | 0.000 | 100% |

On this model, accuracy is gone by 1.0, and the predicted 1.8 produces no usable answers at all. The model is small and weak in TURN's format, and even at 0.6 most samples never reach the answer line, so this says nothing about where the best temperature is. It does say that the literal prediction `T_melt/1.141` would fail badly here. Qwen is not on TURN's model list, and the Llama run is still the test that decides it. The most likely outcome now is a fail, and a GPU run is still worth doing for two reasons. It measures TURN's sample-averaged turning point next to T_melt, which is the ratio that matters. It also tests the per-question setting, which the laptop check did not.

## Smoke run on Kaggle, 23 September 2026

The notebook ran end to end on a Kaggle T4 with Llama-3.2-1B-Instruct, 8 problems, 4 samples and a 256-token cap. It was a pipeline check, and its accuracy numbers carry no weight at that size. The compact record is `data/experiment_a_smoke_llama1b.json`.

- **The log-probability check works.** The slope at `T = 0.5` came out at 2.0001, so this vLLM returned tempered log probabilities, which is what TURN used. The entropy curve is therefore TURN's own quantity.
- **TURN's temperature reproduces.** The sweep gives `t* = 0.6`, inside the 0.6 to 0.7 that Du, Yang and Welleck report for general models, even from only 8 samples per temperature.
- **The ratio that matters is far from 1.141.** T_melt has a median of 1.77, so `T_melt / t*` is 2.95. The single-distribution ratio on the same positions is 1.025, as on the Qwen models. The gap between the two is the difference between one next-token distribution and TURN's sample-averaged curve, which is the transfer that section 7m flagged as untested.
- **The predicted temperature lands in noise.** `T_melt/1.141` is 1.55. From `T = 1.2` upward every sample in the smoke run was unparseable, and the samples at 1.55 are strings of unrelated tokens.
- **Timing.** The whole run took about 4 minutes of generation. With only 32 sequences in flight it reached 734 tokens a second, and the full run batches 6400 sequences, so its throughput should be much higher. The full run is still split into two parts, A and B, so each fits inside a 12-hour session.

## Known limits of this design

- The grid stops at 1.4, which is TURN's grid. If `T_melt/1.141` lands above 1.4, its accuracy is still measured, but the best grid temperature cannot be above 1.4.
- `k = 32` is far below TURN's 256 samples per question. That makes the accuracy at large N noisier than theirs.
- T_melt is a median over 32 greedy positions, and those positions include formatting tokens. A version that reads only answer tokens, or that changes temperature per position, is not attempted.
- MBPP runs model-written code. It belongs in a disposable environment such as Kaggle and nowhere else.
