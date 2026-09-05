# Independent validation audit

Checked 5 September 2026 against commit `fdc01db`, before the corrections recorded below.

## Verdict

The thermodynamic core is mathematically correct and the committed data reproduce the headline
tables. The project is credible as an exploratory study, not yet as a publication-grade empirical
claim. The strongest reasons are small and non-independent samples, a public reproduction command
that starts from derived JSON rather than raw model outputs, and several claims whose wording was
broader than the actual measurement.

This audit corrected the validator, the Schottky peak constant, and the scope of the
phrasing-versus-meaning percentage. It also produced a new negative result in
[`PEAK_OCCUPANCY.md`](PEAK_OCCUPANCY.md).

## Checks that pass

`python src/validate.py` now contains assertions and exits nonzero on failure. On the current tree:

| Check | Measured error | Required |
| --- | ---: | ---: |
| `F = U - TS` | `1.69e-13` absolute | `< 2e-11` |
| `dU/dT = C` | `8.75e-8` relative | `< 1e-6` |
| `dS/dT = C/T` | `2.32e-8` relative | `< 1e-6` |
| predictive entropy = `S(1)` | `1.78e-15` absolute | `< 1e-12` |
| varentropy = `C(1)` | `9.33e-14` absolute | `< 1e-12` |
| two-level analytic heat capacity | `1.07e-14` absolute | `< 2e-14` |
| ideal peak-occupancy law | `1.01e-13` absolute | `< 1e-12` |
| exact ideal `C_max` law | `5.21e-13` absolute | `< 1e-11` |
| density-of-states entropy | `8.05e-6` absolute | `< 1e-5` |
| density-of-states heat capacity | `1.66e-5` absolute | `< 2e-5` |

`python repro.py` reproduces the 12-row melting table, the 58% pooled scatter reduction, and the
reported bootstrap interval. It now asserts those conditions instead of printing them without a
test. All Python files compile and `git diff --check` passes.

The free-form JSON independently recomputes the reported contrasts:

| Model | Free-form phrasing share | Short factual phrasing share |
| --- | ---: | ---: |
| Qwen2.5-1.5B | 94.66% | 13.22% |
| Qwen2.5-0.5B | 90.60% | 4.84% |

These percentages are computed from the **top 12 logits after renormalisation**, not from the full
vocabulary. The public descriptions have been corrected accordingly.

## Problems found

1. **The old CI did not validate its printed diagnostics.** `src/validate.py` printed relative
   derivative errors of `5.6e3` and `1.3e5` but returned success. The blow-up came from dividing by
   almost-zero derivatives at the grid tails. The replacement uses centred finite differences at
   six nondegenerate temperatures and hard tolerances.
2. **The quoted Schottky peak was a grid artefact.** The exact equal-degeneracy value is
   `T*/gap = 0.4167782798...`, not `0.4177`. The README and findings now use the correct value.
3. **The strongest phrasing claim omitted truncation.** The experiment forks and renormalises the
   top 12 candidates. It does not identify what fraction of full-vocabulary entropy is phrasing.
4. **The pooled 58% is descriptive, not a clean replication statistic.** The scatter reduction is
   39% within the five single-seed controlled tiny models and 28% within the seven pretrained
   models. Pooling the two groups yields 58%. The sign is consistent, but the headline magnitude
   depends on pooling heterogeneous groups.
5. **The public one-command reproduction is analysis-stage reproduction.** It checks committed
   derived measurements, not model inference or tiny-model training. The three main probe outputs
   and all trained weights are excluded, so the 3,096-question detector results cannot be rebuilt
   from a fresh clone without downloading models and rerunning the expensive stages.
6. **`data/countries.json` is invalid JSON.** It contains a 301 redirect HTML response and is
   unused by the current pipeline. It should be removed in a cleanup commit.
7. **The semantic experiments remain exploratory.** They use 12–30 prompts per condition, two
   sizes from one model family, deterministic fork continuations, single-linkage clustering, one
   sentence embedding model, and (for detection) an incomplete threshold sensitivity analysis.

## Statistical interpretation

The melting-law data support “including vocabulary size improves the fit in this sample.” They do
not yet justify a universal physical law. The five controlled models use one training seed each,
and the 12 pooled points are not exchangeable: five are related models trained here and seven are
heterogeneous pretrained systems. A publication-grade test needs multiple seeds per vocabulary,
a hierarchical model or cluster bootstrap, and held-out model families.

The free-form contrast is large enough to deserve replication, but its unit of inference is the
question, not each fork or token position. Confidence intervals should resample questions and keep
all forks/positions for a question together.

## Literature boundary

The base thermodynamic framing is not new: [Yang (2024)](https://arxiv.org/abs/2407.21092) defines
partition functions, internal energy, and free energy for language models, while
[Semantic Energy](https://arxiv.org/abs/2508.14496) explicitly uses Boltzmann-inspired logit
energies. [Schottky peak dependence on level degeneracy](https://arxiv.org/abs/1512.05112) is
standard statistical mechanics. [Liao et al. (2025)](https://aclanthology.org/2025.emnlp-main.75/)
also derive a vocabulary-size term while adapting temperature to hold entropy stable during LLM
reinforcement learning. None of these sources, or the targeted searches run for this audit, reports
the real-model top-token occupancy at the heat-capacity maximum tested here.

“No prior result found” is not proof of novelty. The peak-occupancy result should be presented to
experts as a new observation and a request for prior-art correction, not as a guaranteed first.
