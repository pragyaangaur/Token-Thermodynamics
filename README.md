# Temperature spectroscopy of a language model's next-token distribution

[![reproduce](https://github.com/pragyaangaur/Token-Thermodynamics/actions/workflows/repro.yml/badge.svg)](https://github.com/pragyaangaur/Token-Thermodynamics/actions/workflows/repro.yml)

Softmax sampling in a language model is exactly the Boltzmann distribution of statistical mechanics. This repository builds the rest of the thermodynamics on top of that identity and measures what it buys for hallucination detection.

Most of the ideas here were wrong, and the experiments say so. Eight hypotheses were falsified, including one method that worked on the first model, failed to replicate on the second, and has now also failed a clustering-threshold sweep on the first model. It is withdrawn. The negative results are kept in full because they are the most informative part.

Everything runs on a laptop. The headline result reproduces in about one second with no GPU and no model download.

```bash
pip install -r requirements.txt
python repro.py
```

## The identity this is built on

Set the energy of a token to the negative of its logit, $E_i = -\ell_i$. Then sampling at temperature $T$ gives

$$p_i = \frac{e^{-E_i/T}}{Z(T)}, \qquad Z(T) = \sum_j e^{-E_j/T}$$

which is the Boltzmann distribution, so the usual thermodynamic quantities become computable from one forward pass. Two of them are already in use under other names:

- Predictive entropy is exactly $S(T=1)$.
- Varentropy, the variance of surprisal used in adaptive samplers, is exactly the heat capacity $C(T=1)$.

The second follows in one line. Surprisal is $-\log p_i = \beta E_i + \log Z$, so its variance is $\beta^2 \mathrm{Var}(E)$, which is the heat capacity

$$C(T) = \frac{dU}{dT} = \frac{\mathrm{Var}(E)}{T^2}.$$

This relation between surprisal variance and heat capacity is published in Reeb and Wolf (2015), Equation 24, and this repository does not claim it. Both identities were verified numerically to 1 part in $10^{14}$ as a check on the implementation. On 3096 factual questions, all seven common baseline scalars together score the same as those two alone, AUROC 0.8575 against 0.8575, with a paired bootstrap 95% CI of $[-0.0016, +0.0015]$.

## Main results

**The melting law.** Heat capacity peaks where probability mass leaves the top token and floods the vocabulary. Solving $dC/dT = 0$ for one ground state above $V$ states at gap $\Delta$ gives the exact peak condition

$$x^\ast = \frac{2(1+u)}{1-u}, \qquad u = V e^{-x^\ast}, \qquad T_\mathrm{melt} = \frac{\Delta}{x^\ast},$$

where $x^\ast = \log V + c$ with $c$ a slow correction of order one. For a vocabulary of 151936 tokens, $x^\ast = 12.26$.

This is confirmed across 12 models spanning 976x in vocabulary size, including 5 small transformers trained here on identical data where only the vocabulary size changed. Including the $\log V$ term cuts model-to-model scatter by 58 percent, and the bootstrap 95% CI on that improvement is $[-0.269, -0.082]$.

**Free-form and factual generation are opposite regimes.** Forking the top 12 candidate first tokens and continuing each to see where it lands shows that 94.7 percent of the entropy within that renormalised candidate set is pure phrasing carrying no information about meaning. For a short factual answer the figure is 13.2 percent. This replicates on a second model at 90.6 percent and 4.8 percent. These are not percentages of the full-vocabulary entropy.

**Token entropy points the wrong way in the free-form regime, and the direction is not stable.** Reporting signed AUROC, where above 0.5 means the score rises with fabrication, plain token entropy scores 0.098 on Qwen2.5-0.5B for known against fabricated questions. Higher entropy there means the model knows the answer. On Qwen2.5-1.5B it points the right way for one comparison and the wrong way for another. Sampled semantic entropy keeps the correct sign in all four comparisons.

**The entropy turning point is forced to sit below the melting temperature.** Du, Yang and Welleck (2025) choose a sampling temperature at the inflection of the entropy curve, found by sweeping temperatures and generating at each one. Because $S'(T) = C(T)/T$, all three natural ways of writing that inflection reduce to $T C'/C = R$ with $R > 0$, which forces $C'(T) > 0$. The heat capacity is still rising there, and the melting temperature is where it stops rising. So every turning point lies on a rising flank of the heat capacity, with no free parameter, and when the heat capacity has a single peak, which is the usual case, it lies strictly below the melting temperature. On 7 models and 140 distributions from 20 generic prompts the ratio is 1.141, and its per-model medians span only 1.063x. The ratio depends on the kind of text, though. On the MATH problems Du, Yang and Welleck use it is about 1.03, so one constant does not carry across domains. The convention that mixes $\log S$ with a linear $T$ is not scale invariant, and its per-model medians span 11.2x. Sampling at the temperature this predicts does not work. On Llama-3.2-1B-Instruct and their MATH set, $T_\mathrm{melt}/1.141$ is 1.50 and majority-vote accuracy there is exactly 0, against 0.390 at the best swept temperature. The single-distribution ratio does not carry over to their curve, which is averaged over sampled tokens: $T_\mathrm{melt}$ over their turning point is 2.45 on the 1B model and 2.98 on the 3B. The protocol, the decision rule fixed before the run and the result are in [EXPERIMENT_A.md](EXPERIMENT_A.md), and FINDINGS.md section 7q discusses it.

**A measurement error that looked like a refutation.** The cross-model melting test first appeared to fail badly, with a 9.7x spread and the wrong sign. The cause was measuring the energy gap $\Delta$ to the mean logit. Some models have extreme outlier logits that pull the mean far from where the states actually sit, and measured to the mean, the gap comes out 2.7 times too large on Pythia and 8.0 times too large on BLOOMZ. Measuring $\Delta$ to the mode instead, which is what the theory asks for, brings all 12 models into line. Two published uncertainty scores, LogTokU and Semantic Energy, read raw logit values. Neither paper compares across models, so their results stand. Used across the seven model families here, though, both rank the models perfectly by their logit offset (Spearman −1.000) and not by their uncertainty, and measuring the logits from the mode fixes it. This is in FINDINGS.md section 7p.

Full detail, including every falsified hypothesis, is in [FINDINGS.md](FINDINGS.md). A literature audit of which claims are new and which are not is in [NOVELTY.md](NOVELTY.md). A plain English version is in [ELI5.md](ELI5.md).

## Closest prior work

Three papers are close enough that anyone reading this repository should know them first. The comparison in detail is in [NOVELTY.md](NOVELTY.md).

- **Arnold, Holtorf, Schäfer and Lörch (2024)**, *Phase Transitions in the Output Distribution of Large Language Models*, [arXiv 2405.17088](https://arxiv.org/abs/2405.17088). They treat logits as energies, sweep the sampling temperature, compute a heat capacity, and report a high-temperature transition. They work on the full generated sequence, where they note that the Boltzmann mapping does not hold and the heat capacity can go negative. This repository works on the single next-token distribution, where the mapping is exact and the heat capacity cannot be negative.
- **Reeb and Wolf (2015)**, *Tight bound on relative entropy by entropy difference*, [arXiv 1304.0036](https://arxiv.org/abs/1304.0036). Equation 24 is the varentropy and heat capacity identity. Theorem 8 and Corollary 10 give the maximum-heat-capacity state and its bound $\log^2(d-1)/4 + 1$, which the ideal melt model in [PEAK_OCCUPANCY.md](PEAK_OCCUPANCY.md) rederives.
- **Du, Yang and Welleck (2025)**, *Optimizing Temperature for Language Models with Multi-Sample Inference*, [arXiv 2502.05234](https://arxiv.org/abs/2502.05234). They choose a sampling temperature from the turning point of the entropy curve, found by a temperature sweep. The melting temperature is a different point on a related curve and comes from one forward pass. That comparison is in [FINDINGS.md](FINDINGS.md) section 7m. The turning point is provably below the melting temperature. The measured ratio is stable across model families on one kind of text and moves between kinds of text. Sampling at a temperature predicted from the melting temperature was tested against their method and failed (FINDINGS.md section 7q).

The closed form $T_\mathrm{melt} = \Delta/(\log V + c)$ and its test across 12 models with vocabulary size controlled do not appear in any of them.

## Limitations

These matter, and they are stated here rather than buried.

- The free-form results rest on 12 to 30 questions per condition and at most 2 models, both from the Qwen family. The effects are large and the sample sizes are small.
- Meaning clustering uses sentence embeddings at one similarity threshold, and this matters more than I expected. A sensitivity sweep across six thresholds overturned one headline number in FINDINGS.md section 7i and a second one in section 7n, and section 7o shows on the second model that the direction of the withdrawn forking detector also flips with the threshold. Every clustered AUROC in the project should be read as a point on a curve rather than as a value. The detector that scored 0.820 on the clean comparison ranges from 0.562 to 0.914 across the six thresholds.
- The reference semantic entropy is the cheap string-clustering version, not bidirectional entailment. The comparison against it on short factual questions may therefore be unfair to the published method.
- The fabricated-entity conditions use invented names, so any comparison against them partly reads orthography. The clean comparisons use real entities split by obscurity.
- Everything is measured on the first answer token unless stated otherwise, which section 7f shows is a regime-dependent choice.

## Installing

Python 3.12 or newer, because the pinned numpy and scipy require it. There are two requirement files, because checking the main claim should not mean installing PyTorch.

| File | What it covers | Size |
| --- | --- | --- |
| `requirements.txt` | `repro.py`, all analysis, all figures | about 100 MB |
| `requirements-models.txt` | everything above, plus running the models themselves | several GB |

```bash
pip install -r requirements.txt          # enough to reproduce and analyse
pip install -r requirements-models.txt   # only if you want to regenerate from the models
```

## Repository layout

```
repro.py                      reproduces the headline melting-law table from committed data
src/thermo.py                 exact thermodynamics of a logit vector, plus the Schottky reference
src/dos.py                    density-of-states compression, errors of order 1e-5 and about 90x faster
src/features.py               baseline uncertainty scalars, computed on the full vocabulary
src/curvefeats.py             spectral features of a heat-capacity curve
src/probe.py                  Wikidata question probe and grading
src/vocab_experiment.py       trains the 5 tiny models that vary only in vocabulary size
src/multimodel_melt.py        melting law across pretrained model families
src/freeform_degeneracy.py    phrasing versus meaning, free-form against short factual
src/where_meaning.py          position-resolved split of entropy into phrasing and meaning
src/report.py                 detector evaluation, cross-validation and paired bootstraps
src/turning_point.py          entropy turning point against the melting temperature
src/ffh_texts.py              regenerates the free-form detection run keeping raw strings
src/ffh_thr.py                reclusters that run at six thresholds, no model needed
src/experiment_a.py           samples at T_melt/1.141 against TURN, needs a GPU
src/logit_offset.py           raw-logit uncertainty scores across model families
notebooks/                    Kaggle notebook that runs experiment_a.py
scripts/make_archive.sh       builds a release zip from git, for archiving
```

## Regenerating from scratch

The three largest probe outputs are not committed because they total about 170 MB. Everything else needed by `repro.py` is in `data/`. To rebuild the large files:

```bash
pip install -r requirements-models.txt
python src/fetch_data.py                                    # Wikidata ground truth
python src/probe.py --model Qwen/Qwen2.5-1.5B-Instruct --out data/main_1p5b.json
python src/probe.py --model Qwen/Qwen2.5-1.5B-Instruct --out data/fake_1p5b.json --fake
python src/report.py data/main_1p5b.json data/fake_1p5b.json
```

To rebuild the melting-law inputs, which is what `repro.py` reads:

```bash
python src/multimodel_melt.py --out data/multimodel_melt3.json      # downloads 7 models
python src/vocab_experiment.py --out data/vocab_exp.json --epochs 3 # trains 5 tiny models, about 20 minutes
```

The probe runs took about 30 minutes each on an Apple M4 in float32. Float32 is deliberate, because the method reads fine structure in the logits. The noise experiment reported in FINDINGS.md section 6 shows that the integral features tolerate logit noise of $5 \times 10^{-2}$.

## Validation

The implementation is checked against analytic results rather than only against itself. Continuous integration runs `repro.py` and `src/validate.py` on a clean machine on every push. Both scripts contain explicit assertions and exit nonzero if a stated tolerance or headline condition fails.

| Check | Agreement |
| --- | --- |
| $F = U - TS$ | $10^{-11}$ |
| $dU/dT = C$ at six independently differenced temperatures | relative error below $10^{-6}$ |
| Two-level Schottky heat capacity, including the universal peak $T^\ast/\Delta = 0.41678\ldots$ | curve error below $2 \times 10^{-14}$ |
| Density-of-states compression against exact summation | absolute entropy error below $10^{-5}$; heat-capacity error below $2 \times 10^{-5}$ |

```bash
python src/validate.py
```
