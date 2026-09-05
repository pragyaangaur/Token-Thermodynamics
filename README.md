# Temperature spectroscopy of a language model's next-token distribution

Softmax sampling in a language model is exactly the Boltzmann distribution of statistical mechanics. This repository builds the rest of the thermodynamics on top of that identity and measures what it buys for hallucination detection.

Most of the ideas here were wrong, and the experiments say so. Seven hypotheses were falsified, including one method that worked on the first model, failed to replicate on the second, and has been withdrawn. The negative results are kept in full because they are the most informative part.

Everything runs on a laptop. The headline result reproduces in about one second with no GPU.

```bash
pip install -r requirements.txt
python repro.py
```

## The identity this is built on

Set the energy of a token to the negative of its logit. Then sampling at temperature `T` is `p_i = exp(-E_i/T)/Z(T)`, which is the Boltzmann distribution, and the usual thermodynamic quantities become computable from one forward pass. Two of them are already in use under other names:

- Predictive entropy is exactly `S(T=1)`.
- Varentropy, the variance of surprisal used in adaptive samplers, is exactly the heat capacity `C(T=1)`. Surprisal is `beta*E + log Z`, so its variance is `beta^2 Var(E)`, which is `C`.

Both were verified numerically to 1 part in 10^14. On 3096 factual questions, all seven common baseline scalars together score the same as those two alone, AUROC 0.8575 against 0.8575, with a paired bootstrap 95% CI of [-0.0016, +0.0015].

## Main results

**The melting law.** Heat capacity peaks where probability mass leaves the top token and floods the vocabulary. Solving `dC/dT = 0` gives a closed form, `T_melt = Delta / x*(V)` with `x* = log V + c`. This is confirmed across 12 models spanning 976x in vocabulary size, including 5 small transformers trained here on identical data where only the vocabulary size changed. Including the vocabulary term cuts model-to-model scatter by 58 percent, and the bootstrap 95% CI on that improvement is [-0.269, -0.082].

**Free-form and factual generation are opposite regimes.** Forking every top candidate first token and continuing it to see where it lands shows that 94.7 percent of the entropy at the first token of a free-form answer is pure phrasing carrying no information about meaning. For a short factual answer the figure is 13.2 percent. This replicates on a second model at 90.6 percent and 4.8 percent.

**Token entropy points the wrong way in the free-form regime, and the direction is not stable.** Reporting signed AUROC, where above 0.5 means the score rises with fabrication, plain token entropy scores 0.098 on Qwen2.5-0.5B for known against fabricated questions. Higher entropy there means the model knows the answer. On Qwen2.5-1.5B it points the right way for one comparison and the wrong way for another. Sampled semantic entropy keeps the correct sign in all four comparisons.

**A measurement error that looked like a refutation.** The cross-model melting test first appeared to fail badly, with a 9.7x spread and the wrong sign. The cause was measuring the energy gap to the mean logit. Some models have extreme outlier logits that pull the mean far from where the states actually sit, and Pythia's mean logit is 2.2 standard deviations from its mode. Measuring to the mode instead, which is what the theory asks for, brings all 12 models into line. Several published uncertainty measures aggregate logits by mean, and that choice is worth checking before comparing across model families.

Full detail, including every falsified hypothesis, is in [FINDINGS.md](FINDINGS.md). A literature audit of which claims are new and which are not is in [NOVELTY.md](NOVELTY.md). A plain English version is in [ELI5.md](ELI5.md).

## Limitations

These matter, and they are stated here rather than buried.

- The free-form results rest on 12 to 30 questions per condition and at most 2 models, both from the Qwen family. The effects are large and the sample sizes are small.
- Meaning clustering uses sentence embeddings at one similarity threshold. A sensitivity sweep across six thresholds overturned one of the headline numbers, and that correction is recorded in FINDINGS.md section 7i. The same sweep has not been repeated for the detection results in section 7k.
- The reference semantic entropy is the cheap string-clustering version, not bidirectional entailment. The comparison against it on short factual questions may therefore be unfair to the published method.
- The fabricated-entity conditions use invented names, so any comparison against them partly reads orthography. The clean comparisons use real entities split by obscurity.
- Everything is measured on the first answer token unless stated otherwise, which section 7f shows is a regime-dependent choice.

## Repository layout

```
repro.py                 reproduces the headline melting-law table from committed data
src/thermo.py            exact thermodynamics of a logit vector, plus the Schottky reference
src/dos.py               density-of-states compression, 1e-5 accurate and about 90x faster
src/features.py          baseline uncertainty scalars, computed on the full vocabulary
src/curvefeats.py        spectral features of a heat-capacity curve
src/probe.py             Wikidata question probe and grading
src/vocab_experiment.py  trains the 5 tiny models that vary only in vocabulary size
src/multimodel_melt.py   melting law across pretrained model families
src/freeform_degeneracy.py   phrasing versus meaning, free-form against short factual
src/where_meaning.py     position-resolved split of entropy into phrasing and meaning
src/report.py            detector evaluation, cross-validation and paired bootstraps
```

## Regenerating from scratch

The three largest probe outputs are not committed because they total about 170 MB. Everything else needed by `repro.py` is in `data/`. To rebuild the large files:

```bash
python src/fetch_data.py                                    # Wikidata ground truth
python src/probe.py --model Qwen/Qwen2.5-1.5B-Instruct --out data/main_1p5b.json
python src/probe.py --model Qwen/Qwen2.5-1.5B-Instruct --out data/fake_1p5b.json --fake
python src/report.py data/main_1p5b.json data/fake_1p5b.json
```

To rebuild the melting-law inputs, which is what `repro.py` reads:

```bash
python src/multimodel_melt.py --out data/multimodel_melt3.json    # needs 7 models downloaded
python src/vocab_experiment.py --out data/vocab_exp.json --epochs 3   # trains 5 tiny models, about 20 minutes
```

The probe runs took about 30 minutes each on an Apple M4 in float32. Float32 is deliberate, because the method reads fine structure in the logits, though `src/thermo.py` includes a noise robustness check showing the integral features tolerate logit noise of 5e-2.

## Validation

The implementation is checked against analytic results rather than only against itself:

- `F = U - TS` to 1e-11
- `dU/dT = C` to 3e-6 on the temperature grid
- The two-level Schottky heat capacity reproduced to 5e-16, including the universal peak position `T*/gap = 0.4177`
- The density-of-states compression against exact full-vocabulary summation to 1e-5

Run `python src/validate.py` to see all of these.
