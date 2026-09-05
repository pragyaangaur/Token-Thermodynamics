# Novelty audit

Every claim from FINDINGS.md, checked against the literature, September 2026. Verdicts are
my honest read after searching; I could not run an exhaustive review, and "no prior work
found" means exactly that and not "no prior work exists".

## Prior work that is genuinely close, and must be cited

**Semantic Energy: Detecting LLM Hallucination Beyond Entropy** (arXiv 2508.14496).
This is the closest paper to my setup and I did not know about it when I started. It uses
**the same mapping I use**, energy = negative logit, explicitly Boltzmann-inspired, with
`kτ = 1` because that is the training temperature. So the mapping itself is not new.

What it does not do, confirmed by reading the paper: no temperature sweep, no heat
capacity or any thermodynamic derivative, no scale-invariance analysis, no free energy,
no partition function normalisation. It uses the mean energy of a sequence, aggregated
over sampled semantic clusters, so it still needs sampling and clustering.

There is also a real disagreement worth stating. Semantic Energy's motivation is that
softmax throws away the logits' "intensity", meaning the overall scale, and that this
intensity is signal. My section 4 measures that same scale, finds it varies per item at
`sd(log T_melt) = 0.133`, and shows that dividing it out *improves* detection and makes
the measure exactly invariant. We are using the same quantity with opposite sign of
intent. That is a concrete, testable disagreement rather than a duplication.

**Estimating LLM Uncertainty with Logits / LogTokU** (arXiv 2502.00290). Fits a Dirichlet
evidence model to the top-K raw logits and decouples aleatoric from epistemic uncertainty
in closed form with no sampling. This is the same ambition as my single-pass work and the
same raw material. It also treats logit magnitude as evidence strength, so it sits on the
same side of the disagreement above. Different mathematics (evidential/Dirichlet rather
than thermodynamic), and no temperature dependence.

**Neural Thermodynamic Laws for LLM Training** (arXiv 2505.10559). Maps learning rate to
temperature and defines a heat capacity, but for **training dynamics on the loss
landscape**, not for the output distribution at inference. Different object entirely.
Worth citing so the two are not confused.

**Entropy, Thermodynamics and the Geometrization of the Language Model** (arXiv
2407.21092) builds the partition function and free energy for a language model, so the
general framing is not new. **Detecting hallucinations using semantic entropy** (Nature
2024) and **Semantic Entropy Probes** (arXiv 2406.15927) are the baselines I measure
against. **Entropix** is the origin of varentropy as a practical feature.

## Claim-by-claim verdict

| # | Claim | Verdict |
|---|---|---|
| 1 | energy = negative logit; softmax sampling is Boltzmann | **Not novel.** Semantic Energy states it explicitly. Older statistical-physics-of-LLMs work implies it. |
| 2 | predictive entropy = `S(T=1)`, varentropy = `C(T=1)` | **Likely novel as stated for LLMs.** That varentropy relates to heat capacity is standard statistical mechanics (see arXiv 2603.27997 for the general identity). No source found making the connection for the LLM entropy/varentropy pair, and searches of the entropix literature turned up no thermodynamic reading of it. Low-difficulty result, but it does not appear to have been written down, and it is what motivates the temperature sweep. |
| 3 | All 7 standard scalars carry no information beyond `S(1)` and `C(1)` | **Novel measurement**, on this task. Individually unsurprising; I found no paper that measures it directly. |
| 4 | Closed-form melting temperature, `x* = 2(1+u)/(1−u)` | **The mathematics is textbook** — this is the Schottky anomaly peak condition, known since the 1930s. **Applying it to a next-token distribution and testing it on real logits appears novel.** No prior work found on Schottky anomalies or specific heat of an LLM output distribution. |
| 5 | `T_melt log V ≈ Δ` confirmed on real logits across 4 orders of magnitude of `V_eff` | **Novel measurement.** Untested across model families, which is the experiment now running. |
| 6 | `C_max` is exactly scale-invariant, ceiling `(log V)²/4` | **Novel for LLMs.** The invariance is a one-line consequence of the mapping, and the ceiling follows from #4, but I found no work using peak heat capacity as an LLM uncertainty feature. |
| 7 | `C_max` alone beats all 7 baselines at the obscurity task | **Novel result.** Needs replication on more models before I would defend it. |
| 8 | Reading entropy at `T_melt/2` instead of `T=1` | **Likely novel.** Temperature scaling (Guo et al. 2017) fits one global `T` against held-out accuracy; this sets a per-item `T` from the distribution's own structure with no fitting and no labels. Different mechanism, different purpose. |
| 9 | Per-item logit scale variation measured at `sd(log T_melt) = 0.133` | **Novel measurement.** I found no paper that quantifies this. It is the reason #8 works. |
| 10 | 96.7% of top candidate first tokens lead to different answers | **Novel measurement, and the result I am most confident is new.** Semantic entropy assumes paraphrase degeneracy matters; nobody appears to have measured how much of it exists at the token level. Directly relevant to when the sampling-based methods are worth their cost. |
| 11 | Semantic entropy does not beat single-pass entropy on short factual QA | **Contradicts the common reading of a published result**, so it needs care. My reference implementation is discrete (string-normalised clustering, K=10), not the bidirectional-entailment version. Stated as a scoped negative, not a refutation. |
| 12 | Configurational/vibrational entropy split | **Framing is standard physics**, the chain-rule decomposition is standard information theory. The mapping onto semantic entropy appears to be new phrasing rather than new mathematics. |
| 13 | The six falsifications | Negative results on my own hypotheses. Novel only in the sense that nobody had proposed them to falsify. |

## Honest summary

Three things look genuinely new and defensible: the **96.7% degeneracy measurement**
(#10), the **melting temperature applied to and confirmed on real token distributions**
(#4, #5), and **`C_max` as a scale-free knowledge indicator** (#6, #7). The
entropy/varentropy identity (#2) is easy mathematics that nobody seems to have written
down, and it is useful mainly as the door into the rest.

The base mapping is not mine, and Semantic Energy got there first. My contribution sits
in the temperature dependence, which nobody has explored.

---

# Novelty audit, part two: the free-form results

Searched September 2026, same caveat as above.

## The most important correction to my own claims

**The concept I thought I was contributing is the founding motivation of the field I was
measuring against.** The semantic entropy literature states plainly that token-level
uncertainty "conflates semantic uncertainty (uncertainty over the meaning of the
generation) with lexical and syntactic uncertainty (uncertainty over how to phrase the
answer)". That is precisely my phrasing-versus-meaning split, and Kuhn, Gal and Farquhar
built semantic entropy to fix it. It is also already known empirically that
"token-probability based approaches become less effective when the length of generation
increases".

So the idea is not new. What I can find no prior work on is the **quantification**:

- Nobody appears to have measured *how much* of the entropy among the leading next-token
  candidates is phrasing rather than meaning. My numbers are 94.7% for free-form and 13.2%
  for short factual, measured on the renormalised top 12 candidates by forking and continuing
  each. The full-vocabulary entropy fraction remains unmeasured.
- Nobody appears to have mapped it **position by position** along a generated answer, or
  reported that meaning share falls monotonically while token entropy rises.
- I found no prior report of the **anticorrelation** as a measured quantity
  (Spearman −0.230), nor of the specific finding that the highest-entropy positions in a
  free-form answer have a mean meaning share of exactly zero.

The closest related work is **Semantic Entropy Probes** (arXiv 2406.15927), which trains
probes on hidden states at particular token positions to approximate semantic entropy from
one generation. Same goal of cutting the 5-to-10x sampling cost, and it also cares about
position. Different mechanism: it reads internal activations, where I fork the output
distribution and measure where the answer actually diverges. The two are complementary,
and honestly theirs is the more practical method.

## Claim-by-claim verdict

| # | Claim | Verdict |
|---|---|---|
| 14 | Token entropy conflates phrasing with meaning | **Not novel at all.** This is the stated motivation of the entire semantic entropy line of work. I arrived at it independently, which is not the same as it being new. |
| 15 | 94.7% of renormalised top-12 first-token entropy is phrasing, 13.2% for short factual | **Novel as a measurement.** The concept is old, the scoped number does not appear to have been measured. |
| 16 | 96.7% of top candidate first tokens lead to a different answer on factual questions | **Novel measurement**, and still the result I am most confident is new. |
| 17 | Position-resolved meaning share, monotone decreasing, entropy rising | **Novel measurement.** Semantic Entropy Probes look at positions but for a different purpose. |
| 18 | Spearman(renormalised top-8 entropy, meaning share) = −0.230; its top-quintile positions have mean meaning share 0.0000 | **Novel measurement.** The direction is folklore; the scoped number is not published anywhere I can find. |
| 19 | Forking only the first 1 to 2 tokens matches full semantic entropy at 1.6x lower cost | **Novel method**, though a small and lightly-tested one (16 questions per condition). Semantic Entropy Probes achieve a bigger cost saving by a different route. |
| 20 | Plain token entropy is at chance (0.523) separating known from obscure free-form questions | **Novel measurement**, and the most useful negative in the project. |
| 21 | The melting law confirmed across 12 models spanning 976x in vocabulary | **Novel.** The Schottky peak condition is textbook physics from the 1930s; applying it to token distributions and testing it across models, including five trained specifically to vary V, does not appear to have been done. |
| 22 | Measuring the logit gap to the mean is unsafe across model families | **Novel as a stated warning**, with a concrete demonstration (a 9.9x spread and the wrong sign, fixed by using the mode). Several published measures aggregate logits by mean. |
| 23 | Meaning free energy correction to greedy decoding | **Not novel as an idea.** This is the mode-seeking problem that Minimum Bayes Risk decoding addresses. My contribution is only the negative measurement that the correction is worth 0.000 here. |

## Honest summary of the whole project

Ranked by how confident I am that each is both new and worth something:

1. The quantified phrasing-versus-meaning split and its position profile (#15 to #18, #20).
   The concept is old, the numbers are new, and they say something specific about when the
   expensive methods earn their cost.
2. The melting law and its cross-model confirmation (#4, #5, #21).
3. The mean-versus-mode warning (#22), which is small but immediately actionable.
4. `C_max` as a scale-free knowledge indicator (#6, #7), which needs more models.
5. The entropy/varentropy identity (#2), which is easy mathematics that nobody seems to
   have written down, and is useful mainly as the door into the temperature sweep.

Not mine: the energy mapping (Semantic Energy, arXiv 2508.14496 got there first), the
phrasing-versus-meaning concept (the semantic entropy line of work), the Schottky peak
condition (1930s solid state physics), and the mode-seeking critique of greedy decoding
(the MBR decoding literature).
