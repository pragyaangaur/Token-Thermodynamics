# Novelty audit

Every claim from FINDINGS.md, checked against the literature, September 2026. Verdicts are
my honest read after searching; I could not run an exhaustive review, and "no prior work
found" means exactly that and not "no prior work exists".

## Prior work that is genuinely close, and must be cited

**Tight bound on relative entropy by entropy difference**, Reeb and Wolf, IEEE Trans. Inf. Theory 61(3):1458, 2015 (arXiv 1304.0036). Found on 16 September 2026. It settles two of my claims completely.

Section 2.2.2 is titled "Maximum heat capacity in finite dimensions". Equation 24 reads `C(T) = var_{rho_T}(H/T) = var_{rho_T}(-log rho_T)`. That is the varentropy and heat capacity identity of claim #2, in print since 2015, for any thermal state and with no extra conditions.

Theorem 8 and Corollary 10 go further and cover the two laws in `PEAK_OCCUPANCY.md`. They ask which state on `d` dimensions has the largest surprisal variance. Their answer, Equation 21, is the spectrum `(1-r, r/(d-1), ..., r/(d-1))`, which is one large eigenvalue above a completely degenerate bulk and is the same as my ideal melt model. Their optimal `r` solves `(1-2r) log((1-r)(d-1)/r) = 2`. Substituting `d-1 = V` and `r = u/(1+u)` turns that into `x* = 2(1+u)/(1-u)`, which is my peak condition exactly. Their `1-r_d = 1/2 + 1/log(d-1) + O(1/log^2 d)` is my `p_top = 1/2 + 1/x*`, and their bound `C(T) <= N_d = log^2(d-1)/4 + 1` is my `C_max = (x*^2 - 4)/4`. At `V = 151936` mine gives 36.58 and theirs gives 36.59.

So I rederived their extremal state without knowing it. The `(log V)^2/4` ceiling is a known universal bound, and this repository did not find it. The measurement survives: real next-token distributions sit far below that bound.

**Phase Transitions in the Output Distribution of Large Language Models**, Arnold, Holtorf, Schäfer and Lörch, arXiv 2405.17088, May 2024. Also found on 16 September 2026. This is the closest work to the temperature sweep. It removes my central novelty claim, which used to read "my contribution sits in the temperature dependence, which nobody has explored".

Appendix C sets up the same mapping. They write the per-token distribution as `Q_T(x_i | x_1..x_{i-1}) = exp(-E(x_i|.)/T) / Z_i(T)` and say the conditional energies "are typically referred to as logits". They then sweep the sampling temperature, compute a heat capacity, and look for peaks in it to locate critical points. For Pythia 70M they report two transitions, at `T_1* = 0.02` and `T_2* = 0.5`, and they name three phases: frozen, coherent and disordered. The temperature dependence has therefore been explored, and a high-temperature transition has already been reported.

The two works differ in the level they measure at, and my remaining contribution sits in that difference. They define energy on the **full generated sequence**, `E(x) = -log P(x|T=1)` over `N` tokens. Their footnote 5 says this is not a valid Boltzmann construction, because "while the distribution over individual tokens can be expressed as a Boltzmann distribution at varying temperature, the overall distribution P(x|T) cannot". They report the consequence themselves: the sampling mismatch "can lead to the counterintuitive phenomenon of the mean energy of the system increasing with decreasing temperature corresponding to a negative heat capacity", and it is visible in their Figure 3(b).

At the single next-token distribution that defect cannot occur. The softmax is exactly Boltzmann and the energy is the logit, which does not depend on `T`. By Reeb and Wolf Equation 24 the heat capacity is then a variance of surprisal, so it is non-negative by construction. Their estimate costs 20,480 generated outputs per temperature value for one prompt on a 70M model, and mine costs one forward pass and a closed form.

They do not discuss vocabulary size, logit gaps, degeneration or truncation sampling, and the paper has no closed form for any critical temperature. Their `T_2* = 0.5` and my measured `T_melt` of 1.0 to 1.5 are a factor of two to three apart. I do not yet know why, and the two numbers should not be described as agreeing.

Their reference [110] is a Wolfram Community forum post by Sebastián Bahamondes, "Study of the possibility of phase transitions in LLMs". It looked at the low-temperature transition in GPT-2 and, in their words, "speculated on the existence of a phase transition at higher temperatures". It is a forum post, and it should still be cited.

**Optimizing Temperature for Language Models with Multi-Sample Inference**, Du, Yang and Welleck, ICML 2025 (arXiv 2502.05234), code at github.com/StigLidu/TURN. They define an "entropy turning point" as the temperature where `log H(T)` changes from concave to convex. They locate it by sweeping temperatures and generating at each one, then sample at that point for majority voting and best-of-N. They beat fixed-temperature baselines on MATH and MBPP across 13 models.

Their critical point is an inflection of `log H` and mine is the maximum of `dS/dlog T`. These are different points on the same curve, so the comparison is well defined. They have no closed form and they need a sweep. This paper is useful because it supplies a practical downstream task, which the melting law does not yet have.

**Run on 21 September 2026, in FINDINGS.md section 7m.** The relation turns out to be an
identity rather than a measurement. Because `S'(T) = C(T)/T`, each of the three natural
inflection conditions reduces to `T C'/C = R` with `R` strictly positive, which forces
`C'(T) > 0` and puts every such turning point strictly below the heat capacity peak. The
derivation is elementary and I would expect a physicist to call it folklore about the
rising flank of a Schottky anomaly, so I am not claiming the mathematics. The new parts are
that it settles the sign of a published discrepancy in language models, that the measured
ratio is 1.141 with per-model medians spanning only 1.063x across seven families, and that
their own convention, an inflection of `log H` against a linear `T`, is not invariant under
a global rescaling of the logits and varies 11.2x across families. That last point is the
same class of warning as the mean-versus-mode one in claim #22.

**Tested on 25 September 2026, in FINDINGS.md section 7q.** The practical use this paper seemed to offer did not hold. Sampling at `T_melt/1.141` scores exactly 0 on their MATH set with Llama-3.2-1B, because the single-distribution ratio does not carry over to their sample-averaged curve. So the comparison with their method is now a relation between two temperatures and a documented negative result, and it gives the melting law no downstream task.

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
| 2 | predictive entropy = `S(T=1)`, varentropy = `C(T=1)` | **Not novel. Corrected 16 September 2026.** Reeb and Wolf (2015) Equation 24 states `C(T) = var(-log rho_T)` for any thermal state. I had this as "likely novel as stated for LLMs" on the grounds that nobody had written it down for the entropy/varentropy pair. The general statement covers the LLM case with no extra work, so the distinction I was drawing does not hold. Cite Reeb and Wolf and move on. It still motivates the temperature sweep. |
| 3 | All 7 standard scalars carry no information beyond `S(1)` and `C(1)` | **Novel measurement**, on this task. Individually unsurprising; I found no paper that measures it directly. |
| 4 | Closed-form melting temperature, `x* = 2(1+u)/(1−u)` | **The mathematics is textbook**, since this is the Schottky anomaly peak condition, known since the 1930s. **Corrected 16 September 2026:** it is also exactly the optimality condition in Reeb and Wolf Theorem 8, and Arnold et al. (2405.17088) already compute a heat capacity of an LLM output distribution over a temperature sweep. What is left is that they do it at the sequence level where the Boltzmann mapping fails, and neither paper gives a closed form for the critical temperature. Applying it to the next-token distribution and testing it on real logits still appears novel. |
| 5 | `T_melt log V ≈ Δ` confirmed on real logits across 4 orders of magnitude of `V_eff` | **Novel measurement.** Untested across model families, which is the experiment now running. |
| 6 | `C_max` is exactly scale-invariant, ceiling `(log V)²/4` | **The ceiling is not novel. Corrected 16 September 2026.** It is Reeb and Wolf Corollary 10, a universal bound on the heat capacity of any state in `d` dimensions, `C(T) <= log^2(d-1)/4 + 1`. The scale invariance is a one-line consequence of the mapping and stands. Using peak heat capacity as an LLM uncertainty feature still appears novel, and it is now better posed as the normalised ratio `C_max/N(V)`, which is the fraction of the universal bound a real distribution reaches. |
| 7 | `C_max` alone beats all 7 baselines at the obscurity task | **Novel result.** Needs replication on more models before I would defend it. |
| 8 | Reading entropy at `T_melt/2` instead of `T=1` | **Likely novel.** Temperature scaling (Guo et al. 2017) fits one global `T` against held-out accuracy; this sets a per-item `T` from the distribution's own structure with no fitting and no labels. Different mechanism, different purpose. |
| 9 | Per-item logit scale variation measured at `sd(log T_melt) = 0.133` | **Novel measurement.** I found no paper that quantifies this. It is the reason #8 works. |
| 10 | 96.7% of top candidate first tokens lead to different answers | **Novel measurement, and the result I am most confident is new.** Semantic entropy assumes paraphrase degeneracy matters; nobody appears to have measured how much of it exists at the token level. Directly relevant to when the sampling-based methods are worth their cost. |
| 11 | Semantic entropy does not beat single-pass entropy on short factual QA | **Contradicts the common reading of a published result**, so it needs care. My reference implementation is discrete (string-normalised clustering, K=10), not the bidirectional-entailment version. Stated as a scoped negative, not a refutation. |
| 12 | Configurational/vibrational entropy split | **Framing is standard physics**, the chain-rule decomposition is standard information theory. The mapping onto semantic entropy appears to be new phrasing rather than new mathematics. |
| 13 | The six falsifications | Negative results on my own hypotheses. Novel only in the sense that nobody had proposed them to falsify. |

## Honest summary

Rewritten 16 September 2026 after the three references above.

Two things look new and defensible. The first is the **96.7% degeneracy measurement** (#10). The second is the **closed-form melting temperature confirmed on real token distributions across 12 models with vocabulary size controlled** (#4, #5). `C_max` as a scale-free knowledge indicator (#6, #7) still stands as a feature, with the ceiling now attributed to Reeb and Wolf. The entropy/varentropy identity (#2) is published work from 2015 and is useful only as the door into the rest.

Semantic Energy got to the base mapping first. The previous version of this section claimed the temperature dependence as mine on the grounds that nobody had explored it. Arnold et al. explored it in May 2024, so that claim is withdrawn.

Here is what is left, stated as narrowly as I can. Arnold et al. build the heat capacity on the full generated sequence, where they show the Boltzmann mapping does not hold and where they report negative heat capacity as a result. At the single next-token distribution the mapping is exact, so the heat capacity is a surprisal variance and cannot go negative. The measurement costs one forward pass, where theirs costs 20,480 generations per temperature point, and the peak has a closed form in the logit gap and the vocabulary size. This claim is narrower than the one I started with, and I can defend it against a named paper.

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
| 19 | Forking only the first 1 to 2 tokens matches full semantic entropy at 1.6x lower cost | **Withdrawn 21 September 2026.** The clustering-threshold sweep in FINDINGS.md section 7n shows the parity holds only near the threshold originally used, and that the expensive method wins at four of the six thresholds. The cost saving is real and the parity is not. Section 7l had already found it did not replicate on a second model. |
| 20 | Plain token entropy is at chance (0.523) separating known from obscure free-form questions | **Novel measurement**, and the most useful negative in the project. Strengthened 21 September 2026: it involves no clustering, so it is the only number in that section the threshold sweep leaves untouched. |
| 21 | The melting law confirmed across 12 models spanning 976x in vocabulary | **Novel.** The Schottky peak condition is textbook physics from the 1930s; applying it to token distributions and testing it across models, including five trained specifically to vary V, does not appear to have been done. |
| 22 | Measuring the logit gap to the mean is unsafe across model families | **Novel as a stated warning**, with a concrete demonstration (a 9.9x spread and the wrong sign, fixed by using the mode). Several published measures aggregate logits by mean. Checked 23 September 2026 in FINDINGS.md section 7p: LogTokU and Semantic Energy read raw logit values, and used across model families both rank models perfectly by their logit offset. Neither paper compares across models, so no published result changes. |
| 24 | The entropy turning point lies on a rising flank of the heat capacity, so below the melting temperature when the heat capacity has one peak | **The mathematics is elementary** and probably folklore in thermodynamics; I found no statement of it. **Novel as applied here**, because it explains the published factor of two to three and turns a one-forward-pass quantity into an estimate of a temperature that is currently found by sweeping. |
| 25 | The `log H` against linear `T` turning point is not scale invariant and spans 11.2x across model families | **Novel as a stated warning**, same class as #22. |
| 23 | Meaning free energy correction to greedy decoding | **Not novel as an idea.** This is the mode-seeking problem that Minimum Bayes Risk decoding addresses. My contribution is only the negative measurement that the correction is worth 0.000 here. |

## Honest summary of the whole project

Ranked by how confident I am that each is both new and worth something:

1. The quantified phrasing-versus-meaning split and its position profile (#15 to #18, #20).
   The concept is old, the numbers are new, and they say something specific about when the
   expensive methods earn their cost. The absolute values depend heavily on the clustering
   threshold, so the ones to lean on are the threshold-free ones in #20.
2. The melting law and its cross-model confirmation (#4, #5, #21).
3. The mean-versus-mode warning (#22), which is small and immediately actionable.
4. `C_max` as a scale-free knowledge indicator (#6, #7), which needs more models and should now be reported as a fraction of the Reeb and Wolf bound.
5. The entropy/varentropy identity (#2), which is Reeb and Wolf Equation 24 and is useful only as the door into the temperature sweep.

These are not mine:

- the energy mapping (Semantic Energy, arXiv 2508.14496 got there first)
- the phrasing-versus-meaning concept (the semantic entropy line of work)
- the Schottky peak condition (1930s solid state physics)
- the maximum-heat-capacity state and its bound (Reeb and Wolf 2015)
- the heat capacity of an LLM output distribution over a temperature sweep (Arnold et al. 2024)
- the mode-seeking critique of greedy decoding (the MBR decoding literature)
