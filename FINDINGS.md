# Temperature spectroscopy of a language model's next-token distribution

**What this is.** I mapped LLM sampling onto statistical mechanics properly, which is an
identity rather than an analogy, and then asked what the machinery buys. Most of my ideas
were wrong and the experiments say so. The work split into two halves: instrument-building
on short factual answers (sections 1 to 6), and then a set of measurements about free-form
generation that turned out to matter more (sections 7f to 7k).

**The four strongest results.**

1. **Free-form and factual generation are opposite regimes.** Among the renormalised top 12
   candidates at the first token of a free-form answer, 94.7% of the entropy is pure phrasing
   and carries no information about meaning. For a short factual answer it is 13.2%. Measured
   by forking those candidates and continuing each to see where it lands; these percentages
   do not describe the omitted vocabulary tail.
2. **Token entropy is anticorrelated with meaning-relevance in free-form text**
   (Spearman −0.23, and the direction holds at every clustering threshold tested). The
   highest-entropy positions in an answer carry a mean meaning share of 0.0000. On real
   questions, plain token entropy separates known from obscure topics at chance, 0.523.
3. **The sign of the token-entropy signal flips between models.** On Qwen2.5-0.5B, higher
   first-token entropy means the model *knows* the answer (signed AUROC 0.098, strongly
   backwards); on the 1.5B it points the right way for one comparison and the wrong way for
   another. A measure whose direction you cannot predict is not a weak detector, it is not a
   detector at all. Semantic entropy keeps the correct sign in all four comparisons.
4. **Every token distribution has a melting temperature with a closed form**,
   `T_melt = Delta/(log V + c)`, now confirmed across **12 models spanning 976x in
   vocabulary size**, including 5 trained specifically for the test with only V changed.
   Including the `log V` term cuts model-to-model scatter by 58%, bootstrap CI on the
   improvement [−0.269, −0.082].

**Also holds.** Predictive entropy is exactly `S(T=1)` and varentropy is exactly `C(T=1)`,
so the field's two standard scalars are thermodynamic quantities read at one arbitrary
point, and all seven common baselines together score identically to those two alone.
`C_max` is exactly scale-invariant with a known ceiling, and on the cleanest knowledge task
it beats all seven baselines combined. Reading entropy at each item's own melting
temperature is scale-invariant at no cost in accuracy.

**Did not survive.** Discrete band structure in the spectrum. Top-k truncation raising the
melting temperature. Separating paraphrase from meaning by temperature. Energy-gap
clustering. The meaning-free-energy correction to greedy decoding. A multivariate accuracy
gain (one of two models). Any useful gain in selective prediction. And one headline figure
of my own, that 98.8% of meaning sits in the first three tokens, which the threshold
sensitivity analysis in 7i partly overturned.

**A methodological warning worth extracting.** My cross-model melting test first appeared
to fail badly, with a 9.7x spread and the wrong sign. The cause was measuring the energy
gap to the *mean* logit. Some models have extreme outlier logits that drag the mean far
from where the states actually sit. Using the mode instead, which is what the theory asks
for, all 12 models line up. Several published uncertainty measures aggregate logits by
mean, and this is a concrete reason to check that choice before comparing across models.


Working notes. Every number below was produced by the code in `src/` on this machine.
Model: Qwen2.5-1.5B-Instruct (and 0.5B for replication), float32, Apple M4.

## 0. The setup is an identity, not an analogy

For logits `l_i`, set the energy `E_i = -l_i`. Then softmax sampling at temperature
`T` is exactly the Boltzmann distribution `p_i = exp(-E_i/T)/Z(T)`, and the whole
thermodynamic apparatus is computable from one forward pass:

| quantity | formula | meaning for an LLM |
|---|---|---|
| partition function | `Z(T) = sum exp(-E_i/T)` | softmax denominator |
| free energy | `F = -T log Z` | log-sum-exp of the logits |
| internal energy | `U = <E>` | negative mean logit under `p` |
| entropy | `S = -sum p log p` | the usual predictive entropy |
| heat capacity | `C = dU/dT = Var(E)/T^2` | see below |

## 1. The two standard uncertainty scalars are thermodynamic quantities at T=1

Verified numerically to 1e-14 (`src/validate.py`, and the check in this session):

- **predictive entropy = `S(T=1)`**
- **varentropy (the variance of surprisal, the "entropix" feature) = `C(T=1)`**

The proof is one line. Surprisal is `-log p_i = beta*E_i + log Z`, so its variance is
`beta^2 Var(E)`, which is the heat capacity. So the field's two favourite scalars are
the entropy and the heat capacity of the token distribution, both read at the single
arbitrary point `T=1`. The natural generalisation is to read the whole curve.

Empirically this framing is not just cosmetic: on 3096 real questions, all seven
standard baseline scalars together score exactly the same as `S(1)` and `C(1)` alone
(AUROC 0.8575 vs 0.8575, paired bootstrap 95% CI [-0.0016, +0.0015]). The standard
toolkit really is two numbers.

## 2. Every distribution has a melting temperature, and there is a closed form for it

`C(T)` has a large peak where the probability mass leaves the top token and spreads
into the vocabulary bulk. Solving `dC/dT = 0` for a ground state above `V` bulk states
at gap `Delta` gives the exact condition

    x* = 2(1+u)/(1-u),   u = V exp(-x*),   T_melt = Delta / x*

and `x* = log V + c` with `c` a slow O(1) drift. For `V = 151936`, `x* = 12.26`, so

    T_melt  =  Delta / 12.26

This is the temperature at which the entropy gain `log V` outweighs the energy cost of
leaving the top token. Measured on real Qwen logits, `T_melt` sits at 1.6 to 2.6, which
implies a 20 to 32 logit gap between the top token and the vocabulary bulk.

Consequence: top-k truncation replaces `V` by `k`, so it raises the melting temperature
by `x*(V)/x*(k)`. For `k=40` that is 2.66x. This gives a one-line physical account of
why samplers can run at temperature 1.5 to 2 with top-k but produce noise without it:
plain temperature sampling above ~2 crosses the melting point.

## 3. Main empirical result: read the entropy at the melting temperature, not at T=1

3096 factual questions with Wikidata ground truth (country capitals, element symbols,
film directors, book authors), overall accuracy 35.2%. Task is to predict whether the
greedy answer is wrong, from the first answer token's logits alone.

| detector | cost | AUROC |
|---|---|---|
| predictive entropy `S(T=1)` | 1 forward pass | 0.8579 |
| all 7 standard baselines together | 1 forward pass | 0.8579 |
| **`S(T_melt/2)`, a single number** | 1 forward pass | **0.8682** |
| full thermodynamic feature set | 1 forward pass | 0.8688 |

A single scalar read at the distribution's own melting temperature beats the entire
standard baseline suite. The paired bootstrap on the full feature sets gives
+0.0107 AUROC, 95% CI [+0.0056, +0.0159], significant.

The gain is modest. The robustness result below is the larger one.

## 4. Why it helps: real per-item variation in the logit scale

The logit scale is not a constant of the model. Measured across the 3096 real items on
the 1.5B model, `T_melt` runs from 1.49 at the 1st percentile to 2.59 at the 99th, and
`sd(log T_melt) = 0.133`. So each item's distribution has its own thermal scale, varying
by about 13%, and reading every item's entropy at the same fixed `T=1` reads each one at
a different point of its own curve. Reading at a fixed fraction of each item's melting
temperature removes that. This is what the single-feature gain in section 3 is: on both
models, `S(T_melt/2)` beats `S(T=1)` by 1.0 to 1.6 AUROC points, and that is the size of
the correction the measured scale variation predicts.

Pushing the same variation further, detectors trained on the original logits and tested
on the same items rescaled by an unknown per-item `s ~ LogNormal(0, sigma)`:

| sigma | standard baselines | baselines + rescaling augmentation | `S(T_melt)` | thermo shape features |
|---|---|---|---|---|
| 0.00 | 0.8653 | - | 0.8706 | 0.8761 |
| 0.13 (the level actually observed) | ~0.845 | - | 0.8706 | ~0.876 |
| 0.15 | 0.8424 | 0.8560 | 0.8706 | 0.8755 |
| 0.30 | 0.7771 | 0.8424 | 0.8706 | 0.8747 |
| 0.50 | 0.6875 | 0.8184 | **0.8706** | 0.8759 |

The melting-normalised measure is exactly invariant at every sigma, by construction, and
that invariance is free: it costs nothing at sigma = 0. Training the baselines with
rescaling augmentation does not close the gap.

**An important negative, so this is not over-claimed.** AUROC is rank based, so a scale
shift that is the *same for every item* is a monotone reparametrisation and does almost
no damage. Only *per-item* variation hurts. I checked this directly by training the
detector on one model and deploying it on the other:

| feature set | within 1.5B | within 0.5B | 1.5B -> 0.5B | 0.5B -> 1.5B | transfer loss |
|---|---|---|---|---|---|
| entropy `S(1)` alone | 0.8574 | 0.9076 | 0.9081 | 0.8579 | -0.0005 |
| all 7 baselines | 0.8575 | 0.9207 | 0.9180 | 0.8556 | +0.0023 |
| `S(T_melt/2)` alone | 0.8675 | 0.9229 | 0.9237 | 0.8682 | -0.0008 |
| thermo shape only | 0.8663 | 0.9207 | 0.9225 | 0.8679 | -0.0017 |

Transfer loss is essentially zero for every method including plain entropy, because
these two models differ mostly by a near-uniform scale factor (mean `T_melt` 1.95 versus
1.60). So the invariance is not what makes cross-model transfer work between similar
models. It matters when the scale varies from item to item, which it does, at
`sigma = 0.13`, and which is what section 3's gain is measuring.

## 5. Peak heat capacity, a scale-free measure of how sharply the model commits

`C_max = max_T C(T)` is **exactly invariant** under rescaling of the logits: a rescale
maps `C(T) -> C(T/s)`, which cannot change the maximum. Verified numerically, `C_max`
moves by 0.09% across a 16x rescale while `S(T=1)` moves by a factor of 13.

It also has a known ceiling. For an ideal sharp melt, one ground state above `V` bulk
states, `C_max = x*^2 u/(1+u)^2` at the peak condition of section 2, which is close to
`(log V)^2 / 4`. For `V = 151936` that is 36.6. Measured on real Qwen logits the mean is
12.0 for the 1.5B model and 10.1 for the 0.5B, so real melting transitions are about 3x
broader than an ideal two-level melt, and `C_max / 36.6` is a dimensionless sharpness in
[0,1] sitting around 0.33.

Interpretation: a high `C_max` means the chosen token is cleanly separated from the
vocabulary bulk and the model releases its uncertainty over a narrow temperature range.
A low `C_max` means the top of the distribution is smeared into the bulk. This is the
surviving, continuous form of the discrete "band gap" idea that section 6 records as
falsified.

### Detecting that the model has nothing to draw on

Two versions of the task, single forward pass.

**(a) Fabricated entities.** 1000 questions about entities that do not exist ("Who
directed The Pale Harbour of Tirburg?"), so every confident answer is a confabulation by
construction.

| detector | AUROC |
|---|---|
| all standard baselines | 0.8044 |
| thermodynamic shape + scale | 0.8883 |
| all features | 0.9613 |

**This number is inflated by a confound and should not be quoted on its own.** Invented
names have unusual spelling and tokenise unusually, so part of what the detector sees is
orthography rather than absent knowledge. The melting temperature was the single best
feature here (0.8250), and the control below shows that specific feature was reading the
confound.

**(b) The clean version: obscure versus famous real entities.** Same task, real names
only, so no change in the surface distribution. Split the 3096 items by Wikipedia
sitelink count into the bottom and top quartiles (accuracy 32.3% versus 64.3%).

| detector | AUROC |
|---|---|
| entropy alone | 0.7759 |
| all 7 standard baselines | 0.7790 |
| melt-normalised `S`, `C` | 0.8064 |
| thermodynamic shape + scale | 0.8419 |
| all features | **0.8603** |

The gain survives the control: +0.081 AUROC over the full baseline suite, with no
orthographic confound. The single best feature is `C_max` at 0.8076, which alone beats
all seven standard baselines combined. Meanwhile `T_melt`, the best feature on the
fabricated set, collapses to 0.5126 here, confirming it was reading the confound.

So the real effect is carried by the peak heat capacity, not by the melting temperature.

### Where the gain does not show up

On risk-coverage, which is what a practitioner deploying abstention actually cares
about, the improvement is small. Area under the risk-coverage curve, lower is better:

| score | 1.5B | 0.5B |
|---|---|---|
| entropy `S(T=1)` | 0.3674 | 0.6629 |
| max probability | 0.3702 | 0.6591 |
| `S(T_melt/2)` | 0.3644 | 0.6597 |

About one AUROC point translates into almost nothing at the operating points that
matter. The honest summary is that this improves *measurement* and *invariance*
substantially and *selective prediction accuracy* barely.

## 6. What was falsified

Stated in full, because these were the original hypotheses and most of them were wrong.

- **Discrete band structure is not there.** I expected next-token spectra to show a
  resolved Schottky peak from a small set of competing answers sitting below the
  vocabulary melting peak. Only about 10% of real items show a resolved second peak, and
  every discrete peak-structure feature (`npeaks`, `has_low`, `gap_ratio`, `valley`,
  `band_weight`) contributes exactly 0.0000 AUROC beyond entropy. The real spectrum is
  one broad peak with a smooth cold tail. Section 7c says why: bands need groups of
  near-degenerate states and the states are not degenerate.
- **Top-k truncation does not raise the melting temperature.** It lowers it, from 1.505
  to 0.802 at k=8, because truncation cuts the energy gap faster than it cuts `log V`.
  See section 7d, which also gives the corrected physical picture.
- **The paraphrase/meaning temperature separation does not pay on real data.** It works
  in synthetic systems, lifting held-out R^2 from 0.571 to 0.762, and it needs the
  within-meaning logit spread to be under about a tenth of the between-meaning spread.
  Where real degeneracy exists at all, that ratio is 0.642.
- **Energy-gap clustering as a cheap semantic entropy** adds nothing beyond entropy on
  real data (marginal AUROC +0.0000 to -0.0005 for every threshold tried).
- **The melting temperature is fairly stable across first answer tokens**, spanning 1.7x
  between the 1st and 99th percentile. Across ordinary text it is 2.9x, which is larger
  but still not the order-of-magnitude spread I expected.
- **Peak-detection features are numerically fragile.** Under logit noise of 5e-2 the
  continuous integral features keep Spearman 0.96 to 0.99 against their unperturbed
  values, while `npeaks` drops to 0.54. Prefer integrals over peak finding.

## 7. Replication on a second model (Qwen2.5-0.5B-Instruct, same 3096 questions)

Accuracy 13.2%, so a much weaker model and a much more skewed label set.

What replicates:

- `S(T_melt/2)` is again the single best feature, 0.9237, above entropy alone at
  0.9077 and above every individual baseline.
- The rescaling result replicates strongly. At sigma = 0.5 the baselines fall from
  0.9234 to 0.7447, augmentation recovers only to 0.8831, and `S(T_melt)` is exactly
  invariant at 0.9180.

What does not replicate:

- The multivariate gain vanishes. On the 0.5B model, all baselines together score
  0.9220 and baselines plus thermodynamic features score 0.9223, a paired difference of
  -0.0002 with 95% CI [-0.0036, +0.0029]. On the 1.5B model the same comparison was
  +0.0107 and significant.

So the durable finding is the invariance, not the accuracy gain. The accuracy gain is
small and model dependent, and I would not claim it without more models.


## 7b. Semantic entropy, and why the paraphrase correction does not pay here

Total token entropy splits exactly by the chain rule into entropy over meaning clusters
plus the average entropy inside a cluster. In condensed-matter terms that is
configurational entropy plus vibrational entropy. Semantic entropy is the first term.
The idea I set out to test is that the two live at different energy scales, so a
temperature window can separate them from one forward pass instead of by sampling.

**In synthetic systems with known clustering, the separation works and is close to
orthogonal.** The cold window `[0, 0.05] T_melt` correlates +0.75 with the vibrational
term and +0.10 with the configurational term. Adding it to total entropy lifts held-out
R^2 for predicting configurational entropy from 0.571 to 0.762. A sweep over the two
scales gives a clear criterion: the gain is large (+0.08 to +0.25 R^2) when the
within-meaning logit spread is under about 1/20 of the between-meaning spread, small
below 1/5, and zero or negative above that.

**On real data the correction is small, and the reason is measurable.** Reference
semantic entropy on 1100 questions, 10 samples each at T=1, clustered by normalised
string equality:

| predictor of semantic entropy | held-out R^2 |
|---|---|
| single-pass entropy `S(T=1)` alone | 0.6183 |
| all 7 baselines | 0.6965 |
| baselines + spectrum windows | 0.7271 |
| baselines + full shape curve | 0.7298 |

So the spectrum does add information about semantic entropy beyond entropy alone
(0.618 -> 0.730), and beyond the whole baseline suite (0.697 -> 0.730), but the second
gain is small.

**And the honest headline of this section: semantic entropy did not beat plain entropy
on this task at all.** Reference semantic entropy scores AUROC 0.8568 for predicting an
incorrect answer, against 0.8607 for single-pass entropy, while costing 10x the compute.
That is not a refutation of the published method, which was validated on longer
free-form answers with entailment clustering. It says that for short factual answers the
paraphrase component is already small, so there is little nuisance to remove. That is
the same conclusion the synthetic phase diagram reaches from the other direction, and it
is why the temperature separation, which is a way of removing that same nuisance, also
buys little here.

## 7c. The decisive measurement: there is almost no paraphrase degeneracy to remove

The whole configurational/vibrational programme rests on an assumption nobody seems to
have checked: that a chunk of a next-token distribution's entropy is surface variation
between tokens that mean the same thing. I measured it directly.

Method: for 500 questions, take the top 12 candidate first tokens. Force each one, then
continue greedily, so every candidate first token is mapped to the complete answer it
would produce. Candidates landing on the same normalised answer are surface variants of
one meaning. Candidates landing on different answers are different meanings.

| measurement | value |
|---|---|
| distinct meanings among the top 12 candidates | mean 11.45, median 12 |
| items where all 12 candidates give 12 different answers | 70.8% |
| candidates sitting alone in their meaning cluster | **96.7%** |
| items with any within-meaning degeneracy at all | 29.2% |
| median within-meaning logit spread `delta` | 0.000 |
| median between-meaning logit spread | 3.892 |
| median `delta/spread`, among items that have any degeneracy | 0.642 |
| first-token semantic entropy vs token entropy | 1.4432 vs 1.4713 nats |
| Spearman between the two | 0.9928 |

**At the first answer token, the token distribution is already the meaning
distribution.** 96.7% of live candidates lead somewhere different. The paraphrase
component is 0.028 nats out of 1.471, under 2%.

This closes the loop on every negative result above:

- It is why reference semantic entropy does not beat token entropy here (0.8568 vs
  0.8607). There is nothing for it to remove.
- It is why the temperature separation buys nothing on real data even though it works
  cleanly in synthetic systems. The synthetic gain needed `delta/spread` below about
  0.1, and where real degeneracy exists at all the median is 0.642, six times too high.
- It is why the discrete band structure is absent. Bands require groups of
  near-degenerate states, and the states are not degenerate.

I would expect this to break down for longer free-form answers, where a single meaning
really can be phrased many ways, which is the setting semantic entropy was validated in.
For short factual answers it does not hold, and that is a specific, falsifiable claim
about where the sampling-based methods are and are not buying anything.

## 7d. Testing the melting law on real logits

The law from section 2 is `T_melt = Delta / (log V_eff + c)`. Truncating the
distribution to its top `k` tokens sets `V_eff = k`, so the law can be tested by
sweeping `k` over four orders of magnitude on real logits and checking that
`Delta_k / (T_melt * log k)` stays constant. Ten prompts covering factual completion,
narrative, code and open-ended prose.

| k | mean `T_melt` | mean `Delta_k` | `T_melt * log k` | ratio |
|---|---|---|---|---|
| 8 | 0.802 | 2.326 | 1.669 | 1.394 |
| 32 | 0.948 | 4.206 | 3.285 | 1.280 |
| 128 | 1.085 | 6.355 | 5.264 | 1.207 |
| 512 | 1.218 | 8.500 | 7.597 | 1.119 |
| 2048 | 1.308 | 10.666 | 9.970 | 1.070 |
| 8192 | 1.374 | 13.024 | 12.385 | 1.052 |
| 32768 | 1.434 | 15.856 | 14.908 | 1.064 |
| 151936 | 1.505 | 20.169 | 17.961 | 1.123 |

The ratio moves between 1.05 and 1.39 while `k` changes by a factor of 19000 and
`Delta` changes by a factor of 8.7. The law holds. Across the ten prompts at full
vocabulary, `r(T_melt, Delta/log V) = 0.869`, with the measured value 0.891 of the
predicted one.

**The consequence I predicted from the law is wrong, and the reason is worth stating.**
I expected top-k truncation to raise the melting temperature, because it shrinks `V_eff`
in the denominator, and I used that to explain why samplers can run hot with top-k. The
data goes the other way: `T_melt` falls from 1.505 at full vocabulary to 0.802 at k=8.
Truncation removes the far tail, which cuts `Delta` (20.17 down to 2.33) faster than it
cuts `log k`. My synthetic argument held `Delta` fixed while shrinking the bulk, which
is not what truncation does.

The physical picture that survives is different and better. The melting temperature is
not the right safety threshold once you truncate. What matters is what the distribution
melts *into*. Melting into eight plausible continuations is harmless; melting into
151936 tokens is noise. Top-k does not raise the temperature at which mass leaves the
top token, it changes the consequence of that happening.

## 7e. Melting temperature across ordinary text

`T_melt` measured at every position of four passages (472 positions).

| text | median `T_melt` | p5 to p95 | full spread |
|---|---|---|---|
| technical prose | 1.652 | 1.131 to 2.032 | 2.37x |
| Python code | 2.104 | 1.391 to 2.415 | 2.14x |
| simple factual statements | 1.770 | 1.131 to 2.032 | 2.14x |
| open-ended personal prose | 1.652 | 1.191 to 1.897 | 2.21x |
| all positions | - | 1.99x (p95/p5) | 2.91x |

Code sits about 27% hotter than open-ended prose, which fits: the next token in code is
more strongly determined, so the gap to the vocabulary bulk is larger. Within any single
passage the melting point still varies by a factor of about 2, so a single global
sampling temperature does run some positions much closer to their own melting point than
others. Whether correcting that helps generation quality is the experiment in section 9
that I did not run.

## 7f. The prediction I made, tested: free-form answers behave completely differently

Section 7c ended with a falsifiable claim of my own. I said the 96.7% result should break
down for longer free-form answers, where one meaning really can be phrased many ways, and
that this is why semantic entropy was validated in that setting. I tested it.

Same protocol: top 12 candidate first tokens, force each, continue greedily. Two
conditions, 30 questions each, on Qwen2.5-1.5B. Because free-form answers are sentences,
clustering is now by sentence-embedding cosine similarity (all-MiniLM-L6-v2, threshold
0.80), and the short-factual set is re-run through the **same** clustering so the two
numbers are directly comparable.

| | free-form | short factual |
|---|---|---|
| distinct meanings among the top 12 | **2.10** | 9.37 |
| candidates alone in their meaning cluster | **9.2%** | 67.8% |
| items with any degeneracy | 100% | 86.7% |
| median within-meaning logit spread | 8.932 | 4.877 |
| median between-meaning logit spread | 5.952 | 7.088 |
| semantic entropy vs token entropy | 0.0219 vs 0.4098 | 0.6428 vs 0.7408 |
| **share of top-12-renormalised entropy that is pure paraphrase** | **94.7%** | **13.2%** |

**Prediction confirmed, and the size of the effect is larger than I expected.** For
free-form answers, 94.7% of the entropy among the renormalised top 12 next-token candidates is
surface variation carrying no information about meaning. For short factual answers it is
13.2%. The omitted vocabulary tail is not included. The two regimes are not
slightly different, they are opposite.

This is the cleanest statement in the whole project of when the sampling-based methods
earn their cost. Semantic entropy exists to remove exactly this nuisance term. On free-form
answers there is 94.7% of it to remove, which is presumably why it works there. On short
factual answers there is 13.2%, which is why it did not beat single-pass entropy in
section 7b.

Two further consequences, both testable.

**My temperature-separation idea is dead in both regimes, for opposite reasons.** On
factual answers there is no degeneracy to separate. On free-form answers there is plenty,
but look at the two spread rows: within-meaning spread is 8.932 and between-meaning spread
is 5.952, so `delta/spread = 1.04`. Surface variants span a *wider* energy range than
different meanings do. The energy ordering does not align with the semantic hierarchy at
all, so no temperature window can separate them. The criterion from the synthetic phase
diagram needed 0.1 and free-form gives 1.04.

**First-token uncertainty measures should fail on free-form tasks.** At the first token of
a free-form answer the model's semantic entropy is 0.0219 nats, essentially zero: it has
already decided what it is going to say, and is only choosing how to start the sentence.
Any detector reading the first token alone is therefore reading phrasing, not content. The
real content uncertainty must appear later in the sequence. Every number in sections 1
through 5 of this document comes from a first answer token on a short factual task, which
is a setting where that token does carry the meaning, and none of it should be assumed to
transfer to free-form generation.

## 7g. The melting law across seven model families

Section 7d tested the law within one model by truncating its vocabulary. The harder test
is across models that genuinely differ, because there `T_melt` and `Delta` are two
independently measured quantities and the law predicts a fixed ratio between them.

Seven models, one forward pass each on 20 shared prompts: GPT-2, Pythia-160M, SmolLM2-360M,
OPT-125M, Qwen2.5-0.5B, Qwen2.5-1.5B, BLOOMZ-560M. Vocabulary spans 49,152 to 250,880.

| model | V | x*(V) | `T_melt` | `Delta` | `T_melt x*/Delta` |
|---|---|---|---|---|---|
| GPT-2 | 50,257 | 11.186 | 1.040 | 14.03 | 0.793 |
| Pythia-160M | 50,304 | 11.187 | 1.066 | 11.39 | 0.987 |
| SmolLM2-360M | 49,152 | 11.165 | 1.234 | 16.29 | 0.817 |
| OPT-125M | 50,272 | 11.187 | 1.175 | 14.74 | 0.902 |
| Qwen2.5-0.5B | 151,936 | 12.260 | 1.412 | 19.78 | 0.867 |
| Qwen2.5-1.5B | 151,936 | 12.260 | 1.447 | 20.60 | 0.863 |
| BLOOMZ-560M | 250,880 | 12.749 | 1.133 | 18.43 | 0.796 |

**The dimensionless ratio is 0.861 ± 0.064, a spread of 1.25x**, across 5.1x in vocabulary
size and seven different architectures and training regimes. `r(T_melt, Delta/x*) = 0.879`,
against 0.842 for `r(T_melt, Delta)` with the vocabulary term dropped, so the `log V` term
helps, though only slightly.

### The estimator of Delta is the whole game, and getting it wrong looks like a refutation

My first attempt measured `Delta` as the top logit minus the **mean** logit. That gave a
ratio spread of 9.7x and an anticorrelation of −0.27, which reads as a clean refutation of
the law. Switching to the **median** left BLOOMZ off by 8x. Only the **mode** of the logit
distribution works.

This is not fitting until it works. The theory is a two-level idealisation: one ground
state and `V` states at a single energy `Delta` below it. The right estimator of "where the
bulk of states sits" is therefore the peak of the density of states, which is the mode.
Mean and median are corrupted by tail outliers, and the models differ enormously in how
heavy that tail is:

| model | top logit | mode | median | mean | sd | `d_mean/d_median` |
|---|---|---|---|---|---|---|
| GPT-2 | −103.8 | −117.8 | −118.4 | −118.6 | 4.1 | 1.02 |
| Qwen2.5-1.5B | 20.2 | −0.6 | 0.4 | 0.7 | 3.0 | 0.99 |
| Pythia-160M | 837.8 | 826.4 | 823.8 | 807.1 | 80.8 | 2.19 |
| BLOOMZ-560M | 411.3 | 392.2 | 311.5 | 267.8 | 139.6 | 1.41 |

Absolute logit values span −118 to +411 and standard deviations span 3.0 to 139.6, yet
every model's `T_melt` lands between 1.04 and 1.45. That is exactly what the law predicts,
because softmax is shift-invariant and `T_melt` depends only on the gap to the bulk.

The practical lesson is worth stating separately: **any measure that summarises a logit
vector with a mean is not safe to compare across model families.** Pythia's mean sits 2.2
standard-deviation-widths below its mode. Several published uncertainty measures aggregate
logits by mean, and this is a concrete reason to check that choice before comparing models.

## 7h. The log V term, tested properly with models trained for the purpose

Section 7g confirmed the `Delta` dependence across seven pretrained models but could not
test the `log V` term, because real vocabularies cluster between 49k and 251k, so `x*(V)`
only varies by 14% which is inside the model-to-model scatter. The honest fix is to make
models whose vocabularies actually differ.

**Controlled sweep.** One 5.8M-character public-domain corpus, byte-level BPE tokenizers
at V = 257, 512, 2048, 8192, 32768, and an identical 4-layer 256-dim transformer trained
on each for the same number of epochs over the same text, so every model sees the same
characters and the only difference is the vocabulary size. `x*` spans 6.22 to 10.77, a
1.73x range, which is finally enough leverage.

| V | x*(V) | bits/char | `T_melt` | `Delta` | `T/Delta` | `T x*/Delta` |
|---|---|---|---|---|---|---|
| 257 | 6.22 | 1.967 | 1.582 | 12.45 | 0.1271 | 0.790 |
| 512 | 6.84 | 1.830 | 1.738 | 9.21 | 0.1887 | 1.291 |
| 2,048 | 8.13 | 2.069 | 1.119 | 8.44 | 0.1326 | 1.077 |
| 8,192 | 9.44 | 2.070 | 0.721 | 6.99 | 0.1032 | 0.974 |
| 32,768 | 10.77 | 2.041 | 0.834 | 10.19 | 0.0819 | 0.882 |

Within the controlled sweep alone, including the `log V` term cuts the scatter from
CV 0.283 to CV 0.172, a 39% reduction. V = 512 is a visible outlier.

**Pooled with the seven pretrained models: 12 models, V from 257 to 250,880, a 976x range,
`x*` from 6.22 to 12.75.**

| | mean | CV | spread |
|---|---|---|---|
| `T_melt/Delta`, no `log V` term | 0.0966 | 0.363 | 3.07x |
| `T_melt x*(V)/Delta`, with it | **0.9289** | **0.151** | 1.65x |

The `log V` term reduces the scatter by **58%**. The correlation between measured and
predicted melting temperature rises from `r = 0.342` to `r = 0.795`. A 20,000-sample
bootstrap on the difference in coefficient of variation gives −0.191 with 95% CI
[−0.269, −0.082], so the improvement is significant rather than a lucky reparametrisation.

This was listed in section 9 as the sharpest falsifiable claim in the project. It is now
tested and it holds. The constant is 0.93 ± 0.14 rather than 1, which is expected: the
derivation idealises the vocabulary as `V` states at a single energy, and section 5 already
measured real melting transitions to be about 3x broader than that ideal.

## 7i. Where a free-form answer's content is actually decided, and why entropy detectors miss it

Section 7f showed that at the first token of a free-form answer, 94.7% of the renormalised
top-12 candidate entropy is phrasing. The obvious next question is where the content gets
decided instead. So I walked
along each greedy answer and, at **every** position, forked the top 8 alternative tokens,
continued each greedily, and clustered the resulting complete answers by meaning. That
gives a per-position split:

    meaning share(t) = semantic entropy at position t / renormalised top-8 entropy at position t

0 means the choice at that position is pure phrasing. 1 means it fully determines what the
answer says. 12 free-form questions, 480 token positions, Qwen2.5-1.5B.

| positions | n | mean top-8 entropy | mean meaning share | share of the answer's total semantic entropy |
|---|---|---|---|---|
| 0 | 12 | 0.183 | 0.218 | 40.0% |
| 1 to 2 | 24 | 0.271 | 0.081 | 58.8% |
| 3 to 5 | 36 | 0.491 | 0.013 | 1.2% |
| 6 to 11 | 72 | 0.627 | 0.000 | 0.0% |
| 12 to 19 | 96 | 0.755 | 0.000 | 0.0% |
| 20 to 29 | 120 | 0.651 | 0.001 | 0.0% |
| 30 to 59 | 120 | 0.777 | 0.000 | 0.0% |

At this clustering threshold, 98.8% of the semantic entropy sits in the first three tokens
and 96.9% of positions read as pure phrasing. That specific figure turns out to depend on
the threshold, and the sensitivity analysis below says which parts of it survive.

Notice that top-8-renormalised entropy moves the opposite way. It is lowest at position 0 (0.183) where
all the meaning is decided, and highest late in the answer (0.777) where none of it is.

### The consequence for entropy-based hallucination detection

| measurement | value |
|---|---|
| Spearman(top-8 entropy, meaning share) | **−0.230** |
| Spearman(top-8 entropy, semantic entropy) | −0.226 |
| among the 20% highest-entropy positions: mean meaning share | **0.0000** |
| ...fraction of those carrying no meaning information | **100%** |
| among positions that do decide meaning (share > 0.2): mean entropy | 0.294 (overall mean 0.657) |
| ...fraction of those below median entropy | 75% |

Top-8-renormalised entropy is **anticorrelated** with whether a position carries meaning. The positions a
detector flags are precisely the ones that do not matter. Concretely, the highest-entropy
positions in the sample were tokens like ` tiny`, ` weather`, ` contains`, ` sudden`,
` bicycle`, all with entropy near 2.0 nats and a meaning share of exactly 0.000. Meanwhile
` manager` at position 2 had entropy 0.000 and a meaning share of 0.464: the model was
completely certain of the token, and that token determined the whole answer.

This is the most practically useful negative result in the project. Applying token-entropy
uncertainty to long-form generation is measuring phrasing at 97% of positions, and the one
place the content signal lives, the first two or three tokens, is where entropy is lowest.
It suggests why long-form hallucination detection is harder than the short-answer results
in the literature would imply, and it says where to look instead.

**Caveats.** 12 questions, one model, greedy continuation after each fork. Greedy
continuation probably understates divergence, because the model can steer back toward the
same content after a perturbed token.

### The clustering threshold matters, and it partly overturns the headline number

I re-ran the whole measurement saving every forked continuation, then reclustered at six
embedding thresholds. A loose threshold merges answers that differ in detail and would
manufacture the result; a strict one splits paraphrases and would destroy it.

| threshold | mean meaning share | share at pos 0 | pos 1-2 | pos 6+ | % of semantic entropy in pos 0-2 | rho(entropy, share) |
|---|---|---|---|---|---|---|
| 0.65 | 0.0006 | 0.000 | 0.008 | 0.0000 | 47.3 | −0.160 |
| 0.75 | 0.0045 | 0.031 | 0.032 | 0.0000 | 93.7 | −0.207 |
| **0.80** | 0.0178 | 0.218 | 0.081 | 0.0003 | **98.7** | −0.266 |
| 0.85 | 0.0602 | 0.421 | 0.204 | 0.0178 | 41.5 | −0.248 |
| 0.90 | 0.1321 | 0.643 | 0.442 | 0.0541 | 32.1 | −0.238 |
| 0.95 | 0.3620 | 0.828 | 0.618 | 0.2698 | 11.8 | −0.078 |

**The specific claim that 98.8% of the meaning signal sits in the first three tokens does
not survive.** It is a property of the 0.80 threshold. At 0.95 only 11.8% does, and late
positions carry real meaning differences.

Two things do survive at every threshold tested, and these are what I will stand behind:

1. **The gradient is monotone and large.** Meaning share is always highest at position 0,
   lower at positions 1 to 2, and lowest from position 6 on. At threshold 0.80 that runs
   0.218 to 0.081 to 0.0003; at 0.95 it runs 0.828 to 0.618 to 0.270. The ordering never
   flips. Content decisions are concentrated early, at every level of semantic granularity.
2. **Token entropy is anticorrelated with meaning share at every threshold**, rho from
   −0.078 to −0.266. The direction never flips.

The threshold dependence is itself informative rather than just noise. Whether a
late-answer token "changes the meaning" depends on how fine a distinction you are willing
to call a different meaning. The coarse question, is this the same explanation, is settled
in the first few tokens. The fine detail, which particular facts and words, keeps being
decided all the way through. Any semantic-entropy method inherits exactly this ambiguity
from its own clustering step, and the published methods do not usually report sensitivity
to it. On this evidence they should.

## 7j. Meaning free energy: a clean hypothesis, cleanly falsified

Section 7f found enormous phrasing freedom in free-form answers, which suggested a real
problem with greedy decoding. Greedy maximises the probability of a *sequence*. The
probability of a *meaning* is the sum over all its phrasings. In thermodynamic terms the
most probable meaning minimises a free energy

    F(m) = E(m) - T * S_phrasing(m)

so a meaning with many phrasings gets an entropy bonus that greedy decoding ignores. If
`S_phrasing` is large, greedy should sometimes pick a sharp lonely peak over a broad likely
basin. That is the mode-seeking problem Minimum Bayes Risk decoding exists to fix, and I
expected to measure a large effect.

Test: 24 questions per condition, greedy answer plus 24 samples at T=1, all clustered by
meaning. Does the greedy answer's meaning match the meaning carrying the most sampled mass?

| | free-form | short factual |
|---|---|---|
| distinct meanings among 24 samples | 1.71 | 5.29 |
| distinct surface strings / K | **1.000** | 0.234 |
| phrasing degeneracy, log(#phrasings) | 2.444 nats | 0.113 nats |
| greedy meaning == highest-mass meaning | **100.0%** | 95.8% |
| probability mass on the greedy meaning | 0.970 | 0.767 |
| mass left on the table by greedy | **0.000** | 0.003 |

**Falsified.** Greedy picks the most probable meaning essentially always, and the free
energy correction is worth 0.000 on free-form and 0.003 on factual.

The reason is the interesting part, and it confirms the picture from a third independent
angle. On free-form questions every one of the 24 samples was a textually unique string,
with a phrasing degeneracy of 2.44 nats, and yet they collapsed to 1.71 meanings with 97%
of the mass on a single one. Enormous freedom in wording, essentially none in meaning. The
entropy bonus is large but it accrues entirely *inside* one meaning, so there is no
competing meaning for it to tip the balance toward.

Three measurements now agree: 94.7% of renormalised top-12 first-token entropy is phrasing
(7f), 98.8% of the
meaning signal sits in the first three tokens and 96.9% of positions are pure phrasing
(7i), and whole sampled answers are textually unique but semantically identical (7j).

**One important scope limit.** All three used questions the model knows well. Whether
meaning uncertainty appears when the model is fabricating is a different question, and it
is the one that decides whether any of this can detect free-form hallucination at all. That
is section 7k.

## 7k. Free-form hallucination is detectable, and section 7i says where to look

Sections 7f, 7i and 7j all found near-zero meaning uncertainty in free-form answers, but
every one of them used questions the model knows well. The question that decides whether
any of this is useful is whether meaning uncertainty appears when the model is fabricating.

Three matched conditions, 16 free-form questions each, same surface form and length:
KNOWN (common knowledge), OBSCURE (real but very obscure history, linguistics, geology),
FABRICATED (invented entities, so every answer is a confabulation by construction).

| | KNOWN | OBSCURE | FABRICATED |
|---|---|---|---|
| sampled-answer semantic entropy | 0.029 | 0.842 | 1.437 |
| distinct meanings among 16 samples | 1.12 | 4.94 | 7.88 |
| probability mass on the top meaning | 0.992 | 0.742 | 0.562 |
| semantic entropy of the first 3 forked tokens | 0.021 | 0.097 | 0.431 |
| plain token entropy, first 3 tokens | 0.736 | **0.327** | 1.015 |

**Meaning uncertainty appears, and it is large and monotone.** Semantic entropy rises 50x
from KNOWN to FABRICATED. The earlier sections were measuring a regime where the model
happened to know the answer, not a property of free-form generation.

Look at the last row. **Plain token entropy is not monotone**: the model is *most*
confident at the token level on the obscure questions (0.327, below KNOWN's 0.736), because
obscure questions have stereotyped answer openings. This is section 7i's anticorrelation
showing up as a concrete failure.

### A cheaper detector, derived from where the signal lives

Since 7i located the content signal in the first few tokens, fork only those. At each of
the first n positions take the top 8 tokens, continue each for 28 tokens, cluster by
meaning, take the entropy.

| method | generated tokens | KNOWN vs FABRICATED | KNOWN vs OBSCURE | OBSCURE vs FABRICATED | mean |
|---|---|---|---|---|---|
| sampled semantic entropy | 704 | **0.996** | 0.820 | 0.731 | 0.849 |
| fork position 0 only | 224 (3.1x cheaper) | 0.941 | 0.627 | 0.863 | 0.811 |
| **fork positions 0 to 1** | 448 (1.6x cheaper) | 0.973 | 0.762 | 0.828 | **0.854** |
| fork positions 0 to 2 | 672 | 0.953 | 0.719 | 0.844 | 0.839 |
| plain token entropy, pos 0 | 0 (1 forward pass) | 0.801 | 0.523 | 0.793 | 0.706 |
| plain token entropy, pos 0-2 | 0 (1 forward pass) | 0.602 | **0.668** | 0.813 | 0.694 |

Forking the first two tokens matches full semantic entropy on the three-way mean (0.854 vs
0.849) at 1.6x lower cost, and forking only the first token keeps 0.941 of the headline
comparison at 3.1x lower cost. Adding positions beyond the second makes it worse, which is
what 7i predicts.

**The confound, stated plainly.** The FABRICATED questions use invented names, so any
comparison against them is partly reading orthography, exactly as in section 5. The clean
comparison is KNOWN vs OBSCURE, both real. There, **plain token entropy is at chance
(0.523 for position 0, 0.668 across the first three)** while forking the first two tokens
reaches 0.762 and full semantic entropy reaches 0.820. The ordering survives the control
even though the absolute numbers fall.

**Caveats.** 16 questions per condition, one model, one clustering threshold. Section 7i's
sensitivity analysis applies here too and I have not repeated it for these numbers.

### What this adds up to

The physics did not produce the useful method here. The useful method came from a
measurement the physics motivated me to make. Sections 7f and 7i asked where meaning lives
in a token sequence, found it concentrated in the first two or three tokens, and found
entropy pointing the wrong way. Section 7k turns that into a detector that matches the
standard sampling method at lower cost, and explains why the cheap single-pass measures
that work on short factual answers collapse to chance on free-form ones.

## 7l. Replication on a second model: the effect holds, my detector does not, and token entropy is worse than weak

Everything in 7f to 7k was one model. I repeated the two central experiments on
Qwen2.5-0.5B-Instruct.

**The phrasing-versus-meaning split replicates, strongly.**

| | Qwen2.5-1.5B | Qwen2.5-0.5B |
|---|---|---|
| free-form: candidates alone in their meaning cluster | 9.2% | 13.9% |
| free-form: share of token entropy that is pure phrasing | 94.7% | **90.6%** |
| short factual: candidates alone in their cluster | 67.8% | 82.8% |
| short factual: share that is pure phrasing | 13.2% | **4.8%** |

The two regimes are just as far apart on the smaller model.

**Sampled semantic entropy replicates.** Known versus fabricated 0.959, known versus
obscure 0.910 on the 0.5B, against 0.996 and 0.820 on the 1.5B.

**My cheap forking detector does not replicate.** Known versus fabricated falls from 0.973
to 0.668, and known versus obscure from 0.762 to 0.586. On the 0.5B model plain token
entropy has more raw discriminative power than my method does. I am retracting section 7k's
suggestion that forking the first two tokens is a usable substitute for semantic entropy.
On one model it matched; on the next it did not.

### But the token entropy result got stronger, not weaker

Reporting **signed** AUROC, where above 0.5 means the score rises with fabrication, which
is the direction a detector needs:

| measure | model | known vs fabricated | known vs obscure |
|---|---|---|---|
| sampled semantic entropy | 1.5B | 0.996 | 0.820 |
| sampled semantic entropy | 0.5B | 0.959 | 0.910 |
| fork positions 0 to 1 | 1.5B | 0.973 | 0.762 |
| fork positions 0 to 1 | 0.5B | 0.668 | 0.586 |
| **plain token entropy** | **1.5B** | 0.602 | **0.332** |
| **plain token entropy** | **0.5B** | **0.098** | **0.176** |

**The sign of the token-entropy signal flips.** On the 0.5B model, higher first-token
entropy means the model *knows* the answer, with a signed AUROC of 0.098, which is strongly
backwards. On the 1.5B it points the right way for known versus fabricated (0.602) and the
wrong way for known versus obscure (0.332). So its direction is unstable both across models
and across comparisons within a single model.

Earlier sections said token entropy was a weak signal on free-form text. That understated
it. **A measure whose sign you cannot predict is not a weak detector, it is not a detector
at all**, because using it requires labelled data from that exact model and question
distribution to learn which way to read it. Semantic entropy keeps the correct sign in all
four comparisons; my forking method keeps the correct sign in all four but is weak on the
smaller model.

This is the clearest practical conclusion in the project, and it survived the replication
that killed my own proposed method.

## 8. Methods

- Models: Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B-Instruct, float32 on Apple M4 (MPS).
  float32 rather than a lower precision because the whole method reads fine structure in
  the logits, though section 6 shows the integral features tolerate 5e-2 of logit noise.
- Data: 3096 questions built from Wikidata SPARQL ground truth, in four
  relation types (country capital 207, chemical element symbol 174, film director 1559,
  book author 1156). Plus 1000 questions about generated non-existent entities.
- Greedy decoding, up to 12 new tokens. All features come from the logits at the first
  generated token unless marked `sp_` (mean spectrum over the whole answer span).
- The phrasing-versus-meaning percentages in section 7f are computed over the top 12
  candidates after renormalising their logits; they are not fractions of full-vocabulary
  entropy. Meaning clusters use single-linkage cosine similarity at one threshold.
- Grading: normalised string containment against the Wikidata value, with a surname
  match allowed for people. Overall accuracy 35.2% (1.5B) and 13.2% (0.5B).
- Thermodynamic curves on 300 log-spaced inverse temperatures from `T=100` down to
  `T=0.0033`. Computed from a density-of-states compression: the top 1024 logits kept
  exactly, the remaining ~151k binned into 512 histogram bins. On the validation fixture,
  absolute entropy error is below 1e-5 and heat-capacity error below 2e-5; the method is
  about 90x faster than summing over the full vocabulary.
- The implementation is validated against analytic results: `F = U - TS` to 2e-11,
  `dU/dT = C` to 1e-6 relative error at independently differenced temperatures, and the
  two-level Schottky heat-capacity curve to 2e-14, including the universal peak position
  `T*/gap = 0.416778...`.
- Statistics: AUROC with 2000-sample bootstrap intervals for single features, 5-fold by
  4-repeat cross-validated logistic regression for feature sets, and a 4000-sample
  paired bootstrap on out-of-fold scores for comparing two feature sets on the same
  items.

## 9. What I would do next

Updated after the second round of work. Three of the original four items are now done and
are reported above.

1. ~~More models, and models from different vocabulary sizes.~~ **Done, section 7g and 7h.**
   12 models, 976x range, law confirmed.
2. `C_max` still deserves its own study. It is scale-free, it has a known ceiling near
   `(log V)^2/4`, and on the clean obscurity task it beat every standard scalar. Across the
   seven pretrained models the normalised sharpness `4 C_max/(log V)^2` ranges from 0.210
   (BLOOMZ) to 0.326 (SmolLM2), which is suggestive of a link to model quality but is
   confounded by multilinguality and has n=7. That is the experiment to run.
3. Sampling at a fixed fraction of `T_melt` instead of a fixed `T`. Still not run. Section
   7e measured the melting point varying about 2x within a single passage, so a global
   temperature does run some positions much closer to their melting point than others.
4. ~~Whether the paraphrase result breaks down on free-form answers.~~ **Done, section 7f.**
   It does, dramatically, and 7i, 7j and 7k follow from it.

New items, in priority order:

5. **Replicate the free-form results on more models and more questions.** Sections 7f, 7i,
   7j and 7k rest on 12 to 30 questions each and mostly one model. The effects are large,
   but the sample sizes are small and I would not defend the second decimal place of any of
   them.
6. **Repeat the threshold sensitivity analysis for section 7k.** Section 7i showed the
   clustering threshold moves the absolute numbers a lot while preserving the ordering. I
   did not repeat that check for the detection results, and it should be done before anyone
   relies on them.
7. **Test the first-token forking detector against the published cheap methods.** Semantic
   Entropy Probes achieve a larger cost saving by reading hidden states. A head-to-head on
   the same data would say whether forking adds anything or is simply a worse route to the
   same place.
8. **Check whether the mean-versus-mode problem in section 7g affects published measures.**
   Several logit-aggregating uncertainty methods use means. Pythia's mean logit sits 2.2
   standard deviations from its mode. Whether that changes any published cross-model
   comparison is a concrete, checkable question.
