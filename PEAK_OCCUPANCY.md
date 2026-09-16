# The melting peak sits well short of a 50–50 handoff

Written during the validation audit on 5 September 2026. **Corrected on 16 September 2026.**

The original version of this file called the two laws below a new derivation. Both are in fact a rederivation of Theorem 8 and Corollary 10 of Reeb and Wolf, *Tight bound on relative entropy by entropy difference*, IEEE Trans. Inf. Theory 61(3):1458, 2015 (arXiv 1304.0036). The details of the overlap are in `NOVELTY.md`. In short, their maximum-surprisal-variance state is one large eigenvalue above a degenerate bulk, and their optimality condition becomes `x* = 2(1+u)/(1-u)` under the substitution `d-1 = V` and `r = u/(1+u)`. Their bound `log^2(d-1)/4 + 1` is the `C_max` expression below, and at `V = 151936` the two give 36.58 and 36.59.

The measurement on 140 real distributions still stands, and it is the part worth keeping. Now that the ideal model has a name, the measurement also has a sharper meaning: real next-token distributions sit far below a rigorous universal bound on heat capacity.

## Exact prediction of the ideal model, which is the Reeb and Wolf extremal state

For one top token and `V` degenerate bulk tokens at gap `Delta`, write

```text
x = Delta/T
u = V exp(-x)
C = x^2 u / (1 + u)^2
```

At the heat-capacity maximum, the existing derivation gives

```text
x* = 2(1 + u)/(1 - u).
```

Solving this equation for `u` gives `u = (x* - 2)/(x* + 2)`. Two further laws follow immediately:

```text
p_top(T_melt) = 1/(1 + u) = 1/2 + 1/x*
C_max = x*^2 u/(1 + u)^2 = (x*^2 - 4)/4.
```

The second expression is the exact version of the repository's approximate `(log V)^2/4` ceiling, and it is Reeb and Wolf's `N(d)`, the largest heat capacity any state on `d` dimensions can have. The first says the maximum response occurs *before* an equal probability-mass handoff, and it is their `1 - r_d = 1/2 + 1/log(d-1) + O(1/log^2 d)`. For `V = 151936`, `x* = 12.26`, so the ideal model predicts `p_top = 0.5816` at melting.

## The prediction fails cleanly on real logits

`python src/peak_occupancy.py --local-only` reruns the seven cached pretrained models on the same 20
prompts as the melting-law experiment. For every one of the 140 prompt-model distributions, the
measured top-token probability at `T_melt` is below the ideal prediction.

| Model | Ideal `p_top(T_melt)` | Measured median | Interquartile range |
| --- | ---: | ---: | ---: |
| GPT-2 | 0.5894 | 0.0972 | 0.0657–0.1948 |
| Pythia-160M | 0.5894 | 0.1065 | 0.0405–0.1750 |
| SmolLM2-360M | 0.5896 | 0.1751 | 0.0895–0.2228 |
| OPT-125M | 0.5894 | 0.0680 | 0.0441–0.1146 |
| Qwen2.5-0.5B | 0.5816 | 0.1025 | 0.0345–0.1803 |
| Qwen2.5-1.5B | 0.5816 | 0.0970 | 0.0520–0.1341 |
| BLOOMZ-560M | 0.5784 | 0.1350 | 0.0624–0.2067 |

Across all 140 distributions the median is 0.1055, the range is 0.0071–0.4233, and all 140 are below
their ideal-model value. Pooled across prompts and models, `p_top(T_melt)` correlates with ideal-
normalised peak sharpness at Spearman `rho = 0.750`; this correlation is descriptive because prompts
within a model are not independent.

## Interpretation

The vocabulary/gap formula can locate the broad peak without correctly describing the probability
flow at that peak. A real logit spectrum is not one ground state plus one degenerate band: mass has
already moved through many intermediate logit levels by the time global energy variance is maximal.
That explains both observations already in the repository: real `C_max` is far below the ideal
ceiling, and discrete band structure is rarely resolved.

Reading it against Reeb and Wolf makes the statement precise. Their extremal state maximises surprisal variance over every state of the same dimension. A real logit spectrum sitting at `p_top = 0.106` instead of `0.58` therefore measures how far a language model's next-token distribution is from the maximum-fluctuation state its vocabulary allows.

This gives two scale-free shape statistics:

```text
m = p_max(T_melt)
f = C_max / N(V),  with N(V) = log^2(V)/4 + 1 from Reeb and Wolf Corollary 10.
```

Global logit rescaling moves `T_melt` by the same factor and leaves both unchanged. `m` has a direct sampling interpretation, which is the probability still held by the modal token when the distribution's energy fluctuations peak. `f` is the fraction of the universal bound that the distribution actually reaches. It improves on the repository's earlier normalised sharpness, because its denominator is a proven bound where the old one was an approximation. Whether either statistic predicts correctness or generation-quality collapse is untested. That is the next falsifiable experiment, and this file makes no claim about it.

Raw measurements are in `data/peak_occupancy.json`; the generating code is
`src/peak_occupancy.py`.
