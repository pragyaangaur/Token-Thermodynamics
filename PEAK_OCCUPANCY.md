# The melting peak is not a 50–50 handoff

This is a new derivation and falsification produced during the validation audit on 5 September
2026. The exact mathematics is a consequence of the textbook two-level Schottky model; the
application and 140-distribution measurement appear absent from the LLM literature searched for
this audit.

## Exact prediction of the ideal model

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

The second expression is the exact version of the repository's approximate `(log V)^2/4`
ceiling. The first says the maximum response occurs *before* an equal probability-mass handoff. For
`V = 151936`, `x* = 12.26`, so the ideal model predicts `p_top = 0.5816` at melting.

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

This suggests a new scale-free shape statistic:

```text
m = p_max(T_melt).
```

Global logit rescaling moves `T_melt` by the same factor and leaves `m` unchanged. Unlike `C_max`,
`m` has a direct sampling interpretation: the probability still held by the modal token when the
distribution's energy fluctuations peak. Whether `m` predicts correctness or generation-quality
collapse is untested; that is the next falsifiable experiment, not a claim made here.

Raw measurements are in `data/peak_occupancy.json`; the generating code is
`src/peak_occupancy.py`.
