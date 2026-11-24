# Lean 4 check of the exact results

This folder holds a Lean 4 project that proves the exact statements the paper rests on. It depends on Mathlib and Physlib (master) and builds with Lean `v4.36.0`.

```bash
cd lean
lake build TokenThermo
```

The first build needs the Mathlib and Physlib packages. Nothing in the project uses `sorry` or adds axioms beyond the three standard ones (`propext`, `Classical.choice`, `Quot.sound`). `TokenThermo/Axioms.lean` prints the axioms of the main theorems.

## What is checked and what is not

The Lean files cover the algebra and calculus of a finite energy vector `E : ι → ℝ` and of the two-level model. They do not check anything about language models, logits from real networks, the density-of-states compression, the measured ratios, the sampling experiment or the uncertainty scores. Those stay with the Python code and the data.

The definitions in `TokenThermo/Basic.lean` are our own. They use a plain finite sum with the Boltzmann constant set to 1 and temperature as a real number. Physlib has a canonical ensemble, but it is measure theoretic and carries a Boltzmann constant, a phase space unit and a `Temperature` type. `TokenThermo/Bridge.lean` proves that for a finite Physlib ensemble with Boltzmann constant 1 the probability, partition function, mean energy, variance, Shannon entropy and free energy agree with ours.

Results marked new in the table below are not in the paper. They are proved here for the first time in this project, and the paper states them only as formulas or as observations.

## Files

| File | Content |
| --- | --- |
| `TokenThermo/Basic.lean` | Definitions, the Boltzmann form of softmax, the `T = 1` identities, positivity of `C` and `S` |
| `TokenThermo/Deriv.lean` | `dU/dT`, `dS/dT`, `dF/dT`, `d(log Z)/dT`, `dVar/dT`, `dC/dT` |
| `TokenThermo/Ordering.lean` | The ordering theorem for the three inflection conventions |
| `TokenThermo/TwoLevel.lean` | The two-level model: closed form, peak equation, uniqueness, bounds |
| `TokenThermo/TwoLevelOrdering.lean` | Turning points of the two-level model lie below `T_melt` |
| `TokenThermo/Bridge.lean` | Agreement with Physlib's finite canonical ensemble |

## Theorem map

In every row the energy vector is finite and non-empty, `T` is a real temperature and the logit vector is `z` with `E = -z`. A hypothesis `hne` means that the energies are not all equal.

| Lean theorem | Statement | Paper | Python check |
| --- | --- | --- | --- |
| `softmax_eq_prob` | `softmax(z / T) = exp(-E_i / T) / Z(T)` with `E = -z` | Setup, the mapping from logits to energies | `thermo.py` `curves`, which builds `p` from `beta * logit` |
| `S_eq` | `S = log Z + U / T` | Setup, `S = beta U + log Z` | `thermo.py` `curves` computes `S` from `p`, and `validate.py` section 1 checks `F = U - T S` |
| `F_eq_U_sub_TS` | `F = U - T S` | Appendix on validation | `validate.py` section 1, `max abs (F - (U - T S))` |
| `S_one_eq_predEntropy` | `S` at `T = 1` is the predictive entropy of `softmax z` | Setup, "predictive entropy is `S(1)`" | `validate.py` section 1, `abs (predictive entropy - S(1))` |
| `C_one_eq_varentropy` | `C` at `T = 1` is the varentropy of `softmax z` | Setup, "varentropy is `C(1)`" | `validate.py` section 1, `abs (varentropy - C(1))` |
| `var_eq`, `C_nonneg` | `C = Var(E) / T^2` and `C >= 0` | Setup | `thermo.py`, `C = beta^2 * var` |
| `C_eq_zero_iff` | `C = 0` exactly when all energies are equal | none (new) | none |
| `hasDerivAt_U` | `dU/dT = C` | Setup and the validation appendix | `validate.py` section 1, `dU/dT = C` |
| `hasDerivAt_S` | `dS/dT = C / T` | Section on the turning point, first line of the proof | `validate.py` section 1, `dS/dT = C/T` |
| `hasDerivAt_F` | `dF/dT = -S` | none (new) | none |
| `hasDerivAt_C` | `dC/dT = mu3 / T^4 - 2 Var / T^3` with `mu3` the third central moment | none (new) | none |
| `deriv_deriv_S`, `deriv_deriv_logS_exp`, `deriv_deriv_logS` | The second derivatives of `S` against `T`, of `log S` against `log T` and of `log S` against `T` | Proposition on the turning point, the three displayed equivalences | `validate.py` section 4 finds the inflections numerically |
| `ordering_linear`, `ordering_loglog`, `ordering_logS_linear` | With `hne`, at each inflection `T C' / C` equals `1`, `C / S` or `1 + C / S` respectively, and `C' > 0` | Proposition on the turning point | `validate.py` section 4, no turning point with `C' <= 0` out of 208 |
| `turning_point_lt_of_decreasing_after` | If `C' <= 0` from `Tm` on, every first-convention turning point is below `Tm` | The sentence about a single peak | `validate.py` section 4, 106 single-peak cases below `T_melt` |
| `two_level_peak` | For real `g >= 1` the peak equation `x* = 2 (1 + u) / (1 - u)` has exactly one solution `x* > 0`, with `x* > log g` and `x* > 2`, and `C` has a strict global maximum there (new) | Melting law section, stated without proof | `melt_theory.py`, `repro.py` `xstar`, `validate.py` section 5 |
| `two_level_melting` | The general `C` of the energy vector `(0, Delta, ..., Delta)` has its unique maximiser at `T = Delta / x*` (new) | Melting law section | `validate.py` section 2 |
| `C_twoLevel` | The general `C` of that vector equals `x^2 u / (1 + u)^2` with `x = Delta / T` and `u = g exp(-x)` | Melting law section, "the heat capacity is" | `validate.py` section 2, comparison with the Schottky formula |
| `peakOccupancy` | At the peak the top token has probability `1/2 + 1/x*` | `PEAK_OCCUPANCY.md` | `validate.py` section 2, `p_top(T_melt) = 1/2 + 1/x*` |
| `peakHeight` | At the peak `C_max = (x*^2 - 4) / 4` | `PEAK_OCCUPANCY.md` | `validate.py` section 2, `C_max = (x*^2 - 4)/4` |
| `c_lower`, `c_upper`, `c_pos_sq_le` | With `c = x* - log g`, `c x* > 4`, `c x* <= 2 c + 4` and `c^2 <= 2 c + 4`, so `0 < c < 1 + sqrt 5` (new) | Melting law section, "`c` a slowly varying term of order one" | `validate.py` section 5 |
| `peakHeight_bounds` | `(log g)^2 / 4 + 1 - c^2 / 4 < C_max <= (log g)^2 / 4 + 1 - c^2 / 4 + c` (new) | none | `validate.py` section 5 checks the ingredients |
| `two_level_turning_lt_melt` | In the two-level model every turning point of all three conventions lies strictly below `T_melt` (new) | The paper states this only as an observation on synthetic spectra | `validate.py` section 4 |
| `bridge_probability`, `bridge_partitionFunction`, `bridge_meanEnergy`, `bridge_energyVariance`, `bridge_shannonEntropy`, `bridge_helmholtz` | Agreement with Physlib's `CanonicalEnsemble` for finite ensembles with Boltzmann constant 1 | none | none |

## Bounds on the offset

For `g = 151936` the root is `x* = 12.2604`, so `c = 0.3292`. The Lean bounds give `0.3263 < c <= 0.3876`. For `g = 1` the root is `x* = 2.3994`, so `T_melt / Delta = 0.41678`, which is the universal Schottky peak quoted in the paper, and the bounds say `2 < x* <= 1 + sqrt 5`. At the vocabulary sizes of real models the lower bound is within about 1 percent of the truth and the upper bound is looser. Both show `c ~ 4 / log g`.

## Differences found between the paper, the Lean statements and the code

1. **Hypothesis of the ordering theorem.** The paper (Proposition in the turning point section) and `FINDINGS.md` section 7m say the result holds for any distribution that is not a point mass. At finite temperature a softmax has full support, so no distribution is a point mass, and the condition that matters is that the energies are not all equal. For equal energies `C` is identically zero, `S'' = 0` at every temperature, and the equation `T C' / C = 1` is meaningless. The Lean theorems assume `hne`, which says the energies are not all equal. `C_eq_zero_iff` proves this is exactly the failure case. `S > 0` needs only two tokens.
2. **Local maximum.** The paper says that the first temperature above the turning point with `C' = 0` is a local maximum of `C`. The proof only shows `C' > 0` at the turning point. A first critical point above it could in principle be a stationary inflection, so the claim is not proved. The Lean statements stop at `C' > 0` and at the condition that `C` falls beyond some critical point.
3. **Comment in `src/melt_theory.py`.** The closing sentence said that `c` drifts from about 1.7 at `V = 10` to about 0.3 at `V = 1e8`. The table printed just above it gives 1.27 and 0.22. The sentence is corrected in the repository.
4. **`V` against `V - 1`.** The paper and the code use `V` states at the gap `Delta`. A vocabulary of `V` tokens with one ground state has `V - 1` states in the bulk. The Lean theorems are stated for a real `g >= 1`, so they cover both. The difference between `log V` and `log (V - 1)` is below `1 / (V - 1)`, far inside every tolerance used.
5. **Peak in `x` or in `T`.** The paper says `dC/dT = 0` gives the peak equation. Lean proves the statement for `dC/dx` with `x = Delta / T`, and `two_level_melting` transfers it to `T`. The two agree because `x` is a strictly decreasing function of `T`.

No numerical mismatch appeared. `validate.py` and `repro.py` pass, and the numbers they print match the paper (`T / Delta = 0.4168` for the equal-degeneracy Schottky peak, `x* = 12.26` at `V = 151936`, scatter reduced by 58 percent).
