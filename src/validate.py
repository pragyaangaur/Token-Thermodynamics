"""Executable validation checks for the thermodynamic implementation.

Every reported check has an explicit tolerance and raises on failure, so this
file is useful in CI rather than merely printing diagnostics.
"""
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, "src")
from dos import curves_dos, make_dos
from thermo import curves, schottky


def check(name, value, limit):
    print(f"  {name:48s} {value:.3e}  (limit {limit:.1e})")
    if not np.isfinite(value) or value > limit:
        raise AssertionError(f"{name}: {value:.6g} exceeds {limit:.6g}")


print("== 1. exact thermodynamic identities ==")
rng = np.random.default_rng(0)
logits = rng.normal(0, 4, 5000)
temperatures = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
c = curves(logits, 1.0 / temperatures)

check("max |F - (U - T S)|", np.max(np.abs(c["F"] - (c["U"] - c["T"] * c["S"]))), 2e-11)

# Independent centred finite differences at isolated temperatures avoid the
# meaningless relative errors produced near C=0 by the old log-grid check.
du_errors, ds_errors = [], []
for temperature, capacity in zip(temperatures, c["C"]):
    step = temperature * 1e-4
    side = curves(logits, 1.0 / np.array([temperature - step, temperature + step]))
    d_u = (side["U"][1] - side["U"][0]) / (2 * step)
    d_s = (side["S"][1] - side["S"][0]) / (2 * step)
    du_errors.append(abs(d_u - capacity) / capacity)
    ds_errors.append(abs(d_s - capacity / temperature) / (capacity / temperature))
check("max relative error dU/dT = C", max(du_errors), 1e-6)
check("max relative error dS/dT = C/T", max(ds_errors), 1e-6)

# At T=1, predictive entropy is S and varentropy is C.
shifted = logits - logits.max()
p = np.exp(shifted) / np.exp(shifted).sum()
surprisal = -np.log(p)
entropy = -np.sum(p * np.log(p))
varentropy = np.sum(p * (surprisal - np.sum(p * surprisal)) ** 2)
at_one = curves(logits, np.array([1.0]))
check("|predictive entropy - S(1)|", abs(entropy - at_one["S"][0]), 1e-12)
check("|varentropy - C(1)|", abs(varentropy - at_one["C"][0]), 1e-12)


print("\n== 2. analytic two-level Schottky systems ==")
grid_t = np.logspace(-3, 3, 20000)
grid_b = 1.0 / grid_t
max_curve_error = 0.0
for gap, g0, g1 in [(1.0, 1, 1), (3.0, 1, 1), (1.0, 1, 10), (1.0, 1, 100)]:
    system = np.array([0.0] * g0 + [-gap] * g1)
    numeric = curves(system, grid_b)["C"]
    analytic = schottky(grid_t, gap, g0, g1)
    max_curve_error = max(max_curve_error, float(np.max(np.abs(numeric - analytic))))
check("max |C_numeric - C_Schottky|", max_curve_error, 2e-14)

equal = curves(np.array([0.0, -1.0]), grid_b)["C"]
peak_ratio = grid_t[int(np.argmax(equal))]
exact_peak_ratio = 0.4167782798004823
check("|T_peak/gap - 0.416778...|", abs(peak_ratio - exact_peak_ratio), 2e-4)

# Exact peak-occupancy and peak-height laws derived from the stationary condition.
occupancy_errors, height_errors = [], []
for degeneracy in (1, 10, 100, 10000):
    equation = lambda x: x - 2 * (1 + degeneracy * np.exp(-x)) / (1 - degeneracy * np.exp(-x))
    x_star = brentq(equation, np.log(degeneracy) + 1e-9, np.log(degeneracy) + 80)
    u_star = degeneracy * np.exp(-x_star)
    occupancy_errors.append(abs(1 / (1 + u_star) - (0.5 + 1 / x_star)))
    height_errors.append(abs(x_star**2 * u_star / (1 + u_star) ** 2 - (x_star**2 - 4) / 4))
check("max error in p_top(T_melt) = 1/2 + 1/x*", max(occupancy_errors), 1e-12)
check("max error in C_max = (x*^2 - 4)/4", max(height_errors), 1e-11)


print("\n== 3. density-of-states compression ==")
test_logits = np.random.default_rng(7).normal(0, 4, 5000)
test_betas = np.logspace(-2, 2, 200)
exact = curves(test_logits, test_betas)
energies, weights = make_dos(test_logits, K=1024, NB=512)
compressed = curves_dos(energies, weights, test_betas)
check("max absolute entropy error", np.max(np.abs(exact["S"] - compressed["S"])), 1e-5)
check("max absolute heat-capacity error", np.max(np.abs(exact["C"] - compressed["C"])), 2e-5)


print("\nAll validation checks passed.")
