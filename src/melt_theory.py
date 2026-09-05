"""Analytic melting condition.

For a ground state of degeneracy 1 and a bulk of g states at energy Delta, the
heat capacity is the Schottky form C = x^2 u/(1+u)^2 with x = Delta/T, u = g e^-x.
Setting dC/dx = 0 gives the exact peak condition
        x* = 2 (1 + u) / (1 - u),      u = g exp(-x*)
so T_melt = Delta / x*, and x* = log g + c(g) with c a slowly varying O(1) term.
"""
import numpy as np
from scipy.optimize import brentq

def xstar(g):
    f = lambda x: x - 2 * (1 + g * np.exp(-x)) / (1 - g * np.exp(-x))
    lo = np.log(g) + 1e-6
    return brentq(f, lo, lo + 60)

print(f"{'g (bulk size)':>14s} {'x*':>8s} {'log g':>8s} {'c = x* - log g':>15s} {'T_melt*log g/Delta':>20s}")
for g in [10, 100, 1000, 10**4, 10**5, 151936, 10**6, 10**8]:
    x = xstar(g); lg = np.log(g)
    print(f"{g:14d} {x:8.4f} {lg:8.4f} {x-lg:15.4f} {lg/x:20.4f}")
print()
print("So T_melt = Delta / (log V + c), c drifts from ~1.7 at V=10 to ~0.3 at V=1e8.")
print("For a 151936-token vocabulary: x* = %.3f, i.e. T_melt = Delta / %.3f" % (xstar(151936), xstar(151936)))
print("A model melts at sampling temperature T once T * (log V + c) exceeds the")
print("logit gap between the top token and the vocabulary bulk.")
print()
print("consequence: truncating to top-k replaces V by k, so")
for k in [8, 40, 200, 1000, 151936]:
    print(f"   top-k = {k:6d}  ->  melting temperature is {xstar(151936)/xstar(k):.2f}x higher than untruncated")
