"""Controlled test of the central claim, with ground truth known by construction.

Build a token distribution with KNOWN cluster structure:
  - M answer clusters ('meanings'); cluster m sits at logit level L_m
  - inside a cluster, n_m surface variants spread over a small width delta
    (these are tokenisation / paraphrase variants of the same answer)
  - plus a bulk of ~150k junk tokens far below
Then the exact chain rule gives
  S_total = S_config (entropy over clusters, i.e. semantic entropy)
          + S_vib   (average entropy inside a cluster)
Question: can integrating C(T)/T over a temperature window recover S_config
from the spectrum alone, with no knowledge of the clustering?
"""
import sys, os
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from thermo import beta_grid
from dos import make_dos, curves_dos

B = beta_grid(400, 1e-2, 3e2)

def build(rng, M, delta, spread, nvar_max=6, bulk=150000):
    """M clusters, gaps ~ spread, within-cluster width delta."""
    levels = np.sort(rng.uniform(0, spread, M))[::-1]
    logits, cl = [], []
    for m in range(M):
        n = int(rng.integers(1, nvar_max + 1))
        v = levels[m] + rng.uniform(-delta, delta, n)
        logits.append(v); cl.append(np.full(n, m))
    logits.append(rng.normal(-18, 3, bulk)); cl.append(np.full(bulk, -1))
    return np.concatenate(logits), np.concatenate(cl)

def truth(logits, cl):
    p = np.exp(logits - logits.max()); p /= p.sum()
    S_tot = float(-(p * np.log(p + 1e-300)).sum())
    S_cfg, S_vib = 0.0, 0.0
    for c in np.unique(cl):
        m = cl == c
        pc = p[m].sum()
        if pc <= 0: continue
        S_cfg -= pc * np.log(pc)
        q = p[m] / pc
        S_vib += pc * float(-(q * np.log(q + 1e-300)).sum())
    return S_tot, S_cfg, S_vib

def win(T, C, Tm, a, b):
    m = (T >= a * Tm) & (T <= b * Tm)
    return float(np.trapezoid(C[m] / T[m], T[m])) if m.sum() > 1 else 0.0

rng = np.random.default_rng(0)
GRID = [(0.0,0.05),(0.0,0.1),(0.0,0.2),(0.05,0.3),(0.1,0.4),(0.2,0.6),(0.3,0.8),(0.5,1.0),
        (0.05,0.5),(0.1,0.6),(0.02,0.25),(0.0,0.5),(0.0,1.0),(0.15,0.5)]

for regime, delta, spread in [("well separated  (delta=0.15, spread=6)", 0.15, 6.0),
                              ("moderate        (delta=0.5,  spread=5)", 0.50, 5.0),
                              ("overlapping     (delta=1.5,  spread=4)", 1.50, 4.0)]:
    rows = []
    for _ in range(400):
        M = int(rng.integers(1, 9))
        lg, cl = build(rng, M, delta, spread)
        St, Sc, Sv = truth(lg, cl)
        E, W = make_dos(lg); c = curves_dos(E, W, B)
        o = np.argsort(c["T"]); T, C = c["T"][o], c["C"][o]
        Tm = T[int(np.argmax(C))]
        rows.append((St, Sc, Sv, T, C, Tm))
    St = np.array([r[0] for r in rows]); Sc = np.array([r[1] for r in rows]); Sv = np.array([r[2] for r in rows])
    print(f"\n{regime}   S_config mean={Sc.mean():.3f}  S_vib mean={Sv.mean():.3f}")
    print(f"   baseline: rho(total entropy S_total, S_config) = {spearmanr(St, Sc).statistic:+.4f}")
    best = (0, None)
    for a, b in GRID:
        v = np.array([win(r[3], r[4], r[5], a, b) for r in rows])
        if np.std(v) < 1e-12: continue
        rho = spearmanr(v, Sc).statistic
        rho_v = spearmanr(v, Sv).statistic
        if abs(rho) > abs(best[0]): best = (rho, (a, b))
        print(f"     window [{a:4.2f},{b:4.2f}] : rho vs S_config {rho:+.4f}   rho vs S_vib {rho_v:+.4f}")
    print(f"   -> best window {best[1]} rho={best[0]:+.4f}  (vs {spearmanr(St, Sc).statistic:+.4f} for plain entropy)")
