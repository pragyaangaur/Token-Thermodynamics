"""Does energy-gap clustering recover the true semantic (configurational) entropy?"""
import sys, os
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from synth_split import build, truth, win, B
from dos import make_dos, curves_dos
from gapclust import gap_entropy

EPS = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8]
rng = np.random.default_rng(21)
print("recovery of the true configurational entropy S_config")
print(f"{'delta':>6s} {'spread':>7s} {'d/s':>6s} | {'rho(S_tot)':>10s} | " + " ".join(f"eps={e:<5}" for e in EPS) + " | best  bias(best)")
for spread in [3.0, 6.0]:
    for delta in [0.05, 0.15, 0.4, 0.8, 1.5]:
        St, Sc, G = [], [], {e: [] for e in EPS}
        for _ in range(300):
            M = int(rng.integers(1, 9))
            lg, cl = build(rng, M, delta, spread)
            a, b, c = truth(lg, cl); St.append(a); Sc.append(b)
            E, W = make_dos(lg); cv = curves_dos(E, W, B)
            o = np.argsort(cv["T"]); T, C = cv["T"][o], cv["C"][o]
            Tm = T[int(np.argmax(C))]
            top = np.sort(lg)[::-1][:64]
            logZ = float(np.log(np.exp(lg - lg.max()).sum()) + lg.max())
            for e in EPS:
                G[e].append(gap_entropy(top, logZ, Tm, e)[0])
        St = np.array(St); Sc = np.array(Sc)
        rr = {e: spearmanr(np.array(G[e]), Sc).statistic for e in EPS}
        be = max(rr, key=lambda e: rr[e])
        bias = float(np.mean(np.array(G[be]) - Sc))
        print(f"{delta:6.2f} {spread:7.1f} {delta/spread:6.3f} | {spearmanr(St,Sc).statistic:10.3f} | "
              + " ".join(f"{rr[e]:9.3f}" for e in EPS) + f" | {be:<5} {bias:+.3f}")
