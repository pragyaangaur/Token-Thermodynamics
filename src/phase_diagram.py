"""When does the temperature axis separate meaning from surface form?

Sweep the two scales of the problem:
  delta  = spread of surface variants WITHIN one meaning
  spread = spread of energy levels BETWEEN meanings
and measure how much the cold-window nuisance estimate improves prediction of
the configurational (semantic) entropy over plain total entropy.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from synth_split import build, truth, win, B
from dos import make_dos, curves_dos
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold

def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
def fit(X, y):
    return r2(y, cross_val_predict(make_pipeline(StandardScaler(), Ridge(1.0)), X, y,
                                   cv=KFold(5, shuffle=True, random_state=0)))

DELTAS = [0.05, 0.15, 0.4, 0.8, 1.5, 3.0]
SPREADS = [2.0, 4.0, 8.0]
N = 300
rng = np.random.default_rng(11)
print(f"gain in held-out R^2 for predicting semantic entropy, from adding the cold-window feature")
print(f"{'delta':>7s} {'spread':>7s} {'d/s':>6s} | {'R2(S_tot)':>10s} {'R2(+cold)':>10s} {'gain':>8s} | {'rho(cold,S_vib)':>16s} {'rho(cold,S_cfg)':>16s}")
from scipy.stats import spearmanr
for spread in SPREADS:
    for delta in DELTAS:
        St, Sc, Sv, cold, mid = [], [], [], [], []
        for _ in range(N):
            M = int(rng.integers(1, 9))
            lg, cl = build(rng, M, delta, spread)
            a, b, c = truth(lg, cl); St.append(a); Sc.append(b); Sv.append(c)
            E, W = make_dos(lg); cv = curves_dos(E, W, B)
            o = np.argsort(cv["T"]); T, C = cv["T"][o], cv["C"][o]
            Tm = T[int(np.argmax(C))]
            cold.append(win(T, C, Tm, 0.0, 0.05)); mid.append(win(T, C, Tm, 0.1, 0.4))
        St, Sc, Sv = map(np.array, (St, Sc, Sv)); cold = np.array(cold); mid = np.array(mid)
        a = fit(np.c_[St], Sc); b = fit(np.c_[St, cold], Sc)
        print(f"{delta:7.2f} {spread:7.1f} {delta/spread:6.3f} | {a:10.4f} {b:10.4f} {b-a:+8.4f} | "
              f"{spearmanr(cold,Sv).statistic:16.3f} {spearmanr(cold,Sc).statistic:16.3f}")
