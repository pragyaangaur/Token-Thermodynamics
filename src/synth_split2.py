import sys, os
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from synth_split import build, truth, win, B
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from dos import make_dos, curves_dos

def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

rng = np.random.default_rng(1)
for regime, delta, spread in [("well separated (d=0.15,s=6)", .15, 6.), ("moderate (d=0.5,s=5)", .5, 5.), ("overlapping (d=1.5,s=4)", 1.5, 4.)]:
    D = []
    for _ in range(700):
        M = int(rng.integers(1, 9))
        lg, cl = build(rng, M, delta, spread)
        St, Sc, Sv = truth(lg, cl)
        E, W = make_dos(lg); c = curves_dos(E, W, B)
        o = np.argsort(c["T"]); T, C = c["T"][o], c["C"][o]
        Tm = T[int(np.argmax(C))]
        cold = win(T, C, Tm, 0.0, 0.05); cold2 = win(T, C, Tm, 0.0, 0.12)
        mid = win(T, C, Tm, 0.10, 0.40); Cm = C.max()
        D.append((St, Sc, Sv, cold, cold2, mid, Cm, np.interp(1.0, T, C)))
    D = np.array(D)
    St, Sc, Sv, cold, cold2, mid, Cmax, C1 = D.T
    print(f"\n{regime}")
    print(f"   how good is the cold window as an estimate of the NUISANCE term S_vib?")
    print(f"     rho(S_cold[0,0.05], S_vib)    = {spearmanr(cold, Sv).statistic:+.4f}")
    print(f"     rho(S_cold[0,0.05], S_config) = {spearmanr(cold, Sc).statistic:+.4f}   <- near zero means it is a clean nuisance estimate")
    sets = {
        "S_total alone":                 np.c_[St],
        "S_total + C(T=1)":              np.c_[St, C1],
        "S_total + S_cold  (subtract)":  np.c_[St, cold],
        "S_total + S_cold + S_cold2":    np.c_[St, cold, cold2],
        "S_total + S_cold + S_mid":      np.c_[St, cold, mid],
        "S_cold + S_mid only":           np.c_[cold, mid],
    }
    print(f"   predicting S_config (= semantic entropy), 5-fold held out:")
    for n, X in sets.items():
        p = cross_val_predict(make_pipeline(StandardScaler(), Ridge(1.0)), X, Sc, cv=KFold(5, shuffle=True, random_state=0))
        print(f"     {n:32s} R^2={r2(Sc, p):+.4f}  rho={spearmanr(Sc, p).statistic:+.4f}")
    # the explicit physical estimator, no fitting at all
    est = St - cold
    print(f"   unfitted estimator  S_total - S_cold[0,0.05] :  rho={spearmanr(est, Sc).statistic:+.4f}  (S_total alone {spearmanr(St, Sc).statistic:+.4f})")
