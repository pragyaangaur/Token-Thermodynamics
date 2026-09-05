"""Cross-model transfer: train the error detector on one model, deploy on another.

This is the non-synthetic version of the rescaling experiment. Two models have
different logit scales and different competence, so a detector built on
scale-dependent scalars should transfer badly and a scale-free one should not.
"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from analyze import load, BASE, SHAPE, SCALE, mat
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

NORM = ["n_S_at_melt", "n_C_at_melt"] + [f"n_S_{q}melt" for q in (25, 50, 75)] + [f"n_C_{q}melt" for q in (25, 50, 75)]

A = load(["data/main_1p5b.json"]); B = load(["data/main_0p5b.json"])
print(f"1.5B n={len(A)} acc={np.mean([r['correct'] for r in A]):.3f}   "
      f"0.5B n={len(B)} acc={np.mean([r['correct'] for r in B]):.3f}")
ya = np.array([1 - r["correct"] for r in A]); yb = np.array([1 - r["correct"] for r in B])
print(f"mean logit scale proxy (C_max): 1.5B {np.mean([r['t_Cmax'] for r in A]):.2f}   0.5B {np.mean([r['t_Cmax'] for r in B]):.2f}")
print(f"mean T_melt:                    1.5B {np.mean([r['t_Tmain'] for r in A]):.3f}   0.5B {np.mean([r['t_Tmain'] for r in B]):.3f}")

SETS = [("entropy S(1) alone", ["b_entropy"], False),
        ("all 7 baselines", BASE, False),
        ("S(Tmelt/2) alone", ["n_S_50melt"], False),
        ("melt-normalised S,C", NORM, False),
        ("thermo SHAPE only", SHAPE, False),
        ("thermo shape curve", [], True)]

print(f"\n{'feature set':24s} {'within 1.5B':>12s} {'within 0.5B':>12s} {'1.5B->0.5B':>12s} {'0.5B->1.5B':>12s} {'transfer loss':>14s}")
for name, cols, sc in SETS:
    def fit(tr, ytr):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        m.fit(mat(tr, cols, sc), ytr); return m
    from sklearn.model_selection import cross_val_predict
    wa = roc_auc_score(ya, cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), mat(A, cols, sc), ya, cv=5, method="predict_proba")[:, 1])
    wb = roc_auc_score(yb, cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), mat(B, cols, sc), yb, cv=5, method="predict_proba")[:, 1])
    ab = roc_auc_score(yb, fit(A, ya).predict_proba(mat(B, cols, sc))[:, 1])
    ba = roc_auc_score(ya, fit(B, yb).predict_proba(mat(A, cols, sc))[:, 1])
    loss = ((wb - ab) + (wa - ba)) / 2
    print(f"{name:24s} {wa:12.4f} {wb:12.4f} {ab:12.4f} {ba:12.4f} {loss:+14.4f}")
print("\ntransfer loss = mean drop from the within-model cross-validated score")
print("when the detector is trained on the other model instead.")
