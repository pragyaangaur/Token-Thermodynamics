"""E7: can a single forward pass approximate semantic entropy?

Theory. Total token entropy splits exactly as
    S_total = S_over_meaning_clusters + sum_c p_c * S_within_cluster
which in condensed-matter language is configurational entropy plus vibrational
entropy. Semantic entropy is the first term and needs K samples plus a
clustering model. The claim tested here is that the two terms live at different
ENERGY scales, so the heat-capacity spectrum separates them by temperature
alone, from one forward pass:
    S(window) = integral of C(T)/T dT over  [a*T_melt, b*T_melt]
Windows are expressed relative to the melting temperature so the measure does
not depend on the logit scale.
"""
import json, sys, os, pickle, argparse
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def window_entropy(T, C, Tm, a, b):
    m = (T >= a * Tm) & (T <= b * Tm)
    if m.sum() < 2: return 0.0
    return float(np.trapezoid(C[m] / T[m], T[m]))

def main(recs_pkl, sem_json):
    R = pickle.load(open(recs_pkl, "rb"))
    key = {(r["subject"], r["rel"]): r for r in R if not r.get("fake")}
    S = json.load(open(sem_json))
    rows = []
    for s in S:
        r = key.get((s["subject"], s["rel"]))
        if r is None: continue
        rows.append((r, s))
    print(f"matched {len(rows)} items with reference semantic entropy")
    T = rows[0][0]["_T"]
    se = np.array([s["sem_entropy"] for _, s in rows])
    y = np.array([1 - r["correct"] for r, _ in rows])
    print(f"  semantic entropy: mean {se.mean():.3f}  frac>0 {np.mean(se>0):.3f}")
    print(f"  error rate {y.mean():.3f}")
    print(f"  AUROC of reference semantic entropy for error : {roc_auc_score(y, se):.4f}   (needs {len(S[0]['samples'])} samples)")
    ent = np.array([r["b_entropy"] for r, _ in rows])
    var = np.array([r["b_varentropy"] for r, _ in rows])
    print(f"  AUROC of single-pass entropy   S(T=1)          : {roc_auc_score(y, ent):.4f}")
    print(f"  AUROC of single-pass varentropy C(T=1)         : {roc_auc_score(y, var):.4f}")
    print(f"  Spearman(semantic entropy, single-pass entropy): {spearmanr(se, ent).statistic:.4f}")
    print(f"  Spearman(semantic entropy, varentropy C(1))    : {spearmanr(se, var).statistic:.4f}")

    print("\n  which temperature window of C(T) tracks semantic entropy?")
    print("  window [a,b] x T_melt         Spearman vs semantic entropy")
    grid = [(0.0, 0.05), (0.0, 0.1), (0.02, 0.1), (0.05, 0.2), (0.1, 0.3), (0.2, 0.5),
            (0.3, 0.7), (0.5, 1.0), (0.7, 1.3), (1.0, 2.0), (0.0, 1.0), (0.05, 0.5),
            (0.1, 0.5), (0.02, 0.3), (0.01, 0.2)]
    best = None
    feats = {}
    for a, b in grid:
        v = np.array([window_entropy(T, r["_C"], r["t_Tmain"], a, b) for r, _ in rows])
        if np.std(v) < 1e-12: continue
        rho = spearmanr(se, v).statistic
        feats[(a, b)] = v
        print(f"    [{a:4.2f}, {b:4.2f}]                       rho = {rho:+.4f}    AUROC(error) = {roc_auc_score(y, v):.4f}")
        if best is None or abs(rho) > abs(best[0]): best = (rho, a, b, v)
    print(f"  best single window: [{best[1]},{best[2]}]  rho={best[0]:+.4f}")

    print("\n  energy-gap clustering: cut the top of the spectrum wherever the gap")
    print("  between consecutive levels exceeds eps * T_melt, entropy over groups")
    GEPS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8)
    gcols = [f"g_S_{e}" for e in GEPS]
    for c in gcols:
        v = np.array([r[c] for r, _ in rows], float)
        if np.std(v) < 1e-12: continue
        print(f"    {c:10s} rho vs semantic entropy = {spearmanr(se, v).statistic:+.4f}   AUROC(error) = {roc_auc_score(y, v):.4f}")
    G = np.array([[r[c] for c in gcols + [f"g_n_{e}" for e in GEPS]] for r, _ in rows], float)

    print("\n  predicting semantic entropy from the single-pass spectrum (held-out R^2)")
    Xsets = {
        "entropy S(1) only":        np.c_[ent],
        "S(1) + C(1)":              np.c_[ent, var],
        "all baselines":            np.array([[r[c] for c in ("b_entropy","b_varentropy","b_pmax","b_logprob_top1","b_margin_logit","b_top5mass","b_margin_prob")] for r, _ in rows]),
        "spectrum windows":         np.array([feats[k] for k in feats]).T,
        "baselines + windows":      np.hstack([np.array([[r[c] for c in ("b_entropy","b_varentropy","b_pmax","b_logprob_top1","b_margin_logit","b_top5mass","b_margin_prob")] for r, _ in rows]),
                                               np.array([feats[k] for k in feats]).T]),
        "gap clustering":           G,
        "baselines + gap":          np.hstack([np.array([[r[c] for c in ("b_entropy","b_varentropy","b_pmax","b_logprob_top1","b_margin_logit","b_top5mass","b_margin_prob")] for r, _ in rows]), G]),
        "full shape curve":         np.array([r["_shape"] for r, _ in rows]),
        "baselines + shape curve":  np.hstack([np.array([[r[c] for c in ("b_entropy","b_varentropy","b_pmax","b_logprob_top1","b_margin_logit","b_top5mass","b_margin_prob")] for r, _ in rows]),
                                               np.array([r["_shape"] for r, _ in rows])]),
        "EVERYTHING single-pass":   np.hstack([np.array([[r[c] for c in ("b_entropy","b_varentropy","b_pmax","b_logprob_top1","b_margin_logit","b_top5mass","b_margin_prob")] for r, _ in rows]),
                                               G, np.array([feats[k] for k in feats]).T,
                                               np.array([r["_shape"] for r, _ in rows])]),
    }
    from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold
    for name, X in Xsets.items():
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        pred = cross_val_predict(m, X, se, cv=KFold(5, shuffle=True, random_state=0))
        r2 = 1 - np.sum((se - pred) ** 2) / np.sum((se - se.mean()) ** 2)
        rho = spearmanr(se, pred).statistic
        # and: use the PREDICTED semantic entropy as a hallucination detector
        auc = roc_auc_score(y, pred)
        print(f"    {name:26s} R^2={r2:+.4f}  rho={rho:+.4f}   AUROC as detector={auc:.4f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
