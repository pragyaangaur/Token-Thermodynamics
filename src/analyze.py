import json, sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from curvefeats import feats_from_curve
from gapclust import features as gap_features
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE = ["b_entropy","b_logprob_top1","b_pmax","b_margin_logit","b_margin_prob","b_top5mass","b_varentropy"]
SHAPE = ["t_gap_ratio","t_band_weight","t_valley","t_logwidth","t_spec_var","t_spec_skew",
         "t_cold_frac","t_cold10","t_cold25","t_cold50","t_npeaks","t_has_low"]
SCALE = ["t_Cmax","t_logTmain","t_spec_mean"]

def load(paths):
    recs = []
    for p in paths:
        d = json.load(open(p)); T = 1.0 / np.array(d["betas"])
        o = np.argsort(T); Ts = T[o]
        for r in d["results"]:
            C = np.array(r["C"])[o]
            f = feats_from_curve(Ts, C)
            shape = f.pop("_shape")
            Cs = np.array(r["C_span"])[o]
            fs = feats_from_curve(Ts, Cs, prefix="sp_"); fs.pop("_shape")
            rec = {k: r[k] for k in ("rel","subject","gold","pred","correct","abstain","links","fake","top_logits") if k in r}
            rec.update({k: r[k] for k in r if k.startswith("b_")})
            rec.update(f); rec.update(fs)
            U = np.array(r["U"])[o]
            logZ1 = float(r["b_entropy"]) - float(np.interp(1.0, Ts, U))
            rec.update(gap_features(r["top_logits"], logZ1, f["t_Tmain"]))
            # entropy and heat capacity read at the item's OWN melting temperature
            # instead of at the arbitrary T=1. Both are scale-free by construction.
            Sc_ = np.array(r["S"])[o]; Cc_ = np.array(r["C"])[o]
            Tm_ = f["t_Tmain"]
            rec["n_S_at_melt"] = float(np.interp(Tm_, Ts, Sc_))
            rec["n_C_at_melt"] = float(np.interp(Tm_, Ts, Cc_))
            for q in (0.25, 0.5, 0.75):
                rec[f"n_S_{int(q*100)}melt"] = float(np.interp(q * Tm_, Ts, Sc_))
                rec[f"n_C_{int(q*100)}melt"] = float(np.interp(q * Tm_, Ts, Cc_))
            rec["_shape"] = shape
            rec["_C"] = C; rec["_S"] = np.array(r["S"])[o]; rec["_U"] = np.array(r["U"])[o]
            rec["_T"] = Ts; rec["model"] = d["model"]
            recs.append(rec)
    return recs

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(set(y)) < 2: return np.nan
    return roc_auc_score(y, s)

def boot_auc(y, s, n=2000, seed=0):
    y = np.asarray(y); s = np.asarray(s); rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2: continue
        out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])

def cv_auc(X, y, folds=5, reps=4, seed=0):
    X = np.nan_to_num(np.asarray(X, float), nan=0, posinf=0, neginf=0)
    y = np.asarray(y); aucs = []
    for r in range(reps):
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
            m.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))

def mat(recs, cols, with_shape=False):
    X = np.array([[r.get(c, 0.0) for c in cols] for r in recs], float)
    if with_shape:
        X = np.hstack([X, np.array([r["_shape"] for r in recs], float)])
    return np.nan_to_num(X, nan=0, posinf=0, neginf=0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    R = load(a.files)
    print(f"loaded {len(R)} records from {len(a.files)} file(s)")
    json.dump("ok", open("/dev/null","w"))
    import pickle; pickle.dump(R, open("data/records.pkl","wb"))
    print("cached -> data/records.pkl")
