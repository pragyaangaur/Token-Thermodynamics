import json, sys, os, pickle, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from analyze import load, BASE, SHAPE, SCALE, auroc, boot_auc, cv_auc, mat
NORM = ["n_S_at_melt","n_C_at_melt"] + [f"n_S_{q}melt" for q in (25,50,75)] + [f"n_C_{q}melt" for q in (25,50,75)]
GAP = [f"g_S_{e}" for e in (0.02,0.05,0.1,0.2,0.35,0.5,0.8)] + [f"g_n_{e}" for e in (0.02,0.05,0.1,0.2,0.35,0.5,0.8)]
from curvefeats import feats_from_curve

def hdr(s): print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)

def rescale_curve(T, C, s):
    """Rescaling every logit by 1/s maps C(T) -> C(T/s). Exact, no model rerun."""
    lt = np.log(T)
    return np.interp(lt, lt + np.log(s), C, left=C[0], right=C[-1])

def main(files, tag="run"):
    R = load(files)
    real = [r for r in R if not r.get("fake")]
    fake = [r for r in R if r.get("fake")]
    print(f"records: {len(R)}  real={len(real)}  fake={len(fake)}")
    if real:
        print(f"accuracy on real items: {np.mean([r['correct'] for r in real]):.4f}")
        for rel in sorted(set(r['rel'] for r in real)):
            s = [r for r in real if r['rel'] == rel]
            print(f"   {rel:16s} n={len(s):5d} acc={np.mean([x['correct'] for x in s]):.3f}")

    y = np.array([1 - r["correct"] for r in real])       # 1 = error / hallucination
    hdr("E1  mean heat-capacity spectrum, correct vs incorrect")
    T = real[0]["_T"]
    for lab, sub in [("correct", [r for r in real if r["correct"]]),
                     ("incorrect", [r for r in real if not r["correct"]]),
                     ("fake-entity", fake)]:
        if not sub: continue
        Cm = np.mean([r["_C"] for r in sub], 0)
        Sm = np.mean([r["_S"] for r in sub], 0)
        i = int(np.argmax(Cm))
        # entropy released below Tmain/4, scale free
        dS = Cm / T; tot = np.trapezoid(dS, T)
        m = T <= T[i] / 4
        print(f"  {lab:12s} n={len(sub):5d}  Tmain={T[i]:.3f}  Cmax={Cm.max():7.3f}  "
              f"S(T=1)={np.interp(1.0,T,Sm):.3f}  cold25={np.trapezoid(dS[m],T[m])/tot:.5f}")
        sh = np.mean([r["_shape"] for r in sub], 0)
        print("      shape C(T/Tmain)/Cmax on log grid -4..2: " + " ".join(f"{v:.2f}" for v in sh))

    hdr("E2  single-feature AUROC for predicting an incorrect answer")
    rows = []
    for c in BASE + SCALE + SHAPE + GAP + NORM + ["sp_cold25", "sp_cold_frac", "sp_spec_var", "sp_gap_ratio"]:
        v = np.array([r.get(c, 0.0) for r in real], float)
        v = np.nan_to_num(v)
        if np.std(v) < 1e-12: continue
        a = auroc(y, v)
        a2 = max(a, 1 - a)
        lo, hi = boot_auc(y, v if a >= .5 else -v)
        rows.append((a2, c, lo, hi, "base" if c.startswith("b_") else ("span" if c.startswith("sp_") else ("shape" if c in SHAPE else ("gap" if c in GAP else ("norm" if c in NORM else "scale"))))))
    rows.sort(reverse=True)
    for a, c, lo, hi, k in rows:
        print(f"  {a:.4f}  [{lo:.3f},{hi:.3f}]  {k:6s}  {c}")

    hdr("E3  cross-validated detectors (5-fold x 4 reps, logistic regression)")
    sets = {
        "entropy alone":            (["b_entropy"], False),
        "varentropy alone  = C(1)": (["b_varentropy"], False),
        "entropy + varentropy":     (["b_entropy", "b_varentropy"], False),
        "ALL baselines":            (BASE, False),
        "thermo SHAPE only":        (SHAPE, False),
        "thermo shape + scale":     (SHAPE + SCALE, False),
        "gap-clustering only":      (GAP, False),
        "S,C at melting point":     (NORM, False),
        "S(Tmelt) alone":           (["n_S_at_melt"], False),
        "baselines + gap":          (BASE + GAP, False),
        "baselines + thermo":       (BASE + SHAPE + SCALE, False),
        "baselines + shape curve":  (BASE, True),
        "everything":               (BASE + SHAPE + SCALE + GAP, True),
    }
    for name, (cols, sc) in sets.items():
        X = mat(real, cols, with_shape=sc)
        m, s = cv_auc(X, y)
        print(f"  {m:.4f} +- {s:.4f}   {name}  (d={X.shape[1]})")

    hdr("E3a  paired bootstrap: does the thermodynamic feature set beat the baselines?")
    from sklearn.model_selection import cross_val_predict
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.pipeline import make_pipeline as _mp
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.metrics import roc_auc_score as _auc
    def oof(cols, sc=False):
        X = mat(real, cols, sc)
        return cross_val_predict(_mp(_SS(), _LR(max_iter=3000)), X, y, cv=5, method="predict_proba")[:, 1]
    pairs = [("entropy+varentropy", ["b_entropy", "b_varentropy"], False, "ALL baselines", BASE, False),
             ("ALL baselines", BASE, False, "baselines + thermo", BASE + SHAPE + SCALE, False),
             ("ALL baselines", BASE, False, "everything", BASE + SHAPE + SCALE + GAP, True)]
    rngb = np.random.default_rng(0)
    for na, ca, sa, nb, cb, sb in pairs:
        pa, pb = oof(ca, sa), oof(cb, sb)
        aa, ab = _auc(y, pa), _auc(y, pb)
        d = []
        for _ in range(4000):
            i = rngb.integers(0, len(y), len(y))
            if len(set(y[i])) < 2: continue
            d.append(_auc(y[i], pb[i]) - _auc(y[i], pa[i]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = "SIGNIFICANT" if lo > 0 else ("significant (worse)" if hi < 0 else "not significant")
        print(f"  {nb} vs {na}:  {ab:.4f} - {aa:.4f} = {ab-aa:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]  {sig}")

    hdr("E3b  marginal value of each feature BEYOND entropy and varentropy")
    print("  Reference model uses S(T=1) and C(T=1) only. Each row adds one feature.")
    ref, _ = cv_auc(mat(real, ["b_entropy", "b_varentropy"]), y)
    print(f"  reference AUROC (entropy + varentropy) = {ref:.4f}")
    marg = []
    for c in SHAPE + SCALE + GAP + NORM + [x for x in BASE if x not in ("b_entropy", "b_varentropy")]:
        v = np.array([r.get(c, 0.0) for r in real], float)
        if np.std(np.nan_to_num(v)) < 1e-12: continue
        m, sd = cv_auc(mat(real, ["b_entropy", "b_varentropy", c]), y)
        marg.append((m - ref, m, sd, c))
    marg.sort(reverse=True)
    for d, m, sd, c in marg:
        star = "  <-- helps" if d > 2 * sd / np.sqrt(20) else ""
        print(f"  {d:+.4f}  ->{m:.4f} +-{sd:.4f}   {c}{star}")

    hdr("E4  generalisation to a held-out relation type")
    rels = sorted(set(r["rel"] for r in real))
    for name, (cols, sc) in [("ALL baselines", (BASE, False)),
                             ("baselines + thermo", (BASE + SHAPE + SCALE, False)),
                             ("everything", (BASE + SHAPE + SCALE + GAP, True))]:
        accs = []
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        for held in rels:
            tr = [r for r in real if r["rel"] != held]; te = [r for r in real if r["rel"] == held]
            ytr = np.array([1 - r["correct"] for r in tr]); yte = np.array([1 - r["correct"] for r in te])
            if len(set(yte)) < 2 or len(set(ytr)) < 2: continue
            mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            mdl.fit(mat(tr, cols, sc), ytr)
            accs.append((held, roc_auc_score(yte, mdl.predict_proba(mat(te, cols, sc))[:, 1])))
        print(f"  {name:22s} " + "  ".join(f"{h}={v:.3f}" for h, v in accs) + f"   mean={np.mean([v for _,v in accs]):.4f}")

    hdr("E5  robustness to an unknown logit rescaling (miscalibration / temperature shift)")
    print("  Train on the original logits. Test on the SAME items with every logit")
    print("  divided by an unknown per-item s ~ LogNormal(0, sigma). This is exactly")
    print("  what happens across quantisation, distillation, or a changed sampling")
    print("  temperature. Baselines are scale-dependent. Ratio features are not.")
    rng = np.random.default_rng(0)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    RB = ["b_entropy", "b_varentropy", "b_logprob_top1", "b_pmax", "b_margin_logit"]
    idx = np.arange(len(real)); rng.shuffle(idx)
    cut = int(.6 * len(idx)); tr = [real[i] for i in idx[:cut]]; te0 = [real[i] for i in idx[cut:]]
    ytr = np.array([1 - r["correct"] for r in tr]); yte = np.array([1 - r["correct"] for r in te0])
    Tg = real[0]["_T"]

    def rescaled(r, s):
        """Every logit divided by s. Recomputed exactly from the stored curves."""
        n = dict(r)
        Cn = rescale_curve(Tg, r["_C"], s)
        f = feats_from_curve(Tg, Cn); n["_shape"] = f.pop("_shape"); n.update(f)
        Ts = float(s)                      # new beta = 1/s  <=>  read curves at T = s
        S_ = float(np.interp(Ts, Tg, r["_S"]))
        U_ = float(np.interp(Ts, Tg, r["_U"]))
        logit_max = -float(r["_U"][0])     # U -> E_min = -logit_max as T -> 0
        logZ = S_ - U_ / Ts
        n["b_entropy"] = S_
        n["b_varentropy"] = float(np.interp(Ts, Tg, r["_C"]))
        n["b_logprob_top1"] = logit_max / Ts - logZ
        n["b_pmax"] = float(np.exp(min(n["b_logprob_top1"], 0.0)))
        n["b_margin_logit"] = r["b_margin_logit"] / Ts
        return n

    # sanity: s = 1 must reproduce the stored features
    chk = rescaled(te0[0], 1.0)
    print(f"  sanity s=1: entropy {chk['b_entropy']:.4f} vs {te0[0]['b_entropy']:.4f} | "
          f"varent {chk['b_varentropy']:.4f} vs {te0[0]['b_varentropy']:.4f} | "
          f"logp1 {chk['b_logprob_top1']:.4f} vs {te0[0]['b_logprob_top1']:.4f}")
    for sigma in [0.0, 0.15, 0.3, 0.5]:
        te = [rescaled(r, float(np.exp(rng.normal(0, sigma))) if sigma > 0 else 1.0) for r in te0]
        out = {}
        for name, cols, sc in [("baselines", RB, False),
                               ("S(Tmelt)", ["n_S_at_melt"], False),
                               ("thermo SHAPE", SHAPE, False),
                               ("shape curve", [], True)]:
            mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            mdl.fit(mat(tr, cols, sc), ytr)
            out[name] = roc_auc_score(yte, mdl.predict_proba(mat(te, cols, sc))[:, 1])
        # fair-play control: train the baselines WITH the same rescaling as augmentation
        if sigma > 0:
            rg2 = np.random.default_rng(99)
            tra = [rescaled(r, float(np.exp(rg2.normal(0, sigma)))) for r in tr] + tr
            ytra = np.concatenate([ytr, ytr])
            mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            mdl.fit(mat(tra, RB, False), ytra)
            out["baselines+aug"] = roc_auc_score(yte, mdl.predict_proba(mat(te, RB, False))[:, 1])
        print(f"  sigma={sigma:.2f}  " + "   ".join(f"{k}={v:.4f}" for k, v in out.items()))

    if fake:
        hdr("E6  real answerable question vs fabricated entity (forced confabulation)")
        yy = np.array([0] * len(real) + [1] * len(fake))
        allr = real + fake
        rows = []
        for c in BASE + SHAPE + SCALE:
            v = np.nan_to_num(np.array([r.get(c, 0.0) for r in allr], float))
            if np.std(v) < 1e-12: continue
            a = auroc(yy, v); rows.append((max(a, 1 - a), c))
        rows.sort(reverse=True)
        for a, c in rows[:12]: print(f"  {a:.4f}  {c}")
        for name, (cols, sc) in [("ALL baselines", (BASE, False)), ("thermo shape+scale", (SHAPE + SCALE, False)),
                                 ("everything", (BASE + SHAPE + SCALE + GAP, True))]:
            m, s = cv_auc(mat(allr, cols, sc), yy)
            print(f"  CV {m:.4f} +- {s:.4f}  {name}")
    pickle.dump(R, open(f"data/records_{tag}.pkl", "wb"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--tag", default="run")
    a = ap.parse_args(); main(a.files, a.tag)
