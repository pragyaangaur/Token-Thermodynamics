import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from thermo import find_peaks

def feats_from_curve(T, C, prefix="t_"):
    """Interpretable features of a heat-capacity spectrum C(T).

    T ascending. Two families:
      SCALE features  (change if you rescale all logits)
      SHAPE features  (invariant to a global rescaling of the logits, because a
                       rescale logit -> logit/s maps T -> T*s, so any RATIO of
                       characteristic temperatures is unchanged)
    """
    T = np.asarray(T, float); C = np.asarray(C, float)
    o = np.argsort(T); T, C = T[o], C[o]
    f = {}
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    Cmax = float(C.max()); imax = int(np.argmax(C))
    if Cmax <= 0:
        z = {prefix + k: 0.0 for k in ("Cmax","Tmain","logTmain","npeaks","has_low","gap_ratio",
             "band_weight","valley","logwidth","spec_mean","spec_var","spec_skew","cold_frac",
             "cold10","cold25","cold50")}
        z["_shape"] = [0.0] * 24
        return z
    Tmain = float(T[imax])
    f[prefix + "Cmax"] = Cmax
    f[prefix + "Tmain"] = Tmain
    f[prefix + "logTmain"] = float(np.log(Tmain))

    pk = find_peaks(T, C)
    f[prefix + "npeaks"] = float(len(pk))
    low = [p for p in pk if p["T"] < Tmain * 0.85]
    if low:
        p0 = min(low, key=lambda d: d["T"])
        i0 = p0["i"]
        lo, hi = min(i0, imax), max(i0, imax)
        vmin = float(C[lo:hi + 1].min())
        f[prefix + "has_low"] = 1.0
        f[prefix + "gap_ratio"] = float(np.log(Tmain / p0["T"]))          # SHAPE
        f[prefix + "band_weight"] = float(np.log(p0["C"] / Cmax))          # SHAPE
        f[prefix + "valley"] = float(np.log(p0["C"] / (vmin + 1e-300)))    # SHAPE
    else:
        f[prefix + "has_low"] = 0.0
        f[prefix + "gap_ratio"] = 0.0
        f[prefix + "band_weight"] = 0.0
        f[prefix + "valley"] = 0.0

    # width of the main peak in log-T at half maximum  -> SHAPE
    half = Cmax / 2.0
    l = imax
    while l > 0 and C[l] > half: l -= 1
    r = imax
    while r < len(C) - 1 and C[r] > half: r += 1
    f[prefix + "logwidth"] = float(np.log(T[r]) - np.log(T[l]))

    # normalised spectral moments in log-T, weighted by C  -> SHAPE
    x = np.log(T); w = C / (C.sum() + 1e-300)
    m1 = float((w * x).sum())
    m2 = float((w * (x - m1) ** 2).sum())
    m3 = float((w * (x - m1) ** 3).sum()) / (m2 ** 1.5 + 1e-12)
    f[prefix + "spec_mean"] = m1                    # scale
    f[prefix + "spec_var"] = m2                     # SHAPE
    f[prefix + "spec_skew"] = m3                    # SHAPE

    # cold spectral weight: fraction of total entropy that is released below the
    # main melting peak. S(T) = integral C/T dT, so this is a genuine entropy split.
    dS = C / T
    tot = float(np.trapezoid(dS, T)) + 1e-300
    cold = float(np.trapezoid(dS[:imax + 1], T[:imax + 1]))
    f[prefix + "cold_frac"] = cold / tot            # SHAPE
    # same split at a fixed fraction of the main peak temperature
    for q in (0.1, 0.25, 0.5):
        m = T <= Tmain * q
        f[prefix + f"cold{int(q*100)}"] = (float(np.trapezoid(dS[m], T[m])) / tot) if m.sum() > 2 else 0.0

    # the normalised curve sampled on a rescaled axis u = T / Tmain  -> pure SHAPE
    u = np.log(T / Tmain)
    grid = np.linspace(-4.0, 2.0, 24)
    f["_shape"] = np.interp(grid, u, C / Cmax).tolist()
    return f
