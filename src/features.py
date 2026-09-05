"""Scalar features extracted from one logit vector.

Baselines (standard in the uncertainty literature) vs the thermodynamic-spectrum
features proposed here.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from thermo import curves, beta_grid, find_peaks

BETAS = beta_grid(300, 1e-2, 3e2)     # T from 100 down to 0.0033


def baseline_feats(logits):
    lg = np.asarray(logits, dtype=np.float64)
    m = lg.max()
    p = np.exp(lg - m); p /= p.sum()
    sp = np.sort(p)[::-1]
    sl = np.sort(lg)[::-1]
    S = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {
        "b_entropy": S,
        "b_logprob_top1": float(np.log(sp[0])),
        "b_pmax": float(sp[0]),
        "b_margin_logit": float(sl[0] - sl[1]),
        "b_margin_prob": float(sp[0] - sp[1]),
        "b_top5mass": float(sp[:5].sum()),
        # varentropy: Var of surprisal, the entropix feature
        "b_varentropy": float(np.sum(p * (np.log(p + 1e-300) + S) ** 2)),
    }


def thermo_feats(logits, betas=BETAS):
    c = curves(logits, betas)
    T, C = c["T"], c["C"]
    o = np.argsort(T); T, C = T[o], C[o]
    pk = find_peaks(T, C)
    Cmax = float(C.max()) if len(C) else 0.0
    f = {"t_Cmax": Cmax, "t_npeaks": float(len(pk))}

    # global (melting) peak = the largest; it is set by the vocabulary bulk
    if pk:
        main = max(pk, key=lambda d: d["C"])
        f["t_T_main"] = main["T"]
        f["t_C_main"] = main["C"]
    else:
        i = int(np.argmax(C)); f["t_T_main"] = float(T[i]); f["t_C_main"] = float(C[i])

    # low-temperature (semantic) peak = lowest-T peak that is not the main one
    lowpk = [p for p in pk if p["T"] < f["t_T_main"] * 0.9]
    if lowpk:
        low = min(lowpk, key=lambda d: d["T"])
        f["t_T_low"] = low["T"]
        f["t_C_low"] = low["C"]
        f["t_has_low"] = 1.0
        # SPECTRAL GAP RATIO: invariant under global rescaling of the logits
        f["t_gap_ratio"] = f["t_T_main"] / low["T"]
        f["t_log_gap_ratio"] = float(np.log(f["t_gap_ratio"]))
        # valley between the two peaks: how cleanly separated the answer band is
        i0, i1 = low["i"], int(np.argmin(np.abs(T - f["t_T_main"])))
        lo, hi = min(i0, i1), max(i0, i1)
        vmin = float(C[lo:hi + 1].min()) if hi > lo else low["C"]
        f["t_valley_depth"] = float(low["C"] / (vmin + 1e-12))
        f["t_log_valley"] = float(np.log(f["t_valley_depth"] + 1e-12))
        f["t_band_weight"] = float(low["C"] / (Cmax + 1e-12))
    else:
        f.update(t_T_low=f["t_T_main"], t_C_low=f["t_C_main"], t_has_low=0.0,
                 t_gap_ratio=1.0, t_log_gap_ratio=0.0,
                 t_valley_depth=1.0, t_log_valley=0.0, t_band_weight=1.0)

    # low-temperature spectral mass: integral of C/T dT below T=1 is exactly S(T=1)
    # restricted to the answer band, so this is 'entropy that lives in the cold region'
    mlo = T <= 1.0
    if mlo.sum() > 2:
        f["t_S_cold"] = float(np.trapezoid((C[mlo] / T[mlo]), T[mlo]))
    else:
        f["t_S_cold"] = 0.0
    mhi = (T > 1.0) & (T <= 30.0)
    f["t_S_hot"] = float(np.trapezoid((C[mhi] / T[mhi]), T[mhi])) if mhi.sum() > 2 else 0.0
    f["t_cold_frac"] = float(f["t_S_cold"] / (f["t_S_cold"] + f["t_S_hot"] + 1e-12))
    return f


def all_feats(logits):
    d = baseline_feats(logits); d.update(thermo_feats(logits)); return d
