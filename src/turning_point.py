"""Where the entropy turning point sits relative to the melting temperature.

FINDINGS.md item 9. Du, Yang and Welleck (arXiv 2502.05234) pick a sampling
temperature at the turning point of the entropy curve, located by sweeping
temperatures and generating at each one. The melting temperature of this
repository is the peak of the heat capacity and comes from one forward pass.
This file asks how the two points are related.

There is an exact answer, and it does not depend on which convention the
turning point is written in. Using S'(T) = C(T)/T, the three natural
inflection conditions all reduce to the same shape:

    S vs T           S'' = 0          ->  T C'/C = 1
    log S vs log T   d2/dlogT2 = 0    ->  T C'/C = C/S
    log S vs T       d2/dT2 = 0       ->  T C'/C = 1 + C/S

C and S are both strictly positive for any non-degenerate distribution, so the
right-hand side is strictly positive in all three cases, which forces C'(T) > 0.
The heat capacity is still rising there. The melting temperature is where
C'(T) = 0. So every one of these turning points lies on a rising flank of
C, with no assumption about the logit spectrum, and below the melting
temperature whenever C has a single peak. turning_points() only searches below
the peak, so its selftest cannot detect a violation. src/validate.py has the
check that can.

That is the ordering. The size of the gap is an empirical question, and this
script measures it on the seven cached pretrained models.
"""
import json, sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from dos import make_dos, curves_dos

GRID = np.exp(np.linspace(np.log(1e-3), np.log(3e2), 4000))   # betas, descending T


def _rhs(name, C, S):
    if name == "S_vs_T":
        return np.ones_like(C)
    if name == "logS_vs_logT":
        return C / np.maximum(S, 1e-300)
    if name == "logS_vs_T":
        return 1.0 + C / np.maximum(S, 1e-300)
    raise ValueError(name)


def turning_points(logits, K=1024, NB=512):
    """T_melt and the three turning points for one logit vector."""
    E, W = make_dos(np.asarray(logits, np.float64), K=K, NB=NB)
    c = curves_dos(E, W, GRID)
    o = np.argsort(c["T"])
    T, C, S = c["T"][o], c["C"][o], c["S"][o]
    i = int(np.argmax(C))
    Tmelt = float(T[i])
    dCdT = np.gradient(C, T)
    out = dict(Tmelt=Tmelt, Cmax=float(C[i]), S_at_melt=float(S[i]))
    for name in ("S_vs_T", "logS_vs_logT", "logS_vs_T"):
        g = T * dCdT / np.maximum(C, 1e-300) - _rhs(name, C, S)
        # the turning point is the last sign change strictly below the peak
        seg = g[: i + 1]
        sgn = np.sign(seg)
        k = np.where(sgn[:-1] * sgn[1:] < 0)[0]
        if k.size == 0:
            out[name] = None
            continue
        j = int(k[-1])
        w = seg[j] / (seg[j] - seg[j + 1])          # linear interpolation in log T
        lt = np.log(T[j]) + w * (np.log(T[j + 1]) - np.log(T[j]))
        out[name] = float(np.exp(lt))
    return out


def selftest():
    """The ordering theorem, on spectra with no relation to any language model."""
    rng = np.random.default_rng(0)
    bad = 0
    for trial in range(200):
        V = int(rng.integers(50, 4000))
        kind = trial % 4
        if kind == 0:
            lg = rng.normal(0, rng.uniform(0.3, 6.0), V)
        elif kind == 1:
            lg = rng.gumbel(0, rng.uniform(0.5, 4.0), V)
        elif kind == 2:
            lg = np.concatenate([[rng.uniform(4, 20)], np.zeros(V - 1)])   # ideal melt
        else:
            lg = -np.sort(rng.exponential(rng.uniform(0.5, 5.0), V))
        r = turning_points(lg, K=min(1024, V), NB=256)
        for name in ("S_vs_T", "logS_vs_logT", "logS_vs_T"):
            t = r[name]
            if t is not None and not t < r["Tmelt"] * (1 + 1e-9):
                bad += 1
                print(f"  VIOLATION {name} trial={trial} T={t:.6g} Tmelt={r['Tmelt']:.6g}")
    print(f"selftest: 200 synthetic spectra x 3 conventions, {bad} violations of "
          f"T_turn < T_melt")
    assert bad == 0, "ordering theorem violated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/turning_point.json")
    ap.add_argument("--selftest-only", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest_only:
        return
    # model imports live here so validate.py can use turning_points without torch
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from multimodel_melt import PROMPTS, MODELS
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out = []
    for m in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(m)
            model = AutoModelForCausalLM.from_pretrained(
                m, dtype=torch.float32, local_files_only=True).to(dev).eval()
        except Exception as e:
            print(f"SKIP {m}: {type(e).__name__} {str(e)[:90]}", flush=True)
            continue
        V = model.get_output_embeddings().weight.shape[0]
        rows = []
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                lg = model(ids).logits[0, -1].float().cpu().numpy().astype(np.float64)
            lg = lg[np.isfinite(lg)]
            r = turning_points(lg)
            r["prompt"] = p
            rows.append(r)
        rec = dict(model=m, V=int(V), rows=rows)
        line = f"{m:34s} V={V:7d} T_melt={np.median([r['Tmelt'] for r in rows]):6.3f}"
        for name in ("S_vs_T", "logS_vs_logT", "logS_vs_T"):
            v = [r[name] for r in rows if r[name] is not None]
            ratio = [r['Tmelt'] / r[name] for r in rows if r[name] is not None]
            rec[name + "_med"] = float(np.median(v)) if v else None
            rec[name + "_ratio_med"] = float(np.median(ratio)) if ratio else None
            line += f"  {name}={np.median(v):6.3f} (x{np.median(ratio):5.2f})" if v else f"  {name}=none"
        out.append(rec)
        print(line, flush=True)
        del model
        if dev == "mps":
            torch.mps.empty_cache()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print()
    for name in ("S_vs_T", "logS_vs_logT", "logS_vs_T"):
        allr = [r["Tmelt"] / r[name] for o in out for r in o["rows"] if r[name] is not None]
        n_viol = sum(1 for x in allr if x < 1.0)
        allr = np.array(allr)
        print(f"{name:14s} T_melt/T_turn over {len(allr)} distributions: "
              f"median {np.median(allr):.3f}  IQR {np.percentile(allr,25):.3f}-"
              f"{np.percentile(allr,75):.3f}  range {allr.min():.3f}-{allr.max():.3f}  "
              f"violations {n_viol}")
    print("TPDONE")


if __name__ == "__main__":
    main()
