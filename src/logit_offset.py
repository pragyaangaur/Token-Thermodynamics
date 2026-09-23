"""Do published logit-magnitude measures survive a change of model family?

FINDINGS.md item 8. Softmax ignores a constant added to every logit, so any quantity
that describes the model's belief should ignore it too. Section 7g found that the
constant differs wildly between model families: GPT-2's logits sit near -118 and
Pythia-160M's near +826, for the same kind of prediction. Two published uncertainty
measures read raw logit values rather than probabilities:

    LogTokU (Ma et al., arXiv 2502.00290)
        evidence alpha_k = raw logit of the k-th most likely token, top K
        AU = -sum_k (alpha_k/alpha_0) (psi(alpha_k + 1) - psi(alpha_0 + 1))
        EU = K / sum_k (alpha_k + 1)
    Semantic Energy (arXiv 2508.14496)
        token energy = -raw logit of the chosen token

Neither paper compares values across models, so no published result can flip. This
script measures what would happen if someone did, and whether measuring the logits
relative to the mode, which is what the melting law uses, removes the problem.

The prompts and models are the seven of section 7g, one forward pass each, float32.
No sampling is needed, because the question is about the scores and not about
detection accuracy.
"""
import json, sys, os, argparse
import numpy as np
from scipy.special import digamma
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))

KS = (2, 10, 25)


def logtoku(alpha):
    """AU and EU from an evidence vector, exactly as in LogTokU equations 3 to 5."""
    a0 = alpha.sum()
    au = float(-np.sum(alpha / a0 * (digamma(alpha + 1) - digamma(a0 + 1))))
    eu = float(len(alpha) / np.sum(alpha + 1))
    return au, eu


def mode_of(lg, bins=2000):
    h, e = np.histogram(lg, bins=bins)
    i = int(np.argmax(h))
    return float(0.5 * (e[i] + e[i + 1]))


def scores(lg):
    """Every score for one logit vector, raw and relative to the mode."""
    lg = lg[np.isfinite(lg)]
    p = np.exp(lg - lg.max()); p /= p.sum()
    out = dict(entropy=float(-(p[p > 0] * np.log(p[p > 0])).sum()),
               top_logit=float(lg.max()), mode=mode_of(lg), mean=float(lg.mean()))
    out["energy_raw"] = -out["top_logit"]
    out["energy_mode"] = -(out["top_logit"] - out["mode"])
    top = np.sort(lg)[::-1]
    for K in KS:
        raw = top[:K]
        rel = top[:K] - out["mode"]
        out[f"EU_raw_K{K}"] = logtoku(raw)[1] if (raw + 1 > 0).all() else None
        out[f"AU_raw_K{K}"] = logtoku(raw)[0] if (raw > 0).all() else None
        out[f"EU_mode_K{K}"], out[f"AU_mode_K{K}"] = logtoku(rel)[1], logtoku(rel)[0]
    return out


def shift_test():
    """A constant shift changes nothing the model believes, so it should change no score."""
    rng = np.random.default_rng(0)
    lg = rng.gumbel(0, 2.0, 50000) + 5.0
    base, moved = scores(lg), scores(lg + 20.0)
    print("same distribution, every logit shifted by +20:")
    for k in ("entropy", "energy_raw", "energy_mode", "EU_raw_K10", "EU_mode_K10",
              "AU_raw_K10", "AU_mode_K10"):
        print(f"  {k:12s} {base[k]:10.4f} -> {moved[k]:10.4f}")
    return dict(base=base, shifted=moved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/logit_offset.json")
    a = ap.parse_args()
    shift = shift_test()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from multimodel_melt import PROMPTS, MODELS
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    per_model = {}
    for m in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(m)
            model = AutoModelForCausalLM.from_pretrained(
                m, dtype=torch.float32, local_files_only=True).to(dev).eval()
        except Exception as e:
            print(f"SKIP {m}: {type(e).__name__}")
            continue
        rows = []
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                lg = model(ids).logits[0, -1].float().cpu().numpy().astype(np.float64)
            rows.append(scores(lg))
        per_model[m] = rows
        del model
        if dev == "mps":
            torch.mps.empty_cache()

    names = list(per_model)
    med = lambda m, k: (np.median([r[k] for r in per_model[m]])
                        if all(r[k] is not None for r in per_model[m]) else None)
    keys = ["entropy", "mode", "energy_raw", "energy_mode"] + \
           [f"{u}_{v}_K{K}" for K in KS for u in ("EU", "AU") for v in ("raw", "mode")]
    table = {m: {k: med(m, k) for k in keys} for m in names}
    print(f"\n{'model':30s}" + "".join(f"{k:>13s}" for k in keys[:8]))
    for m in names:
        print(f"{m:30s}" + "".join(
            f"{table[m][k]:13.3f}" if table[m][k] is not None else f"{'undefined':>13s}"
            for k in keys[:8]))

    # A cross-model reading should rank models the way their distributions do, so
    # compare each score's model ranking with the ranking by entropy, and with the
    # ranking by where the logits happen to sit (the mode), which carries no meaning.
    print("\nSpearman correlation of the per-model median, across models")
    summary = {}
    for k in keys[2:]:
        ok = [m for m in names if table[m][k] is not None]
        if len(ok) < 4:
            summary[k] = dict(n=len(ok))
            print(f"  {k:14s} defined on only {len(ok)} of {len(names)} models")
            continue
        x = [table[m][k] for m in ok]
        r_ent = spearmanr(x, [table[m]["entropy"] for m in ok])[0]
        r_mode = spearmanr(x, [table[m]["mode"] for m in ok])[0]
        summary[k] = dict(n=len(ok), rho_entropy=float(r_ent), rho_offset=float(r_mode))
        print(f"  {k:14s} n={len(ok)}  with entropy {r_ent:+.3f}   with logit offset "
              f"{r_mode:+.3f}")
    json.dump(dict(shift_test=shift, per_model=per_model, medians=table,
                   spearman=summary), open(a.out, "w"))
    print("OFFSETDONE")


if __name__ == "__main__":
    main()
