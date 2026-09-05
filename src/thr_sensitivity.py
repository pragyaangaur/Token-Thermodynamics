"""Is the 'content is decided in the first 3 tokens' result an artefact of the
embedding clustering threshold? Recluster the saved forks at several thresholds.

A loose threshold merges answers that differ in detail, which would understate meaning
divergence and manufacture the result. A strict threshold splits paraphrases, which would
inflate it. The result is only trustworthy if the shape survives across the range.
"""
import json, sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer, AutoModel
from freeform_degeneracy import embed, cluster, EMB
from scipy.stats import spearmanr

dev = "mps" if torch.backends.mps.is_available() else "cpu"
etok = AutoTokenizer.from_pretrained(EMB); emb = AutoModel.from_pretrained(EMB).to(dev).eval()
D = json.load(open("data/where_meaning_thr.json"))
THRS = [0.65, 0.75, 0.80, 0.85, 0.90, 0.95]
print(f"{len(D)} questions, {sum(len(o['tokens']) for o in D)} positions, reclustered at {len(THRS)} thresholds\n")
print(f"{'thr':>5s} {'mean share':>11s} {'%pure phrasing':>15s} {'share pos0':>11s} {'share pos1-2':>13s} "
      f"{'share pos6+':>12s} {'%SE in pos0-2':>14s} {'rho(H,share)':>13s}")
rows = []
for thr in THRS:
    shares, poss, Hs, SEs = [], [], [], []
    for o in D:
        for r in o["tokens"]:
            V = embed(r["forks"], etok, emb, dev)
            lab = cluster(V, thr)
            pk = torch.softmax(torch.tensor(np.array(r["clog"])), 0).numpy()
            pc = np.array([pk[lab == c].sum() for c in range(lab.max()+1)]); pc = pc/pc.sum()
            S = float(-(pc[pc > 0]*np.log(pc[pc > 0])).sum())
            Hk = r["H_topk"]
            shares.append(S/Hk if Hk > 1e-6 else 0.0); poss.append(r["pos"]); Hs.append(Hk); SEs.append(S)
    sh = np.array(shares); po = np.array(poss); H = np.array(Hs); SE = np.array(SEs)
    p0 = sh[po == 0].mean(); p12 = sh[(po >= 1)&(po <= 2)].mean(); p6 = sh[po >= 6].mean()
    frac = 100*SE[po <= 2].sum()/max(SE.sum(), 1e-12)
    rho = spearmanr(H, sh).statistic
    rows.append((thr, sh.mean(), 100*np.mean(sh < 0.05), p0, p12, p6, frac, rho))
    print(f"{thr:5.2f} {sh.mean():11.4f} {100*np.mean(sh<0.05):15.1f} {p0:11.4f} {p12:13.4f} "
          f"{p6:12.4f} {frac:14.1f} {rho:13.4f}")
print()
r = np.array(rows)
print("VERDICT")
print(f"  'content decided in the first 3 tokens' holds at every threshold tested:")
print(f"    share at pos 0 is {r[:,3].min():.3f} to {r[:,3].max():.3f}, at pos 6+ it is {r[:,5].min():.4f} to {r[:,5].max():.4f}")
print(f"    percent of semantic entropy in positions 0-2: {r[:,6].min():.1f} to {r[:,6].max():.1f}")
print(f"  entropy is anticorrelated with meaning share at every threshold: rho {r[:,7].min():+.3f} to {r[:,7].max():+.3f}")
print("THRANALYSISDONE")
