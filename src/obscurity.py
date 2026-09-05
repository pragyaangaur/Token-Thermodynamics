"""Control for the fabricated-entity confound.

Made-up names have unusual spelling, so a detector might be reading orthography
rather than 'the model has nothing to say about this'. Cleaner test: use only
REAL entities and split them by obscurity, measured by Wikipedia sitelink count.
Obscure real entities are things the model plausibly does not know, with no
change in the surface distribution of names.
"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from analyze import BASE, SHAPE, SCALE, mat, cv_auc
from sklearn.metrics import roc_auc_score
NORM = ["n_S_at_melt","n_C_at_melt"]+[f"n_S_{q}melt" for q in (25,50,75)]+[f"n_C_{q}melt" for q in (25,50,75)]

R = [r for r in pickle.load(open("data/records_both15.pkl","rb")) if not r.get("fake")]
L = np.array([r["links"] for r in R])
print("sitelink counts: p10 %d  median %d  p90 %d" % (np.percentile(L,10), np.median(L), np.percentile(L,90)))
lo, hi = np.percentile(L, 25), np.percentile(L, 75)
obs = [r for r in R if r["links"] <= lo]; fam = [r for r in R if r["links"] >= hi]
print(f"obscure quartile n={len(obs)} acc={np.mean([r['correct'] for r in obs]):.3f}   "
      f"famous quartile n={len(fam)} acc={np.mean([r['correct'] for r in fam]):.3f}")
allr = fam + obs
y = np.array([0]*len(fam) + [1]*len(obs))
print("\ntask: is this entity obscure (model likely has nothing to say)? real names only")
for name, cols, sc in [("all 7 baselines", BASE, False), ("entropy alone", ["b_entropy"], False),
                       ("melt-normalised S,C", NORM, False),
                       ("thermo shape+scale", SHAPE+SCALE, False),
                       ("everything", BASE+SHAPE+SCALE+NORM, True)]:
    m, s = cv_auc(mat(allr, cols, sc), y)
    print(f"  {m:.4f} +- {s:.4f}   {name}")
print("\nsingle features:")
for c in ["b_entropy","b_pmax","t_logTmain","t_Cmax","n_S_50melt","t_cold_frac"]:
    v = np.nan_to_num(np.array([r.get(c,0.0) for r in allr], float))
    a = roc_auc_score(y, v); print(f"  {max(a,1-a):.4f}  {c}")

print("\nand the same comparison on the fabricated set, for reference:")
F = [r for r in pickle.load(open("data/records_both15.pkl","rb")) if r.get("fake")]
allr2 = R + F; y2 = np.array([0]*len(R)+[1]*len(F))
for name, cols, sc in [("all 7 baselines", BASE, False), ("thermo shape+scale", SHAPE+SCALE, False),
                       ("everything", BASE+SHAPE+SCALE+NORM, True)]:
    m, s = cv_auc(mat(allr2, cols, sc), y2)
    print(f"  {m:.4f} +- {s:.4f}   {name}")
