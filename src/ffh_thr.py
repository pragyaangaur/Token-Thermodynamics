"""Clustering-threshold sensitivity for section 7k.

FINDINGS.md item 6, and the second bullet of the README's limitations. Section
7i swept the embedding-similarity threshold and one headline number moved. The
detection results of 7k were left at the single threshold 0.80. This reruns the
whole 7k table, including every AUROC, at six thresholds, from the raw strings
saved by `src/ffh_texts.py`. No language model is loaded, only the embedder.
"""
import json, sys, os, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer, AutoModel
from freeform_degeneracy import embed, cluster, EMB
from sklearn.metrics import roc_auc_score

THRS = [0.65, 0.75, 0.80, 0.85, 0.90, 0.95]
CONDS = ["KNOWN", "OBSCURE", "FABRICATED"]
PAIRS = [("KNOWN", "FABRICATED"), ("KNOWN", "OBSCURE"), ("OBSCURE", "FABRICATED")]


def entropy(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default="data/ffh_texts_1p5b.json")
    ap.add_argument("--out", default="data/ffh_thr.json")
    a = ap.parse_args()
    D = json.load(open(a.texts))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    etok = AutoTokenizer.from_pretrained(EMB, local_files_only=True)
    emb = AutoModel.from_pretrained(EMB, local_files_only=True).to(dev).eval()

    # embed once; only the clustering threshold changes
    cache = {}
    for cond in CONDS:
        for i, r in enumerate(D[cond]):
            cache[(cond, i, "samples")] = embed(r["samples"], etok, emb, dev)
            for pz in r["positions"]:
                cache[(cond, i, "fork", pz["pos"])] = embed(pz["forks"], etok, emb, dev)
    print(f"embedded {len(cache)} text groups\n")

    allrows = {}
    for thr in THRS:
        per = {c: [] for c in CONDS}
        for cond in CONDS:
            for i, r in enumerate(D[cond]):
                lab = cluster(cache[(cond, i, "samples")], thr)
                cnt = np.bincount(lab)
                p = cnt / cnt.sum()
                rec = dict(sem_entropy=entropy(p), n_meanings=int(cnt.size),
                           top_mass=float(p.max()))
                fse, th = [], []
                for pz in r["positions"]:
                    l2 = cluster(cache[(cond, i, "fork", pz["pos"])], thr)
                    pk = torch.softmax(torch.tensor(pz["top_logits"]), 0).numpy()
                    pc = np.array([pk[l2 == c].sum() for c in range(l2.max() + 1)])
                    pc = pc / pc.sum()
                    fse.append(entropy(pc))
                    th.append(pz["tok_H"])
                rec["fork_SE"] = fse
                rec["tok_H"] = th
                for n in (1, 2, 3):
                    rec[f"fork{n}"] = float(np.sum(fse[:n]))
                    rec[f"tokH{n}"] = float(np.sum(th[:n]))
                per[cond].append(rec)
        allrows[thr] = per

        print(f"--- threshold {thr:.2f} ---")
        print(f"{'':44s}" + "".join(f"{c:>12s}" for c in CONDS))
        for key, lab in [("sem_entropy", "sampled-answer semantic entropy"),
                         ("n_meanings", "distinct meanings among 16 samples"),
                         ("top_mass", "probability mass on the top meaning"),
                         ("fork3", "semantic entropy of first 3 forked tokens"),
                         ("tokH3", "plain token entropy, first 3 tokens")]:
            print(f"  {lab:42s}" + "".join(
                f"{np.mean([r[key] for r in per[c]]):12.3f}" for c in CONDS))
        print(f"  {'AUROC':42s}" + "".join(f"{A[0]}v{B[0]:>9s}" for A, B in PAIRS))
        for key, lab in [("sem_entropy", "sampled semantic entropy (16 samples)"),
                         ("fork1", "fork position 0 only"),
                         ("fork2", "fork positions 0 to 1"),
                         ("fork3", "fork positions 0 to 2"),
                         ("tokH1", "plain token entropy, pos 0"),
                         ("tokH3", "plain token entropy, pos 0-2")]:
            line = f"  {lab:42s}"
            for A, B in PAIRS:
                y = np.array([0] * len(per[A]) + [1] * len(per[B]))
                v = np.array([r[key] for r in per[A]] + [r[key] for r in per[B]])
                au = roc_auc_score(y, v)
                line += f"{max(au, 1 - au):12.3f}"
            print(line)
        print()

    json.dump({str(k): v for k, v in allrows.items()}, open(a.out, "w"))

    print("=== stability of each AUROC across the six thresholds ===")
    print(f"{'method':42s}{'pair':22s}{'min':>8s}{'median':>8s}{'max':>8s}{'range':>8s}")
    for key, lab in [("sem_entropy", "sampled semantic entropy"),
                     ("fork1", "fork position 0 only"),
                     ("fork2", "fork positions 0 to 1"),
                     ("fork3", "fork positions 0 to 2"),
                     ("tokH1", "plain token entropy, pos 0"),
                     ("tokH3", "plain token entropy, pos 0-2")]:
        for A, B in PAIRS:
            vals = []
            for thr in THRS:
                per = allrows[thr]
                y = np.array([0] * len(per[A]) + [1] * len(per[B]))
                v = np.array([r[key] for r in per[A]] + [r[key] for r in per[B]])
                au = roc_auc_score(y, v)
                vals.append(max(au, 1 - au))
            vals = np.array(vals)
            print(f"{lab:42s}{A+' vs '+B:22s}{vals.min():8.3f}{np.median(vals):8.3f}"
                  f"{vals.max():8.3f}{vals.max()-vals.min():8.3f}")
    print("FFHTHRDONE")


if __name__ == "__main__":
    main()
