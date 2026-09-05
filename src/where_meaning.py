"""Position-resolved split of token entropy into phrasing and meaning.

Section 7f found that at the FIRST token of a free-form answer, 94.7% of the entropy is
phrasing and the model has already decided what it will say. So where does the content
actually get decided? This walks along the greedy answer and, at every position, forks the
top-K alternatives, continues each, and clusters the resulting complete answers by meaning.

At position t:
    token entropy    H_t  = entropy of the next-token distribution
    semantic entropy S_t  = entropy over meaning clusters of the K forked continuations
    meaning share        = S_t / H_t   (0 = pure phrasing, 1 = pure content)
"""
import json, sys, os, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from freeform_degeneracy import embed, cluster, EMB

QS = [
 "Why is the sky blue?", "What makes a good manager?",
 "Explain why bread rises when you bake it.", "How does a vaccine work?",
 "Why do leaves change colour in autumn?", "What causes traffic jams on motorways?",
 "Explain why the ocean is salty.", "Why do metals conduct electricity?",
 "What are the trade-offs of nuclear power?", "Explain how a refrigerator keeps food cold.",
 "Why is exercise good for mental health?", "What makes a story satisfying to read?",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("-K", type=int, default=8)
    ap.add_argument("--answer-len", type=int, default=40)
    ap.add_argument("--cont", type=int, default=28)
    ap.add_argument("--thr", type=float, default=0.80)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    etok = AutoTokenizer.from_pretrained(EMB); emb = AutoModel.from_pretrained(EMB).to(dev).eval()

    out = []
    for qi, q in enumerate(QS):
        msgs = [{"role": "system", "content": "You are a helpful assistant. Answer in two or three sentences."},
                {"role": "user", "content": q}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(dev)
        with torch.no_grad():
            g = lm.generate(ids, max_new_tokens=a.answer_len, do_sample=False, pad_token_id=tok.pad_token_id)
        ans = g[0, ids.shape[1]:]
        recs = []
        for t in range(len(ans)):
            pre = torch.cat([ids[0], ans[:t]])[None]
            with torch.no_grad():
                lg = lm(pre).logits[0, -1].float()
            p = torch.softmax(lg, -1)
            H = float(-(p[p > 0] * torch.log(p[p > 0])).sum())
            top = torch.topk(lg, a.K)
            clog = top.values.cpu().numpy().astype(np.float64)
            forks = torch.cat([pre.repeat(a.K, 1), top.indices[:, None]], 1)
            with torch.no_grad():
                gg = lm.generate(forks, max_new_tokens=a.cont, do_sample=False, pad_token_id=tok.pad_token_id)
            # cluster the COMPLETE answers (shared prefix + fork + continuation)
            texts = [tok.decode(x, skip_special_tokens=True).strip() for x in gg[:, ids.shape[1]:]]
            lab = cluster(embed(texts, etok, emb, dev), a.thr)
            pk = torch.softmax(torch.tensor(clog), 0).numpy()
            pc = np.array([pk[lab == c].sum() for c in range(lab.max()+1)])
            pc = pc / pc.sum()
            S = float(-(pc[pc > 0] * np.log(pc[pc > 0])).sum())
            Hk = float(-(pk * np.log(pk + 1e-300)).sum())     # entropy restricted to top-K
            recs.append(dict(pos=t, tok=tok.decode(ans[t]), H_full=H, H_topk=Hk, forks=texts, clog=clog.tolist(),
                             S_sem=S, n_clusters=int(lab.max()+1),
                             meaning_share=float(S / Hk) if Hk > 1e-6 else 0.0))
        out.append(dict(q=q, answer=tok.decode(ans, skip_special_tokens=True), tokens=recs))
        ms = np.array([r["meaning_share"] for r in recs])
        print(f"[{qi+1}/{len(QS)}] {q[:38]:40s} mean meaning share {ms.mean():.3f}  "
              f"max {ms.max():.3f} at pos {int(np.argmax(ms))} ({recs[int(np.argmax(ms))]['tok']!r})", flush=True)
        json.dump(out, open(a.out, "w"))

    allr = [r for o in out for r in o["tokens"]]
    ms = np.array([r["meaning_share"] for r in allr]); pos = np.array([r["pos"] for r in allr])
    Hk = np.array([r["H_topk"] for r in allr])
    print(f"\n=== position-resolved meaning share, n={len(allr)} token positions ===")
    print(f"  overall mean meaning share: {ms.mean():.4f}   median {np.median(ms):.4f}")
    print(f"  fraction of positions that are pure phrasing (share < 0.05): {np.mean(ms < 0.05)*100:.1f}%")
    print(f"  fraction that are mostly meaning (share > 0.5):             {np.mean(ms > 0.5)*100:.1f}%")
    print(f"\n  {'position bucket':18s} {'n':>5s} {'mean H(topK)':>13s} {'mean meaning share':>19s}")
    for lo, hi in [(0,1),(1,3),(3,6),(6,12),(12,20),(20,30),(30,60)]:
        m = (pos >= lo) & (pos < hi)
        if m.sum(): print(f"  {f'{lo}-{hi-1}':18s} {m.sum():5d} {Hk[m].mean():13.4f} {ms[m].mean():19.4f}")
    hi_ = sorted(allr, key=lambda r: -r["meaning_share"])[:15]
    print("\n  the 15 positions where the model actually decides content:")
    for r in hi_: print(f"    share {r['meaning_share']:.3f}  pos {r['pos']:3d}  H={r['H_topk']:.3f}  token {r['tok']!r}")
    lo_ = [r for r in sorted(allr, key=lambda r: -r["H_topk"])[:60] if r["meaning_share"] < 0.05][:12]
    print("\n  high-entropy positions that are PURE PHRASING (a detector would flag these wrongly):")
    for r in lo_: print(f"    share {r['meaning_share']:.3f}  H={r['H_topk']:.3f}  pos {r['pos']:3d}  token {r['tok']!r}")
    print("WMDONE")

if __name__ == "__main__":
    main()
