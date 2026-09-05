"""Does the 96.7% result break down on free-form answers?

FINDINGS.md predicted it should: for short factual answers the token distribution is
already the meaning distribution, but for longer free-form answers one meaning can be
phrased many ways, so paraphrase degeneracy should reappear. That is my own falsifiable
prediction and this tests it.

Same protocol as the factual measurement: take the top-M candidate first tokens, force
each, continue greedily. The difference is that answers are now sentences, so clustering
is by sentence-embedding cosine similarity rather than string equality. The factual set
is re-run through the SAME embedding clustering so the comparison is like for like.
"""
import json, sys, os, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

EMB = "sentence-transformers/all-MiniLM-L6-v2"

OPEN_Q = [
 "Why is the sky blue?", "What makes a good manager?",
 "Explain why bread rises when you bake it.", "What are the downsides of living in a big city?",
 "How should someone prepare for a job interview?", "Why do people enjoy horror films?",
 "Describe what happens during a thunderstorm.", "What is the hardest part of learning a language?",
 "Why do some companies fail after growing quickly?", "Explain how a refrigerator keeps food cold.",
 "What makes a piece of music memorable?", "Why is exercise good for mental health?",
 "How do you decide whether a news source is reliable?", "What causes traffic jams on motorways?",
 "Explain why the ocean is salty.", "What are the advantages of working from home?",
 "Why do leaves change colour in autumn?", "How would you explain gravity to a child?",
 "What makes a story satisfying to read?", "Why do metals conduct electricity?",
 "Describe the process of making cheese.", "What should you consider before adopting a dog?",
 "Explain why airplanes can fly.", "What makes a city feel safe to walk in?",
 "Why do we dream?", "How does a vaccine work?",
 "What are the trade-offs of nuclear power?", "Why is it hard to predict the weather?",
 "Explain what inflation means for ordinary people.", "What makes a good teacher?",
]

FACT_Q = [
 "What is the capital city of France?", "Who wrote the novel Lolita?",
 "What is the chemical symbol for gold?", "Who directed the film Jaws?",
 "What is the capital city of Peru?", "Who wrote the book Ubik?",
 "What is the chemical symbol for tin?", "Who directed the film Alien?",
 "What is the capital city of Kenya?", "Who wrote the novel Middlemarch?",
 "What is the chemical symbol for potassium?", "Who directed the film Rashomon?",
 "What is the capital city of Uruguay?", "Who wrote the play Medea?",
 "What is the chemical symbol for tungsten?", "Who directed the film Solaris?",
 "What is the capital city of Nepal?", "Who wrote the novel Beloved?",
 "What is the chemical symbol for antimony?", "Who directed the film Persona?",
 "What is the capital city of Ghana?", "Who wrote the novel Kim?",
 "What is the chemical symbol for mercury?", "Who directed the film Stalker?",
 "What is the capital city of Bolivia?", "Who wrote the novel Cranford?",
 "What is the chemical symbol for zinc?", "Who directed the film Ran?",
 "What is the capital city of Latvia?", "Who wrote the novel Ficciones?",
]

def embed(texts, tok, mdl, dev):
    out = []
    for i in range(0, len(texts), 32):
        b = tok(texts[i:i+32], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = mdl(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        v = (o * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.vstack(out)

def cluster(vecs, thr):
    """single-linkage on cosine similarity"""
    n = len(vecs); lab = list(range(n))
    S = vecs @ vecs.T
    def find(x):
        while lab[x] != x: lab[x] = lab[lab[x]]; x = lab[x]
        return x
    for i in range(n):
        for j in range(i+1, n):
            if S[i, j] >= thr:
                a, b = find(i), find(j)
                if a != b: lab[a] = b
    roots = {}
    out = []
    for i in range(n):
        r = find(i)
        if r not in roots: roots[r] = len(roots)
        out.append(roots[r])
    return np.array(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True); ap.add_argument("-M", type=int, default=12)
    ap.add_argument("--newtok", type=int, default=48); ap.add_argument("--thr", type=float, default=0.80)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    etok = AutoTokenizer.from_pretrained(EMB); emb = AutoModel.from_pretrained(EMB).to(dev).eval()

    results = {}
    for cond, qs, instr, nt in [("free-form", OPEN_Q, "Answer in two or three sentences.", a.newtok),
                                ("short-factual", FACT_Q, "Answer with only the name.", 8)]:
        recs = []
        for qi, q in enumerate(qs):
            msgs = [{"role": "system", "content": "You are a helpful assistant. " + instr},
                    {"role": "user", "content": q}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            ids = enc["input_ids"].to(dev)
            with torch.no_grad():
                lg = lm(ids).logits[0, -1].float()
            top = torch.topk(lg, a.M)
            clog = top.values.cpu().numpy().astype(np.float64)
            pref = torch.cat([ids.repeat(a.M, 1), top.indices[:, None]], 1)
            with torch.no_grad():
                g = lm.generate(pref, max_new_tokens=nt, do_sample=False, pad_token_id=tok.pad_token_id)
            texts = [tok.decode(x, skip_special_tokens=True).strip() for x in g[:, ids.shape[1]:]]
            lab = cluster(embed(texts, etok, emb, dev), a.thr)
            sizes = np.bincount(lab)
            withins = [clog[lab == c].max() - clog[lab == c].min() for c in range(lab.max()+1) if (lab == c).sum() > 1]
            best = np.array([clog[lab == c].max() for c in range(lab.max()+1)])
            p = np.exp(clog - clog.max()); p /= p.sum()
            pc = np.array([p[lab == c].sum() for c in range(lab.max()+1)])
            recs.append(dict(q=q, n_clusters=int(lab.max()+1),
                             singleton_frac=float(np.mean(sizes[lab] == 1)),
                             delta=float(np.mean(withins)) if withins else 0.0,
                             spread=float(best.max()-best.min()) if len(best) > 1 else 0.0,
                             sem_entropy=float(-(pc[pc>0]*np.log(pc[pc>0])).sum()),
                             tok_entropy=float(-(p*np.log(p)).sum()),
                             sizes=sizes.tolist(), sample=texts[0][:90]))
            if (qi+1) % 10 == 0: print(f"  {cond} {qi+1}/{len(qs)}", flush=True)
        results[cond] = recs
        S = np.array([r["singleton_frac"] for r in recs]); nc = np.array([r["n_clusters"] for r in recs])
        d = np.array([r["delta"] for r in recs]); sp = np.array([r["spread"] for r in recs])
        se = np.array([r["sem_entropy"] for r in recs]); te = np.array([r["tok_entropy"] for r in recs])
        m = (sp > 1e-6) & (d > 1e-9)
        print(f"\n=== {cond} (n={len(recs)}, M={a.M}, cluster threshold {a.thr}) ===")
        print(f"  distinct meanings among top {a.M}      : mean {nc.mean():.2f} / {a.M}")
        print(f"  candidates alone in their cluster      : {S.mean()*100:.1f}%")
        print(f"  items with ANY degeneracy              : {np.mean(nc < a.M)*100:.1f}%")
        print(f"  median within-meaning logit spread     : {np.median(d):.3f}")
        print(f"  median between-meaning logit spread    : {np.median(sp):.3f}")
        if m.sum(): print(f"  median delta/spread (where defined)    : {np.median(d[m]/sp[m]):.4f}")
        print(f"  semantic entropy vs token entropy      : {se.mean():.4f} vs {te.mean():.4f}  "
              f"(gap {te.mean()-se.mean():.4f} nats, {100*(te.mean()-se.mean())/te.mean():.1f}%)")
    json.dump(results, open(a.out, "w"))
    A = results["free-form"]; B_ = results["short-factual"]
    sa = np.mean([r["singleton_frac"] for r in A]); sb = np.mean([r["singleton_frac"] for r in B_])
    print(f"\nPREDICTION TEST: free-form singleton fraction {sa*100:.1f}% vs short-factual {sb*100:.1f}%")
    print("  prediction was that free-form shows MORE paraphrase degeneracy (lower singleton fraction)")
    print(f"  VERDICT: {'CONFIRMED' if sa < sb - 0.05 else ('FALSIFIED' if sa > sb - 0.01 else 'INCONCLUSIVE')}")
    print("FFDONE")

if __name__ == "__main__":
    main()
