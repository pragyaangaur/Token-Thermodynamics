"""Reference semantic entropy, so the single-pass thermodynamic split can be
scored against the method it is meant to approximate.

Discrete semantic entropy: sample K answers at T=1, cluster by normalised string
equality (an adequate entailment proxy for short factual answers), take the
entropy over clusters. This is the 'configurational' entropy of the answer
distribution: it counts distinct meanings and ignores paraphrase.
"""
import json, os, re, sys, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer
from probe import TEMPLATES, norm, grade

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--src", default="data/main_1p5b.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("-K", type=int, default=10)
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--max-new", type=int, default=10)
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()

    src = json.load(open(a.src))["results"]
    rng = np.random.default_rng(1); idx = rng.permutation(len(src))[:a.limit]
    items = [src[i] for i in idx]
    print("semantic entropy on", len(items), "items, K =", a.K, flush=True)

    out, t0 = [], time.time()
    for n, it in enumerate(items):
        q, instr = TEMPLATES[it["rel"]]
        msgs = [{"role": "system", "content": "You are a precise factual assistant. " + instr},
                {"role": "user", "content": q.format(s=it["subject"])}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(dev)
        ids = ids.repeat(a.K, 1)
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=a.max_new, do_sample=True, temperature=1.0,
                               top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id)
        texts = [tok.decode(x, skip_special_tokens=True).strip() for x in g[:, ids.shape[1]:]]
        keys = [norm(t) for t in texts]
        vals, cnt = np.unique(keys, return_counts=True)
        p = cnt / cnt.sum()
        SE = float(-(p * np.log(p)).sum())
        out.append(dict(subject=it["subject"], rel=it["rel"], gold=it["gold"],
                        correct=it["correct"], sem_entropy=SE, n_clusters=int(len(vals)),
                        samples=texts[:a.K], frac_top=float(cnt.max() / cnt.sum())))
        if (n + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(items)}  {el:.0f}s  eta={el/(n+1)*(len(items)-n-1)/60:.1f}min", flush=True)
    json.dump(out, open(a.out, "w"))
    print("DONE ->", a.out, " mean SE =", np.mean([r["sem_entropy"] for r in out]))

if __name__ == "__main__":
    main()
