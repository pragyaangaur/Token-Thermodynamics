"""Measure the two energy scales of a real next-token distribution.

For each question, take the top-M candidate first tokens. Force each one and
greedily continue, so every candidate first token is mapped to the complete
answer it would produce. Cluster the candidates by that final normalised answer:
tokens landing on the same answer are SURFACE variants of one meaning, tokens
landing on different answers are different MEANINGS.

Then measure, in logit (energy) units:
    delta  = spread of candidates inside one meaning cluster
    spread = spread between meaning clusters
The ratio decides whether a temperature window can separate the two, which the
synthetic phase diagram says needs roughly delta/spread < 0.1.
"""
import json, os, re, sys, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer
from probe import TEMPLATES, norm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("-M", type=int, default=12)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--max-new", type=int, default=8)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()

    raw = json.load(open("data/wikidata_raw.json"))
    items = [dict(rel=r, **x) for r, v in raw.items() for x in v]
    rng = np.random.default_rng(3); rng.shuffle(items); items = items[:a.limit]
    print("measuring energy scales on", len(items), "items, M =", a.M, flush=True)

    out, t0 = [], time.time()
    for n, it in enumerate(items):
        q, instr = TEMPLATES[it["rel"]]
        msgs = [{"role": "system", "content": "You are a precise factual assistant. " + instr},
                {"role": "user", "content": q.format(s=it["subject"])}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(dev)
        with torch.no_grad():
            lg = model(ids).logits[0, -1].float()
        top = torch.topk(lg, a.M)
        cand = top.indices                                   # (M,)
        clog = top.values.cpu().numpy().astype(np.float64)
        # force each candidate first token, then continue greedily
        pref = torch.cat([ids.repeat(a.M, 1), cand[:, None]], dim=1)
        with torch.no_grad():
            g = model.generate(pref, max_new_tokens=a.max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        answers = [norm(tok.decode(x, skip_special_tokens=True)) for x in g[:, ids.shape[1]:]]
        # cluster candidates by the answer they lead to
        lab, seen = [], {}
        for ansr in answers:
            if ansr not in seen: seen[ansr] = len(seen)
            lab.append(seen[ansr])
        lab = np.array(lab)
        # energy scales, measured in logit units
        withins, means = [], []
        p = np.exp(clog - clog.max()); p /= p.sum()
        pc = []
        for c in range(lab.max() + 1):
            m = lab == c
            v = clog[m]
            if m.sum() > 1: withins.append(float(v.max() - v.min()))
            means.append(float(v.max()))       # cluster energy = its best member
            pc.append(float(p[m].sum()))
        pc = np.array(pc)
        means = np.array(means)
        rec = dict(rel=it["rel"], subject=it["subject"], gold=it["object"],
                   answer_top=answers[0], n_clusters=int(lab.max() + 1),
                   delta=float(np.mean(withins)) if withins else 0.0,
                   delta_max=float(np.max(withins)) if withins else 0.0,
                   spread=float(means.max() - means.min()) if len(means) > 1 else 0.0,
                   gap12=float(means[0] - means[1]) if len(means) > 1 else 0.0,
                   top_gap_within=float(clog[0] - clog[1]),
                   sem_entropy_tok=float(-(pc[pc > 0] * np.log(pc[pc > 0])).sum()),
                   tok_entropy_topM=float(-(p * np.log(p)).sum()),
                   cluster_sizes=[int((lab == c).sum()) for c in range(lab.max() + 1)],
                   answers=list(seen.keys())[:6])
        out.append(rec)
        if (n + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(items)} {el:.0f}s eta={el/(n+1)*(len(items)-n-1)/60:.1f}min", flush=True)
    json.dump(out, open(a.out, "w"))
    D = np.array([r["delta"] for r in out]); S = np.array([r["spread"] for r in out])
    m = S > 1e-6
    print("DONE ->", a.out)
    print(f"  items with >1 meaning among top-{a.M}: {m.mean():.3f}")
    print(f"  median delta  (within-meaning logit spread)  = {np.median(D[m]):.3f}")
    print(f"  median spread (between-meaning logit spread) = {np.median(S[m]):.3f}")
    print(f"  median ratio delta/spread                    = {np.median(D[m]/S[m]):.3f}")

if __name__ == "__main__":
    main()
