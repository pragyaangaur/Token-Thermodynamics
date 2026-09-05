import json, os, re, sys, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer
from dos import make_dos, curves_dos
from thermo import beta_grid
from features import baseline_feats

BETAS = beta_grid(300, 1e-2, 3e2)

TEMPLATES = {
 "capital":       ("What is the capital city of {s}?",            "Answer with only the city name."),
 "element_symbol":("What is the chemical symbol for the element {s}?", "Answer with only the symbol."),
 "director":      ("Who directed the film \"{s}\"?",              "Answer with only the person's full name."),
 "author":        ("Who wrote the book \"{s}\"?",                 "Answer with only the author's full name."),
}

def norm(x):
    x = x.lower().strip()
    x = re.sub(r"[^a-z0-9 ]", " ", x)
    x = re.sub(r"\b(the|a|an|city|of)\b", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def grade(rel, pred, gold):
    p, g = norm(pred), norm(gold)
    if not p: return 0
    if rel == "element_symbol":
        return int(pred.strip().split()[0].strip(".,").lower() == gold.strip().lower()) if pred.strip() else 0
    if g and (g in p or p in g): return 1
    if rel in ("director", "author"):
        gt = g.split()
        if gt and len(gt[-1]) > 2 and gt[-1] in p.split(): return 1
    return 0

ABSTAIN = re.compile(r"\b(i (do not|don't) know|not sure|unsure|unable to|no information|cannot determine|i'm not)\b", re.I)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    print("loaded", a.model, "on", dev, "vocab", model.config.vocab_size, flush=True)

    if a.fake:
        items = [dict(fake=1, **r) for r in json.load(open("data/fake_entities.json"))]
    else:
        raw = json.load(open("data/wikidata_raw.json"))
        items = []
        for rel, recs in raw.items():
            for r in recs:
                items.append(dict(rel=rel, fake=0, **r))
    rng = np.random.default_rng(0); rng.shuffle(items)
    if a.limit: items = items[:a.limit]
    print("items", len(items), flush=True)

    res, t0 = [], time.time()
    for n, it in enumerate(items):
        q, instr = TEMPLATES[it["rel"]]
        msgs = [{"role": "system", "content": "You are a precise factual assistant. " + instr},
                {"role": "user", "content": q.format(s=it["subject"])}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=a.max_new, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True,
                                 pad_token_id=tok.eos_token_id)
        gen = out.sequences[0, ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True).strip()
        lg0 = out.scores[0][0].float().cpu().numpy().astype(np.float64)

        base = baseline_feats(lg0)
        E, W = make_dos(lg0)
        c = curves_dos(E, W, BETAS)
        rec = dict(rel=it["rel"], subject=it["subject"], gold=it["object"], links=it["links"], fake=it["fake"],
                   pred=text, correct=grade(it["rel"], text, it["object"]),
                   abstain=int(bool(ABSTAIN.search(text))),
                   C=c["C"].tolist(), S=c["S"].tolist(), U=c["U"].tolist(),
                   top_logits=np.sort(lg0)[::-1][:64].tolist(), **base)
        # mean thermo curve over the whole answer span (all generated positions)
        Cs = []
        for sc in out.scores[:len(gen)]:
            l = sc[0].float().cpu().numpy().astype(np.float64)
            if not np.isfinite(l).all(): continue
            Ei, Wi = make_dos(l, K=512, NB=256)
            Cs.append(curves_dos(Ei, Wi, BETAS)["C"])
        rec["C_span"] = (np.mean(Cs, 0).tolist() if Cs else rec["C"])
        res.append(rec)
        if (n + 1) % 100 == 0:
            acc = np.mean([r["correct"] for r in res])
            el = time.time() - t0
            print(f"  {n+1}/{len(items)}  acc={acc:.3f}  {el:.0f}s  eta={el/(n+1)*(len(items)-n-1)/60:.1f}min", flush=True)

    json.dump(dict(model=a.model, betas=BETAS.tolist(), results=res), open(a.out, "w"))
    acc = np.mean([r["correct"] for r in res])
    print(f"DONE  n={len(res)}  acc={acc:.4f}  -> {a.out}")
    for rel in sorted(set(r["rel"] for r in res)):
        sub = [r for r in res if r["rel"] == rel]
        print(f"   {rel:16s} n={len(sub):5d} acc={np.mean([r['correct'] for r in sub]):.3f} abstain={np.mean([r['abstain'] for r in sub]):.3f}")

if __name__ == "__main__":
    main()
