"""Regenerate section 7k keeping every raw string, so the clustering threshold
can be swept afterwards without touching the language model again.

FINDINGS.md item 6. Section 7i showed that the embedding-similarity threshold
moves the absolute numbers a lot. Section 7k was run at a single threshold of
0.80 and the check was never repeated. The original run saved only the
clustered summaries, so the sweep needs the texts back.

This script does the generation half and nothing else. `src/ffh_thr.py` does
the clustering sweep.
"""
import json, sys, os, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer
from freeform_hallucination import KNOWN, OBSCURE, FABRICATED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("-K", type=int, default=16)
    ap.add_argument("--nt", type=int, default=44)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, local_files_only=True).to(dev).eval()
    res = {"_meta": dict(model=a.model, K=a.K, nt=a.nt, seed=a.seed)}
    for cond, qs in [("KNOWN", KNOWN), ("OBSCURE", OBSCURE), ("FABRICATED", FABRICATED)]:
        recs = []
        for qi, q in enumerate(qs):
            msgs = [{"role": "system",
                     "content": "You are a helpful assistant. Answer in two or three sentences."},
                    {"role": "user", "content": q}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)
            ids = enc["input_ids"].to(dev)
            with torch.no_grad():
                sm = lm.generate(ids.repeat(a.K, 1), max_new_tokens=a.nt, do_sample=True,
                                 temperature=1.0, top_p=1.0, top_k=0,
                                 pad_token_id=tok.pad_token_id)
            samples = [tok.decode(x, skip_special_tokens=True).strip()
                       for x in sm[:, ids.shape[1]:]]
            with torch.no_grad():
                gr = lm.generate(ids, max_new_tokens=a.nt, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            ans = gr[0, ids.shape[1]:]
            positions = []
            for t in range(min(3, len(ans))):
                pre = torch.cat([ids[0], ans[:t]])[None]
                with torch.no_grad():
                    lg = lm(pre).logits[0, -1].float()
                pp = torch.softmax(lg, -1)
                tokH = float(-(pp[pp > 0] * torch.log(pp[pp > 0])).sum())
                top = torch.topk(lg, 8)
                fk = torch.cat([pre.repeat(8, 1), top.indices[:, None]], 1)
                with torch.no_grad():
                    gg = lm.generate(fk, max_new_tokens=28, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
                txt = [tok.decode(x, skip_special_tokens=True).strip()
                       for x in gg[:, ids.shape[1]:]]
                positions.append(dict(pos=t, tok_H=tokH,
                                      top_logits=[float(v) for v in top.values.cpu()],
                                      forks=txt))
            recs.append(dict(q=q, samples=samples, positions=positions))
            print(f"  {cond} {qi+1}/{len(qs)}", flush=True)
        res[cond] = recs
    json.dump(res, open(a.out, "w"))
    print("FFHTEXTDONE")


if __name__ == "__main__":
    main()
