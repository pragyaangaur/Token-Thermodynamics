"""Greedy decoding maximises sequence probability. But a MEANING's probability is the
sum over all the ways of phrasing it, so in thermodynamic terms the most probable meaning
minimises a free energy

    F(m) = E(m) - T * S_phrasing(m)

where E(m) is the energy (negative log-probability) of the meaning's best phrasing and
S_phrasing(m) is the log of how many comparable phrasings it has. Section 7f measured
94.7% of free-form first-token entropy to be pure phrasing, so S_phrasing is large and
this correction should not be negligible.

Concretely: does the greedy answer's meaning match the meaning that actually carries the
most probability mass? If not, greedy decoding is picking a sharp lonely peak over a
broad likely basin, which is the mode-seeking problem that Minimum Bayes Risk decoding
exists to fix. This measures the size of the effect and splits it by regime.
"""
import json, sys, os, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from freeform_degeneracy import embed, cluster, EMB, OPEN_Q, FACT_Q

def seq_logprob(lm, ids, gen, dev):
    """sum log p of the generated tokens under the model"""
    full = torch.cat([ids[0], gen])[None].to(dev)
    with torch.no_grad():
        lg = lm(full).logits[0].float()
    lp = torch.log_softmax(lg[:-1], -1)
    tgt = full[0, 1:]
    tot = lp[torch.arange(len(tgt)), tgt]
    return float(tot[ids.shape[1]-1:].sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("-K", type=int, default=24)
    ap.add_argument("--thr", type=float, default=0.80)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    etok = AutoTokenizer.from_pretrained(EMB); emb = AutoModel.from_pretrained(EMB).to(dev).eval()

    res = {}
    for cond, qs, instr, nt in [("free-form", OPEN_Q[:24], "Answer in two or three sentences.", 44),
                                ("short-factual", FACT_Q[:24], "Answer with only the name.", 10)]:
        recs = []
        for qi, q in enumerate(qs):
            msgs = [{"role": "system", "content": "You are a helpful assistant. " + instr},
                    {"role": "user", "content": q}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            ids = enc["input_ids"].to(dev)
            with torch.no_grad():
                gr = lm.generate(ids, max_new_tokens=nt, do_sample=False, pad_token_id=tok.pad_token_id)
            greedy = tok.decode(gr[0, ids.shape[1]:], skip_special_tokens=True).strip()
            with torch.no_grad():
                sm = lm.generate(ids.repeat(a.K, 1), max_new_tokens=nt, do_sample=True,
                                 temperature=1.0, top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id)
            samples = [tok.decode(x, skip_special_tokens=True).strip() for x in sm[:, ids.shape[1]:]]
            texts = [greedy] + samples
            lab = cluster(embed(texts, etok, emb, dev), a.thr)
            gl = lab[0]                                   # greedy's meaning cluster
            sl = lab[1:]                                  # sampled meanings
            K = len(sl)
            counts = np.bincount(sl, minlength=lab.max()+1)
            mass = counts / K                             # Monte Carlo P(meaning)
            top_mass = int(np.argmax(counts))
            # phrasing degeneracy: distinct surface strings per meaning
            uniq = {}
            for c, t in zip(sl, samples): uniq.setdefault(c, set()).add(t)
            Sphr = {c: float(np.log(len(v))) for c, v in uniq.items()}
            recs.append(dict(q=q, greedy=greedy[:120],
                             greedy_cluster=int(gl), top_mass_cluster=top_mass,
                             agree=int(gl == top_mass),
                             greedy_mass=float(mass[gl]) if gl < len(mass) else 0.0,
                             top_mass_val=float(mass[top_mass]),
                             n_meanings=int(len(set(sl))),
                             S_phr_greedy=Sphr.get(gl, 0.0),
                             S_phr_top=Sphr.get(top_mass, 0.0),
                             mean_S_phr=float(np.mean(list(Sphr.values()))) if Sphr else 0.0,
                             uniq_frac=float(len(set(samples))/K)))
            if (qi+1) % 8 == 0: print(f"  {cond} {qi+1}/{len(qs)}", flush=True)
        res[cond] = recs
        ag = np.array([r["agree"] for r in recs]); nm = np.array([r["n_meanings"] for r in recs])
        gm = np.array([r["greedy_mass"] for r in recs]); tm = np.array([r["top_mass_val"] for r in recs])
        sp = np.array([r["mean_S_phr"] for r in recs]); uf = np.array([r["uniq_frac"] for r in recs])
        print(f"\n=== {cond}  (K={a.K} samples at T=1, n={len(recs)}) ===")
        print(f"  distinct meanings among samples        : mean {nm.mean():.2f}")
        print(f"  distinct surface strings / K           : {uf.mean():.3f}")
        print(f"  phrasing degeneracy log(#phrasings)    : {sp.mean():.3f} nats")
        print(f"  greedy meaning == highest-mass meaning : {ag.mean()*100:.1f}% of questions")
        print(f"  probability mass on the greedy meaning : {gm.mean():.3f}")
        print(f"  probability mass on the best meaning   : {tm.mean():.3f}")
        print(f"  mass left on the table by greedy       : {(tm-gm).mean():.3f}")
    json.dump(res, open(a.out, "w"))
    A = res["free-form"]; B = res["short-factual"]
    print(f"\nSUMMARY  greedy picks the most probable MEANING:")
    print(f"  free-form     {np.mean([r['agree'] for r in A])*100:5.1f}%   "
          f"phrasing degeneracy {np.mean([r['mean_S_phr'] for r in A]):.2f} nats")
    print(f"  short factual {np.mean([r['agree'] for r in B])*100:5.1f}%   "
          f"phrasing degeneracy {np.mean([r['mean_S_phr'] for r in B]):.2f} nats")
    print("MFEDONE")

if __name__ == "__main__":
    main()
