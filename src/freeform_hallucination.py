"""Can free-form hallucination be detected at all, given sections 7f/7i/7j?

Those sections all used free-form questions the model knows well, and found essentially
zero meaning uncertainty. The open question is whether meaning uncertainty appears when
the model is fabricating. If it stays near zero, free-form hallucination is invisible to
every method in this project. If it rises, then semantic entropy over just the first few
tokens is a cheap and targeted detector, because 7i showed 98.8% of the meaning signal
lives there.

Three conditions, matched in surface form:
  KNOWN     free-form questions about common knowledge
  OBSCURE   free-form questions about real but very obscure things
  FABRICATED free-form questions about things that do not exist
"""
import json, sys, os, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from freeform_degeneracy import embed, cluster, EMB

KNOWN = [
 "Why is the sky blue?", "Why do leaves change colour in autumn?", "How does a vaccine work?",
 "Why is the ocean salty?", "Why do metals conduct electricity?", "How does a refrigerator keep food cold?",
 "Why does bread rise when you bake it?", "Why is exercise good for you?",
 "How do plants make food from sunlight?", "Why do we need sleep?",
 "What causes thunder?", "Why does ice float on water?",
 "How does soap clean things?", "Why do we see a rainbow after rain?",
 "What makes an aeroplane able to fly?", "Why does iron rust?",
]
OBSCURE = [
 "Why did the Kingdom of Kanem-Bornu decline in the seventeenth century?",
 "What role did the Lex Papia Poppaea play in Roman family law?",
 "Why is the mineral cummingtonite significant to petrologists?",
 "How did the Sasanian spahbed system reorganise military command?",
 "What was the economic function of the Hanseatic Kontor at Bryggen?",
 "Why did the Kalmar Union ultimately dissolve?",
 "What distinguishes the Tocharian B dialect from Tocharian A?",
 "How did the Ryukyuan tributary relationship with Ming China work?",
 "What was the significance of the Council of Chalcedon for Miaphysite churches?",
 "Why is the Cambrian Burgess Shale fauna preserved so unusually well?",
 "What was the purpose of the Byzantine institution of the pronoia?",
 "How did the Zanj Rebellion affect Abbasid agriculture?",
 "What is the metallurgical importance of the Widmanstatten pattern?",
 "Why did the Mississippian culture at Cahokia collapse?",
 "What role did the Silla bone-rank system play in Korean society?",
 "How does the Kuroshio Extension influence North Pacific weather?",
]
FABRICATED = [
 "Why did the Verrandine Compact of 1623 fail to hold?",
 "What role did the Thassic Ledger play in medieval Vundmark trade?",
 "Why is the mineral quellonite unusual among silicates?",
 "How did the Orimean tithe system reshape rural Belutania?",
 "What was the economic function of the Grelhaven Exchange?",
 "Why did the Cadesian League ultimately dissolve?",
 "What distinguishes the Narvic B dialect from Narvic A?",
 "How did the Pelgardian tributary system with the Xanheim court work?",
 "What was the significance of the Synod of Tirburg for the Ilmarite churches?",
 "Why is the Sornish Marl fauna preserved so unusually well?",
 "What was the purpose of the Yelesian institution of the kardomy?",
 "How did the Rhunic Revolt affect Malovian agriculture?",
 "What is the metallurgical importance of the Quesel banding pattern?",
 "Why did the Vasdoran settlement at Thasgard collapse?",
 "What role did the Dorvok rank system play in Krastan society?",
 "How does the Belmaric Drift influence southern Ormesia weather?",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("-K", type=int, default=16)
    ap.add_argument("--nt", type=int, default=44)
    ap.add_argument("--thr", type=float, default=0.80)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    etok = AutoTokenizer.from_pretrained(EMB); emb = AutoModel.from_pretrained(EMB).to(dev).eval()
    res = {}
    for cond, qs in [("KNOWN", KNOWN), ("OBSCURE", OBSCURE), ("FABRICATED", FABRICATED)]:
        recs = []
        for qi, q in enumerate(qs):
            msgs = [{"role": "system", "content": "You are a helpful assistant. Answer in two or three sentences."},
                    {"role": "user", "content": q}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            ids = enc["input_ids"].to(dev)
            # (a) sampled-answer semantic entropy, the expensive reference
            with torch.no_grad():
                sm = lm.generate(ids.repeat(a.K, 1), max_new_tokens=a.nt, do_sample=True,
                                 temperature=1.0, top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id)
            samples = [tok.decode(x, skip_special_tokens=True).strip() for x in sm[:, ids.shape[1]:]]
            lab = cluster(embed(samples, etok, emb, dev), a.thr)
            cnt = np.bincount(lab); p = cnt / cnt.sum()
            SE = float(-(p[p > 0] * np.log(p[p > 0])).sum())
            # (b) cheap first-3-token fork semantic entropy, per section 7i
            with torch.no_grad():
                gr = lm.generate(ids, max_new_tokens=a.nt, do_sample=False, pad_token_id=tok.pad_token_id)
            ans = gr[0, ids.shape[1]:]
            fork_SE, tokH = [], []
            for t in range(min(3, len(ans))):
                pre = torch.cat([ids[0], ans[:t]])[None]
                with torch.no_grad():
                    lg = lm(pre).logits[0, -1].float()
                pp = torch.softmax(lg, -1)
                tokH.append(float(-(pp[pp > 0] * torch.log(pp[pp > 0])).sum()))
                top = torch.topk(lg, 8)
                fk = torch.cat([pre.repeat(8, 1), top.indices[:, None]], 1)
                with torch.no_grad():
                    gg = lm.generate(fk, max_new_tokens=28, do_sample=False, pad_token_id=tok.pad_token_id)
                txt = [tok.decode(x, skip_special_tokens=True).strip() for x in gg[:, ids.shape[1]:]]
                l2 = cluster(embed(txt, etok, emb, dev), a.thr)
                pk = torch.softmax(top.values, 0).cpu().numpy()
                pc = np.array([pk[l2 == c].sum() for c in range(l2.max()+1)]); pc /= pc.sum()
                fork_SE.append(float(-(pc[pc > 0] * np.log(pc[pc > 0])).sum()))
            recs.append(dict(q=q, sem_entropy=SE, n_meanings=int(len(set(lab))),
                             top_mass=float(p.max()), fork_SE=fork_SE, tok_H=tokH,
                             fork_SE_sum=float(np.sum(fork_SE)), tokH_sum=float(np.sum(tokH)),
                             sample=samples[0][:110]))
            if (qi+1) % 8 == 0: print(f"  {cond} {qi+1}/{len(qs)}", flush=True)
        res[cond] = recs
        SE = np.array([r["sem_entropy"] for r in recs]); nm = np.array([r["n_meanings"] for r in recs])
        tm = np.array([r["top_mass"] for r in recs]); fs = np.array([r["fork_SE_sum"] for r in recs])
        th = np.array([r["tokH_sum"] for r in recs])
        print(f"\n=== {cond} (n={len(recs)}, K={a.K}) ===")
        print(f"  sampled-answer semantic entropy : {SE.mean():.4f}   distinct meanings {nm.mean():.2f}   top mass {tm.mean():.3f}")
        print(f"  cheap fork SE over first 3 tokens: {fs.mean():.4f}")
        print(f"  token entropy over first 3 tokens: {th.mean():.4f}")
    json.dump(res, open(a.out, "w"))
    from sklearn.metrics import roc_auc_score
    print("\n=== can these separate KNOWN from FABRICATED? ===")
    for key, lab in [("sem_entropy","sampled semantic entropy (16 samples)"),
                     ("fork_SE_sum","fork SE over first 3 tokens (24 short gens)"),
                     ("tokH_sum","plain token entropy, first 3 tokens (1 pass)"),
                     ("top_mass","top meaning mass (16 samples)")]:
        for A,Bn in [("KNOWN","FABRICATED"),("KNOWN","OBSCURE"),("OBSCURE","FABRICATED")]:
            y=np.array([0]*len(res[A])+[1]*len(res[Bn]))
            v=np.array([r[key] for r in res[A]]+[r[key] for r in res[Bn]])
            au=roc_auc_score(y,v); print(f"  {lab:46s} {A:8s} vs {Bn:10s} AUROC {max(au,1-au):.4f}")
    print("FFHDONE")

if __name__ == "__main__":
    main()
