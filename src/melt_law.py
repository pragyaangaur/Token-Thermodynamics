"""Test the melting-temperature law on real logits.

Prediction from the entropy-versus-energy balance: the distribution melts into
the vocabulary bulk when T * log(V_eff) exceeds the energy gap between the top
token and the bulk, so
        T_melt  ~=  Delta / log(V_eff)
Truncating to the top k tokens sets V_eff = k. If the law holds, T_melt should
rise like 1/log(k) as k shrinks, which would say that top-k truncation and
sampling temperature are two ways of moving the same quantity.
"""
import json, sys, os, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer
from dos import make_dos, curves_dos
from thermo import beta_grid
B = beta_grid(400, 1e-2, 1e3)

PROMPTS = [
 "The capital city of France is", "The chemical symbol for gold is",
 "In a shocking turn of events, the mayor announced that", "def fibonacci(n):\n    if n <",
 "The main advantage of this approach is that it", "She opened the door and saw",
 "The mitochondrion is best described as the", "Water freezes at a temperature of",
 "My favourite thing about the summer is", "The following list summarises the key points:\n1.",
]
KS = [8, 16, 32, 64, 128, 512, 2048, 8192, 32768, 151936]

def Tmelt_of(lg):
    E, W = make_dos(lg, K=min(1024, lg.size), NB=512)
    c = curves_dos(E, W, B)
    o = np.argsort(c["T"]); T, C = c["T"][o], c["C"][o]
    return float(T[int(np.argmax(C))])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    rows = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            lg = model(ids).logits[0, -1].float().cpu().numpy().astype(np.float64)
        s = np.sort(lg)[::-1]
        r = {"prompt": p}
        for k in KS:
            sub = s[:k]
            r[f"T_{k}"] = Tmelt_of(sub)
            r[f"D_{k}"] = float(sub[0] - sub.mean())
        rows.append(r)
    print(f"{'prompt':38s} " + " ".join(f"k={k:<6d}" for k in KS))
    for r in rows:
        print(f"{r['prompt'][:37]:38s} " + " ".join(f"{r[f'T_{k}']:8.3f}" for k in KS))
    print()
    print("test of  T_melt * log(k) = Delta_k  (should be roughly constant in k)")
    print(f"{'k':>8s} {'mean T_melt':>12s} {'mean Delta':>11s} {'T*log k':>9s} {'Delta/(T log k)':>16s}")
    for k in KS:
        T = np.mean([r[f"T_{k}"] for r in rows]); D = np.mean([r[f"D_{k}"] for r in rows])
        print(f"{k:8d} {T:12.3f} {D:11.3f} {T*np.log(k):9.3f} {D/(T*np.log(k)):16.3f}")
    # correlation of the law across prompts at full vocabulary
    T = np.array([r["T_151936"] for r in rows]); D = np.array([r["D_151936"] for r in rows])
    pred = D / np.log(151936)
    from scipy.stats import pearsonr
    print(f"\nacross prompts at full vocab: r(T_melt, Delta/log V) = {pearsonr(T, pred).statistic:.4f}")
    print(f"  mean measured {T.mean():.3f}   mean predicted {pred.mean():.3f}   ratio {T.mean()/pred.mean():.3f}")
    json.dump(rows, open(a.out, "w"))
    print("DONE ->", a.out)

if __name__ == "__main__":
    main()
