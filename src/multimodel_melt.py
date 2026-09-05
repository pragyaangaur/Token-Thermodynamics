"""Test the melting law ACROSS model families with different vocabulary sizes.

The law is  T_melt = Delta / x*(V),  where x*(V) solves the Schottky peak condition
    x* = 2(1+u)/(1-u),  u = V exp(-x*)
and x* ~ log V + c. Within one model this was confirmed by truncating the vocabulary.
The sharper test is across models that genuinely have different V, because there the
law predicts a specific ratio between two independently measured quantities.

Falsifiable prediction: T_melt * x*(V) / Delta should be the same constant for every
model, even though V spans 49k to 251k and Delta varies with the model's confidence.
"""
import json, sys, os, argparse
import numpy as np, torch
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer
from dos import make_dos, curves_dos
from thermo import beta_grid
B = beta_grid(500, 5e-3, 1e3)

def xstar(V):
    f = lambda x: x - 2 * (1 + V * np.exp(-x)) / (1 - V * np.exp(-x))
    lo = np.log(V) + 1e-9
    return brentq(f, lo, lo + 80)

PROMPTS = [
 "The capital city of France is", "The chemical symbol for gold is",
 "In a shocking turn of events, the mayor announced that",
 "The main advantage of this approach is that it", "She opened the door and saw",
 "Water freezes at a temperature of", "My favourite thing about the summer is",
 "The largest planet in our solar system is", "He picked up the phone and said",
 "The first step in the process is to", "Scientists have discovered that the",
 "According to the report published last year, the company",
 "The difference between the two methods is that", "It was raining when they arrived at the",
 "The book was written by", "One reason people prefer this is that",
 "After the war ended, the government began to", "The temperature outside was about",
 "In order to solve this problem, you need to", "The cat sat on the",
]

MODELS = ["gpt2", "EleutherAI/pythia-160m", "HuggingFaceTB/SmolLM2-360M",
          "facebook/opt-125m", "Qwen/Qwen2.5-0.5B-Instruct",
          "Qwen/Qwen2.5-1.5B-Instruct", "bigscience/bloomz-560m"]

def Tmelt_of(lg):
    E, W = make_dos(lg, K=min(1024, lg.size), NB=512)
    c = curves_dos(E, W, B)
    o = np.argsort(c["T"]); T, C = c["T"][o], c["C"][o]
    i = int(np.argmax(C))
    return float(T[i]), float(C.max())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out = []
    for m in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(m)
            model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32).to(dev).eval()
        except Exception as e:
            print(f"SKIP {m}: {type(e).__name__} {str(e)[:90]}", flush=True); continue
        V = model.get_output_embeddings().weight.shape[0]
        rows = []
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                lg = model(ids).logits[0, -1].float().cpu().numpy().astype(np.float64)
            lg = lg[np.isfinite(lg)]
            Tm, Cm = Tmelt_of(lg)
            s = np.sort(lg)[::-1]
            hh, ee = np.histogram(lg, bins=400)
            mode = float(0.5 * (ee[:-1] + ee[1:])[int(np.argmax(hh))])
            rows.append(dict(prompt=p, Tmelt=Tm, Cmax=Cm, delta_mode=float(s[0] - mode), mode=mode,
                             delta_mean=float(s[0] - lg.mean()),
                             delta_bulk=float(s[0] - np.median(lg)),
                             delta_p99=float(s[0] - np.percentile(lg, 1)),
                             top1=float(s[0]), sd=float(lg.std())))
        x = xstar(V)
        T = np.array([r["Tmelt"] for r in rows]); D = np.array([r["delta_mode"] for r in rows])
        Cm = np.array([r["Cmax"] for r in rows])
        rec = dict(model=m, V=int(V), logV=float(np.log(V)), xstar=float(x),
                   Tmelt_med=float(np.median(T)), delta_med=float(np.median(D)),
                   ratio_med=float(np.median(T * x / D)), Cmax_med=float(np.median(Cm)),
                   Cmax_ceiling=float(x * x * (lambda u: u / (1 + u) ** 2)(V * np.exp(-x))),
                   rows=rows)
        out.append(rec)
        print(f"{m:34s} V={V:7d} logV={np.log(V):6.3f} x*={x:6.3f}  "
              f"T_melt={np.median(T):6.3f}  Delta={np.median(D):7.3f}  "
              f"T*x*/Delta={rec['ratio_med']:6.4f}  Cmax={np.median(Cm):6.2f}/{rec['Cmax_ceiling']:.1f}", flush=True)
        del model; torch.mps.empty_cache() if dev == "mps" else None
    json.dump(out, open(a.out, "w"))
    if len(out) > 1:
        r = np.array([o["ratio_med"] for o in out])
        print(f"\nACROSS {len(out)} MODELS, V from {min(o['V'] for o in out)} to {max(o['V'] for o in out)}:")
        print(f"  T_melt * x*(V) / Delta : mean {r.mean():.4f}  sd {r.std():.4f}  "
              f"spread {r.max()/r.min():.3f}x")
        V = np.array([o["V"] for o in out]); T = np.array([o["Tmelt_med"] for o in out])
        D = np.array([o["delta_med"] for o in out])
        from scipy.stats import pearsonr
        print(f"  r(T_melt, Delta/x*)    : {pearsonr(T, D/np.array([o['xstar'] for o in out])).statistic:.4f}")
        print(f"  for contrast, r(T_melt, Delta) ignoring V : {pearsonr(T, D).statistic:.4f}")
    print("MMDONE")

if __name__ == "__main__":
    main()
