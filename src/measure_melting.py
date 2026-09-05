"""How much does the melting temperature vary across ordinary text?

A single global sampling temperature assumes every position has the same
thermal scale. T_melt(position) is measurable in one forward pass, so the
assumption is checkable.
"""
import json, sys, os, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer
from dos import make_dos, curves_dos
from thermo import beta_grid
B = beta_grid(300, 1e-2, 3e2)

TEXTS = {
"prose": """The catalyst was prepared by impregnating the support with an aqueous solution of the metal salt, then drying it overnight at 110 degrees and calcining it in flowing air. Activity was measured in a fixed bed reactor at atmospheric pressure. Conversion rose steadily with temperature until about 400 kelvin, after which it fell again because the active phase sintered. The same behaviour appeared in every batch we tested, so we concluded that the loss was intrinsic to the material rather than an artefact of the preparation.""",
"code": """def solve(grid):
    n = len(grid)
    seen = set()
    total = 0
    for i in range(n):
        for j in range(n):
            if (i, j) in seen or grid[i][j] == 0:
                continue
            stack = [(i, j)]
            size = 0
            while stack:
                x, y = stack.pop()
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                size += 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = x + dx, y + dy
                    if 0 <= a < n and 0 <= b < n and grid[a][b]:
                        stack.append((a, b))
            total = max(total, size)
    return total""",
"facts": """Paris is the capital of France. Tokyo is the capital of Japan. The chemical symbol for gold is Au and the symbol for iron is Fe. Water boils at 100 degrees Celsius at sea level. The Pacific Ocean is the largest ocean on Earth. Mount Everest is the highest mountain above sea level. The speed of light in vacuum is about 300000 kilometres per second.""",
"openended": """The best thing about living near the coast is probably the way the weather changes without warning. Some mornings you wake up and the whole bay has gone flat and silver, and by lunchtime there is a wind coming in hard enough to take the chairs off the terrace. People who grew up inland find this unsettling. I have come to like it, though I could not really explain why, and when I try to the explanation always sounds thinner than the feeling.""",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    out = {}
    for name, txt in TEXTS.items():
        ids = tok(txt, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            lg = model(ids).logits[0].float().cpu().numpy().astype(np.float64)
        recs = []
        for t in range(lg.shape[0]):
            E, W = make_dos(lg[t], K=512, NB=256)
            c = curves_dos(E, W, B)
            o = np.argsort(c["T"]); T, C = c["T"][o], c["C"][o]
            i = int(np.argmax(C))
            nxt = tok.decode(ids[0, t + 1]) if t + 1 < ids.shape[1] else ""
            recs.append(dict(pos=t, Tmelt=float(T[i]), Cmax=float(C.max()),
                             S1=float(np.interp(1.0, T, c["S"][o])),
                             C1=float(np.interp(1.0, T, C)), nxt=nxt))
        out[name] = recs
        v = np.array([r["Tmelt"] for r in recs])
        print(f"{name:10s} n={len(v):4d}  T_melt: min {v.min():.3f}  p5 {np.percentile(v,5):.3f}  "
              f"median {np.median(v):.3f}  p95 {np.percentile(v,95):.3f}  max {v.max():.3f}  "
              f"spread {v.max()/v.min():.2f}x  IQR-ratio {np.percentile(v,75)/np.percentile(v,25):.2f}x", flush=True)
    allv = np.array([r["Tmelt"] for v in out.values() for r in v])
    print(f"{'ALL':10s} n={len(allv):4d}  spread {allv.max()/allv.min():.2f}x  "
          f"p95/p5 {np.percentile(allv,95)/np.percentile(allv,5):.2f}x")
    json.dump(out, open(a.out, "w"))
    print("DONE ->", a.out)

if __name__ == "__main__":
    main()
