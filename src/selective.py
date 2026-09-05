"""Risk-coverage: what practitioners actually care about. Answer only the most
confident fraction of questions, measure the error rate on what you answered."""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

def rc(y_err, score):
    """score high = predicted error. Returns coverage grid and risk."""
    o = np.argsort(score)                      # most confident first
    e = np.asarray(y_err)[o]
    cum = np.cumsum(e) / np.arange(1, len(e) + 1)
    cov = np.arange(1, len(e) + 1) / len(e)
    return cov, cum

for tag, path in [("Qwen2.5-1.5B", "data/records_main15.pkl"), ("Qwen2.5-0.5B", "data/records_main05.pkl")]:
    if not os.path.exists(path): continue
    R = [r for r in pickle.load(open(path, "rb")) if not r.get("fake")]
    y = np.array([1 - r["correct"] for r in R])
    print(f"\n{tag}  n={len(R)}  base error rate {y.mean():.4f}")
    print(f"  {'score':22s} {'AURC':>8s} " + " ".join(f"err@{int(c*100)}%cov" for c in (0.2, 0.4, 0.6, 0.8)))
    for name, key, sign in [("entropy S(T=1)", "b_entropy", +1),
                            ("max prob", "b_pmax", -1),
                            ("log p(top1)", "b_logprob_top1", -1),
                            ("varentropy C(T=1)", "b_varentropy", +1),
                            ("S(T_melt/2)", "n_S_50melt", +1),
                            ("S(T_melt)", "n_S_at_melt", +1),
                            ("cold entropy fraction", "t_cold_frac", +1)]:
        s = sign * np.array([r[key] for r in R], float)
        cov, risk = rc(y, s)
        aurc = float(np.trapezoid(risk, cov))
        cells = " ".join(f"{risk[int(c*len(R))-1]:9.4f}" for c in (0.2, 0.4, 0.6, 0.8))
        print(f"  {name:22s} {aurc:8.4f} {cells}")
