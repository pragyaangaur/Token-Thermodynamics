"""Reproduce the headline result: the melting law across 12 models.

This reads the committed result files and reprints the tables in FINDINGS.md sections 7g
and 7h. It needs no GPU, no model downloads and no network, and runs in about a second.

To regenerate the underlying numbers from scratch instead, see the commands in README.md.
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = {
    "gpt2": "GPT-2",
    "EleutherAI/pythia-160m": "Pythia-160M",
    "HuggingFaceTB/SmolLM2-360M": "SmolLM2-360M",
    "facebook/opt-125m": "OPT-125M",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "bigscience/bloomz-560m": "BLOOMZ-560M",
}


def xstar(vocab):
    """Peak condition of the Schottky heat capacity for one ground state above V states.

    Solving dC/dT = 0 gives x* = 2(1+u)/(1-u) with u = V exp(-x*), and T_melt = Delta / x*.
    """
    f = lambda x: x - 2 * (1 + vocab * math.exp(-x)) / (1 - vocab * math.exp(-x))
    lo = math.log(vocab) + 1e-9
    return brentq(f, lo, lo + 80)


def load(name):
    path = os.path.join(HERE, "data", name)
    if not os.path.exists(path):
        sys.exit(f"missing {path}. Run the commands in README.md to regenerate it.")
    with open(path) as fh:
        return json.load(fh)


def main():
    pretrained = load("multimodel_melt3.json")
    tiny = load("vocab_exp.json")

    rows = []
    for rec in pretrained:
        per_prompt = rec["rows"]
        rows.append((
            SHORT.get(rec["model"], rec["model"]),
            rec["V"],
            rec["xstar"],
            float(np.median([r["Tmelt"] for r in per_prompt])),
            float(np.median([r["delta_mode"] for r in per_prompt])),
            "pretrained",
        ))
    for rec in tiny:
        rows.append((f"tiny V={rec['V']}", rec["V"], rec["xstar"], rec["Tmelt"], rec["delta"], "trained here"))
    rows.sort(key=lambda r: r[1])

    print("Melting law: T_melt = Delta / x*(V), with Delta measured to the mode of the logits\n")
    header = f"{'model':22s} {'V':>7s} {'x*':>6s} {'T_melt':>7s} {'Delta':>7s} {'T/D':>8s} {'T x*/D':>8s}  origin"
    print(header)
    print("-" * len(header))
    for name, vocab, x, t, d, origin in rows:
        print(f"{name:22s} {vocab:7d} {x:6.2f} {t:7.3f} {d:7.2f} {t/d:8.5f} {t*x/d:8.4f}  {origin}")

    t_arr = np.array([r[3] for r in rows])
    d_arr = np.array([r[4] for r in rows])
    x_arr = np.array([r[2] for r in rows])
    v_arr = np.array([r[1] for r in rows])

    plain = t_arr / d_arr
    with_v = t_arr * x_arr / d_arr

    print(f"\n{len(rows)} models, vocabulary {v_arr.min()} to {v_arr.max()} "
          f"({v_arr.max()/v_arr.min():.0f}x), x* {x_arr.min():.2f} to {x_arr.max():.2f}\n")
    print(f"  T/Delta      (no log V term) : mean {plain.mean():.4f}  "
          f"CV {plain.std()/plain.mean():.4f}  spread {plain.max()/plain.min():.2f}x")
    print(f"  T x*/Delta   (with log V)    : mean {with_v.mean():.4f}  "
          f"CV {with_v.std()/with_v.mean():.4f}  spread {with_v.max()/with_v.min():.2f}x")
    print(f"  scatter reduced by {100 * (1 - (with_v.std()/with_v.mean()) / (plain.std()/plain.mean())):.0f}%")
    print(f"  r(T_melt, Delta)             : {pearsonr(t_arr, d_arr).statistic:+.4f}")
    print(f"  r(T_melt, Delta/x*)          : {pearsonr(t_arr, d_arr / x_arr).statistic:+.4f}")

    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(20000):
        idx = rng.integers(0, len(t_arr), len(t_arr))
        a, b = plain[idx], with_v[idx]
        diffs.append(b.std() / b.mean() - a.std() / a.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = "significant" if hi < 0 else "not significant"
    print(f"  bootstrap CV(with) - CV(without): {np.mean(diffs):+.4f} "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}")

    print("\nExpected: scatter reduced by 58%, r rises from +0.34 to +0.79, CI [-0.269, -0.082].")


if __name__ == "__main__":
    main()
