"""Laptop check before Experiment A: does accuracy survive at the predicted temperature?

The melt phase of experiment_a.py predicts a sampling temperature near 1.8 on TURN's
MATH prompts for the Qwen models (EXPERIMENT_A.md). This samples 8 answers for each of
the first 40 problems at four temperatures with the slow Hugging Face backend, and
scores single samples and a majority vote of 8 with TURN's parser and grader. It is a
rough check on a small model and not a substitute for the GPU run.

    python src/experiment_a_preflight.py --turn-dir path/to/TURN
"""
import sys, os, json, time, types, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import experiment_a as E

ap = argparse.ArgumentParser()
ap.add_argument("--turn-dir", default="TURN")
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--out", default="data/experiment_a_preflight_acc.json")
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.turn_dir, "MATH"))
from math_utils.grader import grade_answer

cfg = types.SimpleNamespace(task="math", turn_dir=a.turn_dir, n_problems=40, model=a.model)
rows, extra = E.load_task(cfg)
gen = E.HFGen(cfg)
out = dict(_meta=dict(model=a.model, problems=40, samples=8, max_new_tokens=400, seed=0))
for T in [0.6, 1.0, 1.4, 1.8]:
    t0 = time.time()
    res = gen.generate([r["prompt"] for r in rows], [T] * len(rows), 8, 400, extra["stop"],
                       False, 0)
    parsed = [[E.parse_math(s["text"], extra["parse"]) for s in q] for q in res]
    maj = np.mean([E.majority_correct(p, r["answer"], grade_answer)
                   for p, r in zip(parsed, rows)])
    single = np.mean([np.mean([x != "no answer" and grade_answer(r["answer"], x) for x in p])
                      for p, r in zip(parsed, rows)])
    noans = np.mean([x == "no answer" for p in parsed for x in p])
    out[str(T)] = dict(maj8=float(maj), single=float(single), no_answer=float(noans),
                       secs=time.time() - t0)
    print(f"T={T}  majority@8={maj:.3f}  single-sample={single:.3f}  no-answer={noans:.2f}",
          flush=True)
json.dump(out, open(a.out, "w"), indent=1)
