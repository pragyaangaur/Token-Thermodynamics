"""Experiment A: sample at T_melt/1.141 and ask whether it is as good as TURN.

This is the experiment that decides whether the melting law has a practical use.
Du, Yang and Welleck (arXiv 2502.05234, code at github.com/StigLidu/TURN) choose a
sampling temperature for majority voting and best-of-N by sweeping temperatures,
generating at each one, and taking the turning point of log H against T. Section 7m
of FINDINGS.md shows the single-distribution analogue of that turning point lies
below the melting temperature, with T_melt/T_turn = 1.141 in the S vs T convention.
The melting temperature needs one forward pass and no generation.

The question here is practical. On their benchmarks, with their prompts, parser and
grader, is accuracy at T_melt/1.141 as good as accuracy at their swept temperature
and at the best fixed temperature from a grid search?

Two design choices are stated here because the result depends on them.

1. TURN's H(t) is computed their way, from sampled generations, and is never replaced
   by the single-distribution analogue. The 1.141 ratio was measured in the S vs T
   convention, and TURN uses log H against linear T, which on single distributions
   gave ratios of 2.5x to 27x (data/turning_point.json). So nothing guarantees that
   T_melt/1.141 lands near TURN's temperature, and this script measures the gap.
2. TURN's prompts end straight after the question text, so the first generated token
   continues the question and there is no clean first answer token. T_melt is read at
   each of the first 32 positions of one greedy continuation and the median is taken.
   Two variants are sampled: one temperature per task (median over questions, divided
   by the ratio) and one per question. Section 7e found T_melt moves about 2x within
   one passage, and a temperature that changes per position is not attempted here.

Phases, each checkpointed to --out so a Kaggle session can be resumed:

    melt   one greedy pass per question in float32, full vocabulary, T_melt
    turn   TURN's own entropy sweep, logprobs=1000, reproduced from predict.py
    grid   k samples per question at every grid temperature and the predicted ones
    score  majority voting (MATH) or pass@N (MBPP) at each N, and the comparison

MBPP scoring executes model-written code in a subprocess with a timeout. Run it in a
disposable environment such as a Kaggle notebook, never on a machine you care about.
"""
import json, os, sys, argparse, time, subprocess, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from turning_point import turning_points

RATIO = 1.141        # median T_melt/T_turn, S vs T convention, FINDINGS.md section 7m
BETA = {"math": 0.0, "mbpp": 0.1}   # TURN's aggregation adaptor: MJ 0.0, BofN 0.1
BASELINES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]   # the fixed temperatures TURN compares
TOPK = 1000          # TURN computes entropy over the top 1000 tokens at each step
EPS = 0.02           # TURN's epsilon-optimal range for the hit rate
MBPP_URL = ("https://raw.githubusercontent.com/google-research/google-research/"
            "master/mbpp/mbpp.jsonl")
MBPP_STOP = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"]


# ---------------------------------------------------------------- tasks

def load_task(a):
    """Questions, prompts and whatever the grader needs, for MATH or MBPP."""
    if a.task == "math":
        base = os.path.join(a.turn_dir, "MATH")
        rows = [json.loads(l) for l in open(os.path.join(
            base, "downloads/math_splits/test_filtered.jsonl"))][: a.n_problems]
        few = open(os.path.join(base, "math_utils/few_shot_example_for_math.txt")).read()
        instr = json.load(open(os.path.join(base, "math_utils/instruct_prompt.json")))
        name = a.model.rstrip("/").split("/")[-1]
        instruct, parse = "", ["The answer is:", "ки"]
        if name in instr:
            instruct, parse = instr[name]["instruct"], instr[name]["parse_sentence"]
        for r in rows:
            r["prompt"] = few + r["problem"] + instruct
        return rows, dict(parse=parse, stop=["Question 6:"])
    path = os.path.join(a.out, "mbpp.jsonl")
    if not os.path.exists(path):
        urllib.request.urlretrieve(MBPP_URL, path)
    allrows = {r["task_id"]: r for r in map(json.loads, open(path))}
    # the test split is task ids 11 to 510, and TURN evaluates the first 100
    rows = [allrows[i] for i in range(11, 511)][: a.n_problems]
    for r in rows:
        r["prompt"] = f'"""\n{r["text"]}\n{r["test_list"][0]}\n"""\n'
    return rows, dict(stop=MBPP_STOP)


def parse_math(text, parse):
    """TURN's parser, MATH/hugging_inference.py, few-shot branch."""
    try:
        text = text.split("Question 6")[0].strip()
        if parse[0] not in text:
            return "no answer"
        front = text.split(parse[0])[-1].strip().split("\n")[0].strip()
        if parse[1] != "" and parse[1] in front:
            ans = parse[1].join(front.split(parse[1])[:-1]).strip()
        else:
            ans = front
        return ans if len(ans) < 100 else "no answer"
    except Exception:
        return "no answer"


def mbpp_truncate(text):
    cut = min([text.find(s) for s in MBPP_STOP if s in text] + [len(text)])
    return text[:cut]


def mbpp_passes(row, completion, timeout=5.0):
    program = "\n".join([row["prompt"] + mbpp_truncate(completion),
                         row.get("test_setup_code", ""), *row["test_list"]])
    try:
        r = subprocess.run([sys.executable, "-c", program], capture_output=True,
                           timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


# ---------------------------------------------------------------- TURN

def calc_turning_point(x_values, log_y_values):
    """Copied from TURN predict.py (MIT) so the procedure is theirs exactly."""
    dy_dx = np.gradient(log_y_values, x_values)
    d2y_dx2 = np.gradient(dy_dx, x_values)
    mask = d2y_dx2[1:-1] > 0
    if not np.any(mask):
        mask[-1] = True
    return float(x_values[np.where(mask)[0][0] + 1])


def turn_entropy(lp):
    """TURN's per-position entropy from a top-K set of log probabilities."""
    p = np.exp(lp)
    return float(-(lp * p).sum() / p.sum())


def _logsoftmax(x):
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def position_entropies(lp, T, mode):
    """Entropy of the tempered and of the untempered distribution at one position.

    vLLM returns either raw log probabilities or ones computed after temperature
    scaling, depending on its version and on logprobs_mode. TURN ran on a version that
    returned the tempered ones. Whichever is returned, the other is rebuilt inside the
    top-K set, where log-probability differences scale exactly by 1/T.
    """
    lp = np.asarray(lp, np.float64)
    if mode == "processed":
        return turn_entropy(lp), turn_entropy(_logsoftmax(lp * T))
    return turn_entropy(_logsoftmax(lp / T)), turn_entropy(lp)


# ---------------------------------------------------------------- backends

class VLLM:
    def __init__(self, a):
        from vllm import LLM
        kw = dict(model=a.model, dtype=a.dtype_gen, max_logprobs=TOPK, seed=a.seed,
                  tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem,
                  max_model_len=a.max_model_len, enforce_eager=a.eager)
        self.mode = "unknown"
        try:
            self.llm = LLM(logprobs_mode="processed_logprobs", **kw)
            self.mode = "processed"
        except TypeError:
            self.llm = LLM(**kw)

    def generate(self, prompts, temps, n, max_tokens, stop, logprobs, seed):
        from vllm import SamplingParams
        sps = [SamplingParams(temperature=float(t), top_k=-1, top_p=1.0, n=n,
                              max_tokens=max_tokens, stop=stop, seed=seed + i,
                              logprobs=TOPK if logprobs else None)
               for i, t in enumerate(temps)]
        res = self.llm.generate(prompts, sps)
        out = []
        for r in res:
            samples = []
            for o in r.outputs:
                pos = None
                if logprobs and o.logprobs is not None:
                    pos = [dict((int(k), float(getattr(v, "logprob", v)))
                                for k, v in d.items()) for d in o.logprobs]
                samples.append(dict(text=o.text, ntok=len(o.token_ids), pos=pos))
            out.append(samples)
        return out


class HFGen:
    """Slow fallback so the whole pipeline can be smoke-tested on a laptop."""

    def __init__(self, a):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = device()
        self.tok = AutoTokenizer.from_pretrained(a.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.float32).to(self.dev).eval()
        self.mode = "raw"

    def generate(self, prompts, temps, n, max_tokens, stop, logprobs, seed):
        torch = self.torch
        out = []
        for i, (p, t) in enumerate(zip(prompts, temps)):
            torch.manual_seed(seed + i)
            ids = self.tok(p, return_tensors="pt").input_ids.to(self.dev)
            g = self.model.generate(ids, do_sample=True, temperature=float(t), top_k=0,
                                    top_p=1.0, max_new_tokens=max_tokens,
                                    num_return_sequences=n, output_logits=True,
                                    return_dict_in_generate=True,
                                    pad_token_id=self.tok.eos_token_id)
            new = g.sequences[:, ids.shape[1]:]
            samples = []
            for j in range(n):
                toks = new[j].tolist()
                if self.tok.eos_token_id in toks:
                    toks = toks[: toks.index(self.tok.eos_token_id) + 1]
                text = self.tok.decode(toks, skip_special_tokens=True)
                for s in stop:
                    if s in text:
                        text = text[: text.index(s)]
                pos = None
                if logprobs:
                    pos = []
                    for k in range(len(toks)):
                        lp = torch.log_softmax(g.logits[k][j].float().cpu().double(), -1)
                        v, ix = lp.topk(TOPK)
                        pos.append(dict(zip(ix.tolist(), v.tolist())))
                samples.append(dict(text=text, ntok=len(toks), pos=pos))
            out.append(samples)
        return out


def device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------- phases

def phase_melt(a, rows, path):
    """T_melt along the model's own greedy continuation, float32, full vocabulary.

    TURN's prompts end straight after the question text, so the distribution at the
    end of the prompt belongs to the model continuing the question and not to the
    answer. T_melt is therefore read at each of the first --melt-positions positions
    of one greedy continuation, and the question's value is the median over those
    positions. The prompt-end value is kept for comparison. This costs one greedy
    pass per question and no sampling at any temperature.
    """
    if os.path.exists(path):
        return json.load(open(path))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = dict(dtype=torch.float32)
    if device() == "cuda":
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if device() != "cuda":
        model = model.to(device())
    first = next(model.parameters()).device
    out, ntok = [], 0
    t0 = time.time()
    for r in rows:
        ids = tok(r["prompt"], return_tensors="pt").input_ids.to(first)
        with torch.no_grad():
            g = model.generate(ids, do_sample=False, max_new_tokens=a.melt_positions,
                               output_logits=True, return_dict_in_generate=True,
                               pad_token_id=tok.eos_token_id)
        pos = []
        for k, step in enumerate(g.logits):
            lg = step[0].float().cpu().numpy().astype(np.float64)
            lg = lg[np.isfinite(lg)]
            tp = turning_points(lg)
            pos.append(dict(Tmelt=tp["Tmelt"], S_vs_T=tp["S_vs_T"],
                            logS_vs_T=tp["logS_vs_T"], logS_vs_logT=tp["logS_vs_logT"]))
            if k == 0:
                lp = lg - lg.max() - np.log(np.exp(lg - lg.max()).sum())
                top = np.argsort(-lp)[:50]
                top50 = {int(i): float(lp[i]) for i in top}
        ntok += len(pos)
        out.append(dict(Tmelt=float(np.median([q["Tmelt"] for q in pos])),
                        Tmelt_prompt_end=pos[0]["Tmelt"], positions=pos, top50=top50,
                        greedy=tok.decode(g.sequences[0, ids.shape[1]:],
                                          skip_special_tokens=True)))
    res = dict(rows=out, V=int(lg.size), seconds=time.time() - t0,
               greedy_passes=len(rows), greedy_tokens=ntok)
    json.dump(res, open(path, "w"))
    del model
    if device() == "cuda":
        torch.cuda.empty_cache()
    return res


def detect_mode(gen_out, melt, prompt_idx, T):
    """Compare first-position log-prob differences against the float32 forward pass.

    Raw log probabilities reproduce the differences with slope 1, tempered ones with
    slope 1/T. The slope is saved so a reader can check which one vLLM returned.
    """
    xs, ys = [], []
    for samples, qi in zip(gen_out, prompt_idx):
        ref = melt["rows"][qi]["top50"]
        for s in samples:
            if not s["pos"]:
                continue
            got = s["pos"][0]
            common = [int(k) for k in ref if int(k) in got][:20]
            if len(common) < 3:
                continue
            r = np.array([ref[k] if k in ref else ref[str(k)] for k in common])
            g = np.array([got[k] for k in common])
            xs += list(r - r[0]); ys += list(g - g[0])
    if len(xs) < 5:
        return None, "unknown"
    xs, ys = np.array(xs), np.array(ys)
    slope = float((xs * ys).sum() / (xs * xs).sum())
    mode = "raw" if abs(slope - 1) < abs(slope - 1 / T) else "processed"
    return slope, mode


def phase_turn(a, rows, melt, gen, path):
    """TURN's entropy sweep as in predict.py: N samples per temperature, one per question."""
    if os.path.exists(path):
        return json.load(open(path))
    grid = np.round(np.arange(0.1, a.t_max, 0.1), 2)
    rng = np.random.default_rng(a.seed)
    if len(rows) > a.turn_samples:
        pick, n_each = list(rng.choice(len(rows), a.turn_samples, replace=False)), 1
    else:
        pick, n_each = list(range(len(rows))), max(1, a.turn_samples // len(rows))
    H_proc, H_raw, ntok, slope, mode = [], [], 0, None, gen.mode
    for t in grid:
        res = gen.generate([rows[i]["prompt"] for i in pick], [t] * len(pick), n_each,
                           a.max_new_tokens, a.stop, True, a.seed)
        if slope is None and abs(t - 0.5) < 1e-9:
            slope, detected = detect_mode(res, melt, pick, t)
            if mode == "unknown":
                mode = detected
            elif detected != "unknown" and detected != mode:
                print(f"WARNING: backend says {mode} log probs, slope says {detected}")
        m = mode if mode != "unknown" else "processed"
        per_q_proc, per_q_raw = [], []
        for samples in res:
            sp, sr = [], []
            for s in samples:
                ntok += s["ntok"]
                if not s["pos"]:
                    continue
                e = [position_entropies(list(d.values()), t, m) for d in s["pos"]]
                sp.append(np.mean([x[0] for x in e])); sr.append(np.mean([x[1] for x in e]))
            per_q_proc.append(np.mean(sp)); per_q_raw.append(np.mean(sr))
        H_proc.append(float(np.mean(per_q_proc))); H_raw.append(float(np.mean(per_q_raw)))
        print(f"  turn T={t:.1f}  H_tempered={H_proc[-1]:.4f}  H_raw={H_raw[-1]:.4f}",
              flush=True)
    t_star = calc_turning_point(grid, np.log(H_proc))
    t_star_raw = calc_turning_point(grid, np.log(H_raw))
    res = dict(grid=grid.tolist(), H_tempered=H_proc, H_raw=H_raw, mode=mode,
               slope_at_T05=slope, t_star=t_star, t_star_raw=t_star_raw,
               T_pred=round(t_star + BETA[a.task], 2),
               generations=len(grid) * len(pick) * n_each, generated_tokens=ntok)
    json.dump(res, open(path, "w"))
    return res


def phase_grid(a, rows, settings, gen, outdir):
    """k samples per question for every temperature setting, one file per setting."""
    for label, temps in settings.items():
        path = os.path.join(outdir, f"gen_{label}.json")
        if os.path.exists(path):
            continue
        t0 = time.time()
        res = gen.generate([r["prompt"] for r in rows], temps, a.k, a.max_new_tokens,
                           a.stop, False, a.seed)
        json.dump(dict(label=label, temps=[float(t) for t in temps],
                       texts=[[s["text"] for s in samples] for samples in res],
                       ntok=int(sum(s["ntok"] for samples in res for s in samples)),
                       seconds=time.time() - t0), open(path, "w"))
        print(f"  grid {label}: {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- scoring

def majority_correct(answers, gt, grade):
    """TURN's majority vote, MATH/cal_acc.py: cluster by grade_answer, skip no-answer."""
    counts = {}
    for ans in answers:
        if ans == "no answer":
            continue
        for key in counts:
            if grade(key, ans):
                counts[key] += 1
                break
        else:
            counts[ans] = 1
    if not counts:
        return 0.0
    final = max(counts, key=counts.get)
    return float(grade(gt, final))


def pass_at(n, c, k):
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def score_setting(a, rows, texts, grade):
    """Accuracy at each N, averaged over disjoint blocks of the k samples."""
    Ns = [N for N in (1, 2, 4, 8, 16, 32, 64, 128, 256) if N <= a.k]
    acc = {}
    if a.task == "math":
        parsed = [[parse_math(t, a.parse) for t in q] for q in texts]
        for N in Ns:
            per_q = [np.mean([majority_correct(ans[b * N:(b + 1) * N], r["answer"], grade)
                              for b in range(a.k // N)]) for ans, r in zip(parsed, rows)]
            acc[N] = float(np.mean(per_q))
    else:
        correct = [sum(mbpp_passes(r, t) for t in q) for q, r in zip(texts, rows)]
        for N in Ns:
            acc[N] = float(np.mean([pass_at(a.k, c, N) for c in correct]))
    return acc


def phase_score(a, rows, melt, turn, settings, outdir):
    grade = None
    if a.task == "math":
        sys.path.insert(0, os.path.join(a.turn_dir, "MATH"))
        from math_utils.grader import grade_answer as grade
    cache = os.path.join(outdir, "scores.json")
    scores = json.load(open(cache)) if os.path.exists(cache) else {}
    for label in settings:
        if label in scores:
            continue
        g = json.load(open(os.path.join(outdir, f"gen_{label}.json")))
        scores[label] = dict(acc=score_setting(a, rows, g["texts"], grade),
                             ntok=g["ntok"], seconds=g["seconds"])
        json.dump(scores, open(cache, "w"))
    grid_labels = [l for l in settings if l.startswith("T") and float(l[1:]) < a.t_max]
    Tm = np.array([r["Tmelt"] for r in melt["rows"]])
    P = [q for r in melt["rows"] for q in r["positions"]]
    ratio = lambda c: float(np.nanmedian([q["Tmelt"] / q[c] if q[c] else np.nan for q in P]))
    summary = dict(
        model=a.model, task=a.task, n_problems=len(rows), k=a.k,
        T_melt_median=float(np.median(Tm)), T_melt_iqr=np.percentile(Tm, [25, 75]).tolist(),
        T_melt_prompt_end_median=float(np.median([r["Tmelt_prompt_end"]
                                                  for r in melt["rows"]])),
        ratio_S_vs_T_single=ratio("S_vs_T"), ratio_logS_vs_T_single=ratio("logS_vs_T"),
        turn_t_star=turn["t_star"], turn_t_star_raw=turn["t_star_raw"],
        turn_T_pred=turn["T_pred"], logprob_mode=turn["mode"],
        logprob_slope_at_T05=turn["slope_at_T05"],
        ratio_Tmelt_over_turn_t_star=float(np.median(Tm) / turn["t_star"]),
        cost=dict(melt_greedy_passes=melt["greedy_passes"],
                  melt_generated_tokens=melt["greedy_tokens"],
                  turn_generations=turn["generations"],
                  turn_generated_tokens=turn["generated_tokens"]),
        by_N={})
    for N in scores[grid_labels[0]]["acc"]:
        grid = {l: scores[l]["acc"][N] for l in grid_labels}
        best_l = max(grid, key=grid.get)
        base = {l: v for l, v in grid.items() if float(l[1:]) in BASELINES}
        best = grid[best_l]
        row = dict(best_grid_T=float(best_l[1:]), best_grid_acc=best,
                   best_baseline_T=float(max(base, key=base.get)[1:]),
                   best_baseline_acc=max(base.values()))
        cols = dict(turn=f"T{turn['T_pred']:.1f}", melt_task="melt_task",
                    melt_question="melt_question")
        for name, label in cols.items():
            v = scores[label]["acc"][N]
            row[name] = dict(acc=v, drop=best - v, hit=bool(v >= best - EPS))
        summary["by_N"][N] = row
    summary["temperatures"] = {l: settings[l][0] if len(set(settings[l])) == 1
                               else "per question" for l in settings}
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    return summary


def report(s):
    print(f"\n{s['model']}  {s['task']}  {s['n_problems']} problems  k={s['k']}")
    print(f"T_melt median {s['T_melt_median']:.3f}  TURN t* {s['turn_t_star']:.2f} "
          f"(untempered H: {s['turn_t_star_raw']:.2f})  TURN T_pred {s['turn_T_pred']:.2f}  "
          f"log prob mode {s['logprob_mode']}")
    print(f"T_melt / TURN t* = {s['ratio_Tmelt_over_turn_t_star']:.3f}   single-distribution "
          f"ratios: S vs T {s['ratio_S_vs_T_single']:.3f}, log S vs T "
          f"{s['ratio_logS_vs_T_single']:.3f}")
    print(f"T_melt at the end of the prompt, median {s['T_melt_prompt_end_median']:.3f}")
    print(f"cost: T_melt {s['cost']['melt_greedy_passes']} greedy passes, "
          f"{s['cost']['melt_generated_tokens']} generated tokens; TURN {s['cost']['turn_generations']} generations, "
          f"{s['cost']['turn_generated_tokens']} generated tokens")
    print(f"{'N':>4} {'best grid':>14} {'best fixed':>14} {'TURN':>14} "
          f"{'T_melt/1.141':>14} {'per question':>14}")
    for N, r in s["by_N"].items():
        cells = [f"{r['best_grid_acc']:.3f}@{r['best_grid_T']:.1f}",
                 f"{r['best_baseline_acc']:.3f}@{r['best_baseline_T']:.1f}"]
        cells += [f"{r[k]['acc']:.3f} {'hit' if r[k]['hit'] else 'miss'}"
                  for k in ("turn", "melt_task", "melt_question")]
        print(f"{N:>4} " + " ".join(f"{c:>14}" for c in cells))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["math", "mbpp"], default="math")
    ap.add_argument("--turn-dir", default="TURN", help="a clone of StigLidu/TURN")
    ap.add_argument("--out", default="results_a")
    ap.add_argument("--n-problems", type=int, default=None,
                    help="default 200 for MATH and 100 for MBPP, as in TURN")
    ap.add_argument("--melt-positions", type=int, default=32,
                    help="greedy positions per question that T_melt is read at")
    ap.add_argument("--k", type=int, default=32, help="samples per question per setting")
    ap.add_argument("--t-max", type=float, default=1.5, help="grid is 0.1 to t_max - 0.1")
    ap.add_argument("--turn-samples", type=int, default=32, help="TURN predict.py default")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--dtype-gen", default="auto", help="vLLM dtype, use half on a T4")
    ap.add_argument("--tp", type=int, default=1, help="vLLM tensor parallel size")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--eager", action="store_true", help="vLLM enforce_eager")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phases", default="melt,turn,grid,score")
    ap.add_argument("--only", default="",
                    help="comma list of settings to generate, such as T0.1,T0.2 or "
                         "melt_task, so a long run can be split across sessions")
    a = ap.parse_args()
    if a.n_problems is None:
        a.n_problems = 200 if a.task == "math" else 100
    outdir = os.path.join(a.out, a.model.replace("/", "__"), a.task)
    os.makedirs(outdir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(outdir, "args.json"), "w"), indent=1)
    rows, extra = load_task(a)
    a.stop, a.parse = extra["stop"], extra.get("parse")
    phases = a.phases.split(",")

    melt = phase_melt(a, rows, os.path.join(outdir, "melt.json"))
    Tm = np.array([r["Tmelt"] for r in melt["rows"]])
    Tp = [r["Tmelt_prompt_end"] for r in melt["rows"]]
    print(f"melt: {len(rows)} questions, V={melt['V']}, T_melt median {np.median(Tm):.3f} "
          f"IQR {np.percentile(Tm, 25):.3f}-{np.percentile(Tm, 75):.3f}, so T_melt/{RATIO} = "
          f"{np.median(Tm) / RATIO:.3f}; at the prompt end {np.median(Tp):.3f}", flush=True)
    if phases == ["melt"]:
        return

    gen = None
    turn_path = os.path.join(outdir, "turn.json")
    if "turn" in phases and not os.path.exists(turn_path):
        gen = VLLM(a) if a.backend == "vllm" else HFGen(a)
    turn = phase_turn(a, rows, melt, gen, turn_path)
    print(f"turn: t*={turn['t_star']:.2f}  T_pred={turn['T_pred']:.2f}  mode={turn['mode']}  "
          f"T_melt/t* = {np.median(Tm) / turn['t_star']:.3f}", flush=True)

    grid = np.round(np.arange(0.1, a.t_max, 0.1), 2)
    settings = {f"T{t:.1f}": [float(t)] * len(rows) for t in grid}
    # TURN's prediction lands on the 0.1 grid unless beta pushes it past the top
    if f"T{turn['T_pred']:.1f}" not in settings:
        settings[f"T{turn['T_pred']:.1f}"] = [turn["T_pred"]] * len(rows)
    settings["melt_task"] = [round(float(np.median(Tm)) / RATIO, 3)] * len(rows)
    settings["melt_question"] = [round(float(t) / RATIO, 3) for t in Tm]
    if "grid" in phases:
        todo = settings
        if a.only:
            keep = set(a.only.split(","))
            unknown = keep - set(settings)
            if unknown:
                sys.exit(f"unknown settings in --only: {sorted(unknown)}; "
                         f"choose from {list(settings)}")
            todo = {k: v for k, v in settings.items() if k in keep}
        todo = {k: v for k, v in todo.items()
                if not os.path.exists(os.path.join(outdir, f"gen_{k}.json"))}
        if todo and gen is None:
            gen = VLLM(a) if a.backend == "vllm" else HFGen(a)
        phase_grid(a, rows, todo, gen, outdir)
    if "score" in phases:
        missing = [l for l in settings
                   if not os.path.exists(os.path.join(outdir, f"gen_{l}.json"))]
        if missing:
            print(f"not scoring yet, settings still to generate: {missing}")
            return
        report(phase_score(a, rows, melt, turn, settings, outdir))


if __name__ == "__main__":
    main()
