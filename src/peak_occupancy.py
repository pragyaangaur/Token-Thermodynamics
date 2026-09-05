"""Test the top-token occupancy implied by the ideal melting model.

The two-level approximation makes an exact prediction beyond T_melt:

    p_top(T_melt) = 1/2 + 1/x*
    C_max = ((x*)**2 - 4) / 4

This script compares the first law with real next-token distributions. It uses
the same models and prompts as ``multimodel_melt.py`` and requires the model
dependencies. Pass ``--local-only`` to forbid downloads.
"""
import argparse
import gc
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from multimodel_melt import MODELS, PROMPTS, Tmelt_of, xstar


def ideal_laws(vocab_size):
    x = xstar(vocab_size)
    return x, 0.5 + 1.0 / x, (x * x - 4.0) / 4.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/peak_occupancy.json")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--models", nargs="*", default=MODELS)
    args = parser.parse_args()

    output = []
    for model_name in args.models:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_only)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=args.local_only, dtype=torch.float32
        ).eval()
        vocab_size = model.get_output_embeddings().weight.shape[0]
        x, ideal_p, ideal_c = ideal_laws(vocab_size)
        rows = []
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                logits = model(ids).logits[0, -1].numpy().astype(np.float64)
            logits = logits[np.isfinite(logits)]
            t_melt, c_max = Tmelt_of(logits)
            relative = (logits - logits.max()) / t_melt
            p_top = float(1.0 / np.exp(relative).sum())
            rows.append({"prompt": prompt, "T_melt": t_melt, "C_max": c_max, "p_top_at_melt": p_top})

        measured = np.array([row["p_top_at_melt"] for row in rows])
        record = {
            "model": model_name,
            "V": int(vocab_size),
            "xstar": float(x),
            "ideal_p_top_at_melt": float(ideal_p),
            "ideal_C_max": float(ideal_c),
            "measured_p_top_median": float(np.median(measured)),
            "measured_p_top_mean": float(np.mean(measured)),
            "measured_p_top_q25": float(np.percentile(measured, 25)),
            "measured_p_top_q75": float(np.percentile(measured, 75)),
            "rows": rows,
        }
        output.append(record)
        print(
            f"{model_name:35s} V={vocab_size:7d} "
            f"p_top(T_melt)={record['measured_p_top_median']:.4f} "
            f"ideal={ideal_p:.4f}"
        )
        del model, tokenizer
        gc.collect()

    with open(args.out, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
