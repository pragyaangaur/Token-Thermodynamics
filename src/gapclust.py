"""Single-pass estimate of semantic entropy by energy-gap clustering.

Physics assumption being tested: tokens that are near-degenerate in energy are
surface variants of the same meaning, and a real change of meaning costs a
finite energy gap. So cut the top of the spectrum wherever the gap between
consecutive levels exceeds eps, and take the entropy over the resulting groups.
The threshold is expressed in units of the melting temperature, so the estimator
does not depend on the model's logit scale.
"""
import numpy as np

def gap_entropy(top_logits, logZ, Tmelt, eps):
    """top_logits descending; logZ = log partition function at beta = 1."""
    l = np.asarray(top_logits, float)
    p = np.exp(l - logZ)
    thr = eps * Tmelt
    cuts = np.where(np.diff(l) < -thr)[0] + 1
    groups = np.split(p, cuts)
    q = np.array([g.sum() for g in groups])
    rest = max(1.0 - q.sum(), 0.0)          # everything outside the kept top-k
    if rest > 1e-12:
        q = np.append(q, rest)
    q = q / q.sum()
    q = q[q > 0]
    return float(-(q * np.log(q)).sum()), len(groups)

def features(top_logits, logZ, Tmelt, epss=(0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8)):
    out = {}
    for e in epss:
        S, n = gap_entropy(top_logits, logZ, Tmelt, e)
        out[f"g_S_{e}"] = S
        out[f"g_n_{e}"] = float(n)
    return out
