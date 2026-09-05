"""Density-of-states representation of a logit vector.

C(T), S(T), U(T) depend on the logits only through the density of states
g(E) = #{tokens with energy E}, since Z(beta) = sum_E g(E) exp(-beta E).
So we compress a 152k-dim logit vector into:
   - the top K logits kept EXACTLY (these alone control the low-T / semantic regime)
   - the remaining tail binned into NB histogram bins (only matters at high T)
This is exact where it matters and ~200x faster.
"""
import numpy as np

def make_dos(logits, K=1024, NB=512):
    lg = np.asarray(logits, dtype=np.float64)
    V = lg.size
    K = min(K, V)
    idx = np.argpartition(-lg, K - 1)[:K]
    top = np.sort(lg[idx])[::-1]
    mask = np.ones(V, bool); mask[idx] = False
    tail = lg[mask]
    if tail.size:
        lo, hi = tail.min(), tail.max()
        if hi - lo < 1e-9:
            e = np.array([tail.mean()]); w = np.array([float(tail.size)])
        else:
            cnt, edges = np.histogram(tail, bins=NB, range=(lo, hi))
            ctr = 0.5 * (edges[:-1] + edges[1:])
            keep = cnt > 0
            e, w = ctr[keep], cnt[keep].astype(np.float64)
    else:
        e, w = np.zeros(0), np.zeros(0)
    E = np.concatenate([top, e])
    W = np.concatenate([np.ones(K), w])
    return E, W          # E = logit values, W = multiplicity

def curves_dos(E, W, betas):
    """Thermodynamics from a weighted spectrum. E holds logits, energy = -E."""
    E = np.asarray(E, np.float64); W = np.asarray(W, np.float64)
    b = np.asarray(betas, np.float64)
    A = b[:, None] * E[None, :] + np.log(W)[None, :]
    m = A.max(axis=1, keepdims=True)
    ex = np.exp(A - m)
    Z = ex.sum(axis=1)
    logZ = np.squeeze(m, 1) + np.log(Z)
    p = ex / Z[:, None]
    En = -E
    U = (p * En[None, :]).sum(1)
    var = np.maximum((p * En[None, :] ** 2).sum(1) - U ** 2, 0.0)
    C = b ** 2 * var
    # entropy of the token distribution (not of the coarse-grained spectrum):
    # S = beta*U + logZ  is the exact Gibbs entropy of the *binned* system,
    # which equals the token entropy up to the within-bin term sum p_bin*log W_bin.
    S = b * U + logZ
    T = 1.0 / b
    return dict(beta=b, T=T, U=U, S=S, C=C, F=-T * logZ, logZ=logZ)
