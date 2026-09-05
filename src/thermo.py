"""
Thermodynamics of a next-token distribution.

Mapping (exact, not an analogy):
    energy of token i    E_i = -logit_i
    Boltzmann weight     p_i(T) = exp(-E_i/T) / Z(T)          == softmax(logit/T)
    partition function   Z(T)  = sum_i exp(-E_i/T)
    free energy          F(T)  = -T log Z(T)
    internal energy      U(T)  = <E>_p
    entropy              S(T)  = -sum p log p                  (nats)
    heat capacity        C(T)  = dU/dT = Var_p(E) / T^2

Z(beta) = sum_E g(E) exp(-beta E) is the Laplace transform of the density of
states g(E). So sweeping temperature is a spectroscopy of the model's candidate
set: peaks in C(T) mark energy gaps between distinct bands of candidates.
"""
import numpy as np

LOG2 = np.log(2.0)


def _logsumexp(x, axis=-1, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def curves(logits, betas):
    """Full thermodynamic curves for one logit vector over an inverse-temperature grid.

    logits : (V,) float64
    betas  : (B,) float64, beta = 1/T, ascending
    returns dict of (B,) arrays
    """
    logits = np.asarray(logits, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    E = -logits                                    # (V,)
    A = -betas[:, None] * E[None, :]               # (B,V) = beta*logit
    logZ = _logsumexp(A, axis=1)                   # (B,)
    logp = A - logZ[:, None]
    p = np.exp(logp)
    U = np.sum(p * E[None, :], axis=1)
    E2 = np.sum(p * (E[None, :] ** 2), axis=1)
    var = np.maximum(E2 - U ** 2, 0.0)
    C = (betas ** 2) * var                         # C = Var(E)/T^2
    S = -np.sum(p * logp, axis=1)                  # nats
    T = 1.0 / betas
    F = -T * logZ
    return dict(beta=betas, T=T, logZ=logZ, U=U, S=S, F=F, C=C, var=var)


def schottky(T, gap, g0=1.0, g1=1.0):
    """Analytic heat capacity of a two-level system, for validation."""
    T = np.asarray(T, dtype=np.float64)
    x = gap / T
    w = (g1 / g0) * np.exp(-x)
    return (x ** 2) * w / (1.0 + w) ** 2


def beta_grid(n=400, lo=1e-3, hi=1e3):
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))


def find_peaks(T, C, min_ratio=1.15, floor_frac=1e-9):
    """Local maxima of C found in LOG space with ratio-based prominence.

    C(T) for a real vocabulary spans many orders of magnitude: a huge 'melting'
    peak from the ~10^5 bulk tokens sits far above the small Schottky feature
    from the handful of semantically live candidates. This is the same problem as
    measuring an impurity Schottky anomaly against a phonon background, so the
    peak search has to be scale-free: a peak counts if it exceeds BOTH flanking
    minima by at least `min_ratio`.
    """
    C = np.asarray(C, dtype=np.float64)
    n = len(C)
    if n < 3 or not np.any(C > 0):
        return []
    Cmax = C.max()
    floor = Cmax * floor_frac
    L = np.log(np.maximum(C, floor))
    out = []
    for i in range(1, n - 1):
        if not (L[i] > L[i - 1] and L[i] >= L[i + 1]):
            continue
        if C[i] <= floor * 10:
            continue
        l = i
        while l > 0 and L[l - 1] <= L[l]:
            l -= 1
        r = i
        while r < n - 1 and L[r + 1] <= L[r]:
            r += 1
        ratio = np.exp(L[i] - max(L[l], L[r]))
        if ratio < min_ratio:
            continue
        out.append(dict(T=float(T[i]), C=float(C[i]), ratio=float(ratio), i=i))
    out.sort(key=lambda d: -d["C"])
    return out
