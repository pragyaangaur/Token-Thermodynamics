import numpy as np, sys
sys.path.insert(0,'src')
from thermo import curves, beta_grid, find_peaks
from scipy.optimize import brentq

b = beta_grid(4000, 1e-3, 1e3)
def S1(lg): return curves(lg, np.array([1.0]))['S'][0]

print("== fixed derivative check (absolute error, scaled by max C) ==")
rng=np.random.default_rng(0); lg=rng.normal(0,4,5000)
c=curves(lg,b); o=np.argsort(c['T']); T=c['T'][o]
dUdT=np.gradient(c['U'][o],T); C=c['C'][o]
m=(T>3e-2)&(T<3e1)
print(f"  max |dU/dT - C| / max(C) = {np.max(np.abs(dUdT[m]-C[m]))/C.max():.3e}   (grid discretisation only)")

print("\n== THE KEY CONSTRUCTION: identical entropy, different band structure ==")
target = 1.2   # nats at T=1

# A: 'discrete rival'  -> ground state + a SMALL number of close competitors
def A(gap, g1=3):
    return np.array([0.0] + [-gap]*g1)
gapA = brentq(lambda g: S1(A(g)) - target, 1e-4, 50)
lgA = A(gapA)

# B: 'confabulation continuum' -> ground state + MANY states spread over a band
def B(width, n=400):
    return np.array([0.0] + list(-np.linspace(0.05*width, width, n)))
wB = brentq(lambda w: S1(B(w)) - target, 1e-3, 200)
lgB = B(wB)

# C: 'far heavy tail' -> ground state + huge degenerate set at a large gap
def Cc(gap, g1=20000):
    return np.array([0.0] + [-gap]*g1)
gapC = brentq(lambda g: S1(Cc(g)) - target, 1e-4, 60)
lgC = Cc(gapC)

for name, lg, par in [("A discrete-rival ", lgA, f"gap={gapA:.3f}, g1=3"),
                      ("B continuum-band ", lgB, f"width={wB:.3f}, n=400"),
                      ("C far-degenerate ", lgC, f"gap={gapC:.3f}, g1=20000")]:
    cc = curves(lg, b); pk = find_peaks(cc['T'], cc['C'])
    # top-2 logit margin
    s = np.sort(lg)[::-1]; margin = s[0]-s[1]
    peaks = ", ".join(f"(T={p['T']:.3f},C={p['C']:.3f})" for p in pk[:3])
    print(f"  {name} S(1)={S1(lg):.4f}  margin={margin:.3f}  Cmax={cc['C'].max():.3f}  npeaks={len(pk)}  peaks: {peaks}")
    print(f"      {par}")

print("\n  -> Shannon entropy at T=1 is IDENTICAL (1.2000 nats) for all three.")
print("  -> heat-capacity spectra are completely different.")
print("  -> so C(T) is not a re-labelling of entropy; it separates cases entropy merges.")

print("\n== how well do the standard scalars separate these? ==")
for name, lg in [("A",lgA),("B",lgB),("C",lgC)]:
    p = np.exp(lg - np.log(np.sum(np.exp(lg))))
    s = np.sort(p)[::-1]
    print(f"  {name}: p_max={s[0]:.4f}  top2margin_prob={s[0]-s[1]:.4f}  S={-(p*np.log(p)).sum():.4f}  perplexity={np.exp(-(p*np.log(p)).sum()):.3f}")
