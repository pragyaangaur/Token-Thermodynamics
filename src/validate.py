import numpy as np, sys
sys.path.insert(0,'src')
from thermo import curves, schottky, beta_grid, find_peaks

print("== 1. thermodynamic identities on a random 5000-token logit vector ==")
rng = np.random.default_rng(0)
lg = rng.normal(0, 4, 5000)
b = beta_grid(3000, 1e-3, 1e3)
c = curves(lg, b)
# F = U - T S
err = np.max(np.abs(c['F'] - (c['U'] - c['T']*c['S'])))
print(f"  max |F - (U - T S)|            = {err:.3e}")
# C = dU/dT  (finite difference on the T grid, T descending -> reorder)
o = np.argsort(c['T']); T = c['T'][o]; U = c['U'][o]; C = c['C'][o]
dUdT = np.gradient(U, T)
m = (T > 1e-2) & (T < 1e2)
rel = np.max(np.abs(dUdT[m]-C[m])/(np.abs(C[m])+1e-12))
print(f"  max rel err |dU/dT - C|        = {rel:.3e}")
# dS/dT = C/T
dSdT = np.gradient(c['S'][o], T)
rel2 = np.max(np.abs(dSdT[m]-C[m]/T[m])/(np.abs(C[m]/T[m])+1e-12))
print(f"  max rel err |dS/dT - C/T|      = {rel2:.3e}")
# limits
print(f"  S(T->inf)={c['S'][0]:.6f}  vs log(V)={np.log(5000):.6f}")
print(f"  S(T->0)  ={c['S'][-1]:.3e}  (nondegenerate ground state -> 0)")
print(f"  C at both ends: {c['C'][0]:.3e}, {c['C'][-1]:.3e}")

print("\n== 2. two-level system reproduces the analytic Schottky anomaly ==")
for gap,g0,g1 in [(1.0,1,1),(3.0,1,1),(1.0,1,10),(1.0,1,100)]:
    lg = np.array([0.0]*g0 + [-gap]*g1)      # E = 0 (x g0) and E = gap (x g1)
    c = curves(lg, b)
    ana = schottky(c['T'], gap, g0, g1)
    rel = np.max(np.abs(c['C']-ana))/np.max(ana)
    pk = find_peaks(c['T'], c['C'])
    Tstar = pk[0]['T'] if pk else float('nan')
    print(f"  gap={gap:<4} g0={g0:<4} g1={g1:<4} maxerr={rel:.2e}  Tpeak={Tstar:.4f}  Tpeak/gap={Tstar/gap:.4f}  Cmax={max(pk[0]['C'] for _ in [0]) if pk else 0:.4f}")

print("\n== 3. peak position reads off the gap; peak height reads off degeneracy ==")
print("  (a) vary gap, g1/g0 = 1")
for gap in [0.5,1,2,4,8,16]:
    c = curves(np.array([0.0,-gap]), b); pk = find_peaks(c['T'], c['C'])
    print(f"     gap={gap:<5} Tpeak={pk[0]['T']:.4f}  ratio={pk[0]['T']/gap:.4f}  Cmax={pk[0]['C']:.4f}")
print("  (b) fix gap=4, vary number of degenerate excited states g1")
for g1 in [1,2,5,20,100,1000]:
    lg = np.array([0.0]+[-4.0]*g1); c = curves(lg, b); pk = find_peaks(c['T'], c['C'])
    print(f"     g1={g1:<6} Tpeak={pk[0]['T']:.4f}  Cmax={pk[0]['C']:.4f}  S(T=1)={curves(lg,np.array([1.0]))['S'][0]:.4f}")
