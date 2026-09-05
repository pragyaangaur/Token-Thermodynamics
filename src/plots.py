import sys, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))

C_OK, C_BAD, C_FAKE = "#2b6cb0", "#c53030", "#6b46c1"
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})

def band(ax, T, curves, color, label):
    A = np.array(curves)
    m = A.mean(0); lo = np.percentile(A, 25, 0); hi = np.percentile(A, 75, 0)
    ax.plot(T, m, color=color, lw=1.8, label=f"{label} (n={len(A)})")
    ax.fill_between(T, lo, hi, color=color, alpha=.15, lw=0)

def main(pkl, out="figs"):
    R = pickle.load(open(pkl, "rb"))
    os.makedirs(out, exist_ok=True)
    real = [r for r in R if not r.get("fake")]
    fake = [r for r in R if r.get("fake")]
    ok = [r for r in real if r["correct"]]; bad = [r for r in real if not r["correct"]]
    T = real[0]["_T"]
    groups = [("correct", ok, C_OK), ("incorrect", bad, C_BAD)] + ([("fabricated entity", fake, C_FAKE)] if fake else [])

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    ax = axes[0]
    for lab, g, c in groups:
        if g: band(ax, T, [r["_C"] for r in g], c, lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-5, 60)
    ax.set_xlabel("temperature T"); ax.set_ylabel("heat capacity  C(T)")
    ax.set_title("raw spectra"); ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.axvline(1.0, color="k", ls=":", lw=.8)
    ax.text(1.15, 2e-5, "T=1\nentropy and varentropy\nare read here", fontsize=6)

    ax = axes[1]
    grid = np.linspace(-4, 2, 24)
    for lab, g, c in groups:
        if not g: continue
        A = np.array([r["_shape"] for r in g])
        ax.plot(grid, A.mean(0), color=c, lw=1.8, label=lab)
        ax.fill_between(grid, np.percentile(A, 25, 0), np.percentile(A, 75, 0), color=c, alpha=.15, lw=0)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 2); ax.set_xlabel("log(T / T_melt)"); ax.set_ylabel("C / C_max")
    ax.set_title("scale-free shape\n(logit scale divided out)"); ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    for lab, g, c in groups:
        if not g: continue
        v = np.log10([r["t_Tmain"] for r in g])
        ax.hist(v, bins=45, histtype="step", color=c, lw=1.5, density=True, label=lab)
    ax.set_xlabel("log10 melting temperature  T_melt"); ax.set_ylabel("density")
    ax.set_title("per-token melting point\n(a global T ignores this spread)")
    ax.axvline(0, color="k", ls=":", lw=.8); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f"{out}/fig1_spectra.png"); print("wrote", f"{out}/fig1_spectra.png")

    # entropy release curve dS/dlogT  -- where in temperature does uncertainty live
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ax = axes[0]
    for lab, g, c in groups:
        if not g: continue
        A = np.array([r["_C"] for r in g])          # dS/dlogT = C
        band(ax, T, list(A), c, lab)
    ax.set_xscale("log"); ax.set_xlabel("T"); ax.set_ylabel("dS/dlog T  = C(T)")
    ax.set_title("where uncertainty is released"); ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    for lab, g, c in groups:
        if not g: continue
        A = np.array([r["_S"] for r in g])
        band(ax, T, list(A), c, lab)
    ax.set_xscale("log"); ax.set_xlabel("T"); ax.set_ylabel("S(T)  (nats)")
    ax.axvline(1.0, color="k", ls=":", lw=.8); ax.set_title("entropy curve; standard entropy = the value at T=1")
    fig.tight_layout(); fig.savefig(f"{out}/fig2_entropy.png"); print("wrote", f"{out}/fig2_entropy.png")

    # examples
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    picks = []
    for g, c, lab in [(ok, C_OK, "correct"), (bad, C_BAD, "incorrect"), (fake, C_FAKE, "fabricated")]:
        if not g: continue
        gg = sorted(g, key=lambda r: -r["t_npeaks"])[:1] + g[:2]
        picks.append((gg[0], c, lab))
    for ax, (r, c, lab) in zip(axes, picks):
        ax.plot(T, r["_C"], color=c, lw=1.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{lab}: {r['subject'][:32]}\nsaid '{r['pred'][:28]}' / gold '{str(r['gold'])[:20]}'", fontsize=7)
        ax.set_xlabel("T"); ax.set_ylabel("C(T)")
    fig.tight_layout(); fig.savefig(f"{out}/fig3_examples.png"); print("wrote", f"{out}/fig3_examples.png")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "figs")
