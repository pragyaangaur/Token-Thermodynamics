import sys, os, json, pickle
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,"grid.alpha":.25,
                     "axes.spines.top":False,"axes.spines.right":False})
OK,BAD,ACC,PUR="#2b6cb0","#c53030","#2f855a","#6b46c1"
KS=[8,16,32,64,128,512,2048,8192,32768,151936]
D=json.load(open("data/meltlaw_1p5b.json"))
fig,ax=plt.subplots(1,3,figsize=(12.5,3.5))

a=ax[0]
for r in D:
    a.plot(KS,[r[f"T_{k}"] for k in KS],lw=.9,alpha=.55,color="#718096")
a.plot(KS,[np.mean([r[f"T_{k}"] for r in D]) for k in KS],"o-",color=OK,lw=2.2,label="mean")
a.set_xscale("log"); a.set_xlabel("effective vocabulary $V_{eff}$ (top-k truncation)")
a.set_ylabel("$T_{melt}$"); a.set_title("melting point vs effective vocabulary\n(10 prompts)")
a.legend(fontsize=7,frameon=False)

a=ax[1]
rat=[np.mean([r[f"D_{k}"] for r in D])/(np.mean([r[f"T_{k}"] for r in D])*np.log(k)) for k in KS]
a.plot(KS,rat,"o-",color=ACC,lw=2)
a.axhline(1.0,color="k",ls=":",lw=1)
a.set_xscale("log"); a.set_ylim(0,1.8)
a.set_xlabel("$V_{eff}$"); a.set_ylabel(r"$\Delta \,/\, (T_{melt}\log V_{eff})$")
a.set_title("the law $T_{melt}\\log V=\\Delta$\n(flat = law holds)")
a.text(20,0.25,"$V_{eff}$ varies 19000x\n$\\Delta$ varies 8.7x\nratio varies 1.3x",fontsize=7)

a=ax[2]
M=json.load(open("data/melt_1p5b.json"))
labs={"code":"Python code","facts":"simple facts","prose":"technical prose","openended":"open-ended prose"}
cols={"code":PUR,"facts":ACC,"prose":OK,"openended":BAD}
data=[np.array([r["Tmelt"] for r in M[k]]) for k in labs]
bp=a.boxplot(data,vert=True,widths=.55,patch_artist=True,showfliers=False,
             medianprops=dict(color="k",lw=1.4))
for patch,k in zip(bp["boxes"],labs): patch.set_facecolor(cols[k]); patch.set_alpha(.55)
a.set_xticklabels([labs[k] for k in labs],fontsize=7,rotation=12)
a.set_ylabel("$T_{melt}$"); a.set_title("melting point across ordinary text\n(472 token positions)")
a.axhline(1.0,color="k",ls=":",lw=1); a.text(0.6,1.03,"$T=1$",fontsize=6.5)
fig.tight_layout(); fig.savefig("figs/fig5_meltlaw.png"); print("wrote figs/fig5_meltlaw.png")
