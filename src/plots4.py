import sys, os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,"grid.alpha":.25,
                     "axes.spines.top":False,"axes.spines.right":False})
OK,BAD,ACC,PUR,ORG="#2b6cb0","#c53030","#2f855a","#6b46c1","#b7791f"
D=json.load(open("data/multimodel_melt3.json"))
SHORT={"gpt2":"GPT-2","EleutherAI/pythia-160m":"Pythia-160M","HuggingFaceTB/SmolLM2-360M":"SmolLM2-360M",
       "facebook/opt-125m":"OPT-125M","Qwen/Qwen2.5-0.5B-Instruct":"Qwen2.5-0.5B",
       "Qwen/Qwen2.5-1.5B-Instruct":"Qwen2.5-1.5B","bigscience/bloomz-560m":"BLOOMZ-560M"}
fig,ax=plt.subplots(1,3,figsize=(12.6,3.6))

# panel 1: predicted vs measured T_melt
a=ax[0]
T=np.array([np.median([r["Tmelt"] for r in o["rows"]]) for o in D])
P=np.array([np.median([r["delta_mode"] for r in o["rows"]])/o["xstar"] for o in D])
V=np.array([o["V"] for o in D])
sc=a.scatter(P,T,c=np.log10(V),cmap="viridis",s=95,zorder=3,edgecolor="w",linewidth=.8)
lim=[min(P.min(),T.min())*.9, max(P.max(),T.max())*1.1]
a.plot(lim,lim,"k:",lw=1,label="$T_{melt}=\\Delta/x^*$")
k=np.sum(T*P)/np.sum(P*P)
a.plot(lim,[k*l for l in lim],color=ACC,lw=1.6,label=f"fit, slope {k:.3f}")
for p,t,o in zip(P,T,D):
    a.annotate(SHORT[o["model"]],(p,t),fontsize=6,xytext=(4,-9),textcoords="offset points")
a.set_xlabel("predicted  $\\Delta / x^*(V)$"); a.set_ylabel("measured  $T_{melt}$")
a.set_title("melting law across 7 model families"); a.legend(fontsize=7,frameon=False,loc="upper left")
cb=fig.colorbar(sc,ax=a,pad=.02); cb.set_label("$\\log_{10} V$",fontsize=7); cb.ax.tick_params(labelsize=6)

# panel 2: the Delta estimator lesson
a=ax[1]
ests=["mean","median","mode"]
vals={e:[] for e in ests}
for o in D:
    R=o["rows"]; x=o["xstar"]; t=np.median([r["Tmelt"] for r in R])
    vals["mean"].append(t*x/np.median([r["delta_mean"] for r in R]))
    vals["median"].append(t*x/np.median([r["delta_bulk"] for r in R]))
    vals["mode"].append(t*x/np.median([r["delta_mode"] for r in R]))
for i,(e,c) in enumerate(zip(ests,[BAD,ORG,ACC])):
    v=np.array(vals[e]); jit=(np.arange(len(v))-len(v)/2)*0.035
    a.scatter(np.full(len(v),i)+jit,v,color=c,s=55,zorder=3,edgecolor="w",linewidth=.6)
    a.plot([i-.28,i+.28],[v.mean()]*2,color=c,lw=2)
    a.text(i,1.55,f"spread\n{v.max()/v.min():.2f}x",ha="center",fontsize=7,color=c)
a.axhline(1.0,color="k",ls=":",lw=1)
a.set_yscale("log"); a.set_ylim(0.06,2.4)
a.set_xticks(range(3)); a.set_xticklabels([f"$\\Delta$ to {e}" for e in ests],fontsize=8)
a.set_ylabel("$T_{melt}\\,x^*/\\Delta$"); a.set_title("how you measure $\\Delta$ decides\nwhether the law 'holds'")

# panel 3: normalised sharpness
a=ax[2]
names=[SHORT[o["model"]] for o in D]
sharp=np.array([np.median([r["Cmax"] for r in o["rows"]])/o["Cmax_ceiling"] for o in D])
order=np.argsort(sharp)
a.barh(range(len(D)),sharp[order],color=OK,alpha=.8,height=.6)
a.set_yticks(range(len(D))); a.set_yticklabels([names[i] for i in order],fontsize=7)
a.set_xlabel("$4C_{max}/(\\log V)^2$   (1 = ideal sharp melt)")
a.set_title("normalised melting sharpness")
a.set_xlim(0,0.42)
for i,v in enumerate(sharp[order]): a.text(v+.006,i,f"{v:.3f}",va="center",fontsize=6.5)
fig.tight_layout(); fig.savefig("figs/fig6_crossmodel.png"); print("wrote figs/fig6_crossmodel.png")
