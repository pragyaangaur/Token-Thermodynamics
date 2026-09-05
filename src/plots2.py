import sys, os, pickle
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from thermo import curves, beta_grid
from analyze import BASE, SHAPE, SCALE, mat, cv_auc
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,"grid.alpha":.25,
                     "axes.spines.top":False,"axes.spines.right":False})
OK,BAD,FAKE,ACC="#2b6cb0","#c53030","#6b46c1","#2f855a"
NORM=["n_S_at_melt","n_C_at_melt"]+[f"n_S_{q}melt" for q in (25,50,75)]+[f"n_C_{q}melt" for q in (25,50,75)]

fig,ax=plt.subplots(1,3,figsize=(12.5,3.5))

# --- panel 1: scale invariance ---
b=beta_grid(600,1e-3,1e3); rng=np.random.default_rng(0)
lg=np.concatenate([rng.normal(6,1,4),rng.normal(-9,2,20000)])
ss=np.exp(np.linspace(np.log(0.25),np.log(4),40))
S1,Sm,Cm,Tm=[],[],[],[]
for s in ss:
    c=curves(lg/s,b); o=np.argsort(c['T']); T,C,S=c['T'][o],c['C'][o],c['S'][o]
    tm=T[int(np.argmax(C))]
    S1.append(np.interp(1.0,T,S)); Sm.append(np.interp(tm/2,T,S)); Cm.append(C.max()); Tm.append(tm)
a=ax[0]
a.plot(ss,np.array(S1)/S1[len(ss)//2],color=BAD,lw=2,label="entropy $S(T{=}1)$")
a.plot(ss,np.array(Sm)/Sm[len(ss)//2],color=OK,lw=2,label="$S(T_{melt}/2)$")
a.plot(ss,np.array(Cm)/Cm[len(ss)//2],color=ACC,lw=2,ls="--",label="$C_{max}$")
a.set_xscale("log"); a.set_yscale("log")
a.set_xticks([0.25,0.5,1,2,4]); a.set_xticklabels(["0.25","0.5","1","2","4"])
a.set_yticks([0.5,1,2,4,8]); a.set_yticklabels(["0.5","1","2","4","8"])
a.set_xlabel("logit rescaling factor  $1/s$"); a.set_ylabel("value / value at $s{=}1$")
a.set_title("scale invariance\n(flat line = invariant)"); a.legend(fontsize=7,frameon=False)
a.axhline(1,color="k",lw=.6,ls=":")

# --- panel 2: detector AUROC on the obscurity control ---
R=[r for r in pickle.load(open("data/records_both15.pkl","rb")) if not r.get("fake")]
L=np.array([r["links"] for r in R]); lo,hi=np.percentile(L,25),np.percentile(L,75)
allr=[r for r in R if r["links"]>=hi]+[r for r in R if r["links"]<=lo]
y=np.array([0]*len([r for r in R if r["links"]>=hi])+[1]*len([r for r in R if r["links"]<=lo]))
names,vals,errs,cols=[],[],[],[]
for nm,cs,sc,c in [("entropy $S(1)$",["b_entropy"],False,BAD),("all 7 baselines",BASE,False,BAD),
                   ("melt-normalised",NORM,False,OK),("thermo shape+scale",SHAPE+SCALE,False,OK),
                   ("all features",BASE+SHAPE+SCALE+NORM,True,ACC)]:
    m,s=cv_auc(mat(allr,cs,sc),y); names.append(nm); vals.append(m); errs.append(s); cols.append(c)
a=ax[1]
a.barh(range(len(names)),vals,xerr=errs,color=cols,alpha=.85,height=.6)
a.set_yticks(range(len(names))); a.set_yticklabels(names,fontsize=7.5); a.invert_yaxis()
a.set_xlim(0.70,0.90); a.set_xlabel("AUROC"); a.set_title("does the model know this entity?\n(real names only, obscurity control)")
for i,v in enumerate(vals): a.text(v+0.004,i,f"{v:.3f}",va="center",fontsize=7)

# --- panel 3: robustness under per-item rescaling ---
sig=[0.00,0.15,0.30,0.50]
base15=[0.8653,0.8424,0.7771,0.6875]; aug=[np.nan,0.8560,0.8424,0.8184]
smelt=[0.8706]*4; shape=[0.8761,0.8755,0.8747,0.8759]
a=ax[2]
a.plot(sig,base15,"o-",color=BAD,lw=2,label="standard baselines")
a.plot(sig,aug,"s--",color="#dd6b20",lw=1.6,label="baselines + augmentation")
a.plot(sig,smelt,"^-",color=OK,lw=2,label="$S(T_{melt})$")
a.plot(sig,shape,"d-",color=ACC,lw=2,label="thermo shape")
a.axvline(0.133,color="k",ls=":",lw=1)
a.text(0.145,0.735,"measured\nscale variation\nin this model",fontsize=6.5,ha="left")
a.set_ylim(0.66,0.895)
a.set_xlabel("per-item logit rescaling  $\\sigma$"); a.set_ylabel("AUROC")
a.set_title("robustness to unknown logit scale"); a.legend(fontsize=6.5,frameon=False,loc="center left",bbox_to_anchor=(0.0,0.30))
fig.tight_layout(); fig.savefig("figs/fig4_main.png"); print("wrote figs/fig4_main.png")
