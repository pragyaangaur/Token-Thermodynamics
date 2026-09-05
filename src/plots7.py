import sys, os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,"grid.alpha":.25,
                     "axes.spines.top":False,"axes.spines.right":False})
OK,BAD,ACC,PUR,GREY="#2b6cb0","#c53030","#2f855a","#6b46c1","#718096"
D=json.load(open("data/ffh.json"))
fig,ax=plt.subplots(1,3,figsize=(12.6,3.6))
CONDS=["KNOWN","OBSCURE","FABRICATED"]; COLS=[ACC,"#b7791f",BAD]

a=ax[0]
w=0.36; x=np.arange(3)
se=[np.mean([r["sem_entropy"] for r in D[c]]) for c in CONDS]
th=[np.mean([sum(r["tok_H"]) for r in D[c]]) for c in CONDS]
a.bar(x-w/2,se,w,color=OK,label="semantic entropy (meaning)")
a.bar(x+w/2,th,w,color=BAD,alpha=.85,label="token entropy (first 3 tokens)")
for i,(v1,v2) in enumerate(zip(se,th)):
    a.text(i-w/2,v1+.03,f"{v1:.2f}",ha="center",fontsize=7)
    a.text(i+w/2,v2+.03,f"{v2:.2f}",ha="center",fontsize=7)
a.set_xticks(x); a.set_xticklabels(["known","obscure\n(real)","fabricated"],fontsize=8)
a.set_ylabel("nats"); a.set_title("meaning uncertainty rises with fabrication.\ntoken entropy does not even go the right way")
a.legend(fontsize=7,frameon=False,loc="upper left"); a.set_ylim(0,1.85)

a=ax[1]
meth=[("sampled semantic\nentropy (704 tok)",lambda r:r["sem_entropy"],OK),
      ("fork pos 0-1\n(448 tok)",lambda r:sum(r["fork_SE"][:2]),ACC),
      ("fork pos 0 only\n(224 tok)",lambda r:r["fork_SE"][0],PUR),
      ("token entropy\n(1 pass, 0 tok)",lambda r:sum(r["tok_H"]),BAD)]
from sklearn.metrics import roc_auc_score
def auc(A,B,f):
    y=np.array([0]*len(D[A])+[1]*len(D[B])); v=np.array([f(r) for r in D[A]]+[f(r) for r in D[B]])
    s=roc_auc_score(y,v); return max(s,1-s)
names=[m[0] for m in meth]
kf=[auc("KNOWN","FABRICATED",m[1]) for m in meth]
ko=[auc("KNOWN","OBSCURE",m[1]) for m in meth]
yy=np.arange(len(meth))
a.barh(yy-0.19,kf,0.36,color=[m[2] for m in meth],alpha=.9,label="known vs fabricated")
a.barh(yy+0.19,ko,0.36,color=[m[2] for m in meth],alpha=.45,label="known vs obscure (no confound)")
a.axvline(0.5,color=GREY,ls=":",lw=1)
a.set_yticks(yy); a.set_yticklabels(names,fontsize=7); a.invert_yaxis()
a.set_xlim(0.45,1.03); a.set_xlabel("AUROC")
a.set_title("free-form hallucination detection")
for i,(v1,v2) in enumerate(zip(kf,ko)):
    a.text(v1+.008,i-0.19,f"{v1:.3f}",va="center",fontsize=6.5)
    a.text(v2+.008,i+0.19,f"{v2:.3f}",va="center",fontsize=6.5)
a.legend(fontsize=6.5,frameon=False,loc="lower right")

a=ax[2]
T=json.load(open("data/vocab_exp.json")); M=json.load(open("data/multimodel_melt3.json"))
thr=[0.65,0.75,0.80,0.85,0.90,0.95]
p0=[0.0000,0.0307,0.2182,0.4206,0.6428,0.8279]
p12=[0.0077,0.0323,0.0810,0.2040,0.4422,0.6183]
p6=[0.0000,0.0000,0.0003,0.0178,0.0541,0.2698]
a.plot(thr,p0,"o-",color=ACC,lw=2,label="position 0")
a.plot(thr,p12,"s-",color=OK,lw=2,label="positions 1-2")
a.plot(thr,p6,"^-",color=BAD,lw=2,label="positions 6+")
a.set_xlabel("semantic clustering threshold"); a.set_ylabel("meaning share")
a.set_title("robustness: the ordering never flips,\nthe absolute numbers move a lot")
a.legend(fontsize=7,frameon=False)
fig.tight_layout(); fig.savefig("figs/fig9_ffh.png"); print("wrote figs/fig9_ffh.png")
