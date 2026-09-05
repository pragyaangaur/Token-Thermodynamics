import sys, os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,"grid.alpha":.25,
                     "axes.spines.top":False,"axes.spines.right":False})
OK,BAD,ACC,PUR="#2b6cb0","#c53030","#2f855a","#6b46c1"
fig,ax=plt.subplots(1,3,figsize=(12.6,3.6))

# panel 1: free-form vs factual paraphrase share
F=json.load(open("data/freeform_1p5b.json"))
a=ax[0]
conds=["free-form","short-factual"]; labs=["free-form\nanswers","short factual\nanswers"]
share=[]
for c in conds:
    se=np.mean([r["sem_entropy"] for r in F[c]]); te=np.mean([r["tok_entropy"] for r in F[c]])
    share.append([(te-se)/te, se/te])
share=np.array(share)
x=np.arange(2)
a.bar(x,share[:,0],color=BAD,alpha=.85,width=.55,label="phrasing (no meaning)")
a.bar(x,share[:,1],bottom=share[:,0],color=OK,alpha=.9,width=.55,label="meaning")
for i in range(2):
    a.text(i,share[i,0]/2,f"{share[i,0]*100:.1f}%",ha="center",color="white",fontweight="bold",fontsize=10)
    a.text(i,share[i,0]+share[i,1]/2,f"{share[i,1]*100:.1f}%",ha="center",color="white",fontweight="bold",fontsize=9)
a.set_xticks(x); a.set_xticklabels(labs,fontsize=8); a.set_ylabel("share of next-token entropy")
a.set_title("what the model is uncertain about\nat the first answer token")
a.legend(fontsize=7,frameon=False,loc="lower right"); a.set_ylim(0,1.05); a.grid(axis="x")

# panel 2: position-resolved meaning share
W=json.load(open("data/where_meaning.json"))
R=[r for o in W["tokens"] ] if isinstance(W,dict) else [r for o in W for r in o["tokens"]]
pos=np.array([r["pos"] for r in R]); ms=np.array([r["meaning_share"] for r in R])
H=np.array([r["H_topk"] for r in R]); SE=np.array([r["S_sem"] for r in R])
a=ax[1]
bins=[(0,1),(1,3),(3,6),(6,12),(12,20),(20,30),(30,60)]
xs=[]; msb=[]; hb=[]
for lo,hi in bins:
    m=(pos>=lo)&(pos<hi)
    xs.append(f"{lo}-{hi-1}" if hi-lo>1 else str(lo)); msb.append(ms[m].mean()); hb.append(H[m].mean())
xx=np.arange(len(bins))
a.bar(xx,msb,color=OK,alpha=.85,width=.6,label="meaning share")
a2=a.twinx(); a2.plot(xx,hb,"o--",color=BAD,lw=1.8,ms=5,label="token entropy")
a2.set_ylabel("mean token entropy (nats)",color=BAD); a2.tick_params(labelcolor=BAD,labelsize=8); a2.grid(False)
a.set_xticks(xx); a.set_xticklabels(xs,fontsize=7.5); a.set_xlabel("position in the answer")
a.set_ylabel("meaning share",color=OK); a.tick_params(labelcolor=OK)
a.set_title("content is decided in the first 3 tokens\nentropy peaks where nothing is decided")
h1,l1=a.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels()
a.legend(h1+h2,l1+l2,fontsize=7,frameon=False,loc="upper center")

# panel 3: entropy vs meaning share scatter
a=ax[2]
a.scatter(H,ms,s=16,alpha=.5,color=PUR,edgecolor="none")
hi=ms>0.2
a.scatter(H[hi],ms[hi],s=42,color=ACC,zorder=3,edgecolor="w",linewidth=.6,label="positions that decide meaning")
q=np.percentile(H,80)
a.axvspan(q,H.max()*1.05,color=BAD,alpha=.10)
a.text(q*1.02,0.62,"top 20% entropy\n(what a detector flags)\nmean meaning share\n= 0.0000",fontsize=6.5,color=BAD)
a.set_xlabel("token entropy at that position (nats)"); a.set_ylabel("meaning share")
a.set_title(f"Spearman = {__import__('scipy.stats',fromlist=['spearmanr']).spearmanr(H,ms).statistic:+.3f}")
a.legend(fontsize=7,frameon=False,loc="upper right")
fig.tight_layout(); fig.savefig("figs/fig8_freeform.png"); print("wrote figs/fig8_freeform.png")
