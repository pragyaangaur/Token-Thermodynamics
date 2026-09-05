import sys, os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,"grid.alpha":.25,
                     "axes.spines.top":False,"axes.spines.right":False})
OK,BAD,ACC="#2b6cb0","#c53030","#2f855a"
V=json.load(open("data/vocab_exp.json"))
M=json.load(open("data/multimodel_melt3.json"))
fig,ax=plt.subplots(1,3,figsize=(12.6,3.6))

vv=np.array([o["V"] for o in V]); xs=np.array([o["xstar"] for o in V])
T=np.array([o["Tmelt"] for o in V]); D=np.array([o["delta"] for o in V])
bpc=np.array([o["bpc"] for o in V])

a=ax[0]
a.plot(vv,T/D,"o-",color=BAD,lw=2,label="$T_{melt}/\\Delta$  (no $\\log V$ term)")
a.plot(vv,T*xs/D,"s-",color=ACC,lw=2,label="$T_{melt}x^*(V)/\\Delta$  (with it)")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("vocabulary size $V$"); a.set_ylabel("ratio")
a.set_title("controlled sweep: identical models,\nidentical data, only $V$ changes")
a.legend(fontsize=7,frameon=False)

a=ax[1]
pred=D/xs
a.scatter(pred,T,c=np.log10(vv),cmap="plasma",s=110,zorder=3,edgecolor="w",linewidth=.8)
for p,t,v in zip(pred,T,vv): a.annotate(f"V={v}",(p,t),fontsize=6.5,xytext=(5,-8),textcoords="offset points")
lim=[min(pred.min(),T.min())*.85,max(pred.max(),T.max())*1.15]
a.plot(lim,lim,"k:",lw=1,label="$T_{melt}=\\Delta/x^*$")
k=np.sum(T*pred)/np.sum(pred*pred)
a.plot(lim,[k*l for l in lim],color=ACC,lw=1.6,label=f"fit, slope {k:.3f}")
a.set_xlabel("predicted  $\\Delta/x^*(V)$"); a.set_ylabel("measured  $T_{melt}$")
a.set_title("tiny models, $V$ = 257 to 32768"); a.legend(fontsize=7,frameon=False,loc="upper left")

a=ax[2]
tv=np.array([np.median([r["Tmelt"] for r in o["rows"]]) for o in M])
td=np.array([np.median([r["delta_mode"] for r in o["rows"]]) for o in M])
tx=np.array([o["xstar"] for o in M])
a.scatter(td/tx,tv,s=85,color=OK,zorder=3,edgecolor="w",linewidth=.7,label="7 pretrained models")
a.scatter(pred,T,s=85,color=ACC,marker="^",zorder=3,edgecolor="w",linewidth=.7,label="5 controlled tiny models")
allp=np.concatenate([td/tx,pred]); allt=np.concatenate([tv,T])
lim=[allp.min()*.8,allp.max()*1.15]
a.plot(lim,lim,"k:",lw=1)
kk=np.sum(allt*allp)/np.sum(allp*allp)
a.plot(lim,[kk*l for l in lim],color="#4a5568",lw=1.6,label=f"joint fit {kk:.3f}")
a.set_xlabel("predicted  $\\Delta/x^*(V)$"); a.set_ylabel("measured  $T_{melt}$")
a.set_title(f"all 12 models, $V$ spans {int(min(vv.min(),M[0]['V']))} to {int(max(vv.max(),max(o['V'] for o in M)))}")
a.legend(fontsize=7,frameon=False,loc="upper left")
fig.tight_layout(); fig.savefig("figs/fig7_vocab.png"); print("wrote figs/fig7_vocab.png")

r1=T/D; r2=T*xs/D
print(f"\ncontrolled sweep, x* spans {xs.min():.2f} to {xs.max():.2f} ({xs.max()/xs.min():.2f}x)")
print(f"  without log V term : CV {r1.std()/r1.mean():.4f}  spread {r1.max()/r1.min():.3f}x")
print(f"  with    log V term : CV {r2.std()/r2.mean():.4f}  spread {r2.max()/r2.min():.3f}x")
for o in V: print(f"   V={o['V']:6d} bpc={o['bpc']:.3f} T={o['Tmelt']:.3f} D={o['delta']:6.2f} ratio={o['ratio']:.4f}")
