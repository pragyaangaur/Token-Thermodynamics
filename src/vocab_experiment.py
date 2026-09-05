"""Controlled test of the log V term in the melting law.

Comparing pretrained models cannot test it: real vocabularies cluster around 50k
to 152k, so x*(V) varies under 10%, which is inside the model-to-model scatter.
So train identical models on identical data with only the vocabulary size changed.

Protocol: one corpus, BPE tokenizers at V = 128 ... 32768, the same tiny GPT
architecture, and the same number of CHARACTERS seen by every model (equal epochs
over the same text), so the only difference is V. Then measure T_melt and Delta and
check whether  T_melt * x*(V) / Delta  is constant while x*(V) varies by ~1.8x.
"""
import json, os, sys, math, time, glob, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from scipy.optimize import brentq
from dos import make_dos, curves_dos
from thermo import beta_grid
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

B = beta_grid(500, 5e-3, 1e3)

def xstar(V):
    f = lambda x: x - 2 * (1 + V * np.exp(-x)) / (1 - V * np.exp(-x))
    lo = math.log(V) + 1e-9
    return brentq(f, lo, lo + 80)

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.at = nn.MultiheadAttention(d, h, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x, m):
        h = s.ln1(x); a, _ = s.at(h, h, h, attn_mask=m, need_weights=False)
        x = x + a
        return x + s.mlp(s.ln2(x))

class TinyGPT(nn.Module):
    def __init__(s, V, d=256, nl=4, nh=4, ctx=128):
        super().__init__()
        s.ctx = ctx
        s.tok = nn.Embedding(V, d); s.pos = nn.Embedding(ctx, d)
        s.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, V, bias=False)
        s.head.weight = s.tok.weight                       # tied
        s.apply(s._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(s, idx):
        T = idx.shape[1]
        m = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))[None]
        for b in s.blocks: x = b(x, m)
        return s.head(s.ln(x))

def build_tokenizer(V, text, path):
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=V, special_tokens=["<unk>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             show_progress=False)
    tk.train_from_iterator([text[i:i+10000] for i in range(0, len(text), 10000)], tr)
    tk.save(path)
    return tk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=8.0)
    ap.add_argument("--vocabs", default="128,512,2048,8192,32768")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    text = "".join(open(f, encoding="utf-8", errors="ignore").read() for f in sorted(glob.glob("data/corpus/*.txt")))
    n = len(text); split = int(.97 * n)
    print(f"corpus {n/1e6:.2f} M characters, device {dev}", flush=True)
    os.makedirs("data/tok", exist_ok=True)
    PROMPTS = ["The king said to him, ", "It was a dark and ", "She turned to the window and ",
               "In the beginning of the ", "He could not believe what ", "The old man walked slowly ",
               "There was nothing left but ", "They agreed that the best ",
               "When the ship reached the ", "Nobody knew why the "]
    out = []
    for V in [int(v) for v in a.vocabs.split(",")]:
        t0 = time.time()
        tp = f"data/tok/bpe_{V}.json"
        tk = Tokenizer.from_file(tp) if os.path.exists(tp) else build_tokenizer(V, text[:split], tp)
        ids = np.array(tk.encode(text).ids, dtype=np.int64)
        Vr = tk.get_vocab_size()
        cpt = n / len(ids)
        tr_ids = torch.from_numpy(ids[:int(.97*len(ids))])
        va_ids = torch.from_numpy(ids[int(.97*len(ids)):])
        ctx, bs = 128, 48
        steps = int(a.epochs * len(tr_ids) / (ctx * bs))
        model = TinyGPT(Vr, ctx=ctx).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=.1)
        g = torch.Generator().manual_seed(0)
        model.train()
        for st in range(steps):
            i = torch.randint(0, len(tr_ids)-ctx-1, (bs,), generator=g)
            x = torch.stack([tr_ids[j:j+ctx] for j in i]).to(dev)
            y = torch.stack([tr_ids[j+1:j+ctx+1] for j in i]).to(dev)
            loss = F.cross_entropy(model(x).reshape(-1, Vr), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if st % max(1, steps//4) == 0: print(f"   V={Vr} step {st}/{steps} loss {loss.item():.3f}", flush=True)
        model.eval()
        vl = []
        with torch.no_grad():
            for k in range(0, min(len(va_ids)-ctx-1, 60*ctx), ctx):
                x = va_ids[k:k+ctx][None].to(dev); y = va_ids[k+1:k+ctx+1][None].to(dev)
                if x.shape[1] < ctx: break
                vl.append(F.cross_entropy(model(x).reshape(-1, Vr), y.reshape(-1)).item())
        val = float(np.mean(vl)); bpc = val / math.log(2) / cpt
        rows = []
        with torch.no_grad():
            for p in PROMPTS:
                pi = torch.tensor(tk.encode(p).ids[-ctx:], dtype=torch.long)[None].to(dev)
                lg = model(pi)[0, -1].float().cpu().numpy().astype(np.float64)
                E, W = make_dos(lg, K=min(1024, lg.size), NB=512)
                c = curves_dos(E, W, B); o = np.argsort(c["T"]); T, C = c["T"][o], c["C"][o]
                s = np.sort(lg)[::-1]
                rows.append(dict(Tmelt=float(T[int(np.argmax(C))]), Cmax=float(C.max()),
                                 delta_bulk=float(s[0]-np.median(lg)), delta_mean=float(s[0]-lg.mean())))
        x = xstar(Vr)
        T = np.median([r["Tmelt"] for r in rows]); D = np.median([r["delta_bulk"] for r in rows])
        rec = dict(V=Vr, logV=math.log(Vr), xstar=x, chars_per_token=cpt, steps=steps,
                   val_loss=val, bpc=bpc, Tmelt=float(T), delta=float(D),
                   ratio=float(T*x/D), T_over_D=float(T/D),
                   Cmax=float(np.median([r["Cmax"] for r in rows])), rows=rows)
        out.append(rec)
        print(f"  V={Vr:6d} logV={math.log(Vr):5.2f} x*={x:6.3f} bpc={bpc:.3f} "
              f"T_melt={T:6.3f} Delta={D:7.3f} T/D={T/D:.5f} T*x*/D={T*x/D:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(a.out, "w"))
        del model
    print("\n=== does the log V term hold? ===")
    r1 = np.array([o["T_over_D"] for o in out]); r2 = np.array([o["ratio"] for o in out])
    xs = np.array([o["xstar"] for o in out])
    print(f"  x* spans {xs.min():.2f} to {xs.max():.2f}  = {xs.max()/xs.min():.2f}x")
    print(f"  T/Delta        (no log V term) : spread {r1.max()/r1.min():.3f}x  CV {r1.std()/r1.mean():.4f}")
    print(f"  T*x*/Delta     (with log V)    : spread {r2.max()/r2.min():.3f}x  CV {r2.std()/r2.mean():.4f}")
    print(f"  VERDICT: {'log V term CONFIRMED' if r2.std()/r2.mean() < r1.std()/r1.mean() else 'log V term NOT SUPPORTED'}")
    print("VOCABDONE")

if __name__ == "__main__":
    main()
