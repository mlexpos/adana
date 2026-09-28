"""GPU diagnostic for the adana-slq curvature estimator on a real Enoki model.

Trains a few ADana steps (to get a realistic preconditioner r = (sqrt(v)+eps)^{-1/2}), then compares Gauss-Newton
products from finite differences (fd) and from forward-mode AD (jvp): relative difference, symmetry <w,Hu> vs <u,Hw>,
positivity <u,Hu>; then repeats the SLQ estimate of N(Delta) a few times with each mode (spread + bracket gap + time).

usage (repo root):  python scripts/slq_gnvp_check.py --heads 3 --data $DATASETS_DIR/fineweb-100BT/train_0000.bin
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config  # noqa: E402
from config.scaling import compute_dimensions  # noqa: E402
from models.utils import get_model  # noqa: E402
from optim.adana import ADana  # noqa: E402
from optim.slq_torch import GaussNewtonOperator, slq, _dot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--heads", type=int, default=3)
ap.add_argument("--data", required=True)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--slq_batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
dims = compute_dimensions("enoki", a.heads)
cli = ["--model", "enoki", "--n_head", str(dims["n_head"]), "--n_layer", str(dims["n_layer"]), "--n_embd", str(dims["n_embd"]),
       "--qkv_dim", str(dims["head_dim"]), "--mlp_hidden_dim", str(dims["mlp_hidden_dim"]), "--sequence_length", "2048",
       "--init-scheme", "ScaledGPT", "--weight_tying", "False", "--vocab_size", "50304", "--dropout", "0.0", "--z_loss_coeff", "1e-4"]
base = argparse.ArgumentParser(allow_abbrev=False)
base.add_argument("--config_format", default="base")
ns, rem = base.parse_known_args(cli)
args = config.parse_args_with_format(format="base", base_parser=base, args=rem, namespace=ns)
torch.manual_seed(0)
model = get_model(args).to(dev)
data = np.memmap(a.data, dtype=np.uint16, mode="r")
rng = np.random.default_rng(0)


def batch(B):
    ix = rng.integers(0, len(data) - 2050, B)
    x = torch.from_numpy(np.stack([data[i:i + 2048].astype(np.int64) for i in ix])).to(dev)
    y = torch.from_numpy(np.stack([data[i + 1:i + 2049].astype(np.int64) for i in ix])).to(dev)
    return x, y


names = [n for n, _ in model.named_parameters()]
params = [p for _, p in model.named_parameters()]
opt = ADana([{"params": params}], lr=a.lr, delta=8.0, kappa=0.85, weight_decay=0.0)
for it in range(a.steps):
    x, y = batch(a.batch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, targets=y)["loss"]
    opt.zero_grad(); loss.backward(); opt.step()
print(f"after {a.steps} ADana steps: loss {float(loss):.3f}  params {sum(p.numel() for p in params)/1e6:.1f}M")
r = [(torch.sqrt(opt.state[p]["v"]) + 1e-8).rsqrt() for p in params]
rr = torch.cat([x.reshape(-1) for x in r])
print(f"r: median {float(rr.median()):.3g}  max {float(rr.max()):.3g}  (max/median {float(rr.max()/rr.median()):.3g})")
x, y = batch(a.slq_batch)
model.eval()
ops = {m: GaussNewtonOperator(model, names, params, x, y, r=r, mode=m) for m in ("fd", "jvp")}
gen = torch.Generator(device=dev); gen.manual_seed(1)
u = [torch.randn(p.shape, device=dev, generator=gen) for p in params]
w = [torch.randn(p.shape, device=dev, generator=gen) for p in params]
Hu = {m: ops[m](u) for m in ops}
Hw = {m: ops[m](w) for m in ops}
diff = math.sqrt(float(_dot([a_ - b_ for a_, b_ in zip(Hu["fd"], Hu["jvp"])], [a_ - b_ for a_, b_ in zip(Hu["fd"], Hu["jvp"])])))
nj = math.sqrt(float(_dot(Hu["jvp"], Hu["jvp"])))
print(f"|H_fd u - H_jvp u| / |H_jvp u| = {diff / nj:.3e}")
for m in ops:
    s1, s2, q = float(_dot(w, Hu[m])), float(_dot(u, Hw[m])), float(_dot(u, Hu[m]))
    print(f"  {m}: symmetry |<w,Hu>-<u,Hw>|/|<w,Hu>| = {abs(s1 - s2) / max(abs(s1), 1e-30):.3e}   <u,Hu> = {q:.4g}")
    # per-tensor error of fd relative to jvp
    if m == "fd":
        errs = []
        for n, a_, b_ in zip(names, Hu["fd"], Hu["jvp"]):
            errs.append((float((a_ - b_).norm() / (b_.norm() + 1e-30)), n))
        errs.sort(reverse=True)
        print("     worst tensors (fd vs jvp rel err):", ", ".join(f"{n}={e:.2g}" for e, n in errs[:5]))
lr = a.lr
for D in (8.0 / 40, 8.0 / 400):
    for m in ("jvp",):
        vals, gaps, t0 = [], [], time.time()
        for rep in range(a.reps):
            g = torch.Generator(device=dev); g.manual_seed(100 + rep)
            res = slq(ops[m], params, m_max=48, probes=2, eps=1e-9, g=lr, D_check=[D], generator=g)
            print("     Gauss N vs m (probe 0):", [(mm, round(v[0], 1)) for mm, v in res["hist"][0]])
            th = np.maximum(res["nodes"].reshape(-1), 0); wq = res["weights"].reshape(-1)
            vals.append(float(np.sum(wq * lr * th / (lr * th + D)))); gaps.append(float(np.nanmax(res["gap"])))
        print(f"  Delta={D:.3g} {m}: N = {np.round(vals, 1)}  gaps = {np.round(gaps, 3)}  lam_max ~ {res['lam_max']:.3g}  "
              f"({(time.time() - t0) / a.reps:.1f}s per estimate)")

# batch-to-batch spread of N (the estimator is refreshed on a new batch each time) and cost vs SLQ batch / m
for b in (4, 8):
    vals, t0 = [], time.time()
    for rep in range(3):
        xb, yb = batch(b)
        op = GaussNewtonOperator(model, names, params, xb, yb, r=r, mode="jvp")
        g = torch.Generator(device=dev); g.manual_seed(200 + rep)
        res = slq(op, params, m_max=16, probes=1, eps=1e-9, g=lr, D_check=[0.2], generator=g)
        th = np.maximum(res["nodes"].reshape(-1), 0); wq = res["weights"].reshape(-1)
        vals.append(float(np.sum(wq * lr * th / (lr * th + 0.2))))
    print(f"  slq_batch={b}, m=16, 1 probe, new batch each: N(0.2) = {np.round(vals, 1)}  ({(time.time() - t0) / 3:.1f}s each)")
