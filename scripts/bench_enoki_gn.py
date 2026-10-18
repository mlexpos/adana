"""Cost and accuracy of one Gauss-Newton product on an Enoki model, against a training step (for the measurement overhead of
adana-slq / hybrid-slq, and for porting the ladder's v3 measurement to Enoki).

Trains a few ADana steps (bf16 autocast) for a realistic preconditioner r = (sqrt(v)+eps)^{-1/2}, then on one batch of
--slq_batch sequences times r GN(r u) in each variant below, and compares it with the reference (the variant adana-slq runs
today: forward-mode AD, fp32, TF32 off, math attention, chunks of 2 sequences):
    jvp  fp32 / tf32, chunk 2 / 8           (jvp forces the math SDPA backend: fp32 attention matrices in memory)
    fd   fp32 / tf32, chunk 2 / 8           (central differences of the logits, then one reverse pass: default attention)
Reported per variant: seconds per product, microseconds per token, ratio to the training step's cost per token, peak memory,
relative error |Hu - Hu_ref| / |Hu_ref|, and the relative error of the Rayleigh quotient <u, Hu>.  A short SLQ (m = 32, 2 probes) in
the reference and in the fastest accurate variant compares N(Delta) = tr f(H_z), f = g x / (g x + Delta), at a few Delta.
The training step: forward + backward of --train_batch sequences in bf16 autocast (not compiled; the logged runs compile, so the
script also prints the logged iter_dt it should be compared with when --logged_iter_dt is given).

usage (repo root):  python scripts/bench_enoki_gn.py --heads 12 --data $DATASETS_DIR/fineweb-100BT/train_0000.bin"""
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
ap.add_argument("--heads", type=int, default=12)
ap.add_argument("--data", required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--train_batch", type=int, default=16)
ap.add_argument("--slq_batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--g", type=float, default=1e-4, help="gamma0 for N(Delta)")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--variants", default="jvp:fp32:2,jvp:tf32:2,jvp:tf32:8,fd:fp32:2,fd:tf32:2,fd:tf32:8")
ap.add_argument("--slq_m", type=int, default=32)
ap.add_argument("--Ds", default="1e-2,3e-3,1e-3", help="Delta values for N(Delta)")
ap.add_argument("--ckpt", default=None, help="main.pt of a trained run: weights and its optimizer's v (r = (sqrt v + eps)^-1/2)")
a = ap.parse_args()
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


def sync_time(fn, reps):
    fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps, out


names = [n for n, _ in model.named_parameters()]
import models.enoki as _enoki  # noqa: E402


def _rot(q, k, freqs_cis):
    """apply_rotary_emb keeping float64 (the model's version casts to float32, which breaks the fp64 reference)."""
    dt = torch.float64 if q.dtype == torch.float64 else torch.float32
    q2 = q.to(dt).reshape(*q.shape[:-1], -1, 2); k2 = k.to(dt).reshape(*k.shape[:-1], -1, 2)
    f = _enoki._reshape_for_broadcast(freqs_cis.to(dt), q2)
    qo = torch.stack((q2[..., 0] * f[..., 0] - q2[..., 1] * f[..., 1], q2[..., 0] * f[..., 1] + q2[..., 1] * f[..., 0]), -1)
    ko = torch.stack((k2[..., 0] * f[..., 0] - k2[..., 1] * f[..., 1], k2[..., 0] * f[..., 1] + k2[..., 1] * f[..., 0]), -1)
    return qo.reshape(q2.shape).flatten(3), ko.reshape(k2.shape).flatten(3)


_enoki.apply_rotary_emb = _rot
params = [p for _, p in model.named_parameters()]
nparam = sum(p.numel() for p in params)
torch.backends.cuda.matmul.allow_tf32 = True
opt = ADana([{"params": params}], lr=a.lr, delta=8.0, kappa=0.85, weight_decay=0.0)
vck = None
if a.ckpt:
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model"].items()}
    print("load_state_dict:", model.load_state_dict(sd, strict=False))
    osd = ck["optimizer"]; strip = lambda n: n.replace("_orig_mod.", "").replace("module.", "")
    byname = {}
    for g in osd["param_groups"]:
        for i, n in zip(g["params"], g.get("param_names", [None] * len(g["params"]))):
            if n is not None and i in osd["state"] and "v" in osd["state"][i]:
                byname[strip(n)] = osd["state"][i]["v"]
    vck = [byname[n].to(dev).float() for n in names] if all(n in byname and tuple(byname[n].shape) == tuple(q.shape)
                                                          for n, q in zip(names, params)) else None
    print("optimizer v matched by name:", vck is not None, f"({len(byname)} named states, {len(params)} params)")
    a.steps = 0
for it in range(a.steps):
    x, y = batch(a.train_batch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, targets=y)["loss"]
    opt.zero_grad(); loss.backward(); opt.step()
xt, yt = batch(a.train_batch)


def train_step():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l = model(xt, targets=yt)["loss"]
    opt.zero_grad(set_to_none=True); l.backward()
    return l


t_train, _ = sync_time(train_step, a.reps)
us_train = 1e6 * t_train / (a.train_batch * 2048)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(xt, targets=yt)["loss"]
print(f"heads {a.heads}: {nparam / 1e6:.0f}M params; device {torch.cuda.get_device_name()}; loss {float(loss):.3f} ({'checkpoint' if a.ckpt else f'after {a.steps} steps'})")
print(f"training step (bf16 autocast, fwd+bwd, not compiled): {t_train:.3f} s for {a.train_batch} seq = {us_train:.2f} us/token")
opt.zero_grad(set_to_none=True)
if vck is not None:
    r = [(torch.sqrt(v) + 1e-8).rsqrt() for v in vck]
else:
    if a.ckpt:
        for _ in range(5):                                  # no usable v in the checkpoint: a few steps from the trained weights
            x_, y_ = batch(a.train_batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l_ = model(x_, targets=y_)["loss"]
            opt.zero_grad(); l_.backward(); opt.step()
    r = [(torch.sqrt(opt.state[p]["v"]) + 1e-8).rsqrt() for p in params]
x, y = batch(a.slq_batch)
ntok = a.slq_batch * 2048
model.eval()
gen = torch.Generator(device=dev); gen.manual_seed(1)
u = [torch.randn(p.shape, device=dev, generator=gen) for p in params]


def set_prec(prec):
    """fp32 / tf32 / fp64 for the next products (fp64: the model, r and u are converted to double and back afterwards)."""
    tf = prec == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf; torch.backends.cudnn.allow_tf32 = tf
    dt = torch.float64 if prec == "fp64" else torch.float32
    model.to(dt)
    return [ri.to(dt) for ri in r], [ui.to(dt) for ui in u]


Ds = [float(x) for x in a.Ds.split(",")]
ref = None
rows = []
for v in a.variants.split(","):
    mode, prec, ch = v.split(":"); ch = int(ch)
    rv, uv = set_prec(prec)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        op = GaussNewtonOperator(model, names, params, x, y, r=rv, mode=mode, chunk=ch, cache=False)
        t, Hu = sync_time(lambda: op(uv), a.reps)
    except torch.OutOfMemoryError:
        print(f"  {v:14s} out of memory"); model.float(); continue
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    Hu = [h.double() for h in Hu]; u64 = [ui.double() for ui in u]
    if ref is None:
        ref = Hu; err = 0.0; rq_err = 0.0
    else:
        dd = [h - h0 for h, h0 in zip(Hu, ref)]
        err = math.sqrt(float(_dot(dd, dd)) / float(_dot(ref, ref)))
        rq_err = abs(float(_dot(u64, Hu)) / float(_dot(u64, ref)) - 1)
    us = 1e6 * t / ntok
    line = (f"  {v:14s} {t:7.3f} s/product  {us:7.2f} us/token  = {us / us_train:6.1f} x training  peak {mem:5.1f} GiB  "
            f"rel err {err:.1e}  <u,Hu> err {rq_err:.1e}")
    del Hu
    # N(Delta) from a short SLQ with the same probes in every variant
    g2 = torch.Generator(device=dev); g2.manual_seed(7)
    t0 = time.time()
    res = slq(op, params, m_max=a.slq_m, probes=2, eps=0.0, g=a.g, D_check=Ds, generator=g2)
    th = np.maximum(res["nodes"].reshape(-1), 0); w = res["weights"].reshape(-1)
    Nv = [float(np.sum(w * a.g * th / (a.g * th + D))) for D in Ds]
    rows.append((v, t, us, Nv))
    line += f"  | lam_max {float(np.max(th)):.3g}  N(Delta={a.Ds}) " + " ".join(f"{n:.4g}" for n in Nv)
    if len(rows) > 1:
        line += "  (x ref " + " ".join(f"{n / n0:.3f}" for n, n0 in zip(Nv, rows[0][3])) + ")"
    print(line + f"  [SLQ {time.time() - t0:.0f} s]", flush=True)
    model.float()
