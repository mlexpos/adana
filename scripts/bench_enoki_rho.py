"""Can the ladder's measured signal rho-hat = E_delta / tau-hat run inside an Enoki training loop at ~20% overhead?  A benchmark on a
trained checkpoint (weights and its optimizer's v; hybrid_ladder report, sec:floor-signal and sec:rung4).

In the Adam metric z = R g, R = (sqrt(v) + eps)^{-1/2}, with H_z = R GN R on a curvature batch of c sequences:
  E_delta = <z_A, (H_z + delta_z)^{-1} z_B>,  delta_z = m Delta / gamma0          (z_A, z_B: the two halves of a training-size batch)
read by ONE Lanczos recurrence started at z_B (no reorthogonalization, no stored basis; c_j = <z_A, v_j> accumulated on the fly):
  E_delta ~ |z_B| sum_j c_j [Q f(Theta) Q^T e_1]_j,  f(t) = 1 / (t + delta_z),  at m = 16 ... m_max steps (convergence in m).
Noise temperature (C ~ tau GN), in effective samples (n_eff = sequences x T_eff per half):
  tau-hat = tr(R C R) / tr(H_z),   tr(R C R) = (n_eff / 2) |z_A - z_B|^2,   tr(H_z) by Hutchinson (4 probes, curvature batch 8 seq).
rho-hat = E_delta / tau-hat.  Repeated for curvature batches of --cbatch sequences (two independent draws each: the curvature-sampling
spread) and for --pairs independent pairs of halves (the estimation spread).  GN products: forward-mode AD in TF32 (bench_enoki_gn.py:
within 2-3% of float64 on this checkpoint at 1.7x the fp32 speed).
usage (repo root): python scripts/bench_enoki_rho.py --heads 12 --ckpt CKPT/main.pt --data .../train_0000.bin --g0 1.13e-4 --t 2688"""
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
from optim.slq_torch import GaussNewtonOperator, _dot, effective_tokens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--heads", type=int, default=12)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--g0", type=float, required=True, help="gamma0 of the run (the momentum gain)")
ap.add_argument("--g0mult", default="1", help="gamma0 multipliers swept: E_delta at delta_z = m Delta / (mult gamma0), all from one Lanczos run")
ap.add_argument("--variants", default="", help="curvature batch:precision list, e.g. 32:tf32,64:tf32,32:fp32 (overrides --cbatch)")
ap.add_argument("--t", type=int, required=True, help="step of the checkpoint (Delta_t = 8 / (8 + t))")
ap.add_argument("--mdamp", type=float, default=0.01)
ap.add_argument("--half", type=int, default=256, help="sequences per half (Enoki batch 512 = 2 x 256)")
ap.add_argument("--micro", type=int, default=16)
ap.add_argument("--cbatch", default="8,16,32")
ap.add_argument("--draws", type=int, default=2)
ap.add_argument("--pairs", type=int, default=2)
ap.add_argument("--m_max", type=int, default=96)
a = ap.parse_args()
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dims = compute_dimensions("enoki", a.heads)
cli = ["--model", "enoki", "--n_head", str(dims["n_head"]), "--n_layer", str(dims["n_layer"]), "--n_embd", str(dims["n_embd"]),
       "--qkv_dim", str(dims["head_dim"]), "--mlp_hidden_dim", str(dims["mlp_hidden_dim"]), "--sequence_length", "2048",
       "--init-scheme", "ScaledGPT", "--weight_tying", "False", "--vocab_size", "50304", "--dropout", "0.0", "--z_loss_coeff", "1e-4"]
base = argparse.ArgumentParser(allow_abbrev=False)
base.add_argument("--config_format", default="base")
ns, rem = base.parse_known_args(cli)
args = config.parse_args_with_format(format="base", base_parser=base, args=rem, namespace=ns)
model = get_model(args).to(dev)
names = [n for n, _ in model.named_parameters()]; params = [p for _, p in model.named_parameters()]
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
strip = lambda n: n.replace("_orig_mod.", "").replace("module.", "")
model.load_state_dict({strip(k): v for k, v in ck["model"].items()})
byname = {}
for g in ck["optimizer"]["param_groups"]:
    for i, n in zip(g["params"], g.get("param_names", [None] * len(g["params"]))):
        if n is not None and i in ck["optimizer"]["state"] and "v" in ck["optimizer"]["state"][i]:
            byname[strip(n)] = ck["optimizer"]["state"][i]["v"]
R = [(torch.sqrt(byname[n].to(dev).float()) + 1e-8).rsqrt() for n in names]
del ck
data = np.memmap(a.data, dtype=np.uint16, mode="r")
rng = np.random.default_rng(0)


def batch(B):
    ix = rng.integers(0, len(data) - 2050, B)
    x = torch.from_numpy(np.stack([data[i:i + 2048].astype(np.int64) for i in ix])).to(dev)
    y = torch.from_numpy(np.stack([data[i + 1:i + 2049].astype(np.int64) for i in ix])).to(dev)
    return x, y


def half_grad(n):
    """Mean-over-tokens gradient of n fresh sequences (bf16 autocast, as in training), and the mean loss."""
    gs = [torch.zeros_like(p) for p in params]; L = 0.0
    model.train()
    for _ in range(n // a.micro):
        x, y = batch(a.micro)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, targets=y)["loss"]
        model.zero_grad(set_to_none=True); loss.backward()
        for g_, p in zip(gs, params):
            g_.add_(p.grad.float(), alpha=a.micro / n)
        L += float(loss) * a.micro / n
    model.zero_grad(set_to_none=True); model.eval()
    return gs, L


def lanczos_bilinear(op, za, zb, m_max, fs, marks):
    """<za, f(H) zb> for each f in fs at each m in marks, from one Lanczos recurrence started at zb (no reorthogonalization)."""
    bn = math.sqrt(float(_dot(zb, zb)))
    v = [x / bn for x in zb]; vp = [torch.zeros_like(x) for x in zb]; bprev = 0.0
    al, be, c = [], [], []
    out = {}
    for j in range(m_max):
        c.append(float(_dot(za, v)))
        w = op(v)
        if bprev > 0:
            for wi, vpi in zip(w, vp):
                wi.add_(vpi, alpha=-bprev)
        al_ = float(_dot(v, w))
        for wi, vi in zip(w, v):
            wi.add_(vi, alpha=-al_)
        b_ = math.sqrt(max(float(_dot(w, w)), 0.0))
        al.append(al_); be.append(b_)
        m = j + 1
        if m in marks or m == m_max:
            T = np.diag(al) + np.diag(be[:-1], 1) + np.diag(be[:-1], -1)
            th, Q = np.linalg.eigh(T)
            out[m] = [bn * float(np.array(c) @ (Q @ (f(th) * Q[0]))) for f in fs] + [float(th.max()), float(th.min())]
        vp, v, bprev = v, [wi / b_ for wi in w], b_
    return out


Dt = 8.0 / (8.0 + a.t); dz = a.mdamp * Dt / a.g0
GM = [float(x) for x in a.g0mult.split(",")]
fs = [lambda t, s=s: 1.0 / (t + s * dz) for s in [1.0 / g for g in GM]]
marks = [16, 32, 64, 96, 128, 192, 256, 384]
print(f"heads {a.heads}: {sum(p.numel() for p in params) / 1e6:.0f}M params; t {a.t}, Delta {Dt:.3g}, gamma0 {a.g0:.3g}, "
      f"delta_z {dz:.3g}; halves {a.half} seq; device {torch.cuda.get_device_name()}", flush=True)
# T_eff and tr H_z (curvature batch of 8 sequences)
xh, yh = batch(8)
te, _, _ = effective_tokens(model, params, xh, yh)
op8 = GaussNewtonOperator(model, names, params, xh, yh, r=R, mode="jvp", chunk=2, cache=False)
gen = torch.Generator(device=dev); gen.manual_seed(3)
trH = np.mean([float(_dot(z, op8(z))) for z in
               [[(torch.randint(0, 2, p.shape, device=dev, generator=gen) * 2 - 1).float() for p in params] for _ in range(4)]])
print(f"T_eff {te:.0f} tokens/seq; tr H_z {trH:.4g} (Hutchinson, 4 probes)", flush=True)
for pr in range(a.pairs):
    t0 = time.time()
    gA, LA = half_grad(a.half); gB, LB = half_grad(a.half)
    zA = [r * g_ for r, g_ in zip(R, gA)]; zB = [r * g_ for r, g_ in zip(R, gB)]
    P = 0.5 * (LA + LB); neff = a.half * te
    d2 = sum(float(((x - y) ** 2).sum()) for x, y in zip(zA, zB))
    tau = neff / 2 * d2 / trH
    print(f"pair {pr}: halves {time.time() - t0:.0f} s; P {P:.4f}; tau-hat {tau:.4g} (tau/P {tau / P:.3f}); <zA,zB>/|z|^2 "
          f"{float(_dot(zA, zB)) / math.sqrt(float(_dot(zA, zA)) * float(_dot(zB, zB))):.3f}", flush=True)
    vs = [v.split(":") for v in a.variants.split(",")] if a.variants else [(c, "tf32") for c in a.cbatch.split(",")]
    for cs_, prec in vs:
        c = int(cs_); tf = prec == "tf32"
        torch.backends.cuda.matmul.allow_tf32 = tf; torch.backends.cudnn.allow_tf32 = tf
        for dr in range(a.draws):
            xc, yc = batch(c)
            op = GaussNewtonOperator(model, names, params, xc, yc, r=R, mode="jvp", chunk=2, cache=False)
            torch.cuda.synchronize(); t0 = time.time()
            res = lanczos_bilinear(op, zA, zB, a.m_max, fs, marks)
            torch.cuda.synchronize(); dt = time.time() - t0
            mm = sorted(res); last = res[mm[-1]]; k = len(GM)
            print(f"  c {c:3d} seq {prec} draw {dr}: {dt:6.0f} s ({dt / a.m_max:.2f} s/product); Ritz range [{last[k + 1]:.3g}, {last[k]:.3g}]", flush=True)
            for i, g in enumerate(GM):
                print(f"     gamma0 x{g:g} (delta_z {dz / g:.3g}, kappa {last[k] / (dz / g):.2g}): rho-hat(m={mm}) "
                      + " ".join(f"{res[m][i] / tau:.4f}" for m in mm), flush=True)
