"""GPU check of the adana-slq Gauss-Newton product on a real Enoki model:
  (1) at small size, cached-vjp (cache=True) and checkpointed no-cache products agree;
  (2) at the target size, peak memory of one no-cache product (chunk 1, 2048 tokens), on top of the training state
      (fp32 params + grads + ADana m, v) that is resident during a refresh.

usage (repo root):  python scripts/slq_mem_check.py --data $DATASETS_DIR/fineweb-100BT/train_0000.bin --heads 24
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config  # noqa: E402
from config.scaling import compute_dimensions  # noqa: E402
from models.utils import get_model  # noqa: E402
from optim.slq_torch import GaussNewtonOperator  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--heads", type=int, default=24)
ap.add_argument("--check_heads", type=int, default=3)
a = ap.parse_args()
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
data = np.memmap(a.data, dtype=np.uint16, mode="r")
rng = np.random.default_rng(0)


def build(h):
    dims = compute_dimensions("enoki", h)
    cli = ["--model", "enoki", "--n_head", str(dims["n_head"]), "--n_layer", str(dims["n_layer"]), "--n_embd",
           str(dims["n_embd"]), "--qkv_dim", str(dims["head_dim"]), "--mlp_hidden_dim", str(dims["mlp_hidden_dim"]),
           "--sequence_length", "2048", "--init-scheme", "ScaledGPT", "--weight_tying", "False", "--vocab_size", "50304",
           "--dropout", "0.0", "--z_loss_coeff", "1e-4"]
    base = argparse.ArgumentParser(allow_abbrev=False)
    base.add_argument("--config_format", default="base")
    ns, rem = base.parse_known_args(cli)
    args = config.parse_args_with_format(format="base", base_parser=base, args=rem, namespace=ns)
    torch.manual_seed(0)
    return get_model(args).to(dev)


def batch(B):
    ix = rng.integers(0, len(data) - 2050, B)
    x = torch.from_numpy(np.stack([data[i:i + 2048].astype(np.int64) for i in ix])).to(dev)
    y = torch.from_numpy(np.stack([data[i + 1:i + 2049].astype(np.int64) for i in ix])).to(dev)
    return x, y


# (1) exactness at small size
model = build(a.check_heads)
names = [n for n, _ in model.named_parameters()]; params = [p for _, p in model.named_parameters()]
x, y = batch(2)
r = [torch.rand_like(p) + 0.5 for p in params]
u = [torch.randn_like(p) for p in params]
outs = []
for cache in (True, False):
    op = GaussNewtonOperator(model, names, params, x, y, r=r, chunk=1, cache=cache)
    outs.append(torch.cat([o.reshape(-1) for o in op(u)]))
rel = float((outs[0] - outs[1]).norm() / outs[0].norm())
print(f"[{a.check_heads} heads] cached vs checkpointed no-cache GN product: rel diff {rel:.2e}")
del model, params, op, outs, r, u
torch.cuda.empty_cache()

# (2) memory at the target size
model = build(a.heads)
names = [n for n, _ in model.named_parameters()]; params = [p for _, p in model.named_parameters()]
P = sum(p.numel() for p in params)
state = [torch.zeros_like(p) for p in params for _ in range(3)]          # grads, m, v
for p in params:
    p.grad = torch.zeros_like(p)
x, y = batch(8)
r = [torch.rand_like(p) + 0.5 for p in params]
lanczos = [torch.zeros_like(p) for p in params for _ in range(4)]         # Lanczos vectors + per-tensor work
torch.cuda.synchronize(); base = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()
op = GaussNewtonOperator(model, names, params, x[:1], y[:1], r=r, chunk=1, cache=False)
u = [torch.randn_like(p) for p in params]
out = op(u)
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated()
print(f"[{a.heads} heads, {P / 1e6:.0f}M params] resident state {base / 2**30:.1f} GiB, "
      f"peak during one no-cache GN product {peak / 2**30:.1f} GiB (+{(peak - base) / 2**30:.1f} GiB)")
