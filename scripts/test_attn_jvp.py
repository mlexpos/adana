"""Check and time src/optim/attn_jvp.py (fused causal-attention tangent for Gauss-Newton products).
(1) kernel O, dO against an fp64 jvp of explicit softmax attention, IEEE fp32 / TF32 / TF32x3 matmuls, T not a multiple of the tile;
(2) the autograd wrapper: torch.func.jvp (tangent) and torch.autograd.grad (reverse pass) against the same fp64 reference;
(3) time and peak memory of one attention tangent at Enoki's shape (H heads x T 2048 x D 64, B sequences) against the current path,
    torch.func.jvp through the math SDPA backend.
usage (repo root): python scripts/test_attn_jvp.py [--heads 24]"""
import argparse
import math
import os
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from optim.attn_jvp import CausalAttention, attn_jvp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--heads", type=int, default=24)
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)


def ref_attn(q, k, v):
    T = q.shape[-2]
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return torch.softmax(s, -1) @ v


def rel(a_, b_):
    return float((a_.double() - b_.double()).norm() / b_.double().norm())


def inputs(B, H, T, D, dt=torch.float64):
    # q, k of unit-ish rows as after QK-LayerNorm (logits O(sqrt D) scaled to O(1)); tangents of comparable size
    return [torch.randn(B, H, T, D, device=dev, dtype=dt) for _ in range(6)]


print(f"device {torch.cuda.get_device_name()}  torch {torch.__version__}")
print("(1) kernel vs fp64 reference")
for (B, H, T) in ((2, 4, 300), (1, 2, 2048)):
    q, k, v, dq, dk, dv = inputs(B, H, T, 64)
    o_ref, do_ref = torch.func.jvp(ref_attn, (q, k, v), (dq, dk, dv))
    for prec in ("ieee", "tf32", "tf32x3"):
        os.environ["SLQ_ATTN_PREC"] = prec
        o, do = attn_jvp(*(t.float() for t in (q, k, v, dq, dk, dv)))
        print(f"  B{B} H{H} T{T} {prec:7s} O rel err {rel(o, o_ref):.1e}   dO rel err {rel(do, do_ref):.1e}")
    os.environ["SLQ_ATTN_PREC"] = "ieee"
    q32, k32, v32, dq32, dk32, dv32 = (t.float() for t in (q, k, v, dq, dk, dv))
    with sdpa_kernel(SDPBackend.MATH):
        _, do_math = torch.func.jvp(lambda x, y, z: torch.nn.functional.scaled_dot_product_attention(x, y, z, is_causal=True),
                                    (q32, k32, v32), (dq32, dk32, dv32))
    print(f"  B{B} H{H} T{T} math-SDPA fp32 jvp (current path)  dO rel err {rel(do_math, do_ref):.1e}")

print("(2) autograd wrapper (IEEE)")
q, k, v, dq, dk, dv = inputs(2, 4, 300, 64)
sc = 1 / 8.0
_, do_w = torch.func.jvp(lambda x, y, z: CausalAttention.apply(x, y, z, sc), tuple(t.float() for t in (q, k, v)),
                         tuple(t.float() for t in (dq, dk, dv)))
_, do_ref = torch.func.jvp(ref_attn, (q, k, v), (dq, dk, dv))
print(f"  torch.func.jvp tangent rel err {rel(do_w, do_ref):.1e}")
g = torch.randn_like(q)
qs = [t.float().requires_grad_() for t in (q, k, v)]
gw = torch.autograd.grad(CausalAttention.apply(*qs, sc), qs, g.float())
qr = [t.clone().requires_grad_() for t in (q, k, v)]
gr = torch.autograd.grad(ref_attn(*qr), qr, g)
print("  reverse pass rel err (dq, dk, dv): " + " ".join(f"{rel(x, y):.1e}" for x, y in zip(gw, gr)))
_, fn = torch.func.vjp(lambda x, y, z: CausalAttention.apply(x, y, z, sc), *(t.float() for t in (q, k, v)))
print("  torch.func.vjp rel err (dq, dk, dv):   " + " ".join(f"{rel(x, y):.1e}" for x, y in zip(fn(g.float()), gr)))

print(f"(3) one attention tangent at Enoki shape: {a.heads} heads x T 2048 x D 64")
for B in (2, 8):
    q, k, v, dq, dk, dv = inputs(B, a.heads, 2048, 64, torch.float32)
    for tf in (False, True, "x3"):
        torch.backends.cuda.matmul.allow_tf32 = bool(tf)
        os.environ.pop("SLQ_ATTN_PREC", None)
        if tf == "x3":
            os.environ["SLQ_ATTN_PREC"] = "tf32x3"
        rows = []
        for name in ("math-jvp", "triton"):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                if name == "math-jvp":
                    def f():
                        with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
                            return torch.func.jvp(lambda x, y, z: torch.nn.functional.scaled_dot_product_attention(x, y, z, is_causal=True),
                                                  (q, k, v), (dq, dk, dv))[1]
                else:
                    def f():
                        return attn_jvp(q, k, v, dq, dk, dv)[1]
                f(); torch.cuda.synchronize(); t0 = time.time()
                for _ in range(5):
                    f()
                torch.cuda.synchronize()
                rows.append(f"{name} {1e3 * (time.time() - t0) / 5:7.2f} ms peak {torch.cuda.max_memory_allocated() / 2 ** 30:5.2f} GiB")
            except torch.OutOfMemoryError:
                rows.append(f"{name} OOM")
        print(f"  B{B} {'tf32x3' if tf == 'x3' else 'tf32' if tf else 'fp32'}: " + "   ".join(rows))
print("done")
