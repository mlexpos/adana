"""Causal attention with a fused forward-mode derivative, for Gauss-Newton products (slq_torch.GaussNewtonOperator, attn="triton").

torch.func.jvp cannot go through the flash / memory-efficient SDPA kernels, so the products used the math backend: fp32 T x T
attention matrices per head, chunks of 2 sequences.  Here the tangent of O = softmax(s Q K^T + causal) V along (dQ, dK, dV) is
computed tile by tile with the online softmax of flash attention.  With P = softmax(S), dS = s (dQ K^T + Q dK^T),
    dO = P dV + (P o dS) V - diag(rowsum(P o dS)) O,
so per query tile the kernel keeps, besides the running max m and sum l, the unnormalised accumulators sum p v, sum p dv,
sum (p ds) v and the row sums of p ds, all rescaled together when m moves; nothing of size T x T is stored.  Matmuls run in TF32
when torch.backends.cuda.matmul.allow_tf32 is set and in tf32x3 otherwise (Triton's three-pass split-operand product, fp32
accuracy on tensor cores); SLQ_ATTN_PREC = ieee | tf32 | tf32x3 overrides.
The primal forward and the reverse pass (the vjp half of the product) use PyTorch's memory-efficient SDPA, which supports fp32.
triton_causal_sdpa() patches F.scaled_dot_product_attention for the duration of a product; calls it does not cover (masks,
dropout, other dtypes, head dims that are not a power of two >= 16) fall through to the original function."""
import contextlib
import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_SDPA = F.scaled_dot_product_attention


@triton.jit
def _causal_attn_jvp(Q, K, V, DQ, DK, DV, O, DO, scale, sb, sh, st, H, T,
                     D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    base = (bh // H).to(tl.int64) * sb + (bh % H).to(tl.int64) * sh
    rm = pid_m * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, D)
    mrow = rm < T
    qp = base + rm[:, None].to(tl.int64) * st + rd[None, :]
    q = tl.load(Q + qp, mask=mrow[:, None], other=0.0)
    dq = tl.load(DQ + qp, mask=mrow[:, None], other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    c_i = tl.zeros([BM], tl.float32)
    ao = tl.zeros([BM, D], tl.float32)
    ab = tl.zeros([BM, D], tl.float32)
    ac = tl.zeros([BM, D], tl.float32)
    hi = tl.minimum((pid_m + 1) * BM, T)
    for n0 in range(0, hi, BN):
        cn = n0 + rn
        mcol = cn < T
        kp = base + cn[:, None].to(tl.int64) * st + rd[None, :]
        k = tl.load(K + kp, mask=mcol[:, None], other=0.0)
        dk = tl.load(DK + kp, mask=mcol[:, None], other=0.0)
        v = tl.load(V + kp, mask=mcol[:, None], other=0.0)
        dv = tl.load(DV + kp, mask=mcol[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision=PREC) * scale
        ds = (tl.dot(dq, tl.trans(k), input_precision=PREC) + tl.dot(q, tl.trans(dk), input_precision=PREC)) * scale
        ok = (rm[:, None] >= cn[None, :]) & mcol[None, :]
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        pds = p * tl.where(ok, ds, 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        c_i = c_i * alpha + tl.sum(pds, 1)
        ao = ao * alpha[:, None] + tl.dot(p, v, input_precision=PREC)
        ab = ab * alpha[:, None] + tl.dot(p, dv, input_precision=PREC)
        ac = ac * alpha[:, None] + tl.dot(pds, v, input_precision=PREC)
        m_i = m_new
    o = ao / l_i[:, None]
    do = (ab + ac) / l_i[:, None] - (c_i / l_i)[:, None] * o
    tl.store(O + qp, o, mask=mrow[:, None])
    tl.store(DO + qp, do, mask=mrow[:, None])


def _prec():
    # with TF32 off the kernel runs tf32x3: fp32 accuracy on tensor cores (IEEE fp32 in Triton is ~75x slower on H100)
    p = os.environ.get("SLQ_ATTN_PREC", "")
    return p if p else ("tf32" if torch.backends.cuda.matmul.allow_tf32 else "tf32x3")


def attn_jvp(q, k, v, dq, dk, dv, scale=None):
    """(O, dO) of causal attention at (q, k, v) along (dq, dk, dv); all (B, H, T, D), computed in fp32.  A torch.library op, so
    that inside torch.func transforms the dispatcher unwraps functorch tensors before the Triton launch."""
    return _attn_jvp_op(q, k, v, dq, dk, dv, 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale), _prec())


@torch.library.custom_op("slq_attn::causal_jvp", mutates_args=())
def _attn_jvp_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor,
                 scale: float, prec: str) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, T, D = q.shape
    ts = [t.float().contiguous() for t in (q, k, v, dq, dk, dv)]
    o, do = torch.empty_like(ts[0]), torch.empty_like(ts[0])
    key = (D, torch.cuda.current_device())
    for BM, BN, ns in ([_CFG[key]] if key in _CFG else _CONFIGS):      # largest tiles the GPU's shared memory holds
        try:
            _causal_attn_jvp[(triton.cdiv(T, BM), B * H)](*ts, o, do, scale, ts[0].stride(0), ts[0].stride(1), ts[0].stride(2), H, T,
                                                         D=D, BM=BM, BN=BN, PREC=prec, num_warps=4, num_stages=ns)
            _CFG[key] = (BM, BN, ns)
            break
        except triton.runtime.errors.OutOfResources:
            continue
    else:
        raise RuntimeError(f"attn_jvp: no tile configuration fits (head dim {D})")
    return o.to(q.dtype), do.to(q.dtype)


@_attn_jvp_op.register_fake
def _(q, k, v, dq, dk, dv, scale, prec):
    return torch.empty_like(q), torch.empty_like(q)


_CONFIGS = [(64, 64, 2), (64, 32, 2), (64, 32, 1), (32, 32, 1)]
_CFG = {}


class CausalAttention(torch.autograd.Function):
    """Causal SDPA whose forward-mode derivative is the Triton kernel above (primal and reverse pass: memory-efficient SDPA)."""

    @staticmethod
    def forward(q, k, v, scale):
        return _SDPA(q, k, v, is_causal=True, scale=scale)

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v, scale = inputs
        ctx.save_for_backward(q, k, v)
        ctx.save_for_forward(q, k, v)
        ctx.scale = scale

    @staticmethod
    def backward(ctx, go):
        q, k, v = ctx.saved_tensors
        _, fn = torch.func.vjp(lambda a, b, c: _SDPA(a, b, c, is_causal=True, scale=ctx.scale), q, k, v)
        return (*fn(go), None)

    @staticmethod
    def jvp(ctx, dq, dk, dv, _dscale):
        q, k, v = ctx.saved_tensors
        z = lambda t, ref: torch.zeros_like(ref) if t is None else t
        return attn_jvp(q, k, v, z(dq, q), z(dk, k), z(dv, v), ctx.scale)[1]


def _covered(q, k, v, attn_mask, dropout_p, is_causal, kw):
    D = q.shape[-1]
    return (is_causal and attn_mask is None and dropout_p == 0.0 and not kw and q.dtype == torch.float32 and q.dim() == 4
            and q.shape == k.shape == v.shape and D >= 16 and D & (D - 1) == 0 and q.is_cuda)


@contextlib.contextmanager
def triton_causal_sdpa():
    """Route covered causal SDPA calls through CausalAttention while the context is open."""
    orig = F.scaled_dot_product_attention

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        if _covered(q, k, v, attn_mask, dropout_p, is_causal, kw):
            return CausalAttention.apply(q, k, v, 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale)
        return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)

    F.scaled_dot_product_attention = sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig
