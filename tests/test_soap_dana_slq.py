"""CPU unit tests for soap-dana-slq.  Run from the repo root:  python tests/test_soap_dana_slq.py"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from optim.slq_torch import GaussNewtonOperator, slq   # noqa: E402
from optim.soap import SOAP                            # noqa: E402
from optim.soap_dana_slq import SOAPDanaSLQ, _rot, _unrot  # noqa: E402

torch.set_default_dtype(torch.float64)


class _Blk(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = nn.Module(); self.attn.c_attn = nn.Linear(d, 3 * d, bias=False)
        self.mlp = nn.Module(); self.mlp.c_fc = nn.Linear(d, d)


class TinyQKV(nn.Module):
    """Enoki-style names (transformer.wte, transformer.h.0.attn.c_attn, lm_head) and the Enoki forward interface."""

    def __init__(self, V=11, d=4):
        super().__init__()
        self.d = d
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(V, d)
        self.transformer.h = nn.ModuleList([_Blk(d)])
        self.lm_head = nn.Linear(d, V, bias=False)

    def forward(self, idx, targets=None, get_logits=False):
        x = self.transformer.wte(idx)
        b = self.transformer.h[0]
        q, k, v = b.attn.c_attn(b.ln_1(x)).chunk(3, -1)
        T = idx.shape[1]
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d)
        att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf")).softmax(-1)
        x = x + att @ v
        x = x + torch.tanh(b.mlp.c_fc(x))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return {"logits": logits if get_logits else None, "loss": loss}


def _model(seed=0):
    torch.manual_seed(seed)
    m = TinyQKV()
    # a fresh LayerNorm outputs zero-mean features, so every QKV gradient has an exact null direction whose rotated
    # coordinate is rounding noise (which g / (|g| + eps) then amplifies differently in float32 and float64)
    with torch.no_grad():
        ln = m.transformer.h[0].ln_1
        ln.weight.uniform_(0.5, 1.5); ln.bias.uniform_(-0.5, 0.5)
    return m, [n for n, _ in m.named_parameters()], [p for _, p in m.named_parameters()]


def _batch(B=8, T=6):
    x = torch.randint(0, 11, (B, T))
    return x, (x + 1) % 11


def _train(o, model, params, steps, refresh=False, seed=1):
    torch.manual_seed(seed)
    losses = []
    for _ in range(steps):
        x, y = _batch()
        o.zero_grad()
        out = model(x[:4], targets=y[:4]); (out["loss"] / 2).backward()
        half = [p.grad.clone() for p in params]
        out2 = model(x[4:], targets=y[4:]); (out2["loss"] / 2).backward()
        if hasattr(o, "set_split_stats"):
            o.set_split_stats(half, [p.grad for p in params])
        if refresh and o.needs_refresh():
            o.refresh(model, x, y)
        o.step()
        losses.append(float(out["loss"].detach() + out2["loss"].detach()) / 2)
    return losses


def test_block_names_and_first_step():
    model, names, params = _model()
    o = SOAPDanaSLQ([{"params": params, "param_names": names}], lr=0.01, max_precond_dim=8)
    assert o.nsplit[names.index("transformer.h.0.attn.c_attn.weight")] == 3 and sum(o.nsplit) == len(params) + 2
    assert [n for n in o.bnames if "c_attn" in n] == [f"transformer.h.0.attn.c_attn_{c}.weight" for c in "qkv"]
    assert len(o.ratio) == len(o.bnames) and set(o.agroup[[i for i, n in enumerate(o.bnames) if "c_attn" in n]]) == {"attn"}
    before = [p.detach().clone() for p in params]
    _train(o, model, params, 1)
    # first call only initialises SOAP's preconditioner (the projection never uses its own gradient)
    assert o.t == 0 and all(torch.equal(a, p) for a, p in zip(before, params))
    b = o.state[params[names.index("transformer.wte.weight")]]["blocks"][0]
    assert b["QL"] is None and b["QR"].shape == (4, 4)                     # vocab side > max_precond_dim is left out


def test_matches_soap_without_momentum():
    # with no long momentum (no refresh: r_T = 0), un-split QKV and no weight decay, the step is SOAP with beta1 = 0
    m1, names, p1 = _model()
    m2, _, p2 = _model()
    kw = dict(precondition_frequency=3, max_precond_dim=8)
    o1 = SOAPDanaSLQ([{"params": p1, "param_names": names}], lr=0.02, epsilon=1e-12, beta2=0.9, split_qkv=False, **kw)
    o2 = SOAP(p2, lr=0.02, betas=(0.0, 0.9), eps=1e-12, weight_decay=0.0, **kw)
    _train(o1, m1, p1, 11)
    _train(o2, m2, p2, 11)
    for a, b in zip(p1, p2):                    # SOAP computes its eigenbases in float32
        assert float((a - b).norm() / b.norm()) < 1e-6, float((a - b).norm() / b.norm())


def test_rz_is_symmetric_square_root_of_preconditioner():
    model, names, params = _model()
    o = SOAPDanaSLQ([{"params": params, "param_names": names}], lr=0.02, precondition_frequency=3)
    _train(o, model, params, 7)
    r = o._slq_r()
    assert sum(callable(ri) for ri in r) == 4                                # wte, c_attn, c_fc, lm_head
    torch.manual_seed(3)
    U = [torch.randn_like(p) for p in params]; W = [torch.randn_like(p) for p in params]
    ap = lambda ri, x: ri(x) if callable(ri) else ri * x
    lhs = sum(float((ap(ri, u) * w).sum()) for ri, u, w in zip(r, U, W))
    rhs = sum(float((u * ap(ri, w)).sum()) for ri, u, w in zip(r, U, W))
    assert abs(lhs - rhs) < 1e-10 * abs(lhs)
    # R R U = P^{-1} U, block by block (q, k, v rows of the fused matrix each with their own basis)
    for i, (p, u, ri) in enumerate(zip(params, U, r)):
        st = o.state[p]
        want = torch.cat([_unrot(_rot(ub, b["QL"], b["QR"]) / o._den(b, st["step"], o.epsilon), b["QL"], b["QR"])
                          for ub, b in zip(o._chunks(i, u), st["blocks"])], 0)
        got = ap(ri, ap(ri, u))
        assert float((got - want).norm() / want.norm()) < 1e-10


def test_split_quadrature_vs_dense():
    model, names, params = _model()
    o = SOAPDanaSLQ([{"params": params, "param_names": names}], lr=0.02)
    _train(o, model, params, 5)
    x, y = _batch()
    op = GaussNewtonOperator(model, names, params, x, y, r=o._slq_r())
    sizes = [p.numel() for p in params]
    n = sum(sizes)
    M = np.zeros((n, n))
    for j in range(n):
        e = torch.zeros(n); e[j] = 1.0
        u, k = [], 0
        for p, s in zip(params, sizes):
            u.append(e[k:k + s].view_as(p)); k += s
        M[:, j] = torch.cat([q.reshape(-1) for q in op(u)]).numpy()
    assert np.abs(M - M.T).max() < 1e-8 * np.abs(M).max()
    lam, V = np.linalg.eigh(0.5 * (M + M.T))
    g, D = 0.7, 0.05
    fM = V @ np.diag(g * np.maximum(lam, 0) / (g * np.maximum(lam, 0) + D)) @ V.T
    res, res1 = [slq(op, params, m_max=n, probes=1, eps=1e-9, g=g, D_check=[D], generator=torch.Generator().manual_seed(7),
                     splits=sp) for sp in (o.nsplit, None)]
    gen = torch.Generator(); gen.manual_seed(7)
    z = torch.cat([(torch.randint(0, 2, p.shape, generator=gen, dtype=torch.int8).double() * 2 - 1).reshape(-1)
                   for p in params]).numpy()
    th, w, wT = res["nodes"][0], res["weights"][0], res["block"][0]
    f = g * np.maximum(th, 0) / (g * np.maximum(th, 0) + D)
    exact = z @ fM @ z
    assert abs(f @ w - exact) / exact < 1e-6
    assert wT.shape[1] == len(o.bnames)
    # per-block split (q, k, v row chunks of the fused matrix): z_T^T f(H) z
    k, T = 0, 0
    for p, s, ns in zip(params, sizes, o.nsplit):
        for c in range(ns):
            lo, hi = k + c * s // ns, k + (c + 1) * s // ns
            zT = np.zeros_like(z); zT[lo:hi] = z[lo:hi]
            assert abs(f @ wT[:, T] - zT @ fM @ z) <= 1e-3 * abs(exact), o.bnames[T]   # as test_adana_slq
            T += 1
        k += s
    # the blocks partition the unsplit per-tensor weights
    qi = names.index("transformer.h.0.attn.c_attn.weight")
    assert np.allclose(res["block"][0][:, o.bidx[qi]].sum(1), res1["block"][0][:, qi], atol=1e-10)


def test_basis_refresh_keeps_momentum():
    model, names, params = _model()
    o = SOAPDanaSLQ([{"params": params, "param_names": names}], lr=0.02, precondition_frequency=1000)
    _train(o, model, params, 6)
    b = o.state[params[names.index("transformer.h.0.mlp.c_fc.weight")]]["blocks"][0]
    m_par = _unrot(b["m"], b["QL"], b["QR"]).clone()
    QL0 = b["QL"].clone()
    o._refresh_basis(b)
    assert not torch.allclose(QL0, b["QL"])
    assert torch.allclose(b["QL"].T @ b["QL"], torch.eye(4), atol=1e-12)
    assert float((_unrot(b["m"], b["QL"], b["QR"]) - m_par).norm()) < 1e-12 * float(m_par.norm())


def test_momentum_matches_parameter_space_dana():
    # fixed amplification A per block, the same gradients fed to A = 0 and A > 0: the A > 0 step equals
    # u + A m with m <- (1 - Delta) m + Delta u kept in parameter coordinates (through basis refreshes)
    torch.manual_seed(5)
    names = ["transformer.h.0.attn.c_attn.weight", "transformer.h.0.mlp.c_fc.weight", "transformer.h.0.ln_1.weight"]
    shapes = [(12, 4), (6, 4), (4,)]
    ps = [[torch.randn(s, requires_grad=True) for s in shapes] for _ in range(2)]
    with torch.no_grad():
        for a, b in zip(*ps):
            b.copy_(a)
    os_ = [SOAPDanaSLQ([{"params": p, "param_names": names}], lr=0.03, precondition_frequency=2) for p in ps]
    A = np.array([0.0, 1.5])
    for k, o in enumerate(os_):
        o._compute_ratio = (lambda kk: lambda D: setattr(os_[kk], "A", np.full(len(os_[kk].bnames), A[kk])))(k)
    m_ref = [torch.zeros(s) for s in shapes]
    for t in range(9):
        gs = [torch.randn(s) for s in shapes]
        before = [[q.detach().clone() for q in p] for p in ps]
        for o, p in zip(os_, ps):
            for q, g in zip(p, gs):
                q.grad = g.clone()
            o.step()
        if t == 0:
            continue
        Dm = os_[0].delta / (os_[0].delta + t)
        for i in range(len(shapes)):
            u = -(ps[0][i].detach() - before[0][i]) / 0.03
            m_ref[i] = (1 - Dm) * m_ref[i] + Dm * u
            want = before[1][i] - 0.03 * (u + A[1] * m_ref[i])
            assert float((ps[1][i].detach() - want).norm() / want.norm()) < 1e-10, (t, names[i])
    assert os_[1].n_basis > 0


def test_refresh_and_train_smoke():
    model, names, params = _model()
    o = SOAPDanaSLQ([{"params": params, "param_names": names}], lr=0.02, kprime=16.0, alloc="waterfill_v2", m_max=16,
                    probes=1, slq_batch=4, batch_seqs=8, max_gap=3, precondition_frequency=3, weight_decay=0.01,
                    wd_decaying=True, wd_ts=5.0)
    losses = _train(o, model, params, 30, refresh=True)
    assert o.n_refresh >= 3 and np.isfinite(losses).all() and np.mean(losses[-5:]) < np.mean(losses[:5]), losses
    assert o.n_basis > 0 and (o.A > 0).any() and o.NT.shape == (len(o.bnames),)
    d = o.diagnostics()
    assert np.isfinite(d["slq/N"]) and "slq_type/ratio/attn.c_attn_q" in d and d["slq/last_accepted"] == 1.0
    assert all(np.isfinite(v) for v in o.mz2) and (o.mz2 > 0).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
