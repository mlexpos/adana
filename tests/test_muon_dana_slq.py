"""CPU unit tests for muon-dana-slq.  Run from the repo root:  python tests/test_muon_dana_slq.py"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from optim.muon import zeropower_via_newtonschulz5          # noqa: E402
from optim.muon_dana_slq import MuonDanaSLQ, _polar_factors, _polar_pow   # noqa: E402
from optim.slq_torch import GaussNewtonOperator, slq          # noqa: E402
from tests.test_soap_dana_slq import TinyQKV, _batch          # noqa: E402

torch.set_default_dtype(torch.float64)


def _model(V=11, seed=0):
    torch.manual_seed(seed)
    m = TinyQKV(V=V)
    with torch.no_grad():
        ln = m.transformer.h[0].ln_1
        ln.weight.uniform_(0.5, 1.5); ln.bias.uniform_(-0.5, 0.5)
    return m, [n for n, _ in m.named_parameters()], [p for _, p in m.named_parameters()]


def _train(o, model, params, steps, refresh=False, seed=1, V=11):
    torch.manual_seed(seed)
    losses = []
    for _ in range(steps):
        x = torch.randint(0, V, (8, 6)); y = (x + 1) % V
        o.zero_grad()
        out = model(x[:4], targets=y[:4]); (out["loss"] / 2).backward()
        half = [p.grad.clone() for p in params]
        out2 = model(x[4:], targets=y[4:]); (out2["loss"] / 2).backward()
        o.set_split_stats(half, [p.grad for p in params])
        if refresh and o.needs_refresh():
            o.refresh(model, x, y)
        o.step()
        losses.append(float(out["loss"].detach() + out2["loss"].detach()) / 2)
    return losses


def test_polar_derivative():
    """A = D polar(G): finite differences of U V^T, symmetric PSD, A G = 0 (scale invariance), (A^1/2)^2 = A."""
    rng = np.random.default_rng(0)
    polar = lambda M: (lambda u, s, vt: u @ vt)(*np.linalg.svd(M, full_matrices=False))
    for shape in ((5, 8), (8, 5), (6, 6)):
        G = rng.normal(size=shape)
        f = _polar_factors(torch.tensor(G), 1e-12)
        E1, E2 = (torch.tensor(rng.normal(size=shape)) for _ in range(2))
        h = 1e-6
        fd = (polar(G + h * E1.numpy()) - polar(G - h * E1.numpy())) / (2 * h)
        A1 = _polar_pow(f, E1, 1.0).double().numpy()
        assert np.linalg.norm(A1 - fd) / np.linalg.norm(fd) < 1e-5, shape
        A2 = _polar_pow(f, E2, 1.0).double().numpy()
        assert abs((A1 * E2.numpy()).sum() - (E1.numpy() * A2).sum()) < 1e-6 * np.linalg.norm(A1) * float(E2.norm())
        assert (A1 * E1.numpy()).sum() > 0
        assert float(_polar_pow(f, torch.tensor(G), 1.0).norm()) < 1e-8 * np.linalg.norm(A1)
        assert np.allclose(_polar_pow(f, _polar_pow(f, E1, 0.5), 0.5).double().numpy(), A1, atol=1e-8 * np.abs(A1).max())


def test_first_step_is_dmuon_update():
    """No long momentum yet: hidden matrices move by lr * 0.2 sqrt(max dim) * NS5(g) (d-muon with momentum 0); wte and
    lm_head (first dim >= 10000) get the RMSProp step instead."""
    V = 10001
    model, names, params = _model(V=V)
    o = MuonDanaSLQ([{"params": params, "param_names": names}], lr=0.01, beta2=0.999)
    assert [n for n, u in zip(names, o.use_muon) if not u] == [n for n in names if p_dim(model, n) < 2 or
                                                                 "wte" in n or "lm_head" in n]
    x = torch.randint(0, V, (4, 6)); y = (x + 1) % V
    o.zero_grad(); model(x, targets=y)["loss"].backward()
    before = [p.detach().clone() for p in params]
    grads = [p.grad.clone() for p in params]
    o.step()
    for n, p, b, g, um in zip(names, params, before, grads, o.use_muon):
        if um:
            want = b - 0.01 * 0.2 * math.sqrt(max(p.shape)) * zeropower_via_newtonschulz5(g.bfloat16(), 5).double()
        else:
            want = b - 0.01 * g / (g.abs() + 1e-8)           # RMSProp at step 1: v_hat = g^2
        assert torch.allclose(p.detach(), want, atol=1e-6), n


def p_dim(model, name):
    return dict(model.named_parameters())[name].dim()


def test_quadrature_operator_is_symmetric_square_root():
    """R = A^{1/2} from the current gradient: R R = A (= c D polar(G) on matrices), R symmetric, and the per-tensor
    quadrature of R GN R matches a dense eigendecomposition."""
    model, names, params = _model()
    o = MuonDanaSLQ([{"params": params, "param_names": names}], lr=0.02)
    _train(o, model, params, 5)
    x, y = _batch()
    o.zero_grad(); model(x, targets=y)["loss"].backward()
    r = o._slq_r()
    ap = lambda ri, u: ri(u) if callable(ri) else ri * u
    torch.manual_seed(3)
    for i, (p, ri) in enumerate(zip(params, r)):
        U, W = torch.randn_like(p), torch.randn_like(p)
        assert abs(float((ap(ri, U) * W).sum() - (U * ap(ri, W)).sum())) < 1e-8 * float(ap(ri, U).norm() * W.norm())
        if o.use_muon[i]:
            f = _polar_factors(p.grad, o.polar_floor)
            want = o._scale(p) * _polar_pow(f, U, 1.0)
            assert torch.allclose(ap(ri, ap(ri, U)), want, atol=1e-8 * float(want.abs().max())), names[i]
    op = GaussNewtonOperator(model, names, params, x, y, r=r)
    sizes = [p.numel() for p in params]
    n = sum(sizes)
    M = np.zeros((n, n))
    for j in range(n):
        e = torch.zeros(n); e[j] = 1.0
        u, k = [], 0
        for p, s_ in zip(params, sizes):
            u.append(e[k:k + s_].view_as(p)); k += s_
        M[:, j] = torch.cat([q.reshape(-1) for q in op(u)]).numpy()
    assert np.abs(M - M.T).max() < 1e-8 * np.abs(M).max()
    lam, Vv = np.linalg.eigh(0.5 * (M + M.T))
    g, D = 0.7, 0.05
    fM = Vv @ np.diag(g * np.maximum(lam, 0) / (g * np.maximum(lam, 0) + D)) @ Vv.T
    res = slq(op, params, m_max=n, probes=1, eps=1e-9, g=g, D_check=[D], generator=torch.Generator().manual_seed(7))
    gen = torch.Generator(); gen.manual_seed(7)
    z = torch.cat([(torch.randint(0, 2, p.shape, generator=gen, dtype=torch.int8).double() * 2 - 1).reshape(-1)
                   for p in params]).numpy()
    th, w, wT = res["nodes"][0], res["weights"][0], res["block"][0]
    f_ = g * np.maximum(th, 0) / (g * np.maximum(th, 0) + D)
    exact = z @ fM @ z
    assert abs(f_ @ w - exact) / exact < 1e-6
    k = 0
    for T, s_ in enumerate(sizes):
        zT = np.zeros_like(z); zT[k:k + s_] = z[k:k + s_]; k += s_
        assert abs(f_ @ wT[:, T] - zT @ fM @ z) <= 1e-3 * abs(exact), names[T]


def test_refresh_and_train_smoke():
    model, names, params = _model()
    o = MuonDanaSLQ([{"params": params, "param_names": names}], lr=0.02, kprime=16.0, alloc="global", m_max=16,
                    probes=1, slq_batch=4, batch_seqs=8, max_gap=3, weight_decay=0.01, wd_decaying=True, wd_ts=5.0)
    losses = _train(o, model, params, 30, refresh=True)
    assert o.n_refresh >= 3 and np.isfinite(losses).all() and np.mean(losses[-5:]) < np.mean(losses[:5]), losses
    assert (o.A > 0).any() and np.isfinite(o.diagnostics()["slq/N"])
    try:
        MuonDanaSLQ([{"params": params, "param_names": names}], lr=0.02, alloc="waterfill_v2")
        raise AssertionError("water-filling must be refused")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
