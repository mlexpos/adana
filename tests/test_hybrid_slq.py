"""CPU unit tests for hybrid-slq.  Run from the repo root:  python tests/test_hybrid_slq.py"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from optim.adana_slq import ADanaSLQ                          # noqa: E402
from optim.hybrid_slq import HybridSLQ                         # noqa: E402
from optim.muon import zeropower_via_newtonschulz5            # noqa: E402
from tests.test_muon_dana_slq import _model, _train             # noqa: E402

torch.set_default_dtype(torch.float64)


def test_first_step():
    """No quadrature yet (A = 0): hidden matrices move by s msign_lr c NS5(g + beta buf), buf = g; every other tensor by
    s va lr g / (sqrt(v) + eps), v = Delta_1 g^2."""
    V = 10001
    model, names, params = _model(V=V)
    o = HybridSLQ([{"params": params, "param_names": names}], lr=0.01, msign_lr=0.003, beta=0.9, va=0.5)
    x = torch.randint(0, V, (4, 6)); y = (x + 1) % V
    o.zero_grad(); model(x, targets=y)["loss"].backward()
    before = [p.detach().clone() for p in params]
    grads = [p.grad.clone() for p in params]
    o.step()
    D1 = 8.0 / 9.0
    for n, p, b, g, um in zip(names, params, before, grads, o.use_muon):
        assert um == (p.ndim >= 2 and "wte" not in n and "lm_head" not in n), n
        if um:
            want = b - 0.003 * 0.2 * math.sqrt(max(p.shape)) * zeropower_via_newtonschulz5((1.9 * g).bfloat16(), 5).double()
        else:
            want = b - 0.01 * 0.5 * g / (torch.sqrt(D1 * g * g) + 1e-8)
        assert torch.allclose(p.detach(), want, rtol=1e-6, atol=1e-12), n


def test_first_step_adamw_plain():
    """plain='adamw': the non-hidden tensors take d-muon's AdamW step (bias-corrected first step = va lr sign-like g/|g|)."""
    V = 10001
    model, names, params = _model(V=V)
    o = HybridSLQ([{"params": params, "param_names": names}], lr=0.01, msign_lr=0.003, va=2.0, plain="adamw")
    x = torch.randint(0, V, (4, 6)); y = (x + 1) % V
    o.zero_grad(); model(x, targets=y)["loss"].backward()
    before = [p.detach().clone() for p in params]
    grads = [p.grad.clone() for p in params]
    o.step()
    for n, p, b, g, um in zip(names, params, before, grads, o.use_muon):
        if not um:
            a1, a2 = 0.2 * g, 0.001 * g * g
            want = b - 0.01 * 2.0 * (a1 / (a2.sqrt() + 1e-8)) / (0.2 / 0.001 ** 0.5)
            assert torch.allclose(p.detach(), want, rtol=1e-6, atol=1e-12), n


def test_no_muon_tensors_is_adana_slq():
    """With no tensor on the msign path and va = 1, hybrid-slq is adana-slq step for step (same rule, refreshes, moments)."""
    kw = dict(lr=0.02, s=0.5, alloc="waterfill", m_max=12, probes=1, slq_batch=8, first_refresh=2, teff=1.0, batch_seqs=8)
    m1, n1, p1 = _model(seed=3); m2, n2, p2 = _model(seed=3)
    a = ADanaSLQ([{"params": p1, "param_names": n1}], **kw)
    h = HybridSLQ([{"params": p2, "param_names": n2}], msign_lr=0.0, va=1.0, **kw)
    h.use_muon = [False] * len(h.plist)
    l1 = _train(a, m1, p1, 8, refresh=True); l2 = _train(h, m2, p2, 8, refresh=True)
    assert np.allclose(l1, l2, rtol=1e-6), (l1, l2)          # adana-slq passes its scalars as float32
    assert a.n_refresh > 0 and np.allclose(a.A, h.A)
    for x1, x2 in zip(p1, p2):
        assert torch.allclose(x1, x2, rtol=1e-6, atol=1e-7)


def test_trains_with_refresh():
    """Hidden matrices on msign, waterfill rule live: finite, decreasing loss and a nonzero long momentum after a refresh."""
    model, names, params = _model(seed=5)
    o = HybridSLQ([{"params": params, "param_names": names}], lr=0.01, msign_lr=0.01, s=0.5, alloc="waterfill",
                  m_max=12, probes=1, slq_batch=8, first_refresh=2, teff=1.0, batch_seqs=8)
    assert any(o.use_muon) and not all(o.use_muon)
    losses = _train(o, model, params, 30, refresh=True)
    assert all(np.isfinite(losses))
    assert np.mean(losses[-5:]) < np.mean(losses[:5]), losses
    assert o.n_refresh > 0 and float(np.max(o.A)) > 0
    d = o.diagnostics()
    assert d["hyb/msign_over_gamma0"] == 1.0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print(k, "ok")
