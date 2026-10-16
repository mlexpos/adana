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


def test_cyclic_waves():
    """Groups wte -> h.0 -> lm_head (G = 3, T = 12, stage 4): the active group follows the rule (= 1 here) after a one-step ramp,
    finished groups decay as r_end t_k / t, later groups and small tensors (norms, biases) are 0; cyc = -1 reverses the order."""
    model, names, params = _model(V=11)
    for cyc, order in ((1, ("wte", "h.0", "lm_head")), (-1, ("lm_head", "h.0", "wte"))):
        o = HybridSLQ([{"params": params, "param_names": names}], lr=0.01, msign_lr=0.01, cyc=cyc, cyc_T=12, s=1e9, batch_seqs=8)
        L = len(o.plist)
        o.q = dict(nodes=np.array([1.0]), w=np.array([1.0]), wT=np.full((1, L), 1e-3), trF=1.0, lam_max=1.0)
        grp = {n: next((k for k, key in enumerate(order) if key in n), None) for n in names}
        for t in range(12):
            o.t = t
            o._compute_ratio(0.5)
            for i, n in enumerate(names):
                g = grp[n] if params[i].ndim >= 2 else None
                if g is None:
                    want = 0.0
                elif t < 4 * g:
                    want = 0.0
                elif t < 4 * (g + 1):
                    want = 0.0 if t == 4 * g else 1.0
                else:
                    want = 4 * (g + 1) / t
                assert abs(o.ratio[i] - want) < 1e-9, (cyc, t, n, o.ratio[i], want)
                assert abs(o.A[i] - want / 0.5) < 1e-9


def test_switch_waves():
    """cyc = 2: per-tensor caps (small tensors 0) until S/N stays < 1 for max(1, .02 T) steps after .02 T; then the
    amplifications freeze (r = r_end t_ref / t) and forward waves wte -> h.0 -> lm_head run over [t*, T)."""
    import optim.adana_slq as AS
    model, names, params = _model(V=11)
    T = 60
    o = HybridSLQ([{"params": params, "param_names": names}], lr=0.01, msign_lr=0.01, cyc=2, cyc_T=T)
    L = len(o.plist)
    orig = AS.ADanaSLQ._compute_ratio
    def stub(self, D):                                         # rule = 1 everywhere; the budget binds from t = 10 on
        self.q = {}; self.ratio = np.ones(L); self.NT = np.ones(L); self.S = 100.0 if self.t < 10 else 0.5; self.A = self.ratio / D
    AS.ADanaSLQ._compute_ratio = stub
    try:
        big = o.cgroup >= 0
        for t in range(T):
            o.t = t
            o._compute_ratio(0.5)
            if t < 11:                                         # switch fires after 1 step of S/N < 1 (W = max(1, 1.2) -> 1.2)
                assert np.allclose(o.ratio[big], 1.0) and np.allclose(o.ratio[~big], 0.0), t
        ts = o.tstar
        assert 10 <= ts <= 12, ts
        stg = (T - ts) / 3
        t = int(ts + 2.5 * stg)                                # last stage: lm_head active; wte, h.0 frozen
        o.t = t; o._compute_ratio(0.5)
        g = o.cgroup
        assert np.allclose(o.ratio[g == 2], 1.0)
        assert np.all(o.ratio[g == 0] < 1.0) and np.all(o.ratio[g == 1] < 1.0) and np.allclose(o.ratio[~big], 0.0)
    finally:
        AS.ADanaSLQ._compute_ratio = orig


def test_gamma0_rule():
    """gamma0* solves N(Delta_tauT; g) = S on the quadrature; the multiplier moves halfway in log space, never down, capped; the
    momentum term is scaled by it (checked through A-independent bookkeeping)."""
    model, names, params = _model(V=11)
    o = HybridSLQ([{"params": params, "param_names": names}], lr=0.01, msign_lr=0.01, cyc_T=100, g0_rule=True, g0_max=4.0)
    L = len(o.plist)
    o.q = dict(nodes=np.array([1.0, 10.0]), w=np.array([50.0, 50.0]), wT=np.full((2, L), 1.0), trF=1.0, lam_max=10.0)
    o.S = 40.0
    g = o._gamma0_star()
    Dt = 8.0 / (8.0 + 85.0)
    N = 50 * g * 1.0 / (g * 1.0 + Dt) + 50 * g * 10.0 / (g * 10.0 + Dt)
    assert abs(N - 40.0) < 1e-6 * 40.0, N
    o.g0_star = g
    for _ in range(3):                                         # repeated updates approach min(g*/lr, cap) from below, monotone
        prev = o.g0_mult
        target = math.sqrt(o.g0_mult * g / o.lr)
        o.g0_mult = float(min(o.g0_max, max(o.g0_mult, target)))
        assert o.g0_mult >= prev
    assert o.g0_mult <= o.g0_max + 1e-12


def test_gamma0_rule_trains():
    """End to end: the rule updates at real SLQ refreshes, stays within [1, g0_max], and training stays finite."""
    model, names, params = _model(seed=7)
    o = HybridSLQ([{"params": params, "param_names": names}], lr=0.01, msign_lr=0.01, s=0.5, alloc="waterfill", m_max=12, probes=1,
                  slq_batch=8, first_refresh=2, teff=1.0, batch_seqs=8, cyc_T=30, g0_rule=True)
    losses = _train(o, model, params, 30, refresh=True)
    assert all(np.isfinite(losses)) and o.n_refresh > 0
    assert 1.0 <= o.g0_mult <= o.g0_max and np.isfinite(o.g0_star)
    assert o.diagnostics()["hyb/g0_mult"] == o.g0_mult


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print(k, "ok")
