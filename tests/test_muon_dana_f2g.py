"""CPU unit tests for muon-dana-f2g.  Run from the repo root:  python tests/test_muon_dana_f2g.py"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from optim.muon import zeropower_via_newtonschulz5                       # noqa: E402
from optim.muon_dana_f2g import MuonDanaF2G, dns, h_seq, h_tok, kappa, lanczos, rule_r   # noqa: E402
from tests.test_soap_dana_slq import TinyQKV                             # noqa: E402

torch.set_default_dtype(torch.float64)


def _model(V=11, seed=0):
    torch.manual_seed(seed)
    m = TinyQKV(V=V)
    with torch.no_grad():
        ln = m.transformer.h[0].ln_1
        ln.weight.uniform_(0.5, 1.5); ln.bias.uniform_(-0.5, 0.5)
    names = [n for n, _ in m.named_parameters()]
    params = [p for _, p in m.named_parameters()]
    return m, names, params


def _opt(params, names, **kw):
    groups = [dict(params=params, param_names=names)]
    return MuonDanaF2G(groups, lr=0.05, weight_decay=0.01, **kw)


def _batch(B=8, T=6, V=11):
    x = torch.randint(0, V, (B, T))
    return x, (x + 1) % V


def test_kappa_matches_lyapunov():
    """kappa(h; r, Delta) = h^2 E[theta^2] / v for the per-mode DANA recursion (LaProp order, unit noise)."""
    for h, D, eps, r in [(3.0, 0.05, 0.01, 0.0), (3.0, 0.05, 0.01, 0.4), (50.0, 0.01, 0.003, 1.0), (0.2, 0.2, 0.02, 0.7)]:
        M = np.array([[1 - eps * h * (1 + r), -(eps * r / D) * (1 - D)], [D * h, 1 - D]])
        w = np.array([-eps * (1 + r), D])
        S = np.linalg.solve(np.eye(4) - np.kron(M, M), np.outer(w, w).ravel()).reshape(2, 2)
        ref = h * h * S[0, 0]
        got = float(kappa(np.array(h), D, eps, r))
        assert abs(got - ref) < 1e-9 * abs(ref), (h, D, eps, r, got, ref)
    # small eps h: kappa ~ eps h (Delta + r) / (2 Delta)
    h, D, eps, r = 1.0, 0.1, 1e-4, 0.3
    assert abs(float(kappa(np.array(h), D, eps, r)) / (eps * h * (D + r) / (2 * D)) - 1) < 1e-2


def test_rule_limits():
    w = np.array([[1.0, 0.1, 0.01, 0.0]]); tau = np.array([[0.3, 0.3, 0.4, 0.0]])
    a0, eps, D = np.array([2.0]), np.array([0.01]), 0.05
    r, K0 = rule_r(w, tau, a0, D, eps, np.array([0.0]), 0.2)
    assert r[0] == 0.0                                           # E below the floor: no momentum
    r, _ = rule_r(w, tau, a0, D, eps, np.array([1e9]), 0.2)
    assert r[0] == 1.0                                           # huge excess: full momentum (stable here)
    r, K0 = rule_r(w, tau, a0, D, eps, np.array([0.0]) + 3.0 * K0, 0.2)
    assert 0.0 < r[0] < 1.0
    # K(r) at the chosen r meets the target K0 + c (E - K0) = 1.4 K0
    h = a0[0] * np.sqrt(w[0]); K = lambda rr: 0.5 * np.sum(np.where(tau[0] > 0, tau[0] * kappa(h, D, eps[0], rr), 0.0))
    assert abs(K(r[0]) / (1.4 * K0[0]) - 1) < 1e-6


def test_ns_jvp_matches_finite_differences():
    torch.manual_seed(0)
    G, E = torch.randn(6, 9), torch.randn(6, 9)
    e = 1e-6
    fd = (zeropower_via_newtonschulz5(G + e * E, 5) - zeropower_via_newtonschulz5(G - e * E, 5)) / (2 * e)
    got = dns(G, E, 5)
    assert torch.allclose(got, fd, atol=1e-6), float((got - fd).abs().max())


def test_lanczos_quadrature_exact_at_full_dimension():
    torch.manual_seed(0)
    A = torch.randn(5, 4)
    C = torch.randn(20, 20); H = C @ C.T / 20
    op = lambda V: (H @ V.reshape(-1)).reshape(V.shape)
    th, tau, m = lanczos(op, A, 20)
    ev, Q = torch.linalg.eigh(H)
    ref = float(((Q.T @ A.reshape(-1)) ** 2 @ ev.clamp_min(0).sqrt()))
    assert abs(float(np.sum(tau * np.sqrt(np.maximum(th, 0)))) / ref - 1) < 1e-8
    assert abs(float(np.sum(tau)) / float((A * A).sum()) - 1) < 1e-10


def test_capture_and_gn_products():
    """B^T X over a sequence is the gradient of that sequence's summed sampled-label loss; h_seq is the per-sequence
    estimate (1/N) sum_a <S_a, U> S_a; h_tok and h_seq are symmetric PSD."""
    model, names, params = _model()
    o = _opt(params, names, seqs=4, chunk=2)
    x, y = _batch()
    torch.manual_seed(5)
    X, B, seq, nseq, ntok = o._capture(model, x, y)
    labels = torch.cat(o._last_labels)
    j = 0                                                        # c_attn
    W = o.plist[o.fidx[j]]
    S = []
    for a in range(nseq):
        logits = model(x[a:a + 1], targets=y[a:a + 1], get_logits=True)["logits"]
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[a].reshape(-1), reduction="sum")
        S.append(torch.autograd.grad(loss, W)[0])
        Sa = B[j][seq == a].T @ X[j][seq == a]
        assert torch.allclose(Sa, S[-1], atol=1e-10), float((Sa - S[-1]).abs().max())
    U, V = torch.randn_like(W), torch.randn_like(W)
    ref = sum(float((Sa * U).sum()) * Sa for Sa in S) / ntok
    assert torch.allclose(h_seq(X[j], B[j], seq, nseq, U, ntok), ref, atol=1e-10)
    for f in (lambda Z: h_seq(X[j], B[j], seq, nseq, Z, ntok), lambda Z: h_tok(X[j], B[j], Z, ntok)):
        assert abs(float((V * f(U)).sum()) - float((U * f(V)).sum())) < 1e-10
        assert float((U * f(U)).sum()) >= -1e-12


def test_zero_r_is_momentum_free_muon_and_dmuon_adamw():
    model, names, params = _model()
    o = _opt(params, names, warm=10 ** 9)
    x, y = _batch()
    before = [p.detach().clone() for p in params]
    out = model(x, targets=y); out["loss"].backward()
    grads = [p.grad.detach().clone() for p in params]
    o.step()
    lr, wd, b1, b2, eps = 0.05, 0.01, 0.8, 0.999, 1e-8
    for i, (p, p0, g) in enumerate(zip(params, before, grads)):
        if o.use_muon[i]:
            c = 0.2 * math.sqrt(max(p.shape[:2]))
            u = zeropower_via_newtonschulz5(g.bfloat16(), 5).to(p.dtype)
            ref = p0 * (1 - lr * wd) - lr * c * u                 # m = Delta u, but r = 0
        else:
            m1, m2 = (1 - b1) * g, (1 - b2) * g * g
            scale = (1 - b1) / (1 - b2) ** 0.5
            ref = p0 * (1 - lr * wd) - (lr / scale) * m1 / (eps + m2.sqrt())
        assert torch.allclose(p.detach(), ref, atol=1e-12), names[i]
    assert sum(o.use_muon) == 2                                   # c_attn and c_fc (wte / lm_head excluded)


def test_hidden_lr_mult_scales_only_the_hidden_step():
    res = {}
    for mult in (1.0, 0.25):
        model, names, params = _model()
        o = _opt(params, names, warm=10 ** 9, hidden_lr_mult=mult)
        x, y = _batch()
        before = [p.detach().clone() for p in params]
        model(x, targets=y)["loss"].backward()
        o.step()
        res[mult] = [(p.detach() - b0 * (1 - 0.05 * 0.01)) for p, b0 in zip(params, before)]   # step net of weight decay
    for i in range(len(params)):
        scale = 0.25 if o.use_muon[i] else 1.0
        assert torch.allclose(res[0.25][i], scale * res[1.0][i], atol=1e-12), names[i]


def test_fixed_schedule_control():
    model, names, params = _model()
    o = _opt(params, names, R=5, warm=5, k=6, seqs=4, chunk=2, fixed=(0.5, 1.0))
    _train(o, model, params, 12)
    assert np.allclose(o.r, 0.5 / (1 + 12 / 8.0))                # applied: the fixed schedule at t = 12
    assert o.r_rule is not None and np.all((o.r_rule >= 0) & (o.r_rule <= 1))   # the rule ran in shadow
    assert "f2g/r_rule_shadow_mean" in o.diagnostics()


def test_rmax_lifts_the_cap():
    w = np.array([[1.0, 0.1, 0.01, 0.0]]); tau = np.array([[0.3, 0.3, 0.4, 0.0]])
    a0, eps, D = np.array([2.0]), np.array([0.01]), 0.05
    h = a0[0] * np.sqrt(w[0]); K = lambda rr: 0.5 * np.sum(np.where(tau[0] > 0, tau[0] * kappa(h, D, eps[0], rr), 0.0))
    assert rule_r(w, tau, a0, D, eps, np.array([1e9]), 0.2)[0][0] == 1.0             # default: Nesterov cap
    assert rule_r(w, tau, a0, D, eps, np.array([1e9]), 0.2, rmax=3.0)[0][0] == 3.0   # lifted
    K0 = K(0.0); E = K0 + (K(2.0) - K0) / 0.2                                        # target K(2)
    r = rule_r(w, tau, a0, D, eps, np.array([E]), 0.2, rmax=3.0)[0][0]
    assert abs(r - 2.0) < 1e-6 and rule_r(w, tau, a0, D, eps, np.array([E]), 0.2)[0][0] == 1.0
    model, names, params = _model()
    o = _opt(params, names, R=5, warm=5, k=6, seqs=4, chunk=2, fixed=(2.5, 0.0), rmax=4.0)
    _train(o, model, params, 8)
    assert np.allclose(o.r, 2.5) and np.all(o.r_rule <= 4.0)                         # fixed r = 2.5 applied
    o = _opt(params, names, R=5, warm=5, k=6, seqs=4, chunk=2, fixed=(2.5, 0.0))
    _train(o, model, params, 8)
    assert np.allclose(o.r, 1.0)                                                     # default still capped at 1


def test_filter_rule_and_step():
    w = np.array([[1.0, 0.1, 0.01, 0.0]]); tau = np.array([[0.3, 0.3, 0.4, 0.0]])
    a0, eps, D = np.array([2.0]), np.array([0.01]), 0.05
    h = a0[0] * np.sqrt(w[0]); K = lambda rr: 0.5 * np.sum(np.where(tau[0] > 0, tau[0] * kappa(h, D, eps[0], rr), 0.0))
    E = K(0.0) + (K(0.5) - K(0.0)) / 0.2
    r0 = rule_r(w, tau, a0, D, eps, np.array([E]), 0.2)[0][0]
    r1 = rule_r(w, tau, a0, D, eps, np.array([E]), 0.2, x0=0.02)[0][0]
    assert abs(r0 - 0.5) < 1e-6 and r1 > r0                                          # filtered kernel is cheaper
    # optimizer: identical to the unfiltered one until the first eigenbasis, then m is filtered in the K-FAC basis
    model, names, params = _model()
    o = _opt(params, names, R=5, warm=5, k=6, seqs=4, chunk=2, fixed=(1.0, 0.0), filt_x0=0.02, filt_every=1)
    _train(o, model, params, 12)
    st = o.state[o.plist[o.fidx[0]]]
    assert "filt_U" in st and st["filt_w"].shape == st["m"].shape
    assert o.psi_mean is not None and np.all((o.psi_mean > 0) & (o.psi_mean <= 1))
    m = torch.randn_like(st["m"]); D = 8.0 / (8.0 + 12)
    mf, pm = o._filtered(st, m, 0.05, 2.0, D)
    U, V = st["filt_U"], st["filt_V"]
    psi = 1.0 / (1.0 + (0.05 * 2.0 * st["filt_w"].sqrt() / (0.02 * D)) ** 2)
    assert torch.allclose(mf, (U @ (psi * (U.T @ m.float() @ V)) @ V.T).to(m.dtype), atol=1e-6) and abs(pm - float(psi.mean())) < 1e-9
    assert torch.allclose(o._filtered(st, m, 0.05, 2.0, 1e9)[0], m.float().to(m.dtype), atol=1e-5)   # Delta -> inf: psi -> 1
    assert "f2g_type/psi/attn.c_q" in o.diagnostics() or any(k.startswith("f2g_type/psi/") for k in o.diagnostics())


def test_highpass_filter():
    model, names, params = _model()
    o = _opt(params, names, R=5, warm=5, k=6, seqs=4, chunk=2, fixed=(1.0, 0.0), filt_x0=1e-9, filt_every=1, filt_shape="hs")
    _train(o, model, params, 12)
    st = o.state[o.plist[o.fidx[0]]]; m = torch.randn_like(st["m"]); D = 8.0 / 20
    mf, pm = o._filtered(st, m, 0.05, 2.0, D)                       # tiny cutoff: every mode with curvature passes
    keep = (st["filt_w"] > 0).float()
    U, V = st["filt_U"], st["filt_V"]
    assert torch.allclose(mf, (U @ (keep * (U.T @ m.float() @ V)) @ V.T).to(m.dtype), atol=1e-5)
    o.filt_x0 = 1e9                                                 # huge cutoff: nothing passes, no momentum
    mf, pm = o._filtered(st, m, 0.05, 2.0, D)
    assert pm == 0.0 and float(mf.abs().max()) == 0.0
    o.filt_shape = "hp"; o.filt_x0 = 1.0
    x = 0.05 * 2.0 * st["filt_w"].sqrt() / (1.0 * D)
    assert abs(o._filtered(st, m, 0.05, 2.0, D)[1] - float((x * x / (1 + x * x)).mean())) < 1e-6


def test_logtime_weight_decay():
    """With r = 0 and zero gradients the step is pure decay: d-muon's (1 - lr wd) or ADana's (1 - wd s / (1 + t / ts))."""
    for wd_decaying, factor in ((False, 1 - 0.05 * 0.01), (True, 1 - 0.01 * 1.0 / (1 + 1 / 3.0))):
        model, names, params = _model()
        o = _opt(params, names, warm=10 ** 9, wd_decaying=wd_decaying, wd_ts=3.0)
        for p in params:
            p.grad = torch.zeros_like(p)
        before = [p.detach().clone() for p in params]
        o.step()
        for i, (p, b0) in enumerate(zip(params, before)):
            assert torch.allclose(p.detach(), b0 * factor, atol=1e-12), (wd_decaying, names[i])


def _train(o, model, params, steps, seed=1):
    torch.manual_seed(seed)
    losses = []
    for _ in range(steps):
        x, y = _batch()
        o.zero_grad()
        out = model(x[:4], targets=y[:4]); (out["loss"] / 2).backward()
        half = [p.grad.clone() if p.grad is not None else None for p in params]
        out2 = model(x[4:], targets=y[4:]); (out2["loss"] / 2).backward()
        o.set_split_stats(half, [p.grad for p in params])
        if o.needs_refresh():
            o.refresh(model, x, y)
        o.step()
        losses.append(float(out["loss"] + out2["loss"]) / 2)
    return losses


def test_smoke_with_refreshes_and_state_dict():
    for polar in ("ns", "svd"):
        model, names, params = _model()
        o = _opt(params, names, R=5, warm=5, k=6, seqs=4, chunk=2, polar=polar)
        losses = _train(o, model, params, 40)
        assert all(np.isfinite(losses))
        assert o.param_groups[0]["f2g_nref"] == 8 and o.param_groups[0]["f2g_nL"] >= 3
        assert np.all((o.r >= 0) & (o.r <= 1))
        d = o.diagnostics()
        assert all(np.isfinite(v) for v in d.values()), d
        sd = o.state_dict()
        o2 = _opt(params, names, R=5, warm=5, k=6, seqs=4, chunk=2, polar=polar)
        o2.load_state_dict(sd)
        assert o2.t == o.t == 40
        assert torch.equal(o2.state[params[o.fidx[0]]]["f2g"], o.state[params[o.fidx[0]]]["f2g"])
        assert losses[-1] < losses[0]


def test_bf16_store_matches_fp32():
    """Numerator and Lanczos nodes with bf16-stored layer vectors agree with full precision to bf16 accuracy."""
    out = {}
    for store in ("fp32", "bf16"):
        model, names, params = _model()
        o = _opt(params, names, R=5, warm=5, k=6, seqs=4, chunk=2, store=store)
        torch.manual_seed(3)
        _train(o, model, params, 5, seed=1)                      # one refresh at t = 5
        out[store] = (o.last_N.copy(), np.stack([o.state[o.plist[i]]["f2g_w"].numpy() for i in o.fidx]))
    (N32, w32), (N16, w16) = out["fp32"], out["bf16"]
    assert np.allclose(N16, N32, rtol=3e-2), (N16, N32)
    assert np.allclose(w16[:, -1], w32[:, -1], rtol=3e-2)       # largest node


def test_white_whitens_kronecker_noise():
    """Pure-noise gradients with Kronecker covariance: the whitened raw-gradient buffer has unit-RMS entries, and the
    spectral map has singular values <= 1 (linear below the cap)."""
    model, names, params = _model()
    o = _opt(params, names, warm=10 ** 9, white=0.5, white_vfl=0.0, white_every=5, white_rate=0.02, fixed=(0.0, 0.0))
    torch.manual_seed(4)
    cov = {}
    for i in o.fidx:
        m, n = params[i].shape
        cov[i] = (torch.linspace(0.2, 5.0, m).sqrt(), torch.linspace(0.1, 3.0, n).sqrt())
    for _ in range(300):
        half, full = [None] * len(params), [None] * len(params)
        for i in o.fidx:
            a, b = cov[i]
            g1 = a[:, None] * torch.randn(params[i].shape) * b[None, :]
            g2 = a[:, None] * torch.randn(params[i].shape) * b[None, :]
            full[i] = 0.5 * (g1 + g2); half[i] = 0.5 * g1
        for q in params:
            q.grad = None
        for i in o.fidx:
            params[i].grad = full[i].clone()
        o.set_split_stats(half, full)
        o.step()
    for i in o.fidx:
        st = o.state[params[i]]
        UL, UR, el, er = (st[k].double() for k in ("w_UL", "w_UR", "w_el", "w_er"))
        sd = (torch.outer(el / el.mean(), er / er.mean()) * float(st["w_s"])).sqrt()
        X = UL @ ((UL.T @ st["w_mG"].double() @ UR) / sd) @ UR.T
        rms = float(X.square().mean().sqrt())
        assert 0.8 < rms < 1.25, (names[i], rms)
    O = o._white_dirs(o.t + 1, 8.0 / (8.0 + o.t + 1))
    for i, Oi in O.items():
        assert float(torch.linalg.matrix_norm(Oi.double(), ord=2)) <= 1.0 + 1e-6
    assert all(0.0 <= c <= 1.0 for c in o.capfrac.values())


def test_white_training_and_state_dict():
    model, names, params = _model()
    o = _opt(params, names, R=5, warm=5, k=6, seqs=4, chunk=2, white=0.5, fixed=(1.0, 0.0), white_every=5)
    losses = _train(o, model, params, 40)
    assert all(np.isfinite(losses)) and losses[-1] < losses[0]
    assert all("w_mG" in o.state[params[i]] and "w_UL" in o.state[params[i]] for i in o.fidx)
    d = o.diagnostics()
    assert "f2g/white_capfrac" in d and all(np.isfinite(v) for v in d.values()), d
    o2 = _opt(params, names, R=5, warm=5, k=6, seqs=4, chunk=2, white=0.5, fixed=(1.0, 0.0), white_every=5)
    o2.load_state_dict(o.state_dict())
    assert torch.equal(o2.state[params[o.fidx[0]]]["w_mG"], o.state[params[o.fidx[0]]]["w_mG"])
    assert o2.param_groups[0]["f2g_cw"] == o.param_groups[0]["f2g_cw"] > 0
    # r = 0: the whitened variant is momentum-free Muon, like the sign-SVD buffer variant
    res = {}
    for white in (0.0, 0.5):
        model, names, params = _model()
        o = _opt(params, names, R=5, warm=5, k=6, seqs=4, chunk=2, white=white, fixed=(0.0, 0.0))
        res[white] = _train(o, model, params, 15)
    assert np.allclose(res[0.0], res[0.5])


def test_white_buffer_ignores_gradient_clipping():
    """The whitened buffer takes the unclipped gradient from set_split_stats: scaling p.grad after it (as gradient clipping
    does) leaves the buffer and the whitened momentum direction exactly unchanged."""
    out = []
    for clip in (1.0, 0.3):
        model, names, params = _model()
        o = _opt(params, names, warm=10 ** 9, white=0.5, white_every=5, fixed=(1.0, 0.0))
        torch.manual_seed(4)
        for _ in range(30):
            half, full = [None] * len(params), [None] * len(params)
            for i in o.fidx:
                g1, g2 = torch.randn(params[i].shape) + 0.3, torch.randn(params[i].shape) + 0.3
                full[i] = 0.5 * (g1 + g2); half[i] = 0.5 * g1
            for q in params:
                q.grad = None
            for i in o.fidx:
                params[i].grad = full[i].clone()
            o.set_split_stats(half, full)
            for i in o.fidx:
                params[i].grad.mul_(clip)                       # gradient clipping happens after the split statistics
            o.step()
        O = o._white_dirs(o.t + 1, 8.0 / (8.0 + o.t + 1))
        out.append(([o.state[params[i]]["w_mG"].clone() for i in o.fidx], [O[i].clone() for i in o.fidx]))
    (m1, O1), (m2, O2) = out
    assert all(torch.equal(a_, b_) for a_, b_ in zip(m1, m2))
    assert all(torch.allclose(a_, b_, atol=1e-12) for a_, b_ in zip(O1, O2))


def test_rawbuf_msign_and_ramp():
    """msign: O = msign of the raw buffer; ramp: on pure noise the edge estimate tau matches the buffer noise's top singular
    value and nothing is capped; both train, and neither feels gradient clipping."""
    model, names, params = _model()
    o = _opt(params, names, warm=10 ** 9, rawbuf="ramp", fixed=(1.0, 0.0), ramp_iters=8)
    torch.manual_seed(4)
    cov = {i: (torch.linspace(0.2, 5.0, params[i].shape[0]).sqrt(), torch.linspace(0.1, 3.0, params[i].shape[1]).sqrt()) for i in o.fidx}
    rat = {i: [] for i in o.fidx}
    for t in range(500):
        half, full = [None] * len(params), [None] * len(params)
        for i in o.fidx:
            a, b = cov[i]
            g1 = a[:, None] * torch.randn(params[i].shape) * b[None, :]; g2 = a[:, None] * torch.randn(params[i].shape) * b[None, :]
            full[i] = 0.5 * (g1 + g2); half[i] = 0.5 * g1
        for q in params:
            q.grad = None
        for i in o.fidx:
            params[i].grad = full[i].clone()
        o.set_split_stats(half, full)
        o.step()
        if t >= 200:                                    # tiny test matrices: single top singular values scatter ~20%
            for i in o.fidx:
                st = o.state[params[i]]
                rat[i].append(float(torch.linalg.matrix_norm(st["w_mG"].double(), ord=2)) / math.sqrt(float(st["r_tau2"])))
    for i in o.fidx:
        assert 0.85 < np.mean(rat[i]) < 1.1, (names[i], np.mean(rat[i]))
    assert max(o.capfrac.values()) < 0.1
    O = o._raw_dirs(o.t + 1, 8.0 / (8.0 + o.t + 1))
    for Oi in O.values():
        assert float(torch.linalg.matrix_norm(Oi.double(), ord=2)) <= 1.0 + 1e-6
    for mode in ("msign", "ramp"):
        model, names, params = _model()
        o = _opt(params, names, R=5, warm=5, k=6, seqs=4, chunk=2, rawbuf=mode, fixed=(1.0, 0.0))
        losses = _train(o, model, params, 30)
        assert all(np.isfinite(losses)) and losses[-1] < losses[0], (mode, losses[0], losses[-1])
        if mode == "msign":
            i = o.fidx[0]
            O = o._raw_dirs(o.t + 1, 0.1)
            ref = zeropower_via_newtonschulz5(o.state[params[i]]["w_mG"].bfloat16(), 5).float()
            assert torch.allclose(O[i], ref)
        d = o.diagnostics()
        assert all(np.isfinite(v) for v in d.values()), d


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
