"""CPU unit tests for adana-slq.  Run from the repo root:  python -m pytest -q tests/test_adana_slq.py
(or `python tests/test_adana_slq.py`)."""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from optim.adana import ADana                      # noqa: E402
from optim.adana_slq import ADanaSLQ               # noqa: E402
from optim.slq_torch import GaussNewtonOperator, slq, effective_tokens  # noqa: E402

torch.set_default_dtype(torch.float64)
# the compiled per-shape kernels are exercised on GPU runs; tests call the eager functions
ADana._get_compiled_fn = lambda self, nd: self._update_param_compiled
ADanaSLQ._get_compiled_fn = lambda self, nd: self._update_slq


class TinyLM(nn.Module):
    """Same forward interface as the Enoki model: forward(idx, targets, get_logits) -> dict(logits, loss)."""

    def __init__(self, V=11, d=6):
        super().__init__()
        self.wte = nn.Embedding(V, d)
        self.h = nn.Linear(d, d)
        self.lm_head = nn.Linear(d, V, bias=False)

    def forward(self, idx, targets=None, get_logits=False):
        x = self.wte(idx)
        x = x + torch.tanh(self.h(x))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return {"logits": logits if get_logits else None, "loss": loss}


def _setup(seed=0, B=4, T=5, V=11, d=6):
    torch.manual_seed(seed)
    model = TinyLM(V, d)
    x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    return model, names, params, x, y


def _dense(op, params):
    """Dense matrix of a list-of-tensors operator (small models only)."""
    sizes = [p.numel() for p in params]
    P = sum(sizes)
    M = np.zeros((P, P))
    for j in range(P):
        e = torch.zeros(P); e[j] = 1.0
        u, k = [], 0
        for p, n in zip(params, sizes):
            u.append(e[k:k + n].view_as(p)); k += n
        M[:, j] = torch.cat([o.reshape(-1) for o in op(u)]).numpy()
    return M, sizes


def test_fd_gn_matches_exact():
    model, names, params, x, y = _setup()
    torch.manual_seed(1)
    u = [torch.randn_like(p) for p in params]
    got = {m: torch.cat([o.reshape(-1) for o in GaussNewtonOperator(model, names, params, x, y, r=None, fd_rel=1e-4,
                                                                    mode=m)(u)]) for m in ("fd", "jvp")}

    def f(*ps):
        return torch.func.functional_call(model, dict(zip(names, ps)), (x,), dict(get_logits=True))["logits"]
    _, Ju = torch.func.jvp(f, tuple(p.detach() for p in params), tuple(u))
    logits = f(*[p.detach() for p in params])
    pr = torch.softmax(logits, -1)
    hv = (pr * Ju - pr * (pr * Ju).sum(-1, keepdim=True)) / y.numel()
    _, vjp = torch.func.vjp(f, *[p.detach() for p in params])
    want = torch.cat([g.reshape(-1) for g in vjp(hv)])
    for m in got:
        rel = float((got[m] - want).norm() / want.norm())
        assert rel < (1e-3 if m == "fd" else 1e-10), (m, rel)


def test_lanczos_quadrature_vs_dense():
    model, names, params, x, y = _setup()
    r = [torch.rand_like(p) + 0.5 for p in params]
    op = GaussNewtonOperator(model, names, params, x, y, r=r, fd_rel=1e-4)
    M, sizes = _dense(op, params)
    M = 0.5 * (M + M.T)
    lam, U = np.linalg.eigh(M)
    lam = np.maximum(lam, 0.0)
    g, D = 0.7, 0.05
    fM = U @ np.diag(g * lam / (g * lam + D)) @ U.T
    gen = torch.Generator(); gen.manual_seed(7)
    res = slq(op, params, m_max=60, probes=1, eps=1e-9, g=g, D_check=[D], generator=gen)
    gen2 = torch.Generator(); gen2.manual_seed(7)
    z = torch.cat([(torch.randint(0, 2, p.shape, generator=gen2, dtype=torch.int8).double() * 2 - 1).reshape(-1)
                   for p in params]).numpy()
    th, w, wT = res["nodes"][0], res["weights"][0], res["block"][0]
    f = g * np.maximum(th, 0) / (g * np.maximum(th, 0) + D)
    exact = z @ fM @ z
    assert abs(f @ w - exact) / exact < 1e-3, (f @ w, exact)
    # per-tensor split: z_T^T f(H) z  (projection on one side)
    k = 0
    for T, n in enumerate(sizes):
        zT = np.zeros_like(z); zT[k:k + n] = z[k:k + n]; k += n
        ex = zT @ fM @ z
        assert abs(f @ wT[:, T] - ex) <= 1e-3 * abs(exact), (T, f @ wT[:, T], ex)


def test_matches_adana_when_amplification_forced():
    torch.manual_seed(0)
    kappa, delta = 0.85, 8.0
    m1, m2 = TinyLM(), TinyLM()
    m2.load_state_dict(m1.state_dict())
    specs = lambda m: [{"params": list(m.parameters()), "param_names": [n for n, _ in m.named_parameters()]}]
    o1 = ADana(specs(m1), lr=1e-2, delta=delta, kappa=kappa, weight_decay=0.1, wd_decaying=True, wd_ts=10.0)
    o2 = ADanaSLQ(specs(m2), lr=1e-2, delta=delta, weight_decay=0.1, wd_decaying=True, wd_ts=10.0)

    def forced(D):                                   # ADana's alpha(t) = 1 + (1+t)^{1-kappa}, same for every tensor
        o2.A = np.full(len(o2.plist), 1.0 + (1.0 + o2.t) ** (1 - kappa))
    o2._compute_ratio = forced
    for it in range(6):
        x = torch.randint(0, 11, (3, 5)); y = torch.randint(0, 11, (3, 5))
        for m, o in ((m1, o1), (m2, o2)):
            o.zero_grad(); m(x, targets=y)["loss"].backward(); o.step()
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(a, b, atol=1e-10), float((a - b).abs().max())


def test_waterfill_spends_budget():
    model, names, params, x, y = _setup()
    o = ADanaSLQ([{"params": params, "param_names": names}], lr=0.1, s=0.25, alloc="waterfill", batch_seqs=4)
    L = len(params)
    rng = np.random.default_rng(0)
    o.q = dict(nodes=rng.uniform(0.1, 10, 20), w=np.ones(20), wT=np.abs(rng.normal(size=(20, L))) / L, trF=10.0,
               lam_max=10.0)
    o.q["w"] = o.q["wT"].sum(1)
    o.n_split = 1; o.eX = np.ones(L); o.eD = np.ones(L); o.eDz = np.full(L, 1e-6)
    o.mz2 = rng.uniform(0.1, 1.0, L); o.teff = 1.0
    o._compute_ratio(0.5)
    NTc = np.maximum(o.NT, 1e-2)
    spent = float(np.sum(o.afrac * NTc))
    assert (np.all(o.afrac <= 1 + 1e-12) and (abs(spent - o.S) / o.S < 1e-6 or np.all(o.afrac == 1))), (spent, o.S)


def test_split_half_noise_scale():
    torch.manual_seed(0)
    P, B, sig = 2000, 64, 0.5
    p = nn.Parameter(torch.zeros(P))
    o = ADanaSLQ([{"params": [p], "param_names": ["w"]}], lr=0.1, kprime=1.0, batch_seqs=B)
    grad = torch.randn(P) * 0.05
    for t in range(400):
        g1 = grad + sig * torch.randn(P) / math.sqrt(B / 2)
        g2 = grad + sig * torch.randn(P) / math.sqrt(B / 2)
        o.t = t
        o.set_split_stats([0.5 * g1], [0.5 * (g1 + g2)])
    bnr = 4 * o.eX.sum() / o.eD.sum()
    want = B * float(grad.norm() ** 2) / (sig ** 2 * P)
    assert abs(bnr - want) / want < 0.15, (bnr, want)


def test_refresh_and_train_smoke():
    model, names, params, x, y = _setup(B=8, T=6)
    o = ADanaSLQ([{"params": params, "param_names": names}], lr=0.05, kprime=16.0, alloc="waterfill", m_max=8,
                 probes=1, slq_batch=4, batch_seqs=8, max_gap=3)
    losses = []
    for it in range(12):
        xb = torch.randint(0, 11, (8, 6)); yb = (xb + 1) % 11
        o.zero_grad()
        out = model(xb[:4], targets=yb[:4]); (out["loss"] / 2).backward()
        half = [p.grad.clone() for p in params]
        out2 = model(xb[4:], targets=yb[4:]); (out2["loss"] / 2).backward()
        o.set_split_stats(half, [p.grad for p in params])
        if o.needs_refresh():
            o.refresh(model, xb, yb)
        o.step()
        losses.append(float(out["loss"] + out2["loss"]) / 2)
    assert o.n_refresh >= 2 and np.isfinite(losses).all() and losses[-1] < losses[0], losses
    d = o.diagnostics()
    assert "slq/N" in d and np.isfinite(d["slq/N"])


def test_effective_tokens_limits():
    model, names, params, x, y = _setup(B=8, T=6)
    te, _, _ = effective_tokens(model, params, x, y)
    assert 1.0 <= te <= 6.0 * 3, te


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
