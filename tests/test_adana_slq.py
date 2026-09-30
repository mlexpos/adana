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


def test_jvp_nocache_matches_cached():
    # the no-cache path (J^T by ordinary reverse mode, optional activation checkpointing) equals the cached vjp closure
    model, names, params, x, y = _setup()
    torch.manual_seed(2)
    u = [torch.randn_like(p) for p in params]
    r = [torch.rand_like(p) + 0.5 for p in params]
    a, b = (torch.cat([o.reshape(-1) for o in GaussNewtonOperator(model, names, params, x, y, r=r, chunk=2,
                                                                     cache=c)(u)]) for c in (True, False))
    assert float((a - b).norm() / a.norm()) < 1e-12
    assert all(p.grad is None for p in params)


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


def test_waterfill_leftover_budget():
    """Only tiny tensors carry signal: they are capped, and the rest of the budget is spread uniformly over the tensors
    without signal instead of being left unspent.  A binding budget is unaffected."""
    c = np.array([1.0, 0.0, 0.0, 0.5]); NTc = np.array([1e-2, 10.0, 30.0, 1e-2])
    a = ADanaSLQ._waterfill(c, NTc, 20.0, leftover=True)
    assert np.allclose(a[[0, 3]], 1.0) and np.allclose(a[[1, 2]], (20.0 - 0.02) / 40.0), a
    assert abs(a @ NTc - 20.0) < 1e-9
    assert np.allclose(ADanaSLQ._waterfill(c, NTc, 20.0, leftover=False)[[1, 2]], 0.0)
    c2 = np.array([1.0, 0.2, 0.0, 0.5])
    assert np.allclose(ADanaSLQ._waterfill(c2, NTc, 5.0, leftover=True),
                       ADanaSLQ._waterfill(c2, NTc, 5.0, leftover=False), atol=1e-9)


def test_version_defaults():
    """'waterfill' (and the other allocations) keep the v1 estimator: rejection at gap 0.5, Aitken on, no leftover.
    'waterfill_v2' takes the Gauss bound, never rejects and spends the leftover.  Same resolution on the command line."""
    import argparse
    from config import base
    model, names, params, x, y = _setup()
    spec = [{"params": params, "param_names": names}]
    for alloc, (ait, rej, left) in {"waterfill": (True, 0.5, False), "independent": (True, 0.5, False),
                                    "waterfill_v2": (False, -1.0, True)}.items():
        o = ADanaSLQ(spec, lr=0.1, alloc=alloc)
        assert (o.aitken, o.reject_gap, o.leftover) == (ait, rej, left), alloc
        a = base.parse_args(argparse.ArgumentParser(allow_abbrev=False), ["--opt", "adana-slq", "--slq_alloc", alloc],
                            argparse.Namespace())
        assert (a.slq_aitken, a.slq_reject_gap) == (ait, rej), alloc
    a = base.parse_args(argparse.ArgumentParser(allow_abbrev=False),
                        ["--opt", "adana-slq", "--slq_alloc", "waterfill_v2", "--slq_aitken", "--slq_reject_gap", "0.3"],
                        argparse.Namespace())
    assert (a.slq_aitken, a.slq_reject_gap) == (True, 0.3)


def test_gauss_radau_bracket_contains_trace():
    """Gauss is an upper and Gauss-Radau (node at 0) a lower bound on z^T f(H) z at every m.  For a diagonal operator a
    Rademacher probe returns the trace exactly, so the bracket must contain N(D) itself, even far from convergence."""
    lam = [torch.tensor(np.logspace(-6, 2, 300)), torch.tensor(np.logspace(-4, 1, 200))]
    op = lambda u: [l * ui for l, ui in zip(lam, u)]
    g, D = 1.0, 1e-3
    exact = sum(float((g * l / (g * l + D)).sum()) for l in lam)
    prev = float("inf")
    for m in (2, 4, 8, 16, 32):
        gen = torch.Generator(); gen.manual_seed(1)
        res = slq(op, lam, m_max=m, probes=1, eps=0.0, g=g, D_check=[D], generator=gen)
        lo, hi = float(res["N_lo"][0, 0]), float(res["N_hi"][0, 0])
        assert lo <= exact * (1 + 1e-9) and exact <= hi * (1 + 1e-9), (m, lo, exact, hi)
        assert hi <= prev * (1 + 1e-9), (m, hi, prev)          # Gauss decreases monotonically in m
        prev = hi
        th, w, wT = res["nodes"][0], res["weights"][0], res["block"][0]
        f = g * np.maximum(th, 0) / (g * np.maximum(th, 0) + D)
        assert abs(f @ w - hi) <= 1e-9 * hi
        assert abs((f @ wT).sum() - hi) <= 1e-6 * hi               # per-tensor Gauss values sum to the Gauss bound


def test_typed_allocation():
    names = ["transformer.wte.weight", "transformer.h.0.attn.c_attn.weight", "transformer.h.0.attn.c_proj.weight",
             "transformer.h.0.mlp.c_fc.weight", "transformer.h.0.mlp.c_proj.weight", "transformer.h.0.ln_1.weight",
             "lm_head.weight", "transformer.h.0.attn.q_layernorm.weight", "transformer.ln_f.weight"]
    params = [nn.Parameter(torch.zeros(3)) for _ in names]
    o = ADanaSLQ([{"params": params, "param_names": names}], lr=0.1, s=0.25, alloc="typed", batch_seqs=64,
                 type_frac="attn=0.45,mlp=0.45,vocab=0.1,tiny=0")
    assert list(o.agroup) == ["vocab", "attn", "attn", "mlp", "mlp", "tiny", "vocab", "tiny", "tiny"], list(o.agroup)
    rng = np.random.default_rng(1)
    wT = np.abs(rng.normal(size=(10, len(names)))) * np.array([500, 30, 10, 40, 40, 0.1, 900.0, 0.1, 0.1])
    o.q = dict(nodes=rng.uniform(1, 50, 10), w=wT.sum(1), wT=wT, trF=1.0, lam_max=50.0)
    o.teff = 1.0
    o.n_split = 1; o.eX = np.ones(len(names)); o.eD = np.ones(len(names)); o.eDz = np.full(len(names), 1e-9)
    o.mz2 = np.array([1.0, 1.0, 1e-12, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0])      # attn.c_proj has ~no signal
    o._compute_ratio(0.01)
    NTc = np.maximum(o.NT, 1e-2)
    assert o.ratio[2] == 0.0                                              # G_T <= 0 -> no long momentum
    for g, pi in o.type_frac.items():
        m = o.agroup == g
        share = float((o.ratio[m] * NTc[m]).sum() / o.S)
        pos = o.G[m] > 0
        full = np.all(o.ratio[m][pos] >= o.cap - 1e-12)
        assert share <= pi + 1e-9 and (abs(share - pi) < 1e-6 or full or pi == 0), (g, share, pi)
    assert o.ratio[o.agroup == "tiny"].max() == 0.0


def test_independent_allocation():
    names = ["transformer.wte.weight", "transformer.h.0.attn.c_attn.weight", "transformer.h.0.mlp.c_fc.weight", "lm_head.weight"]
    params = [nn.Parameter(torch.zeros(3)) for _ in names]
    o = ADanaSLQ([{"params": params, "param_names": names}], lr=0.1, s=0.25, alloc="independent", batch_seqs=64)
    rng = np.random.default_rng(2)
    wT = np.abs(rng.normal(size=(10, len(names)))) * np.array([300, 0.5, 0.5, 900.0])
    o.q = dict(nodes=rng.uniform(1, 50, 10), w=wT.sum(1), wT=wT, trF=1.0, lam_max=50.0)
    o.teff = 1.0
    o._compute_ratio(0.01)
    NTc = np.maximum(o.NT, 1e-2)
    assert np.allclose(o.ratio, np.minimum(o.S / NTc, o.cap))           # each tensor against its own N_T, full budget
    assert o.ratio[1] == o.cap and o.ratio[3] < o.cap                      # small blocks at the cap, lm_head limited
    assert np.isclose(o.spent, (o.ratio * o.NT).sum() / (2 * 64))
    o.s = float("inf"); o._compute_ratio(0.01)
    assert np.all(o.ratio == o.cap)                                        # s = inf: Nesterov everywhere


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
    o = ADanaSLQ([{"params": params, "param_names": names}], lr=0.05, kprime=16.0, alloc="waterfill_v2", m_max=8,
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
    # waterfill_v2: every refresh adopted, N at the Gauss bound, bracket logged
    assert o.n_reject == 0 and d["slq/last_accepted"] == 1.0 and d["slq/last_aitken"] == 1.0
    assert d["slq/last_N_lo"] <= d["slq/last_N_hi"] * (1 + 1e-6)          # closed bracket: equal up to roundoff


def test_effective_tokens_limits():
    model, names, params, x, y = _setup(B=8, T=6)
    te, _, _ = effective_tokens(model, params, x, y)
    assert 1.0 <= te <= 6.0 * 3, te


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
