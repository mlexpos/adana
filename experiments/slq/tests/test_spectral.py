"""Tests for adana/spectral.py (spectral preconditioning + DANA-SLQ).  Run from experiments/:
    /opt/e-py/bin/python -m pytest -q tests/test_spectral.py"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adana import optim_tree as O, slq as SLQ, spectral as SP, tasks, trainer  # noqa: E402

KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s", "f", "C", "bk", "pt", "tau", "alloc", "T", "clip",
        "sched_mode", "gns", "gnsb", "sb", "seps", "rms", "mu", "perhead", "nh")


def _spd(rng, n):
    A = rng.normal(size=(n, n))
    return A @ A.T + 0.1 * np.eye(n)


def _mpow(M, e):
    d, Q = np.linalg.eigh(M)
    return (Q * d ** e) @ Q.T


def test_sandwich_matches_direct_powers():
    rng = np.random.default_rng(0)
    L, R, U = _spd(rng, 5), _spd(rng, 4), rng.normal(size=(5, 4))
    dL, QL = np.linalg.eigh(L); dR, QR = np.linalg.eigh(R)
    for e in (-0.25, -0.125, 0.25):
        got = np.asarray(SP.sandwich(jnp.asarray(QL), jnp.asarray(dL ** e), jnp.asarray(U), jnp.asarray(QR), jnp.asarray(dR ** e)))
        assert np.allclose(got, _mpow(L, e) @ U @ _mpow(R, e), atol=1e-4), e


def test_one_step_shampoo_is_msign():
    """With L = G G^T and R = G^T G, L^{-1/4} G R^{-1/4} = U V^T; the Newton-Schulz msign approximates it."""
    rng = np.random.default_rng(1)
    G = rng.normal(size=(6, 6))
    u, _, vt = np.linalg.svd(G)
    sh = _mpow(G @ G.T, -0.25) @ G @ _mpow(G.T @ G, -0.25)
    assert np.allclose(sh, u @ vt, atol=1e-6)
    ns = np.asarray(SP.msign(jnp.asarray(G, jnp.float32)))
    assert np.linalg.norm(ns - u @ vt) / np.linalg.norm(u @ vt) < 0.35          # quintic NS: singular values ~ 0.7-1.2
    assert np.allclose(np.linalg.svd(ns, compute_uv=False), 1.0, atol=0.35)


def _tiny():
    return tasks.modarith(p=11, T=6, d_model=8, n_layers=1, n_heads=2, n_eval=64)


def _hp(**kw):
    base = dict(lr=1e-2, b1=0.9, b2=0.99, eps=1e-8, delta=4.0, cap=1.0, s=0.5, T=0.0, clip=0.0, sched_mode=1.0,
                gnsb=0.0, sb=0.95, seps=1e-6, rms=1.0, mu=0.95, perhead=0.0, nh=2.0, alloc=4.0)
    base.update(kw)
    return {k: jnp.asarray(v, jnp.float32) for k, v in base.items()}


def _trained_state(opt, hp, steps=12, B=16):
    task = _tiny()
    params = task["init"](jax.random.PRNGKey(0))
    st = O.init(opt, params, Q=64)
    kinds = SP.leaf_kinds(params)
    for n in range(steps):
        batch = task["sample"](jax.random.PRNGKey(100 + n), B)
        h = B // 2
        g1 = jax.grad(task["loss"])(params, jax.tree.map(lambda a: a[:h], batch))
        g2 = jax.grad(task["loss"])(params, jax.tree.map(lambda a: a[h:], batch))
        if n == 1 or (n > 0 and n % SP.KEIG == 0):
            st = SP.eig_refresh(st, hp, kinds)
        params, st = O.step(opt, params, st, g1, g2, g1, hp, B)
    return task, params, st


def test_Rz_operator_symmetric_and_quadrature_matches_dense():
    for opt, ph in (("spd_slq", 0.0), ("spd_slq", 1.0), ("soap_dana_slq", 0.0)):
        hp = _hp(perhead=ph)
        task, params, st = _trained_state(opt, hp)
        assert float(st["eig_ok"]) == 1.0
        batch = task["sample"](jax.random.PRNGKey(7), 32)
        Rz = lambda u: SP.apply_Rz(opt, st, hp, u)
        op = jax.jit(lambda u: Rz(task["gnvp"](params, batch, Rz(u))))
        flat, unravel = ravel_pytree(params)
        P = flat.size
        a, b = np.random.default_rng(2).normal(size=(2, P))
        opv = lambda x: np.asarray(ravel_pytree(op(unravel(jnp.asarray(x, jnp.float32))))[0], np.float64)
        sab, sba = a @ opv(b), opv(a) @ b
        assert abs(sab - sba) <= 1e-3 * max(abs(sab), 1e-12), (opt, sab, sba)
        M = np.stack([opv(e) for e in np.eye(P)], 1)
        lam = np.clip(np.linalg.eigvalsh(0.5 * (M + M.T)), 0, None)
        g, D = 1e-2, 1e-3
        exact = float(np.sum(g * lam / (g * lam + D)))
        qn, qw, _ = SLQ.quadrature_tree(op, params, jax.random.PRNGKey(3), m=64, probes=16)
        th = np.maximum(np.asarray(qn), 0)
        est = float(np.sum(np.asarray(qw) * g * th / (g * th + D)))
        assert abs(est - exact) / exact < 0.1, (opt, ph, est, exact)


def test_all_spectral_optimizers_smoke():
    task = _tiny()
    ev = [0, 10, 30]
    cfg = dict(adaptive=True, p=1, m_max=16, chunk=8, eps=0.05, b0=16, b_max=16, refresh0=True, map_configs=True,
               max_gap=2000, debias=False)
    sp = dict(T=30.0, clip=1.0, sched_mode=1.0, eps=1e-8, b2=0.99, sb=0.95, seps=1e-6, rms=1.0, nh=2.0, cap=1.0, tau=0.0)
    for opt in SP.SPECTRAL:
        if opt in SP.SLQ_SPECTRAL:
            rows = [dict(sp, lr=3e-3, delta=4.0, gnsb=32.0, s=0.5, alloc=a, perhead=ph) for a in (0.0, 4.0) for ph in (0.0, 1.0)]
            r = trainer.run(task, opt, "slq", rows, KEYS, S=1, B=8, n_max=30, eval_pts=ev, log=None, slq=cfg)
            assert np.isfinite(r["g3"]).all() and (r["g3"][-1] > 0).any(), opt
        else:
            rows = [dict(sp, lr=l, b1=0.9, mu=0.95) for l in (1e-3, 3e-3)]
            r = trainer.run(task, opt, "slq", rows, KEYS, S=1, B=8, n_max=30, eval_pts=ev, log=None)
        risk = np.asarray(r["risk"])
        assert np.isfinite(risk).all(), opt
        assert (risk[-1] < risk[0]).all(), (opt, risk[:, 0])


def test_step_preconditioner_is_square_of_quadrature_coordinates():
    """P^{-1} used by the step equals P^{-1/2} P^{-1/2} used by the quadrature, and P P^{-1} = I, for every leaf."""
    for opt in ("spd_slq", "soap_dana_slq"):
        hp = _hp(perhead=1.0)
        task, params, st = _trained_state(opt, hp)
        kinds = SP.leaf_kinds(params)
        vl = jax.tree.leaves(st["v"])
        bc2 = 1 - hp["b2"] ** jnp.maximum(st["n"].astype(jnp.float32), 1.0)
        rot = opt in SP.ROTATED
        rng = np.random.default_rng(4)
        for i, x in enumerate(jax.tree.leaves(params)):
            U = jnp.asarray(rng.normal(size=x.shape), jnp.float32)
            P = lambda V, e: SP.pe(rot, st, hp, kinds[i][0], i, V, e, vl[i], bc2)
            full, half2 = P(U, -1.0), P(P(U, -0.5), -0.5)
            assert np.allclose(np.asarray(full), np.asarray(half2), rtol=1e-3, atol=1e-5 * float(jnp.abs(full).max())), (opt, i)
            back = P(P(U, -1.0), 1.0)
            assert np.allclose(np.asarray(back), np.asarray(U), rtol=1e-2, atol=1e-3 * float(jnp.abs(U).max())), (opt, i)
