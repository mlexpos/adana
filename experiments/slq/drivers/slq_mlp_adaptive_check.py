"""Exact check of the ADAPTIVE SLQ refresh (bracket-stopped Lanczos, estimation batch grown until N(F_b)/b <= 1/2,
deterministic-equivalent de-biasing) on the small deep MLP of slq_mlp_check.py: at the same checkpoints, run the
trainer's adaptive refresh for target Deltas and compare the de-biased N_eff (global and per tensor) with the exact
population reference (GN on 65536 samples).  Output runs/slq/mlp_adaptive_check.npz
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from adana import tasks, slq, trainer, optim_tree as O

root = os.path.expanduser("~/dana-exp"); out_dir = os.path.join(root, "runs", "slq")
logf = open(os.path.join(out_dir, "log.txt"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

task = tasks.deep_mlp(v=32, width=32, depth=3, alpha=1.0, teacher_width=64, teacher_depth=3, n_eval=8192)
B, lr, g2 = 16, 0.5, 0.25
params = task["init"](jax.random.PRNGKey(0))
_, unravel = ravel_pytree(params)
sizes = [params[k].size for k in sorted(params)]
grad = jax.jit(jax.grad(task["loss"]))
step = jax.jit(lambda p, k: jax.tree.map(lambda a, g: a - lr * g, p, grad(p, task["sample"](k, B))))
refresh = trainer.make_refresh_adaptive(task, "dana_slq", P=4, m_max=256, chunk=8, eps=0.05, b0=64, b_max=16384, log=log)
Dt = np.logspace(-5, -1, 9)
res = dict(Dt=Dt)
k = jax.random.PRNGKey(1); n = 0
t0 = time.time()
for c in (0, 100, 1000, 10000):
    while n < c:
        k, kk = jax.random.split(k); params = step(params, kk); n += 1
    # population reference (65536 samples)
    fl0 = ravel_pytree(params)[0]; Pn = fl0.size; GN = np.zeros((Pn, Pn))
    bref = task["sample"](jax.random.PRNGKey(99_000 + c), 65536)
    for i in range(0, 65536, 4096):
        sub = jax.tree.map(lambda a: a[i:i + 4096], bref)
        Js = np.asarray(jax.jacfwd(lambda f: task["apply"](unravel(f), sub))(fl0)); GN += Js.T @ Js
    ev, U = np.linalg.eigh(GN / 65536); ev = np.maximum(ev, 0)
    offs = np.cumsum([0] + sizes)
    ref = np.array([np.sum(g2 * ev / (g2 * ev + D)) for D in Dt])
    refT = np.array([[np.sum(g2 * ev / (g2 * ev + D) * np.sum(U[offs[t]:offs[t + 1]] ** 2, 0)) for t in range(len(sizes))] for D in Dt])
    P_ = jax.tree.map(lambda a: a[None, None], params)                   # (S=1, G=1, ...)
    st = jax.vmap(jax.vmap(lambda p: O.init("dana_slq", p, Q=4 * 256)))(P_)
    hp = {"lr": jnp.full((1, 1), g2), "delta": jnp.full((1, 1), 4.0), "s": jnp.zeros((1, 1)), "C": jnp.zeros((1, 1))}
    est, estT, ms, bs = [], [], [], []
    for i, D in enumerate(Dt):
        keys = jax.random.split(jax.random.PRNGKey(500 + i + c), 1)
        st2, info = refresh(P_, st, hp, keys, 64, np.full((1, 1, 1, 1), D / 2), np.full((1, 1, 1, 1), D),
                            np.full((1, 1), g2), np.zeros((1, 1)))
        qn = np.asarray(st2["qn"][0, 0], np.float64); qw = np.asarray(st2["qw"][0, 0], np.float64)
        qT = np.asarray(st2["qT"][0, 0], np.float64); b = float(st2["qb"][0, 0])
        Dp = slq.debias_shift(qn, qw, g2, D, b)
        f = g2 * np.maximum(qn, 0) / (g2 * np.maximum(qn, 0) + Dp)
        est.append(f @ qw); estT.append(f @ qT); ms.append(info["m"]); bs.append(info["b"])
    res[f"ref_c{c}"] = ref; res[f"refT_c{c}"] = refT; res[f"est_c{c}"] = np.array(est); res[f"estT_c{c}"] = np.array(estT)
    res[f"m_c{c}"] = np.array(ms); res[f"b_c{c}"] = np.array(bs)
    log(f"   adaptive mlpcheck n={c}: " + " | ".join(f"D={D:.0e} N={r:.0f} est/ref {e / r:.3f} m={m} b={b}"
                                                   for D, r, e, m, b in zip(Dt, ref, est, ms, bs)) + f" ({time.time()-t0:.0f}s)")
np.savez_compressed(os.path.join(out_dir, "mlp_adaptive_check.npz"), **res)
log("adaptive mlpcheck done")
