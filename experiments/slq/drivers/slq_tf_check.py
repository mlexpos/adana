"""Effective sample count of the transformer GN estimate.  Train modarith briefly with SGD; at checkpoints compute the raw
SLQ N_eff(Delta) of the GN matrix on estimation batches of b sequences (b = 8..512), and the deterministic-equivalent
de-biased value with sample counts b, b*T, b*T*(V-1).  The right count is the one making small-b estimates agree with
the largest b (raw N_b is b-independent once the operator is not rank-limited).  Output runs/e3/slq_tf_check.npz
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import jax, jax.numpy as jnp
from adana import tasks, slq

root = os.path.expanduser("~/dana-exp"); out_dir = os.path.join(root, "runs", "e3")
logf = open(os.path.join(out_dir, "log.txt"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
task = tasks.modarith(p=97, T=32, zipf=1.2, eta_sig=0.7)
T, V = 31, 97                                    # predicted positions per sequence, vocabulary
B, lr, g2 = 16, 0.125, 0.125
params = task["init"](jax.random.PRNGKey(0))
grad = jax.jit(jax.grad(task["loss"]))
step = jax.jit(lambda p, k: jax.tree.map(lambda a, g: a - lr * g, p, grad(p, task["sample"](k, B))))
Dg = np.logspace(-5, -1, 9)
res = dict(Dg=Dg)
k = jax.random.PRNGKey(1); n = 0; t0 = time.time()
for c in (200, 3000):
    while n < c:
        k, kk = jax.random.split(k); params = step(params, kk); n += 1
    for b in (8, 32, 128, 512):
        batch = task["sample"](jax.random.PRNGKey(7000 + b + c), b)
        mv = jax.jit(lambda u: task["gnvp"](params, batch, u))
        qn, qw, _ = slq.quadrature_tree(mv, params, jax.random.PRNGKey(3 + b), m=128, probes=4)
        qn, qw = np.asarray(qn, np.float64), np.asarray(qw, np.float64)
        raw = np.array([slq.fsum(qn, qw, g2, D) for D in Dg])
        res[f"raw_c{c}_b{b}"] = raw
        for name, cnt in (("b", b), ("bT", b * T), ("bTV", b * T * (V - 1))):
            res[f"deb_{name}_c{c}_b{b}"] = np.array([slq.neff_debiased(qn, qw, g2, D, cnt) for D in Dg])
        log(f"   tf check n={c} b={b}: raw N_eff(D) " + " ".join(f"{D:.0e}:{x:.0f}" for D, x in zip(Dg, raw))
            + f" | lam_max {qn.max():.3g} ({time.time()-t0:.0f}s)")
    ref = res[f"raw_c{c}_b512"]
    for name in ("b", "bT", "bTV"):
        log(f"   tf check n={c} de-biased with count={name}: ratio to raw(b=512) at b=8/32/128: " + " | ".join(
            " ".join(f"{x:.2f}" for x in res[f"deb_{name}_c{c}_b{b}"][::2] / np.maximum(ref[::2], 1e-9)) for b in (8, 32, 128)))
np.savez_compressed(os.path.join(out_dir, "slq_tf_check.npz"), **res)
log("tf check done")
