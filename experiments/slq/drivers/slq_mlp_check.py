"""Exact check of the SLQ N_eff estimator on a small deep MLP (P ~ 3.3k parameters): at training checkpoints, form the
GN matrix J^T J / b on an estimation batch explicitly, diagonalize, and compare exact N_eff(Delta) (global and per
tensor) with SLQ estimates (m, probes) computed matrix-free with GN-vector products on the same batch.  A large-batch
GN (b_ref) gives the 'population' reference, separating sampling bias (finite b) from quadrature/probe error.
Output: runs/slq/mlpcheck.npz
"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from adana import tasks, slq, trainer
from adana.sim_ls import eval_points

root = os.path.expanduser("~/dana-exp"); out_dir = os.path.join(root, "runs", "slq"); os.makedirs(out_dir, exist_ok=True)
logf = open(os.path.join(out_dir, "log.txt"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

task = tasks.deep_mlp(v=32, width=32, depth=3, alpha=1.0, teacher_width=64, teacher_depth=3, n_eval=8192)
B, lr = 16, 0.5
params = task["init"](jax.random.PRNGKey(0))
flat0, unravel = ravel_pytree(params)
P = flat0.size
names = list(params.keys())                      # dict leaf order = sorted keys (jax flattens dicts by sorted key)
names = sorted(names)
sizes = [params[k].size for k in names]
log(f"mlpcheck: P={P}, leaves {dict(zip(names, sizes))}")
grad = jax.jit(jax.grad(task["loss"]))
step = jax.jit(lambda p, k: jax.tree.map(lambda a, g: a - lr * g, p, grad(p, task["sample"](k, B))))

Dg = np.logspace(-6, 0, 25)
g2 = 0.25                                         # step used inside N_eff (any g2: N_eff depends on g2*lam/Delta)
ckpts = [0, 100, 1000, 10000]
res = dict(Dg=Dg, g2=g2, ckpts=np.array(ckpts), names=np.array(names), sizes=np.array(sizes))
k = jax.random.PRNGKey(1)
n = 0
t0 = time.time()
for c in ckpts:
    while n < c:
        k, kk = jax.random.split(k); params = step(params, kk); n += 1
    risk = float(task["evaluate"](params)["risk"])
    for b in (256, 1024, 8192, 65536):
        batch = task["sample"](jax.random.PRNGKey(1000 + c + b), b)
        f = lambda flat: task["apply"](unravel(flat), batch)
        J = np.asarray(jax.jacfwd(f)(ravel_pytree(params)[0]) if b <= 8192 else 0)
        if b <= 8192:
            GN = J.T @ J / b
        else:                                     # population reference by accumulation
            GN = np.zeros((P, P))
            for i in range(0, b, 4096):
                sub = jax.tree.map(lambda a: a[i:i + 4096], batch)
                Js = np.asarray(jax.jacfwd(lambda fl: task["apply"](unravel(fl), sub))(ravel_pytree(params)[0]))
                GN += Js.T @ Js
            GN /= b
        ev, U = np.linalg.eigh(GN)
        ev = np.maximum(ev, 0)
        fl = lambda D: g2 * ev / (g2 * ev + D)
        res[f"exact_c{c}_b{b}"] = np.array([fl(D).sum() for D in Dg])
        # exact per-tensor: tr[f(GN)]_TT = sum_i f(lam_i) |U_{T,i}|^2
        offs = np.cumsum([0] + sizes)
        res[f"exactT_c{c}_b{b}"] = np.array([[np.sum(fl(D) * np.sum(U[offs[t]:offs[t + 1]] ** 2, 0)) for t in range(len(sizes))] for D in Dg])
        res[f"lam_c{c}_b{b}"] = ev
        if b in (1024, 8192):
            mv = jax.jit(lambda u: task["gnvp"](params, batch, u))
            for m in (16, 32, 64):
                for p in (1, 4, 16):
                    qn, qw, qT = slq.quadrature_tree(mv, params, jax.random.PRNGKey(7 + m + p + c), m=m, probes=p)
                    res[f"slq_c{c}_b{b}_m{m}_p{p}"] = np.array([float(slq.neff_quad(qn, qw, g2, D)) for D in Dg])
                    res[f"slqT_c{c}_b{b}_m{m}_p{p}"] = np.array([np.asarray(slq.neff_quad(qn, qT, g2, D)) for D in Dg])
                    res[f"slqlam_c{c}_b{b}_m{m}_p{p}"] = float(jnp.max(qn))
    log(f"   ckpt n={c}: risk {risk:.4e}; lam_max(b=8192) {res[f'lam_c{c}_b8192'][-1]:.4g}; "
        f"N_eff(1e-3) exact b=1024/8192/65536: {res[f'exact_c{c}_b1024'][12]:.1f}/{res[f'exact_c{c}_b8192'][12]:.1f}/{res[f'exact_c{c}_b65536'][12]:.1f}; "
        f"SLQ(b=8192,m=32,p=4) {res[f'slq_c{c}_b8192_m32_p4'][12]:.1f} ({time.time()-t0:.0f}s)")
np.savez_compressed(os.path.join(out_dir, "mlpcheck.npz"), **res)
log("mlpcheck done")
