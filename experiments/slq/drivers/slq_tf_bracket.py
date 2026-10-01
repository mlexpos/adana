"""Certified bracket for the transformer's N_eff: Gauss (upper) / Gauss-Radau (lower) bounds from Lanczos up to m=512 on the
GN matrix of b sequences (b = 64..2048), at two SGD checkpoints; plus the split-half extrapolation 2 N_b - N_{b/2}.
Decides whether the large SLQ N_eff (vs the shadow's) is real or quadrature inflation.  Output runs/e3/slq_tf_bracket.npz
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
B, lr, g2 = 16, 0.125, 0.125
M, P = 512, 4
params = task["init"](jax.random.PRNGKey(0))
grad = jax.jit(jax.grad(task["loss"]))
step = jax.jit(lambda p, k: jax.tree.map(lambda a, g: a - lr * g, p, grad(p, task["sample"](k, B))))
Dg = np.array([1e-2, 1e-3, 1e-4, 1e-5])
res = dict(Dg=Dg)


def lanczos_coeffs(prm, batch, key):
    """P probes (sequential) of M Lanczos steps; returns al, be (P, M), zn2 (P,)"""
    mv = lambda u: task["gnvp"](prm, batch, u)
    leaves, tdef = jax.tree.flatten(prm)
    L = len(leaves)
    chunk = jax.jit(lambda ls, al, be, C, j0: slq.lanczos_chunk_tree(mv, ls, al, be, C, j0, 64))
    out_al, out_be, out_zn = [], [], []
    for i in range(P):
        ks = jax.random.split(jax.random.fold_in(key, i), len(leaves))
        z = jax.tree.unflatten(tdef, [jax.random.rademacher(kk, x.shape, jnp.float32) for kk, x in zip(ks, leaves)])
        ls = slq.lanczos_state_tree(z)
        al = jnp.zeros((M,)); be = jnp.zeros((M,)); C = jnp.zeros((M, L))
        for j0 in range(0, M, 64):
            ls, al, be, C = chunk(ls, al, be, C, j0)
        out_al.append(np.asarray(al, np.float64)); out_be.append(np.asarray(be, np.float64)); out_zn.append(float(ls["zn"]) ** 2)
    return np.stack(out_al), np.stack(out_be), np.array(out_zn)


k = jax.random.PRNGKey(1); n = 0; t0 = time.time()
for c in (200, 3000):
    while n < c:
        k, kk = jax.random.split(k); params = step(params, kk); n += 1
    Nfull = {}
    for b in (64, 128, 512, 2048):
        batch = task["sample"](jax.random.PRNGKey(9000 + c), b)          # nested: first b/2 of the b=... batch not required
        al, be, zn2 = lanczos_coeffs(params, batch, jax.random.PRNGKey(17 + c))
        for m in (64, 128, 256, 512):
            (thG, qG), (thR, qR) = slq.gauss_radau(al, be, m)
            U = np.array([np.mean(zn2 * slq.fsum(thG, qG, g2, D)) for D in Dg])
            Lb = np.array([np.mean(zn2 * slq.fsum(thR, qR, g2, D)) for D in Dg])
            res[f"U_c{c}_b{b}_m{m}"] = U; res[f"L_c{c}_b{b}_m{m}"] = Lb
        Nfull[b] = 0.5 * (res[f"U_c{c}_b{b}_m512"] + res[f"L_c{c}_b{b}_m512"])
        log(f"   tf bracket n={c} b={b}: " + " | ".join(
            f"D={D:.0e}: m64 [{res[f'L_c{c}_b{b}_m64'][i]:.0f},{res[f'U_c{c}_b{b}_m64'][i]:.0f}] "
            f"m512 [{res[f'L_c{c}_b{b}_m512'][i]:.0f},{res[f'U_c{c}_b{b}_m512'][i]:.0f}]" for i, D in enumerate(Dg))
            + f" ({time.time()-t0:.0f}s)")
    for b in (128, 512, 2048):
        ext = 2 * Nfull[b] - Nfull[b // 2 if b // 2 in Nfull else 64]
        res[f"ext_c{c}_b{b}"] = ext
        log(f"   tf bracket n={c} split-half extrapolation 2N_{b} - N_{b//2 if b//2 in Nfull else 64}: "
            + " ".join(f"D={D:.0e}:{x:.0f}" for D, x in zip(Dg, ext)))
    log(f"   tf bracket n={c}: N_eff/B at B={B} (midpoint, b=2048, m=512): " + " ".join(f"D={D:.0e}:{x / B:.1f}" for D, x in zip(Dg, Nfull[2048])))
np.savez_compressed(os.path.join(out_dir, "slq_tf_bracket.npz"), **res)
log("tf bracket done")
