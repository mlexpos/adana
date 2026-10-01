"""Correlation length of the gradient noise within sequences (transformer, smooth modular arithmetic).
T_eff = effective number of independent tokens per sequence; B_eff = B T_eff enters the momentum budget.
(1) residual-stream proxy per layer: delta^l_t = dL/dh^l_t (zero probes at the input and after every block),
    minibatch mean per position removed; T_eff^l = T' / R^l, R^l = E||sum_t d_t||^2 / E sum_t ||d_t||^2.
(2) ground truth per parameter tensor: random-sign halves split by SEQUENCES vs by TOKENS (balanced masks);
    E||gA-gB||^2 = (4/(BT')^2) sum_i E||G_i||^2 (seq) resp. (4/(BT')^2) sum_{i,t} E||g_it||^2 (tok) -> T_eff = T' E_tok/E_seq.
(3) global T_eff over all parameters.  Output runs/e3/corr_length.npz
"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import jax, jax.numpy as jnp

from adana import tasks

root = os.path.expanduser("~/dana-exp"); out_dir = os.path.join(root, "runs", "e3")
logf = open(os.path.join(out_dir, "log.txt"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
task = tasks.modarith(p=97, T=32, zipf=1.2, eta_sig=0.7)
L, D, T = task["n_layers"], task["d_model"], task["seq_len"]
Tp = T - 1
Btr, lr = 16, 0.125
Bm, M = 128, 32
params = task["init"](jax.random.PRNGKey(0))
grad = jax.jit(jax.grad(task["loss"]))
step = jax.jit(lambda p, k: jax.tree.map(lambda a, g: a - lr * g, p, grad(p, task["sample"](k, Btr))))
probe_grad = jax.jit(jax.grad(task["loss_probe"], argnums=2))
gw = jax.jit(jax.grad(task["loss_w"]))
names = ["/".join(str(getattr(k, "key", k)) for k in path) for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]]
sq = lambda t: np.array([float(jnp.sum(x * x)) for x in jax.tree.leaves(t)])
res = dict(names=np.array(names))
k = jax.random.PRNGKey(1); n = 0; t0 = time.time()
for c in (100, 1000, 5000, 15000):
    while n < c:
        k, kk = jax.random.split(k); params = step(params, kk); n += 1
    Rn = np.zeros(L + 1); Rd = np.zeros(L + 1); Eseq = 0; Etok = 0
    for j in range(M):
        batch = task["sample"](jax.random.PRNGKey(50_000 + 97 * c + j), Bm)
        # (1) residual-stream probes
        probes = [jnp.zeros((Bm, T, D)) for _ in range(L + 1)]
        dl = probe_grad(params, batch, probes)
        for l, d in enumerate(dl):
            d = np.asarray(d[:, :Tp], np.float64)
            d = d - d.mean(0, keepdims=True)                      # remove the signal (minibatch mean per position)
            Rn[l] += np.mean(np.sum(d.sum(1) ** 2, -1)); Rd[l] += np.mean(np.sum(d ** 2, (1, 2)))
        # (2) sequence split vs token split
        half = Bm // 2
        wA = jnp.concatenate([jnp.ones((half, Tp)), jnp.zeros((half, Tp))]); wB = 1 - wA
        Eseq = Eseq + sq(jax.tree.map(lambda a, b: a - b, gw(params, batch, wA), gw(params, batch, wB)))
        perm = jax.random.permutation(jax.random.PRNGKey(123 + j + c), Bm * Tp)
        mA = jnp.zeros(Bm * Tp).at[perm[: Bm * Tp // 2]].set(1.0).reshape(Bm, Tp)
        Etok = Etok + sq(jax.tree.map(lambda a, b: a - b, gw(params, batch, mA), gw(params, batch, 1 - mA)))
    Tl = Tp * Rd / Rn
    Tt = Tp * Etok / np.maximum(Eseq, 1e-30)
    Tg = Tp * Etok.sum() / Eseq.sum()
    res[f"Tlayer_c{c}"] = Tl; res[f"Ttensor_c{c}"] = Tt; res[f"Tglobal_c{c}"] = Tg
    res[f"Eseq_c{c}"] = Eseq; res[f"Etok_c{c}"] = Etok
    risk = float(task["evaluate"](params)["risk"])
    log(f"   corr n={c} (excess CE {risk:.3f}): T'={Tp} | global T_eff (per-tensor split ratio, all params) = {Tg:.2f} | "
        f"residual-stream proxy T_eff: " + " ".join(f"{'emb' if l == 0 else 'blk' + str(l)}={Tl[l]:.2f}" for l in range(L + 1))
        + f" ({time.time()-t0:.0f}s)")
    order = np.argsort(-Eseq)
    log(f"   corr n={c} per-tensor T_eff (largest-noise tensors first): "
        + " ".join(f"{names[i]}={Tt[i]:.1f}" for i in order[:14]))
np.savez_compressed(os.path.join(out_dir, "corr_length.npz"), **res)
log("corr length done")
