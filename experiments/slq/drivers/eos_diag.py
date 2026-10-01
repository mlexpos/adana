"""Per-tensor edge-of-stability diagnostic.
For a tuned, scheduled optimizer (SGD with clipping / Adam / LaProp) run at peak lr x {1/4, 1/2, 1, 2}, measure at
log-spaced checkpoints, on a fixed batch, the top eigenvalue of every parameter tensor's Hessian diagonal block H_TT
(for Adam/LaProp: of the preconditioned block P^-1/2 H P^-1/2, P = sqrt(v_hat)+eps), by warm-started power iteration
with Hessian-vector products restricted to the tensor; plus the global top eigenvalue.
  x_T = lambda_T / threshold,  threshold = 2/eta_eff (SGD; eta_eff = sigma(n) lr <clip factor>) or 38/(sigma(n) lr) (Adam/LaProp,
  beta1 = 0.9: 2(1+b1)/((1-b1) eta)).
EoS: x_T ~ 1 and d log lambda_T / d log lr ~ -1;  stiff: x_T < 1 and lambda_T independent of lr.
usage: python drivers/eos_diag.py --task modarith --B 16 --nmax 30000 --opt adam
Output runs/<e3|slq>/eos_<task>_B<B>_<opt>.npz
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import jax, jax.numpy as jnp
from adana import tasks, optim_tree as O
from adana.sim_ls import eval_points

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["modarith", "mlp"], required=True)
ap.add_argument("--B", type=int, required=True)
ap.add_argument("--nmax", type=int, required=True)
ap.add_argument("--opt", choices=["sgd", "adam", "laprop"], required=True)
ap.add_argument("--b_eval", type=int, default=512)
ap.add_argument("--iters", type=int, default=6)
ap.add_argument("--mults", type=str, default="", help="comma-separated lr multiples (ramp mode; output suffix _ramp)")
args = ap.parse_args()
root = os.path.expanduser("~/dana-exp")
if args.task == "modarith":
    task = tasks.modarith(p=97, T=32, zipf=1.2, eta_sig=0.7); sub, clip = "e3", 1.0
    base = os.path.join(root, "runs", "e3", f"{task['name']}_B{args.B}_sched_base.npz")
else:
    task = tasks.deep_mlp(v=256, width=128, depth=3, alpha=1.0, teacher_width=256, teacher_depth=3, n_eval=16384)
    sub, clip = "slq", 0.0
    base = os.path.join(root, "runs", "slq", f"{task['name']}_B{args.B}_sched_base.npz")
out_dir = os.path.join(root, "runs", sub)
logf = open(os.path.join(out_dir, "log.txt"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

zb = np.load(base, allow_pickle=True)
fin = zb[f"{args.opt}_risk"][-1].mean(0); fin = np.where(np.isfinite(fin), fin, np.inf)
lr0 = float(json.loads(str(zb[f"{args.opt}_hp"]))[int(np.argmin(fin))]["lr"])
mults = np.array([float(x) for x in args.mults.split(",")]) if args.mults else np.array([0.25, 0.5, 1.0, 2.0]); R = len(mults)
i1 = int(np.argmin(np.abs(mults - 1.0)))                                     # index of the tuned lr
sfx = "_ramp" if args.mults else ""
N = args.nmax
hp = {"lr": jnp.asarray(lr0 * mults, jnp.float32), "b1": jnp.full(R, 0.9), "b2": jnp.full(R, 0.99), "eps": jnp.full(R, 1e-8),
      "T": jnp.full(R, float(N)), "clip": jnp.full(R, clip)}
p0 = task["init"](jax.random.PRNGKey(0))
params = jax.tree.map(lambda a: jnp.broadcast_to(a, (R,) + a.shape).copy(), p0)
st = jax.vmap(lambda p: O.init(args.opt, p))(params)
leaves0, tdef = jax.tree.flatten(p0)
names = ["/".join(str(getattr(k, "key", k)) for k in path) for path, _ in jax.tree_util.tree_flatten_with_path(p0)[0]]
L = len(leaves0)
grad = jax.grad(task["loss"])


def one_step(p, s, h, batch):
    g = grad(p, batch)
    gn = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree.leaves(g)))
    newp, news = O.step(args.opt, p, s, g, g, g, h, args.B)
    return newp, news, jnp.minimum(1.0, jnp.where(h["clip"] > 0, h["clip"] / jnp.maximum(gn, 1e-30), 1.0))


vstep = jax.vmap(one_step, in_axes=(0, 0, 0, None))


@jax.jit
def advance(p, s, key, n0, n1, cf):
    def body(n, c):
        p, s, cf = c
        batch = task["sample"](jax.random.fold_in(key, n), args.B)
        p, s, clf = vstep(p, s, hp, batch)
        return p, s, 0.98 * cf + 0.02 * clf
    return jax.lax.fori_loop(n0, n1, body, (p, s, cf))


evb = task["sample"](jax.random.PRNGKey(424242), args.b_eval)
hvp = lambda p, u: jax.jvp(lambda q: grad(q, evb), (p,), (u,))[1]


def precond(s, n):
    if args.opt == "sgd":
        return None
    bc2 = 1 - 0.99 ** jnp.maximum(n, 1.0)
    return jax.tree.map(lambda v: (jnp.sqrt(v / bc2) + 1e-8) ** -0.5, s["v"])


def block_power(p, r, U, iters):
    """top eigenvalue of every diagonal block (and globally) of r H r; U: list of L+1 warm-start pytrees"""
    out, newU = [], []
    mask = lambda u, T: jax.tree.unflatten(tdef, [x if i == T else jnp.zeros_like(x) for i, x in enumerate(jax.tree.leaves(u))])
    op = (lambda u: hvp(p, u)) if r is None else (lambda u: jax.tree.map(lambda a, b: a * b, r, hvp(p, jax.tree.map(lambda a, b: a * b, r, u))))
    for T in range(L + 1):
        u = U[T]
        for _ in range(iters):
            w = op(u)
            if T < L:
                w = mask(w, T)
            nw = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree.leaves(w)))
            lam = sum(jnp.sum(a * b) for a, b in zip(jax.tree.leaves(u), jax.tree.leaves(w)))
            # a vanishing block (e.g. hidden layers of an MLP with zero readout at init) must not zero the warm start
            ok = nw > 1e-20
            u = jax.tree.map(lambda x, x0: jnp.where(ok, x / jnp.where(ok, nw, 1.0), x0), w, u)
        out.append(lam); newU.append(u)
    return jnp.stack(out), newU


bp = jax.jit(jax.vmap(block_power, in_axes=(0, 0 if args.opt != "sgd" else None, 0, None)), static_argnums=(3,))


def init_U(key):
    Us = []
    for T in range(L + 1):
        ks = jax.random.split(jax.random.fold_in(key, T), L)
        u = [jax.random.normal(k, (R,) + x.shape) * (1.0 if (T == L or i == T) else 0.0) for i, (k, x) in enumerate(zip(ks, leaves0))]
        nrm = jnp.sqrt(sum(jnp.sum(x * x, axis=tuple(range(1, x.ndim))) for x in u))
        Us.append(jax.tree.unflatten(tdef, [x / nrm.reshape((R,) + (1,) * (x.ndim - 1)) for x in u]))
    return Us


def gnvp_op(p, r):
    if r is None:
        return lambda u: task["gnvp"](p, evb, u)
    return lambda u: jax.tree.map(lambda a, b: a * b, r, task["gnvp"](p, evb, jax.tree.map(lambda a, b: a * b, r, u)))


def trace_hutch(p, r, key, n_probe=8):
    """Hutchinson estimate of tr(r GN r) globally and per tensor (Rademacher probes)"""
    op = gnvp_op(p, r)
    def one(k):
        ks = jax.random.split(k, L)
        z = jax.tree.unflatten(tdef, [jax.random.rademacher(kk, x.shape, jnp.float32) for kk, x in zip(ks, jax.tree.leaves(p))])
        w = op(z)
        return jnp.stack([jnp.sum(a * b) for a, b in zip(jax.tree.leaves(z), jax.tree.leaves(w))])
    per = jax.vmap(one)(jax.random.split(key, n_probe)).mean(0)
    return jnp.concatenate([per, per.sum()[None]])


th = jax.jit(jax.vmap(trace_hutch, in_axes=(0, 0 if args.opt != "sgd" else None, None)))
ev = eval_points(N, 30)
U = init_U(jax.random.PRNGKey(5))
cf = jnp.ones(R)
key = jax.random.PRNGKey(11)
rec = dict(n=[], lam=[], thr=[], risk=[], sig=[], cf=[], trF=[])
prev = 0; t0 = time.time()
for n in ev:
    if n > prev:
        params, st, cf = advance(params, st, key, prev, int(n), cf); prev = int(n)
    r = precond(st, float(n)) if args.opt != "sgd" else None
    lam, U = bp(params, r, U, args.iters if n > 0 else 20)
    rec["trF"].append(np.asarray(th(params, r, jax.random.PRNGKey(int(n) + 7))))
    Tt = float(N); wu = max(0.02 * Tt, 1.0)
    sig = (n + 1) / wu if n < wu else 0.1 + 0.45 * (1 + np.cos(np.pi * min(max((n - wu) / (Tt - wu), 0), 1)))
    eta = np.asarray(lr0 * mults * sig)
    thr = 2.0 / (eta * np.asarray(cf)) if args.opt == "sgd" else 38.0 / eta
    risk = np.asarray(jax.vmap(task["evaluate"])(params)["risk"])
    rec["n"].append(n); rec["lam"].append(np.asarray(lam)); rec["thr"].append(thr); rec["risk"].append(risk)
    rec["sig"].append(sig); rec["cf"].append(np.asarray(cf))
    if len(rec["n"]) % 5 == 0 or n == ev[-1]:
        x = np.asarray(lam)[i1, :L] / thr[i1]
        log(f"   eos {task['name']} B{args.B} {args.opt} n={n}: risk(lr x1) {risk[i1]:.4f} | global lam/thr "
            + " ".join(f"{m:g}x:{np.asarray(lam)[k, L] / thr[k]:.2f}" for k, m in enumerate(mults))
            + f" | tensors x_T (lr x1) min/med/max {x.min():.2f}/{np.median(x):.2f}/{x.max():.2f} ({time.time()-t0:.0f}s)")
out = {k: np.asarray(v) for k, v in rec.items()}
lam = out["lam"]            # (n, R, L+1)
late = slice(len(out["n"]) // 2, None)
def fit_slope(T):
    y = np.nanmedian(lam[late, :, T], 0); ok = np.isfinite(y) & (y > 0) & np.isfinite(out["risk"][-1]) & (out["risk"][-1] < 10)
    return np.polyfit(np.log(mults[ok]), np.log(y[ok]), 1)[0] if ok.sum() >= 2 else np.nan
slope = np.array([fit_slope(T) for T in range(L + 1)])
xmed = np.nanmedian(lam[late, i1, :] / out["thr"][late, i1][:, None], 0)
np.savez_compressed(os.path.join(out_dir, f"eos_{task['name']}_B{args.B}_{args.opt}{sfx}.npz"), **out, names=np.array(names + ["GLOBAL"]),
                    mults=mults, lr0=lr0, slope=slope, xmed=xmed, meta=json.dumps(dict(task=task["name"], B=args.B, opt=args.opt, nmax=N)))
trF = out["trF"]          # (n, R, L+1)
def fit_tr(T):
    y = np.nanmedian(trF[late, :, T], 0); ok = np.isfinite(y) & (y > 0) & np.isfinite(out["risk"][-1]) & (out["risk"][-1] < 10)
    return np.polyfit(np.log(mults[ok]), np.log(y[ok]), 1)[0] if ok.sum() >= 2 else np.nan
tr_slope = np.array([fit_tr(T) for T in range(L + 1)])
teff = float(task["seq_len"] - 1) if "seq_len" in task else 1.0          # ~tokens per sequence (measured T_eff ~ 0.8-1.0 T')
Beff = args.B * teff
stoch_thr = 2 * Beff / np.nanmedian(trF[late, i1, L])                       # stochastic threshold at the tuned lr (global)
lip_thr = 2 / np.nanmedian(lam[late, i1, L])
eta_late = np.median(lr0 * np.asarray(out["sig"])[late] * (np.asarray(out["cf"])[late, i1] if args.opt == "sgd" else 1.0))
np.savez_compressed(os.path.join(out_dir, f"eos_{task['name']}_B{args.B}_{args.opt}{sfx}.npz"), **out, names=np.array(names + ["GLOBAL"]),
                    mults=mults, lr0=lr0, slope=slope, xmed=xmed, tr_slope=tr_slope, stoch_thr=stoch_thr, lip_thr=lip_thr,
                    eta_late=eta_late, meta=json.dumps(dict(task=task["name"], B=args.B, opt=args.opt, nmax=N)))
log(f"eos {task['name']} B{args.B} {args.opt}: GLOBAL response of lambda_max {slope[L]:.2f} vs of tr F {tr_slope[L]:.2f} | "
    f"late thresholds at tuned lr: Lipschitz 2/lambda_max {lip_thr:.3g}, stochastic 2B_eff/trF {stoch_thr:.3g} (B_eff={Beff:g}); "
    f"median effective step {eta_late:.3g} (peak lr {lr0:.3g})")
# ramp: per lr multiple, the time course of the Lipschitz ratio chi_L = eta lam_max/2 (SGD; eta = sigma lr clipfactor) and the
# stochastic ratio chi_S = eta trF/(2 B_eff), and the final risk -> which ratio crosses 1 before the breakdown
eta_t = lr0 * np.asarray(out["sig"])[:, None] * mults[None, :] * (np.asarray(out["cf"]) if args.opt == "sgd" else 1.0)
chiL = eta_t * lam[:, :, L] / 2; chiS = eta_t * trF[:, :, L] / (2 * Beff)
np.savez_compressed(os.path.join(out_dir, f"eos_{task['name']}_B{args.B}_{args.opt}{sfx}.npz"), **out, names=np.array(names + ["GLOBAL"]),
                    mults=mults, lr0=lr0, slope=slope, xmed=xmed, tr_slope=tr_slope, stoch_thr=stoch_thr, lip_thr=lip_thr,
                    eta_late=eta_late, chiL=chiL, chiS=chiS, Beff=Beff, meta=json.dumps(dict(task=task["name"], B=args.B, opt=args.opt, nmax=N)))
for k, m in enumerate(mults):
    fin = np.isfinite(chiL[:, k]) & np.isfinite(chiS[:, k])
    first = lambda c: (int(np.asarray(out["n"])[np.argmax(c > 1)]) if (c > 1).any() else -1)
    log(f"   ramp {task['name']} B{args.B} {args.opt} lr x{m:g} (peak {lr0*m:.3g}): final risk {out['risk'][-1][k]:.4g} | "
        f"chi_L late med {np.nanmedian(chiL[late, k]):.2f} max {np.nanmax(chiL[fin, k]) if fin.any() else np.nan:.2f} | "
        f"chi_S late med {np.nanmedian(chiS[late, k]):.3f} max {np.nanmax(chiS[fin, k]) if fin.any() else np.nan:.3f} "
        f"first n>1: {first(chiS[fin, k])} | first finite-risk loss n: {int(np.asarray(out['n'])[np.argmax(~np.isfinite(out['risk'][:, k]) | (out['risk'][:, k] > 10))]) if (~np.isfinite(out['risk'][:, k]) | (out['risk'][:, k] > 10)).any() else -1}")
order = np.argsort(xmed[:L])
log(f"eos {task['name']} B{args.B} {args.opt} (lr0 {lr0:.3g}): second-half medians. GLOBAL x={xmed[L]:.2f} slope={slope[L]:.2f} | per tensor (x, dlog lam/dlog lr): "
    + " ".join(f"{names[i]}=({xmed[i]:.2f},{slope[i]:.2f})" for i in order))
log(f"eos done ({time.time()-t0:.0f}s)")
