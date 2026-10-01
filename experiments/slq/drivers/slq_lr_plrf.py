"""PLRF: choose the SGD step from the SLQ quadrature.  gamma_2 = tau * g2max_hat, where g2max_hat solves the exact
averaged-gradient threshold  sum_k w_k g th_k/(2B-(B+1) g th_k) = 1  on a deflated quadrature of K (exact operator,
Gaussian probes) or of the empirical K_b (b=4096).  Arms: SGD at several temperatures tau; DANA with N_eff from the
same quadrature (s=1/4, and saturating C=1 above the line).  Same keys as E1 -> common random numbers with the E1
grid-tuned SGD / oracle.  Output runs/slq/lr_<inst>_B<B>.npz
usage: python drivers/slq_lr_plrf.py --alpha 1.0 --beta 0.7 --B 2 64 1024 32768
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana.plrf import plrf_reduce
from adana.sim_ls import make_configs, simulate
from adana import theory, slq

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--beta", type=float, required=True)
ap.add_argument("--d", type=int, default=4096)
ap.add_argument("--v", type=int, default=None)
ap.add_argument("--B", type=int, nargs="+", required=True)
ap.add_argument("--nmax", type=int, default=1_000_000)
ap.add_argument("--seeds", type=int, default=4)
args = ap.parse_args()
v = args.v or 2 * args.d
root = os.path.expanduser("~/dana-exp"); out_dir = os.path.join(root, "runs", "slq")
logf = open(os.path.join(out_dir, "log.txt"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
inst = plrf_reduce(v, args.d, args.alpha, args.beta, seed=0, cache_dir=os.path.join(root, "cache"))
lam = np.asarray(inst["lam"], np.float64); d = len(lam); itag = f"a{args.alpha}_b{args.beta}_d{args.d}_v{v}"
above = 2 * args.alpha > 1
M, P = 64, 4


def quad(mv, seed):
    """deflated Gauss quadrature (nodes, weights/P) and raw weights, Gaussian probes"""
    rng = np.random.default_rng(seed); Z = rng.standard_normal((P, d))
    co = [slq.lanczos_full_np(mv, z, M) for z in Z]
    m = min(len(c[0]) for c in co)
    al = np.stack([c[0][:m] for c in co]); be = np.stack([c[1][:m] for c in co])
    th, wd, wr, nd = slq.ritz_quadrature_np(al, be, np.sum(Z * Z, 1), m)
    return th.reshape(-1), wd.reshape(-1) / P, wr.reshape(-1) / P


Xb = {}
def emp_mv(seed, b=4096):
    X = (np.random.default_rng(7000 + seed).standard_normal((b, d)) * np.sqrt(lam)).astype(np.float32)
    return lambda u: (X.T @ (X @ u.astype(np.float32)) / b).astype(np.float64)


for B in args.B:
    tag = f"lr_{itag}_B{B}"
    path = os.path.join(out_dir, tag + ".npz")
    if os.path.exists(path):
        log(f"skip {tag}"); continue
    gtrue = theory.g2_max(lam, B)
    rows, quads, ratios = [], [], {}
    for s_ in range(args.seeds):
        qe = quad(lambda u: lam * u, 11 * s_ + 1); qb = quad(emp_mv(s_), 11 * s_ + 2)
        ge = float(slq.g2max_quad(qe[0], qe[1], B)); gb = float(slq.g2max_quad(qb[0], qb[1], B))
        ratios.setdefault("exact", []).append(ge / gtrue); ratios.setdefault("emp", []).append(gb / gtrue)
        for src, gh, q in (("K", ge, qe), ("Kb", gb, qb)):
            for tau in (1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 0.9):
                rows.append(dict(g2=tau * gh, mode="sgd", seed=s_, fac=tau, grp=f"sgd_{src}", tau=tau)); quads.append(q[:2])
        for tau in (1 / 8, 1 / 4, 1 / 2):
            rows.append(dict(g2=tau * ge, mode="slq", s=0.25, delta=4.0, seed=s_, fac=tau, grp="dana_K", tau=tau)); quads.append(qe[:2])
            if above:
                rows.append(dict(g2=tau * ge, mode="slq", s=min(0.25, 1.0 / B), delta=4.0, seed=s_, fac=tau, grp="dana_K_sat1", tau=tau))
                quads.append(qe[:2])
    Q = max(len(q[0]) for q in quads)
    qn = np.stack([np.pad(q[0], (0, Q - len(q[0]))) for q in quads]).astype(np.float32)
    qw = np.stack([np.pad(q[1], (0, Q - len(q[1]))) for q in quads]).astype(np.float32)
    hp = make_configs(rows)
    nmax = 2 * args.nmax if B == 2 else args.nmax
    log(f"start {tag}: G={len(rows)}; g2max_hat/g2max exact-K {np.mean(ratios['exact']):.3f}+-{np.std(ratios['exact']):.3f}, "
        f"K_4096 {np.mean(ratios['emp']):.3f}+-{np.std(ratios['emp']):.3f}")
    t1 = time.time()
    out = simulate(inst, hp, B, nmax, key=1000 + B, n_pts=90, quad=(qn, qw))
    meta = dict(alpha=args.alpha, beta=args.beta, d=args.d, v=v, B=B, g2max=gtrue, Pstar=inst["Pstar"], nmax=nmax, seeds=args.seeds,
                ratio_exact=ratios["exact"], ratio_emp=ratios["emp"], runtime=time.time() - t1)
    np.savez_compressed(path, **out, **{f"hp_{k}": v_ for k, v_ in hp.items()}, fac=np.array([r["fac"] for r in rows]),
                        grp=np.array([r["grp"] for r in rows]), lam=lam, meta=json.dumps(meta))
    S = args.seeds; G = len(rows) // S
    E = out["E"][-1].reshape(S, G); al = out["alive"][-1].reshape(S, G).all(0); Em = np.where(al, E.mean(0), np.nan)
    grp = [r["grp"] for r in rows[:G]]; tau = [r["tau"] for r in rows[:G]]
    for g in sorted(set(grp)):
        log(f"   {g:12s}: " + " | ".join(f"tau={t:.3g}: {e:.2e}" for gg, t, e in zip(grp, tau, Em) if gg == g))
    e1 = os.path.join(root, "runs", "e1", f"{itag}_B{B}.npz")
    if os.path.exists(e1):
        z1 = np.load(e1, allow_pickle=True); m1 = json.loads(str(z1["meta"])); S1 = m1["seeds"]; G1 = len(z1["grp"]) // S1
        E1 = np.where(z1["alive"][-1].reshape(S1, G1).all(0), z1["E"][-1].reshape(S1, G1).mean(0), np.inf)
        g1 = z1["grp"][:G1]; f1 = z1["fac"][:G1]; ms = g1 == "sgd"
        log(f"   E1 grid-tuned SGD (exact threshold x fac): best {E1[ms].min():.2e} at fac {f1[ms][np.argmin(E1[ms])]:.3g}")
    log(f"done {tag} ({time.time()-t1:.0f}s)")
