"""SLQ estimator of N_eff on PLRF: (a) accuracy vs the exact N_eff(Delta) over Delta, Lanczos steps m, probes p,
reorthogonalization, exact K vs empirical K_b (b samples); (b) closed loop: DANA with g3 = g2 min{s_eff B / Nhat, 1}
where Nhat comes from ONE SLQ quadrature per run, compared with the true-N_eff rule and the shadow rule
(same keys as E1 -> common random numbers with the E1 SGD / oracle runs).
usage: python drivers/slq_plrf.py --part acc|loop --alpha 1.0 --beta 0.7 [--v 8192] --B 2 64 1024 32768
Output: runs/slq/acc_<inst>.npz, runs/slq/<inst>_B<B>.npz
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana.plrf import plrf_reduce
from adana.sim_ls import make_configs, simulate
from adana import theory, slq

ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=["acc", "loop"], required=True)
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--beta", type=float, required=True)
ap.add_argument("--d", type=int, default=4096)
ap.add_argument("--v", type=int, default=None)
ap.add_argument("--B", type=int, nargs="+", default=[64])
ap.add_argument("--nmax", type=int, default=1_000_000)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--m", type=int, default=64)
ap.add_argument("--probes", type=int, default=4)
ap.add_argument("--b_emp", type=int, default=4096)
args = ap.parse_args()
v = args.v or 2 * args.d
root = os.path.expanduser("~/dana-exp")
out_dir = os.path.join(root, "runs", "slq"); os.makedirs(out_dir, exist_ok=True)
logf = open(os.path.join(out_dir, "log.txt"), "a")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

inst = plrf_reduce(v, args.d, args.alpha, args.beta, seed=0, cache_dir=os.path.join(root, "cache"))
lam = np.asarray(inst["lam"], np.float64); d = len(lam)
itag = f"a{args.alpha}_b{args.beta}_d{args.d}_v{v}"
exact_mv = lambda u: lam * u


def emp_op(b, seed):
    X = (np.random.default_rng(seed).standard_normal((b, d)) * np.sqrt(lam)).astype(np.float32)
    return lambda u: (X.T @ (X @ u.astype(np.float32)) / b).astype(np.float64)


if args.part == "acc":
    g2 = 0.25 * theory.g2_max(lam, 64)
    Dg = np.logspace(-7, 0, 36)
    Ntrue = np.array([np.sum(g2 * lam / (g2 * lam + D)) for D in Dg])
    res = {}
    t0 = time.time()
    for m in (8, 16, 32, 64, 128, 256):
        for p in (1, 4, 16):
            est = []
            for rep in range(8):
                qn, qw = slq.quadrature_np(exact_mv, d, m=m, probes=p, seed=1000 * rep + m + p)
                est.append([slq.neff_from_quad(qn, qw, g2, D) for D in Dg])
            res[f"exact_m{m}_p{p}"] = np.array(est)
    for m in (32,):
        est = []
        for rep in range(8):
            qn, qw = slq.quadrature_np(exact_mv, d, m=m, probes=4, seed=77 + rep, reorth=True)
            est.append([slq.neff_from_quad(qn, qw, g2, D) for D in Dg])
        res[f"exact_m{m}_p4_reorth"] = np.array(est)
    Ntrue_emp = {}
    for b in (256, 1024, 4096, 16384):
        est, tru = [], []
        for rep in range(4):
            mv = emp_op(b, 5000 + rep)
            qn, qw = slq.quadrature_np(mv, d, m=64, probes=4, seed=rep)
            est.append([slq.neff_from_quad(qn, qw, g2, D) for D in Dg])
        res[f"emp_b{b}_m64_p4"] = np.array(est)
        # exact N_eff of the empirical operator (one rep) to separate sampling bias from quadrature error
        X = np.random.default_rng(5000).standard_normal((b, d)) * np.sqrt(lam)
        ev = np.linalg.eigvalsh(X.T @ X / b) if b <= 16384 else None
        Ntrue_emp[b] = np.array([np.sum(g2 * np.maximum(ev, 0) / (g2 * np.maximum(ev, 0) + D)) for D in Dg])
        log(f"   acc {itag}: b={b} done ({time.time()-t0:.0f}s)")
    np.savez_compressed(os.path.join(out_dir, f"acc_{itag}.npz"), Dg=Dg, Ntrue=Ntrue, g2=g2, lam=lam,
                        **res, **{f"Ntrue_emp_b{b}": x for b, x in Ntrue_emp.items()})
    band = (Ntrue >= 1) & (Ntrue <= d / 2)
    for k, e in res.items():
        r = np.abs(np.log(e[:, band] / Ntrue[band]))
        log(f"   acc {itag} {k:22s}: median |log ratio| {np.median(r):.3f}  90% {np.quantile(r, .9):.3f}  max {r.max():.3f}")
    log(f"acc {itag} done ({time.time()-t0:.0f}s)")
else:
    for B in args.B:
        tag = f"{itag}_B{B}"
        path = os.path.join(out_dir, tag + ".npz")
        if os.path.exists(path):
            log(f"skip {tag}"); continue
        gm = theory.g2_max(lam, B)
        rows, quads = [], []
        for s in range(args.seeds):
            for C in (None, 1.0, 2.0):
                se = 0.25 if C is None else min(0.25, C / B)
                lab = "" if C is None else f"_sat{C:g}"
                for grp, mode in (("theory", "theory"), ("slq", "slq"), ("slq32", "slq"), ("slq_emp", "slq"), ("shadow", "shadow_batch")):
                    rows.append(dict(g2=0.25 * gm, mode=mode, s=se, delta=4.0, seed=s, fac=0.25, grp=grp + lab))
                    if grp == "slq":
                        quads.append(slq.quadrature_np(exact_mv, d, m=args.m, probes=args.probes, seed=100 * s + 1))
                    elif grp == "slq32":
                        q = slq.quadrature_np(exact_mv, d, m=32, probes=args.probes, seed=100 * s + 3)
                        quads.append((np.pad(q[0], (0, (args.m - 32) * args.probes)), np.pad(q[1], (0, (args.m - 32) * args.probes))))
                    elif grp == "slq_emp":
                        quads.append(slq.quadrature_np(emp_op(args.b_emp, 900 + s), d, m=args.m, probes=args.probes, seed=100 * s + 2))
                    else:
                        quads.append((np.zeros(args.m * args.probes), np.zeros(args.m * args.probes)))
        qn = np.stack([q[0] for q in quads]).astype(np.float32); qw = np.stack([q[1] for q in quads]).astype(np.float32)
        hp = make_configs(rows)
        nmax = 2 * args.nmax if B == 2 else args.nmax
        log(f"start {tag}: G={len(rows)}, nmax={nmax}")
        t1 = time.time()
        out = simulate(inst, hp, B, nmax, key=1000 + B, n_pts=90, quad=(qn, qw))
        meta = dict(alpha=args.alpha, beta=args.beta, d=args.d, v=v, B=B, g2max=gm, Pstar=inst["Pstar"], P0=inst["P0"],
                    nmax=nmax, seeds=args.seeds, m=args.m, probes=args.probes, b_emp=args.b_emp, runtime=time.time() - t1)
        np.savez_compressed(path, **out, **{f"hp_{k}": v_ for k, v_ in hp.items()}, fac=np.array([r["fac"] for r in rows]),
                            grp=np.array([r["grp"] for r in rows]), lam=lam, meta=json.dumps(meta))
        E = out["E"][-1]
        for grp in sorted(set(r["grp"] for r in rows)):
            msk = np.array([r["grp"] == grp for r in rows])
            ok = out["alive"][-1][msk]
            log(f"   {grp:16s}: seed-mean final excess {np.mean(E[msk]) if ok.all() else np.nan:.3e}  (alive {ok.sum()}/{msk.sum()})")
        log(f"done {tag} ({time.time()-t1:.0f}s)")
