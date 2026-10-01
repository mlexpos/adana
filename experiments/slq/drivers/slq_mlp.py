"""Deep-MLP teacher-student panel for the SLQ estimator.
Per batch size B: tuned SGD / Adam / LaProp baselines, oracle DANA grid, and the adaptive rules with the two N_eff
estimators — shadow buffer (1 GN-vector product per step) vs SLQ quadrature (refreshed on a geometric schedule) —
each global and per-tensor, for SGD-DANA and LaProp-DANA v2.1.  Output runs/slq/<task>_B<B>.npz
usage: python drivers/slq_mlp.py --B 8 128 1024 --nmax 30000
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana import tasks, trainer
from adana.sim_ls import eval_points

KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s", "f", "C", "bk", "pt", "tau")
ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, nargs="+", required=True)
ap.add_argument("--nmax", type=int, default=30000)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--m", type=int, default=64)
ap.add_argument("--p", type=int, default=4)
ap.add_argument("--b_est", type=int, default=1024)
ap.add_argument("--auto", action="store_true", help="SGD step from the quadrature: g2 = tau * g2max_hat (SGD and SGD-DANA); "
                "output <tag>_auto.npz")
ap.add_argument("--adaptive", action="store_true", help="only the adaptive-SLQ arms (bracket-stopped m, grown b, de-biased), "
                "reusing lr_edge / LaProp lr from the existing panel file; output <tag>_ad.npz")
args = ap.parse_args()
task = tasks.deep_mlp(v=256, width=128, depth=3, alpha=args.alpha, teacher_width=256, teacher_depth=3, n_eval=16384)
root = os.path.expanduser("~/dana-exp"); out_dir = os.path.join(root, "runs", "slq"); os.makedirs(out_dir, exist_ok=True)
logf = open(os.path.join(out_dir, "log.txt"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
fin = lambda r: np.where(np.isfinite(r[-1].mean(0)), r[-1].mean(0), np.inf)
S = args.seeds
for B in args.B:
    tag = f"{task['name']}_B{B}" + ("_ad" if args.adaptive else "") + ("_auto" if args.auto else "")
    path = os.path.join(out_dir, tag + ".npz")
    if os.path.exists(path):
        log(f"skip {tag}"); continue
    ev = eval_points(args.nmax, 50)
    run = lambda opt, mode, rows, **kw: trainer.run(task, opt, mode, rows, KEYS, S=S, B=B, n_max=args.nmax, eval_pts=ev, log=None, **kw)
    t0 = time.time(); res = {}; tim = {}
    def arm(name, opt, mode, rows, **kw):
        t = time.time(); r = run(opt, mode, rows, **kw); tim[name] = time.time() - t
        res[name] = r; res[name + "_hp"] = rows
        log(f"   {tag} {name:12s}: best {np.min(fin(r['risk'])):.4e} ({len(rows)} cfgs, {tim[name]:.0f}s)")
        return r
    if args.auto:
        acfg = dict(adaptive=True, p=args.p, m_max=256, chunk=8, eps=0.05, b0=256, b_max=16384, refresh0=True)
        rows = [dict(lr=0.0, tau=t, s=0.0, delta=4.0, cap=1.0, pt=0.0, grp="sgd_auto") for t in (0.25, 0.5, 1.0, 1.5, 2.0)] + \
               [dict(lr=0.0, tau=t, s=0.25, delta=4.0, cap=1.0, pt=pt, grp="dana_auto") for pt in (0.0, 1.0) for t in (0.25, 0.5, 1.0)]
        r = arm("auto", "dana_slq", "slq", rows, slq=acfg)
        info = [dict(n=x["n"], m=x["m"], b=x["b"], g2max=x["g2max"]) for x in r["refresh_info"]]
        z0 = np.load(os.path.join(out_dir, f"{task['name']}_B{B}.npz"))
        log(f"   {tag}: grid-tuned SGD best {np.min(fin(z0['sgd_risk'])):.4e} at lr {z0['lr_edge']:.3g}-edge grid; "
            f"g2max_hat trajectory (median): " + " ".join(f"{x['n']}:{x['g2max']:.3g}" for x in info[::6]))
        flat = dict(n=ev, info=json.dumps(info), time=json.dumps(tim), meta=json.dumps(dict(task=task["name"], B=B, seeds=S, **acfg)),
                    auto_hp=json.dumps(rows))
        for kk, vv in r.items():
            if kk != "refresh_info":
                flat[f"auto_{kk}"] = vv
        np.savez_compressed(path, **flat)
        log(f"done {tag} ({time.time()-t0:.0f}s)")
        continue
    if args.adaptive:
        z0 = np.load(os.path.join(out_dir, f"{task['name']}_B{B}.npz"))
        le, lb = float(z0["lr_edge"]), float(z0["lap_best_lr"])
        acfg = dict(adaptive=True, p=args.p, m_max=256, chunk=8, eps=0.05, b0=256, b_max=16384)
        drows = [dict(lr=f * le, s=s_, delta=4.0, cap=1.0, fac=f, pt=pt) for pt in (0.0, 1.0) for f in (0.125, 0.25, 0.5) for s_ in (0.0625, 0.25, 1.0)]
        lrows = [dict(lr=lrm * lb, lrm=lrm, s=s_, f=0.5, b2=0.99, eps=1e-8, delta=4.0, cap=1.0, bk=1.0, C=1e9, pt=pt)
                 for pt in (0.0, 1.0) for lrm in (0.3, 1.0) for s_ in (0.25, 1.0)]
        r1 = arm("slq_ad", "dana_slq", "slq", drows, slq=acfg)
        r2 = arm("lpd2_slq_ad", "lpd2_slq", "slq", lrows, slq=acfg)
        info = {k: [dict(n=x["n"], m=x["m"], b=x["b"], ratio=x["ratio"]) for x in r["refresh_info"]] for k, r in (("slq_ad", r1), ("lpd2_slq_ad", r2))}
        cost = {k: dict(refreshes=len(v), gnvp_samples=int(sum(args.p * x["m"] * x["b"] for x in v)), train_samples=args.nmax * B)
                for k, v in info.items()}
        log(f"   {tag} adaptive cost: " + " | ".join(f"{k}: {c['gnvp_samples']:.3g} sample-GNVPs ({c['refreshes']} refreshes; "
                                                    f"final m={info[k][-1]['m']}, b={info[k][-1]['b']})" for k, c in cost.items())
            + f" | training samples {args.nmax * B:.3g}")
        flat = dict(n=ev, lr_edge=le, lap_best_lr=lb, cost=json.dumps(cost), info=json.dumps(info), time=json.dumps(tim),
                    meta=json.dumps(dict(task=task["name"], B=B, seeds=S, **acfg)))
        for k, v in res.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk != "refresh_info":
                        flat[f"{k}_{kk}"] = vv
            elif isinstance(v, list):
                flat[k] = json.dumps(v)
        np.savez_compressed(path, **flat)
        log(f"done {tag} ({time.time()-t0:.0f}s)")
        continue
    sgrid = list(2.0 ** np.arange(-4, 5))
    r = arm("sgd", "sgd", "adapt", [dict(lr=l) for l in sgrid])
    init = r["risk"][0].mean(0); ok = np.isfinite(r["risk"][-1].mean(0)) & (r["risk"][-1].mean(0) < 0.9 * init)
    le = float(np.asarray(sgrid)[ok].max()); res["lr_edge"] = le
    agrid = list(np.logspace(-4, -0.5, 8))
    arm("adam", "adam", "adapt", [dict(lr=l, b1=0.9, b2=0.99, eps=1e-8) for l in agrid])
    rl = arm("laprop", "laprop", "adapt", [dict(lr=l, b1=0.9, b2=0.99, eps=1e-8) for l in agrid])
    lb = float(np.asarray(agrid)[int(np.argmin(fin(rl["risk"])))]); res["lap_best_lr"] = lb
    arm("oracle", "dana", "oracle", [dict(lr=0.25 * le, c=float(c), kappa=k, delta=4.0, cap=1.0, fac=0.25)
                                     for k in (0.25, 0.5, 1.0) for c in np.logspace(-3, 0.5, 6)])
    drows = [dict(lr=f * le, s=s_, delta=4.0, cap=1.0, fac=f, pt=pt) for pt in (0.0, 1.0) for f in (0.125, 0.25, 0.5) for s_ in (0.0625, 0.25, 1.0)]
    cfg = dict(m=args.m, p=args.p, b=args.b_est)
    arm("shadow", "dana_shadow", "shadow", drows)
    arm("slq", "dana_slq", "slq", drows, slq=cfg)
    lrows = [dict(lr=lrm * lb, lrm=lrm, s=s_, f=0.5, b2=0.99, eps=1e-8, delta=4.0, cap=1.0, bk=1.0, C=1e9, pt=pt)
             for pt in (0.0, 1.0) for lrm in (0.3, 1.0) for s_ in (0.25, 1.0)]
    arm("lpd2", "lpd2", "shadow", lrows)
    arm("lpd2_slq", "lpd2_slq", "slq", lrows, slq=cfg)
    R = len([x for x in trainer.refresh_schedule(args.nmax)])
    cost = dict(steps=args.nmax, B=B, train_samples=args.nmax * B, shadow_gnvp_samples=args.nmax * B,
                lpd2_gnvp_samples=2 * args.nmax * B, slq_refreshes=R, slq_gnvp_samples=R * args.m * args.p * args.b_est)
    log(f"   {tag} cost (sample-GNVPs): shadow {cost['shadow_gnvp_samples']:.3g} | v2 shadow+power {cost['lpd2_gnvp_samples']:.3g} | "
        f"SLQ {cost['slq_gnvp_samples']:.3g} ({R} refreshes) | training samples {cost['train_samples']:.3g}")
    flat = dict(n=ev, lr_edge=le, lap_best_lr=lb, cost=json.dumps(cost), time=json.dumps(tim), meta=json.dumps(dict(task=task["name"], B=B, seeds=S, **cfg)))
    for k, v in res.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}_{kk}"] = vv
        elif isinstance(v, list):
            flat[k] = json.dumps(v)
    np.savez_compressed(path, **flat)
    log(f"done {tag} ({time.time()-t0:.0f}s)")
