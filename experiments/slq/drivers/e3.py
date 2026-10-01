"""E3: small transformer on smooth modular arithmetic.
--phase sgd : SGD lr sweep (-> lr_edge), adaptive-DANA (global and per-leaf) at fac*lr_edge, oracle-DANA grid.
--phase adam: Adam sweep, LaProp sweep (-> lr_edge_lp), adaptive LaProp-DANA at fac*lr_edge_lp, oracle LaProp-DANA grid.
Output runs/e3/<task>_B<B>_<phase>.npz
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana import tasks, trainer
from adana.sim_ls import eval_points

KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s")
VARIANTS = {
    "v1": dict(p=97, T=32, zipf=1.2, eta_sig=0.7),
    "v2": dict(p=113, T=64, zipf=1.05, eta_sig=1.0),
    "v3": dict(p=97, T=32, zipf=0.9, eta_sig=0.7),
}
ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="v1")
ap.add_argument("--phase", choices=["sgd", "adam", "shadow", "lpd", "satsh"], required=True)
ap.add_argument("--B", type=int, nargs="+", required=True)
ap.add_argument("--nmax", type=int, default=30000)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--delta", type=float, default=4.0)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--dmodel", type=int, default=128)
ap.add_argument("--out", default="runs/e3")
args = ap.parse_args()
root = os.path.expanduser("~/dana-exp"); os.makedirs(os.path.join(root, args.out), exist_ok=True)
logf = open(os.path.join(root, args.out, "log.txt"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

task = tasks.modarith(d_model=args.dmodel, n_layers=args.layers, **VARIANTS[args.variant])
ev = eval_points(args.nmax, 40)
best = lambda r: float(np.nanmin(np.where(np.isfinite(r[-1].mean(0)), r[-1].mean(0), np.inf)))


def sweep(opt, rows, g3mode="adapt"):
    t0 = time.time()
    out = trainer.run(task, opt, g3mode, rows, KEYS, S=args.seeds, B=B, n_max=args.nmax, eval_pts=ev, log=None)
    log(f"   {opt}/{g3mode}: {len(rows)} configs, best final excess {best(out['risk']):.4f} ({time.time()-t0:.0f}s)")
    return out


def edge(risk, grid):
    fin = risk[-1].mean(0); init = risk[0].mean(0)
    ok = np.isfinite(fin) & (fin < 0.9 * init)
    return float(np.asarray(grid)[ok].max())


for B in args.B:
    tag = f"{task['name']}_B{B}_{args.phase}"
    path = os.path.join(root, args.out, tag + ".npz")
    if os.path.exists(path):
        log(f"skip {tag}"); continue
    log(f"start {tag} (Bayes {task['bayes']:.4f})")
    res = {}
    if args.phase == "sgd":
        grid = list(2.0 ** np.arange(-4, 4))
        res["sgd"] = sweep("sgd", [dict(lr=l) for l in grid]); res["sgd_grid"] = grid
        le = edge(res["sgd"]["risk"], grid); res["lr_edge"] = le
        log(f"   lr_edge={le}")
        rows = [dict(lr=f * le, s=s, delta=args.delta, cap=1.0, fac=f) for f in (0.125, 0.25, 0.5) for s in (0.125, 0.25, 0.5, 1.0)]
        res["adg"] = sweep("dana", rows, "adapt_global"); res["adg_hp"] = rows
        res["adl"] = sweep("dana", rows, "adapt"); res["adl_hp"] = rows
        orows = [dict(lr=0.25 * le, c=float(c), kappa=k, delta=args.delta, cap=1.0, fac=0.25)
                 for k in (0.25, 0.5, 1.0) for c in np.logspace(-3, 0.5, 6)]
        res["orc"] = sweep("dana", orows, "oracle"); res["orc_hp"] = orows
    elif args.phase == "lpd":
        zb = np.load(os.path.join(root, args.out, f"{task['name']}_B{B}_adam.npz"))
        fin = zb["lap_risk"][-1].mean(0); fin = np.where(np.isfinite(fin), fin, np.inf)
        lb = float(np.asarray(zb["lap_grid"])[int(np.argmin(fin))]); res["lap_best_lr"] = lb
        rows = [dict(lr=f * lb, s=s_, b2=0.99, eps=1e-8, delta=args.delta, cap=1.0, fac=f) for f in (1.0, 0.3, 0.1) for s_ in (0.016, 0.0625, 0.25)]
        res["lpd"] = sweep("laprop_dana", rows, "adapt_global"); res["lpd_hp"] = rows
        orows = [dict(lr=f * lb, c=float(c), kappa=k, b2=0.99, eps=1e-8, delta=args.delta, cap=1.0, fac=f)
                 for f in (1.0, 0.3) for k in (0.25, 0.5, 1.0) for c in np.logspace(-3, 0.5, 5)]
        res["lpo"] = sweep("laprop_dana", orows, "oracle"); res["lpo_hp"] = orows
    elif args.phase == "satsh":
        base = os.path.join(root, args.out, f"{task['name']}_B{B}_sgd.npz")
        le = float(np.load(base)["lr_edge"]); res["lr_edge"] = le
        rows = [dict(lr=f * le, s=min(s0, C / B), delta=args.delta, cap=1.0, fac=f, C=C, s0=s0)
                for f in (0.0625, 0.125, 0.25) for s0 in (0.03, 0.125) for C in (0.5, 1.0, 2.0)]
        res["ssh"] = sweep("dana_shadow", rows, "shadow"); res["ssh_hp"] = rows
        res["sshm"] = sweep("dana_shadow", rows, "shadow_mono"); res["sshm_hp"] = rows
    elif args.phase == "shadow":
        base = os.path.join(root, args.out, f"{task['name']}_B{B}_sgd.npz")
        le = float(np.load(base)["lr_edge"]); res["lr_edge"] = le
        rows = [dict(lr=f * le, s=s_, delta=args.delta, cap=1.0, fac=f) for f in (0.0625, 0.125, 0.25) for s_ in (0.03, 0.125, 0.5)]
        res["sh"] = sweep("dana_shadow", rows, "shadow"); res["sh_hp"] = rows
        res["shm"] = sweep("dana_shadow", rows, "shadow_mono"); res["shm_hp"] = rows
    else:
        grid = list(np.logspace(-4, -1.5, 8))
        res["adam"] = sweep("adam", [dict(lr=l, b1=0.9, b2=0.99, eps=1e-8) for l in grid]); res["adam_grid"] = grid
        res["lap"] = sweep("laprop", [dict(lr=l, b1=0.9, b2=0.99, eps=1e-8) for l in grid]); res["lap_grid"] = grid
        le = edge(res["lap"]["risk"], grid); res["lr_edge"] = le
        log(f"   laprop lr_edge={le}")
        # LaProp's step is lr*(1-b1)*sum b1^k ~ lr per unit u: the DANA outer loop uses g2 = fac * lr_edge
        pass   # LaProp-DANA arms: see --phase lpd
    flat = {}
    for k, v in res.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}_{kk}"] = vv
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            flat[k] = json.dumps(v)
        else:
            flat[k] = np.asarray(v)
    np.savez_compressed(path, n=ev, bayes=task["bayes"], meta=json.dumps(dict(task=task["name"], B=B, nmax=args.nmax,
                        seeds=args.seeds, phase=args.phase)), **flat)
    log(f"done {tag}")
