"""E2: nonlinear PLRF-like panel. Phase 1: SGD lr sweep (also finds the largest stable SGD lr, lr_edge).
Phase 2: adaptive-DANA (s grid) and oracle DANA (c, kappa grid) at g2 = fac * lr_edge, fac in {1/8, 1/4, 1/2}.
Output runs/e2/<task>_B<B>.npz
usage: python drivers/e2.py --task nrf --alpha 1.0 --beta 0.7 --B 4 64 1024 --nmax 100000
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import jax
from adana import tasks, trainer
from adana.sim_ls import eval_points

KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s")
ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["nrf", "nrf_tanh", "2layer"])
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--beta", type=float, default=0.7)
ap.add_argument("--d", type=int, default=1024)
ap.add_argument("--v", type=int, default=2048)
ap.add_argument("--B", type=int, nargs="+", required=True)
ap.add_argument("--nmax", type=int, default=100000)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--delta", type=float, default=4.0)
ap.add_argument("--out", default="runs/e2")
ap.add_argument("--phase", default="sgd", choices=["sgd", "adam", "shadow", "lpd", "satsh"])
args = ap.parse_args()
root = os.path.expanduser("~/dana-exp"); os.makedirs(os.path.join(root, args.out), exist_ok=True)
logf = open(os.path.join(root, args.out, "log.txt"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

if args.task == "nrf":
    task = tasks.nonlinear_rf(v=args.v, d=args.d, alpha=args.alpha, beta=args.beta, act="relu", n_eval=16384)
    lr_grid = 2.0 ** np.arange(-4, 5)
elif args.task == "nrf_tanh":
    task = tasks.nonlinear_rf(v=args.v, d=args.d, alpha=args.alpha, beta=args.beta, act="tanh", n_eval=16384)
    lr_grid = 2.0 ** np.arange(-4, 5)
else:
    task = tasks.two_layer(v=args.v, m=args.d, alpha=args.alpha, teacher_width=args.v, n_eval=16384)
    lr_grid = 2.0 ** np.arange(-3, 7)

def run_adam_phase(B):
    tag = f"{task['name']}_B{B}_adam"
    path = os.path.join(root, args.out, tag + ".npz")
    if os.path.exists(path):
        log(f"skip {tag}"); return
    ev = eval_points(args.nmax, 50)
    grid = list(np.logspace(-4, -0.5, 8))
    t0 = time.time()
    adam = trainer.run(task, "adam", "adapt", [dict(lr=l, b1=0.9, b2=0.99, eps=1e-8) for l in grid], KEYS, S=args.seeds, B=B, n_max=args.nmax, eval_pts=ev, log=None)
    lap = trainer.run(task, "laprop", "adapt", [dict(lr=l, b1=0.9, b2=0.99, eps=1e-8) for l in grid], KEYS, S=args.seeds, B=B, n_max=args.nmax, eval_pts=ev, log=None)
    fin = lap["risk"][-1].mean(0); ok = np.isfinite(fin) & (fin < 10 * lap["risk"][0].mean(0))
    le = float(np.asarray(grid)[ok].max())
    np.savez_compressed(path, n=ev, adam_risk=adam["risk"], adam_grid=grid, lap_risk=lap["risk"], lap_grid=grid, lr_edge=le,
                        meta=json.dumps(dict(task=task["name"], B=B, nmax=args.nmax, seeds=args.seeds, phase="adam")))
    best = lambda r: np.nanmin(np.where(np.isfinite(r[-1].mean(0)), r[-1].mean(0), np.inf))
    log(f"{tag}: best final risk  Adam {best(adam['risk']):.4e} | LaProp {best(lap['risk']):.4e} ({time.time()-t0:.0f}s)")


def run_shadow_phase(B):
    tag = f"{task['name']}_B{B}_shadow"
    path = os.path.join(root, args.out, tag + ".npz")
    if os.path.exists(path):
        log(f"skip {tag}"); return
    base = os.path.join(root, args.out, f"{task['name']}_B{B}.npz")
    le = float(np.load(base)["lr_edge"]) if os.path.exists(base) else None
    if le is None:
        log(f"no base run for {tag}"); return
    ev = eval_points(args.nmax, 50)
    rows = [dict(lr=f * le, s=s_, delta=args.delta, cap=1.0, fac=f) for f in (0.125, 0.25, 0.5) for s_ in (0.016, 0.0625, 0.25, 1.0)]
    t0 = time.time()
    sh = trainer.run(task, "dana_shadow", "shadow", rows, KEYS, S=args.seeds, B=B, n_max=args.nmax, eval_pts=ev, log=None)
    shm = trainer.run(task, "dana_shadow", "shadow_mono", rows, KEYS, S=args.seeds, B=B, n_max=args.nmax, eval_pts=ev, log=None)
    np.savez_compressed(path, n=ev, sh_risk=sh["risk"], sh_g3=sh["g3"], sh_nhb=sh["nhb"], sh_hp=json.dumps(rows),
                        shm_risk=shm["risk"], shm_g3=shm["g3"], shm_hp=json.dumps(rows),
                        meta=json.dumps(dict(task=task["name"], B=B, nmax=args.nmax, seeds=args.seeds, phase="shadow")))
    best = lambda r: np.nanmin(np.where(np.isfinite(r[-1].mean(0)), r[-1].mean(0), np.inf))
    log(f"{tag}: best final risk  shadow {best(sh['risk']):.4e} | shadow-mono {best(shm['risk']):.4e} ({time.time()-t0:.0f}s)")


def run_lpd_phase(B):
    tag = f"{task['name']}_B{B}_lpd"
    path = os.path.join(root, args.out, tag + ".npz")
    base = os.path.join(root, args.out, f"{task['name']}_B{B}_adam.npz")
    if os.path.exists(path) or not os.path.exists(base):
        log(f"skip {tag}"); return
    zb = np.load(base)
    fin = zb["lap_risk"][-1].mean(0); fin = np.where(np.isfinite(fin), fin, np.inf)
    lb = float(np.asarray(zb["lap_grid"])[int(np.argmin(fin))])
    ev = eval_points(args.nmax, 50)
    rows = [dict(lr=f * lb, s=s_, b2=0.99, eps=1e-8, delta=args.delta, cap=1.0, fac=f) for f in (1.0, 0.3, 0.1) for s_ in (0.016, 0.0625, 0.25)]
    t0 = time.time()
    lpd = trainer.run(task, "laprop_dana", "adapt_global", rows, KEYS, S=args.seeds, B=B, n_max=args.nmax, eval_pts=ev, log=None)
    orows = [dict(lr=f * lb, c=float(c), kappa=k, b2=0.99, eps=1e-8, delta=args.delta, cap=1.0, fac=f)
             for f in (1.0, 0.3) for k in (0.25, 0.5, 1.0) for c in np.logspace(-3, 0.5, 5)]
    lpo = trainer.run(task, "laprop_dana", "oracle", orows, KEYS, S=args.seeds, B=B, n_max=args.nmax, eval_pts=ev, log=None)
    np.savez_compressed(path, n=ev, lap_best_lr=lb, lpd_risk=lpd["risk"], lpd_g3=lpd["g3"], lpd_nhb=lpd["nhb"], lpd_hp=json.dumps(rows),
                        lpo_risk=lpo["risk"], lpo_hp=json.dumps(orows),
                        meta=json.dumps(dict(task=task["name"], B=B, nmax=args.nmax, seeds=args.seeds, phase="lpd")))
    best = lambda r: np.nanmin(np.where(np.isfinite(r[-1].mean(0)), r[-1].mean(0), np.inf))
    log(f"{tag}: best final risk  LaProp {float(np.min(fin)):.4e} (lr {lb:.3g}) | adaptive LaProp-DANA {best(lpd['risk']):.4e} | oracle LaProp-DANA {best(lpo['risk']):.4e} ({time.time()-t0:.0f}s)")


def run_satsh_phase(B):
    tag = f"{task['name']}_B{B}_satsh"
    path = os.path.join(root, args.out, tag + ".npz")
    base = os.path.join(root, args.out, f"{task['name']}_B{B}.npz")
    if os.path.exists(path) or not os.path.exists(base):
        log(f"skip {tag}"); return
    le = float(np.load(base)["lr_edge"])
    ev = eval_points(args.nmax, 50)
    rows = [dict(lr=f * le, s=min(s0, C / B), delta=args.delta, cap=1.0, fac=f, C=C, s0=s0)
            for f in (0.125, 0.25, 0.5) for s0 in (0.0625, 0.25) for C in (0.5, 1.0, 2.0)]
    t0 = time.time()
    sh = trainer.run(task, "dana_shadow", "shadow", rows, KEYS, S=args.seeds, B=B, n_max=args.nmax, eval_pts=ev, log=None)
    shm = trainer.run(task, "dana_shadow", "shadow_mono", rows, KEYS, S=args.seeds, B=B, n_max=args.nmax, eval_pts=ev, log=None)
    np.savez_compressed(path, n=ev, ssh_risk=sh["risk"], ssh_g3=sh["g3"], ssh_hp=json.dumps(rows),
                        sshm_risk=shm["risk"], sshm_hp=json.dumps(rows),
                        meta=json.dumps(dict(task=task["name"], B=B, nmax=args.nmax, seeds=args.seeds, phase="satsh")))
    best = lambda r: np.nanmin(np.where(np.isfinite(r[-1].mean(0)), r[-1].mean(0), np.inf))
    log(f"{tag}: best final risk  sat-shadow {best(sh['risk']):.4e} | sat-shadow-mono {best(shm['risk']):.4e} ({time.time()-t0:.0f}s)")


for B in args.B:
    if args.phase == "satsh":
        run_satsh_phase(B); continue
    if args.phase == "lpd":
        run_lpd_phase(B); continue
    if args.phase == "adam":
        run_adam_phase(B); continue
    if args.phase == "shadow":
        run_shadow_phase(B); continue
    tag = f"{task['name']}_B{B}"
    path = os.path.join(root, args.out, tag + ".npz")
    if os.path.exists(path):
        log(f"skip {tag}"); continue
    ev = eval_points(args.nmax, 50)
    t0 = time.time()
    sgd_rows = [dict(lr=float(l)) for l in lr_grid]
    sgd = trainer.run(task, "sgd", "adapt", sgd_rows, KEYS, S=args.seeds, B=B, n_max=args.nmax, eval_pts=ev, log=log)
    fin = sgd["risk"][-1].mean(0)
    ok = np.isfinite(fin) & (fin < 10 * sgd["risk"][0].mean(0))
    lr_edge = float(lr_grid[ok].max())
    log(f"{tag}: SGD done ({time.time()-t0:.0f}s); lr_edge={lr_edge}; final risk by lr: {np.round(fin, 6)}")
    dana_rows, ora_rows = [], []
    for fac in (0.125, 0.25, 0.5):
        for s in (0.004, 0.016, 0.0625, 0.25, 1.0):
            dana_rows.append(dict(lr=fac * lr_edge, s=s, delta=args.delta, cap=1.0, fac=fac))
        for kp in (0.25, 0.5, 0.75, 1.0):
            for c in np.logspace(-3, 1, 7):
                ora_rows.append(dict(lr=fac * lr_edge, c=float(c), kappa=kp, delta=args.delta, cap=1.0, fac=fac))
    t1 = time.time()
    ad = trainer.run(task, "dana", "adapt", dana_rows, KEYS, S=args.seeds, B=B, n_max=args.nmax, eval_pts=ev, log=log)
    log(f"{tag}: adaptive done ({time.time()-t1:.0f}s)")
    t2 = time.time()
    orc = trainer.run(task, "dana", "oracle", ora_rows, KEYS, S=args.seeds, B=B, n_max=args.nmax, eval_pts=ev, log=log)
    log(f"{tag}: oracle done ({time.time()-t2:.0f}s)")
    np.savez_compressed(path, n=ev, sgd_risk=sgd["risk"], sgd_lr=lr_grid, lr_edge=lr_edge,
                        ad_risk=ad["risk"], ad_g3=ad["g3"], ad_nhb=ad["nhb"],
                        ad_hp=json.dumps(dana_rows), or_risk=orc["risk"], or_hp=json.dumps(ora_rows),
                        meta=json.dumps(dict(task=task["name"], B=B, nmax=args.nmax, seeds=args.seeds, delta=args.delta)))
    best = lambda r: np.nanmin(np.where(np.isfinite(r[-1].mean(0)), r[-1].mean(0), np.inf))
    log(f"{tag}: best final risk  SGD {best(sgd['risk']):.4e} | adaptive {best(ad['risk']):.4e} | oracle {best(orc['risk']):.4e}")
