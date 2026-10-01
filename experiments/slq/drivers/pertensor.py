"""Per-tensor adaptive momentum: shadow rule (SGD-DANA, opt dana_shadow) and LaProp-DANA v2.1 (opt lpd2) with
pt=1: each parameter tensor has its own shadow energy and gamma_3, budget split s/L (L = number of tensors).
Global counterparts are already on disk (<task>_B<B>_shadow.npz / _lpd2g.npz, same seeds).  Output <task>_B<B>_pt.npz
usage: python drivers/pertensor.py --task modarith --B 16 --nmax 30000
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana import tasks, trainer
from adana.sim_ls import eval_points

KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s", "f", "C", "bk", "pt")
ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["2layer", "modarith"])
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--B", type=int, nargs="+", required=True)
ap.add_argument("--nmax", type=int, required=True)
args = ap.parse_args()
if args.task == "2layer":
    task = tasks.two_layer(v=512, m=256, alpha=args.alpha, teacher_width=512, n_eval=16384); sub, S, nev = "e2", 3, 50
    FACS = (0.125, 0.25, 0.5)          # as in the E2 global shadow phase
else:
    task = tasks.modarith(p=97, T=32, zipf=1.2, eta_sig=0.7); sub, S, nev = "e3", 2, 40
    FACS = (0.0625, 0.125, 0.25)       # as in the E3 global shadow phase
out_dir = os.path.join(os.path.expanduser("~/dana-exp"), "runs", sub)
logf = open(os.path.join(out_dir, "log.txt"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
best = lambda r: float(np.nanmin(np.where(np.isfinite(r[-1].mean(0)), r[-1].mean(0), np.inf)))

for B in args.B:
    tag = f"{task['name']}_B{B}_pt"
    path = os.path.join(out_dir, tag + ".npz")
    if os.path.exists(path):
        log(f"skip {tag}"); continue
    base_sgd = os.path.join(out_dir, f"{task['name']}_B{B}" + ("_sgd" if args.task == "modarith" else "") + ".npz")
    zs = np.load(base_sgd)
    le = float(zs["lr_edge"])
    za = np.load(os.path.join(out_dir, f"{task['name']}_B{B}_adam.npz"))
    fin = za["lap_risk"][-1].mean(0); fin = np.where(np.isfinite(fin), fin, np.inf)
    lb = float(np.asarray(za["lap_grid"])[int(np.argmin(fin))])
    ev = eval_points(args.nmax, nev)
    t0 = time.time()
    log(f"start {tag}: SGD lr_edge {le:.3g}, LaProp best lr {lb:.3g}")
    srows = [dict(lr=f * le, s=s_, delta=4.0, cap=1.0, fac=f, pt=1.0) for f in FACS for s_ in (0.125, 0.5, 2.0)]
    sh = trainer.run(task, "dana_shadow", "shadow", srows, KEYS, S=S, B=B, n_max=args.nmax, eval_pts=ev, log=None)
    shm = trainer.run(task, "dana_shadow", "shadow_mono", srows, KEYS, S=S, B=B, n_max=args.nmax, eval_pts=ev, log=None)
    base = dict(b2=0.99, eps=1e-8, delta=4.0, cap=1.0, bk=1.0, f=0.5, C=1e9, pt=1.0)
    lrows = [dict(base, lr=lrm * lb, lrm=lrm, s=s_) for lrm in (0.3, 1.0) for s_ in (0.25, 1.0, 4.0)]
    lp = trainer.run(task, "lpd2", "shadow", lrows, KEYS, S=S, B=B, n_max=args.nmax, eval_pts=ev, log=None)
    np.savez_compressed(path, n=ev, sh_risk=sh["risk"], sh_g3=sh["g3"], sh_hp=json.dumps(srows), shm_risk=shm["risk"], shm_g3=shm["g3"],
                        shm_hp=json.dumps(srows), lpt_risk=lp["risk"], lpt_g3=lp["g3"], lpt_g2e=lp["g2e"], lpt_hp=json.dumps(lrows),
                        meta=json.dumps(dict(task=task["name"], B=B, nmax=args.nmax, seeds=S, phase="pt")))
    for nm, r, rows in (("shadow-pt", sh, srows), ("shadow-pt-mono", shm, srows), ("lpd2-pt", lp, lrows)):
        fr = r["risk"][-1].mean(0)
        for i, row in enumerate(rows):
            log(f"   {nm} " + " ".join(f"{k}={row[k]:.3g}" for k in ("lr", "s") ) + (f" lrm={row['lrm']}" if "lrm" in row else f" fac={row['fac']}") + f": {fr[i]:.4g}")
    log(f"{tag}: best final risk  shadow-pt {best(sh['risk']):.4e} | shadow-pt-mono {best(shm['risk']):.4e} | LaProp-DANA v2.1-pt {best(lp['risk']):.4e} ({time.time()-t0:.0f}s)")
