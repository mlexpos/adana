"""Distributing the momentum budget across tensors (SGD-DANA, gamma_3 from the SLQ quadrature, per-tensor N_T):
alloc 0 = global (budget proportional to N_T; same gamma_3 for every tensor), 1 = equal split s/L,
2 = water-filling a_T = min{1, nu G_T/N_T} with G_T = EMA <g1_T, g2_T> (split-batch mean-gradient energy),
3 = water-filling with G_T = D^2 (||y_T||^2 - (N_T/B) Temp/(2 g2)) (buffer signal, dynamics-filtered noise removed).
Budget: sum_T a_T N_T = s B.  gamma_2 fixed = fac * lr_edge (tuned SGD edge from the existing runs).
usage: python drivers/alloc.py --task modarith --B 16 --nmax 30000 | --task mlp --B 8 --nmax 30000
Output runs/<e3|slq>/<task>_B<B>_alloc.npz
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana import tasks, trainer
from adana.sim_ls import eval_points

KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s", "f", "C", "bk", "pt", "tau", "alloc")
ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["modarith", "mlp"], required=True)
ap.add_argument("--B", type=int, nargs="+", required=True)
ap.add_argument("--nmax", type=int, required=True)
args = ap.parse_args()
root = os.path.expanduser("~/dana-exp")
if args.task == "modarith":
    task = tasks.modarith(p=97, T=32, zipf=1.2, eta_sig=0.7); sub, S, nev = "e3", 2, 40
    base = lambda B: os.path.join(root, "runs", "e3", f"{task['name']}_B{B}_sgd.npz")
    FACS = (0.125, 0.25)
else:
    task = tasks.deep_mlp(v=256, width=128, depth=3, alpha=1.0, teacher_width=256, teacher_depth=3, n_eval=16384)
    sub, S, nev = "slq", 3, 50
    base = lambda B: os.path.join(root, "runs", "slq", f"{task['name']}_B{B}.npz")
    FACS = (0.125, 0.25)
out_dir = os.path.join(root, "runs", sub); os.makedirs(out_dir, exist_ok=True)
logf = open(os.path.join(out_dir, "log.txt"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
NAMES = {0: "global", 1: "equal", 2: "wf-split", 3: "wf-buffer"}
for B in args.B:
    tag = f"{task['name']}_B{B}_alloc"
    path = os.path.join(out_dir, tag + ".npz")
    if os.path.exists(path):
        log(f"skip {tag}"); continue
    le = float(np.load(base(B))["lr_edge"])
    rows = [dict(lr=f * le, fac=f, s=s_, delta=4.0, cap=1.0, alloc=float(a)) for a in (0, 1, 2, 3) for f in FACS for s_ in (0.0625, 0.25, 1.0)]
    cfg = dict(adaptive=True, p=4, m_max=128, chunk=16, eps=0.05, b0=64 if args.task == "modarith" else 256,
               b_max=4096 if args.task == "modarith" else 8192, refresh0=True, map_configs=args.task == "modarith")
    ev = eval_points(args.nmax, nev)
    t0 = time.time()
    log(f"start {tag}: lr_edge {le:.3g}, {len(rows)} configs")
    r = trainer.run(task, "dana_slq", "slq", rows, KEYS, S=S, B=B, n_max=args.nmax, eval_pts=ev, log=None, slq=cfg)
    fin = r["risk"][-1].mean(0)
    np.savez_compressed(path, n=ev, risk=r["risk"], g3=r["g3"], nhb=r["nhb"], eG=r["eG"], hp=json.dumps(rows), lr_edge=le,
                        info=json.dumps([dict(n=x["n"], m=x["m"], b=x["b"]) for x in r["refresh_info"]]),
                        meta=json.dumps(dict(task=task["name"], B=B, seeds=S, nmax=args.nmax, **cfg)))
    for a in (0, 1, 2, 3):
        idx = [i for i, h in enumerate(rows) if h["alloc"] == a]
        cells = " ".join(f"(f{rows[i]['fac']:g},s{rows[i]['s']:g}) {fin[i]:.4g}" for i in idx)
        log(f"   {tag} {NAMES[a]:9s}: best {np.nanmin(fin[idx]):.4e} | {cells}")
    log(f"done {tag} ({time.time()-t0:.0f}s)")
