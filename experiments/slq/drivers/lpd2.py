"""LaProp-DANA v2 (optim 'lpd2'): consistent z-coordinates, stability-tracked step g2_n = min(lr, f*2/lam_hat),
shadow momentum with the saturating-batch law.  Compared against tuned LaProp and oracle LaProp-DANA already on disk
(<task>_B<B>_adam.npz / _lpd.npz).  Output runs/<e2|e3>/<task>_B<B>_lpd2.npz
usage: python drivers/lpd2.py --task 2layer --alpha 1.0 --B 4 64 1024 --nmax 50000
       python drivers/lpd2.py --task modarith --B 16 --nmax 30000
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana import tasks, trainer
from adana.sim_ls import eval_points

KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s", "f", "C", "bk")
ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["nrf", "2layer", "modarith"])
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--B", type=int, nargs="+", required=True)
ap.add_argument("--nmax", type=int, required=True)
ap.add_argument("--seeds", type=int, default=0)
ap.add_argument("--grid", default="main", choices=["main", "his", "g"], help="his: larger s and lower lr ceiling")
ap.add_argument("--quick", action="store_true", help="smoke test: 3 configs, 1 seed")
args = ap.parse_args()

if args.task == "nrf":
    task = tasks.nonlinear_rf(v=2048 if args.alpha == 1.0 else 4096, d=1024, alpha=args.alpha, beta=0.7 if args.alpha == 1.0 else 0.5,
                              act="relu", n_eval=16384)
    sub, S, nev = "e2", args.seeds or 3, 50
elif args.task == "2layer":
    task = tasks.two_layer(v=512, m=256, alpha=args.alpha, teacher_width=512, n_eval=16384)
    sub, S, nev = "e2", args.seeds or 3, 50
else:
    task = tasks.modarith(p=97, T=32, zipf=1.2, eta_sig=0.7)
    sub, S, nev = "e3", args.seeds or 2, 40
root = os.path.expanduser("~/dana-exp")
out_dir = os.path.join(root, "runs", sub)
logf = open(os.path.join(out_dir, "log.txt"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

best = lambda r: float(np.nanmin(np.where(np.isfinite(r[-1].mean(0)), r[-1].mean(0), np.inf)))
for B in args.B:
    tag = f"{task['name']}_B{B}_lpd2" + ({"main": "", "his": "s", "g": "g"}[args.grid]) + ("_quick" if args.quick else "")
    path = os.path.join(out_dir, tag + ".npz")
    if os.path.exists(path):
        log(f"skip {tag}"); continue
    za = np.load(os.path.join(out_dir, f"{task['name']}_B{B}_adam.npz"))
    fin = za["lap_risk"][-1].mean(0); fin = np.where(np.isfinite(fin), fin, np.inf)
    lb = float(np.asarray(za["lap_grid"])[int(np.argmin(fin))])
    base = dict(b2=0.99, eps=1e-8, delta=4.0, cap=1.0, bk=1.0)
    rows = []
    for lrm in (1.0, 1e4):                            # lr ceiling: LaProp best lr, or none (fully adaptive)
        for f in (0.25, 0.5):                         # fraction of the stability limit
            for s_, C in ((0.125, 1e9), (0.03, 1e9), (0.125, 2.0), (0.0, 1.0)):   # s=0: no momentum (step schedule only)
                rows.append(dict(base, lr=lrm * lb, lrm=lrm, f=f, s=s_, C=C))
    if args.grid == "his":
        rows = [dict(base, lr=lrm * lb, lrm=lrm, f=f, s=s_, C=1e9)
                for lrm, f, s_ in [(1.0, 0.5, 0.25), (1.0, 0.5, 0.5), (1.0, 0.5, 1.0), (1.0, 0.25, 0.5),
                                   (0.3, 0.5, 0.125), (0.3, 0.5, 0.5), (0.3, 0.5, 1.0), (0.3, 0.5, 0.0)]]
    if args.grid == "g":      # guarded shadow (v2.1): frozen default + backoff ablation + lower ceiling
        rows = [dict(base, lr=lrm * lb, lrm=lrm, f=0.5, s=s_, C=1e9, bk=bk)
                for lrm, s_, bk in [(1.0, 0.25, 1.0), (1.0, 0.25, 0.5), (1.0, 0.125, 1.0), (1.0, 0.125, 0.5),
                                    (0.3, 0.25, 1.0), (0.3, 0.25, 0.5), (0.3, 0.5, 1.0), (0.3, 0.125, 0.5)]]
    if args.quick:
        rows = [r for r in rows if r["f"] == 0.5]
    ev = eval_points(args.nmax, nev)
    t0 = time.time()
    log(f"start {tag}: LaProp best lr {lb:.3g} (final {float(np.min(fin)):.4g}), {len(rows)} configs")
    r = trainer.run(task, "lpd2", "shadow", rows, KEYS, S=1 if args.quick else S, B=B, n_max=args.nmax, eval_pts=ev, log=None)
    np.savez_compressed(path, n=ev, risk=r["risk"], g3=r["g3"], nhb=r["nhb"], g2e=r["g2e"], nres=r.get("nres", np.zeros(1)), hp=json.dumps(rows), lap_best_lr=lb,
                        lap_best=float(np.min(fin)), meta=json.dumps(dict(task=task["name"], B=B, nmax=args.nmax, seeds=S, phase="lpd2")))
    fr = r["risk"][-1].mean(0)
    for i, row in enumerate(rows):
        log(f"   lrm={row['lrm']:g} f={row['f']} s={row['s']} C={row['C']}: final {fr[i]:.4g}  g2e/lr_lap {float(np.mean(r['g2e'][-1][..., i]))/lb:.3g}"
            f"  g3/g2 {float(np.mean(r['g3'][-1][:, i]))/max(float(np.mean(r['g2e'][-1][..., i])), 1e-30):.3g}"
            f"  resets {np.asarray(r['nres'][-1][..., i]).tolist() if 'nres' in r else '-'}")
    log(f"{tag}: best final risk  LaProp {float(np.min(fin)):.4e} | LaProp-DANA v2 {best(r['risk']):.4e} ({time.time()-t0:.0f}s)")
