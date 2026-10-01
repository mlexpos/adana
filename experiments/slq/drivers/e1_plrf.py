"""E1: PLRF — adaptive-DANA (buffer energy) vs tuned SGD vs oracle DANA grid vs theory rule (true N_eff).
One vmapped simulation per (instance, B). Output: runs/e1/<tag>_B<B>.npz
usage: python drivers/e1_plrf.py --alpha 1.0 --beta 0.7 --d 4096 --v 8192 --B 2 64 1024 32768 --nmax 1000000
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana.plrf import plrf_reduce
from adana.sim_ls import make_configs, simulate
from adana import theory

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--beta", type=float, required=True)
ap.add_argument("--d", type=int, default=4096)
ap.add_argument("--v", type=int, default=None)
ap.add_argument("--B", type=int, nargs="+", required=True)
ap.add_argument("--nmax", type=int, default=1_000_000)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--delta", type=float, default=4.0)
ap.add_argument("--inst_seed", type=int, default=0)
ap.add_argument("--out", default="runs/e1")
ap.add_argument("--tag", default="")
ap.add_argument("--arms", default="main", choices=["main", "shadow", "sat", "hig2", "lowg2"])
args = ap.parse_args()
v = args.v or 2 * args.d
root = os.path.expanduser("~/dana-exp")
os.makedirs(os.path.join(root, args.out), exist_ok=True)
logf = open(os.path.join(root, args.out, "log.txt"), "a")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); logf.write(line + "\n"); logf.flush()

t0 = time.time()
inst = plrf_reduce(v, args.d, args.alpha, args.beta, seed=args.inst_seed, cache_dir=os.path.join(root, "cache"))
lam = inst["lam"]; trK = float(lam.sum())
log(f"instance a={args.alpha} b={args.beta} d={args.d} v={v}: trK={trK:.3f} lam1={lam[0]:.3f} P*={inst['Pstar']:.3e} P0={inst['P0']:.3e} ({time.time()-t0:.0f}s)")
k3 = 1.0 / (2 * args.alpha)
kappas = sorted(set([0.0, round(max(0.0, k3 - 0.2), 3), round(k3, 3), round(min(1.0, k3 + 0.2), 3), 1.0]))
cgrid = np.logspace(-4, 1.5, 12)

for B in args.B:
    tag = f"{args.tag}a{args.alpha}_b{args.beta}_d{args.d}_v{v}_B{B}"
    path = os.path.join(root, args.out, tag + ".npz")
    if os.path.exists(path):
        log(f"skip {tag} (exists)"); continue
    gm = theory.g2_max(lam, B)
    rows = []
    if args.arms == "shadow":      # follow-up arms, same keys as the main run (common random numbers)
        for s in range(args.seeds):
            for f in (1 / 4, 1 / 2):
                for sc in (0.125, 0.25, 0.5, 1.0):
                    rows.append(dict(g2=f * gm, mode="shadow_batch", s=sc, delta=args.delta, seed=s, fac=f, grp="shadow"))
                    rows.append(dict(g2=f * gm, mode="shadow_exact", s=sc, delta=args.delta, seed=s, fac=f, grp="shadow_exact"))
                for sc in (0.125, 0.25, 0.5):
                    rows.append(dict(g2=f * gm, mode="adapt_mono", s=sc, delta=args.delta, seed=s, fac=f, grp="adapt_mono"))
    if args.arms == "sat":         # saturating-batch rule: s_eff = min(s, C/B)  (gamma3/gamma2 = min(sB, C)/N_eff)
        for s in range(args.seeds):
            for f in (1 / 4, 1 / 2):
                for C in (0.5, 1.0, 2.0, 4.0):
                    se = min(0.25, C / B)
                    rows.append(dict(g2=f * gm, mode="shadow_batch", s=se, delta=args.delta, seed=s, fac=f, grp=f"sat_shadow_C{C:g}"))
                    rows.append(dict(g2=f * gm, mode="adapt", s=se, delta=args.delta, seed=s, fac=f, grp=f"sat_buffer_C{C:g}"))
                    rows.append(dict(g2=f * gm, mode="adapt_mono", s=se, delta=args.delta, seed=s, fac=f, grp=f"sat_mono_C{C:g}"))
    if args.arms == "hig2":        # DANA arms at SGD-like step sizes (fair comparison below the high-d line)
        for s in range(args.seeds):
            for f in (0.7, 0.85):
                for sc in (0.004, 0.016, 0.0625, 0.25):
                    rows.append(dict(g2=f * gm, mode="shadow_batch", s=sc, delta=args.delta, seed=s, fac=f, grp="shadow"))
                    rows.append(dict(g2=f * gm, mode="adapt", s=sc, delta=args.delta, seed=s, fac=f, grp="adapt"))
                for kp in (0.5, 1.0):
                    for c in np.logspace(-4, 0, 5):
                        rows.append(dict(g2=f * gm, mode="oracle", c=c, kappa=kp, delta=args.delta, seed=s, fac=f, grp="oracle"))
    if args.arms == "lowg2":       # variance-dominated regime: extend SGD below the grid edge; DANA arms at small g2
        for s in range(args.seeds):
            for f in (1 / 32, 1 / 64, 1 / 128, 1 / 256):
                rows.append(dict(g2=f * gm, mode="sgd", delta=args.delta, seed=s, fac=f, grp="sgd"))
            for f in (1 / 16, 1 / 32, 1 / 64):
                for sc in (0.004, 0.016, 0.0625):
                    rows.append(dict(g2=f * gm, mode="shadow_batch", s=sc, delta=args.delta, seed=s, fac=f, grp="shadow"))
                    rows.append(dict(g2=f * gm, mode="adapt", s=sc, delta=args.delta, seed=s, fac=f, grp="adapt"))
                for kp in (0.5, 1.0):
                    for c in np.logspace(-4, 0, 5):
                        rows.append(dict(g2=f * gm, mode="oracle", c=c, kappa=kp, delta=args.delta, seed=s, fac=f, grp="oracle"))
    for s in (range(args.seeds) if args.arms == "main" else []):
        for f in (1 / 16, 1 / 8, 1 / 4, 1 / 2, 0.7, 0.85):
            rows.append(dict(g2=f * gm, mode="sgd", delta=args.delta, seed=s, fac=f, grp="sgd"))
        for f in (1 / 4, 1 / 2):
            for sc in (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0):
                rows.append(dict(g2=f * gm, mode="adapt", s=sc, delta=args.delta, seed=s, fac=f, grp="adapt"))
            for sc in (0.25, 0.5, 1.0):
                rows.append(dict(g2=f * gm, mode="theory", s=sc, delta=args.delta, seed=s, fac=f, grp="theory"))
            for kp in kappas:
                for c in cgrid:
                    rows.append(dict(g2=f * gm, mode="oracle", c=c, kappa=kp, delta=args.delta, seed=s, fac=f, grp="oracle"))
    hp = make_configs(rows)
    G = len(rows)
    log(f"start {tag}: G={G} configs, g2max={gm:.4f}, nmax={args.nmax}")
    t1 = time.time()
    out = simulate(inst, hp, B, args.nmax, key=1000 + B, n_pts=90, log=None)
    dt = time.time() - t1
    meta = dict(alpha=args.alpha, beta=args.beta, d=args.d, v=v, B=B, g2max=gm, trK=trK, lam1=float(lam[0]),
                Pstar=inst["Pstar"], P0=inst["P0"], delta=args.delta, nmax=args.nmax, seeds=args.seeds, runtime=dt)
    np.savez_compressed(path, **out, **{f"hp_{k}": v_ for k, v_ in hp.items()},
                        fac=np.array([r["fac"] for r in rows]), grp=np.array([r["grp"] for r in rows]),
                        lam=lam, meta=json.dumps(meta))
    fin = out["E"][-1]
    for grp in sorted(set(r["grp"] for r in rows)):
        m = np.array([r["grp"] == grp for r in rows]) & out["alive"][-1]
        log(f"   {grp:7s}: best final excess {np.min(fin[m]) if m.any() else np.nan:.3e}  (alive {m.sum()})")
    log(f"done {tag} in {dt:.0f}s")
