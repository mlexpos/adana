"""Scheduled comparison (warmup 2% -> cosine to 10% peak on the step / temperature; transformer clipped at global norm 1):
  --arm base : SGD, Adam, LaProp with the schedule, peak lr tuned on a grid
  --arm dana : auto SGD-DANA   gamma_2 = sigma(n) tau gamma2max_hat(F),   gamma_3 from N_eff;  alloc global / water-filling
  --arm lpd  : auto LaProp-DANA gamma_2 = sigma(n) tau gamma2max_hat(F_z), NO lr ceiling;       alloc global / water-filling
  --arm cmp_lpd   : LaProp-DANA-SLQ (sched_mode 1, k' = --kp), lr x s x alloc grid (--lrs, --svals, --allocs)
  --arm cmp_adana : SLQ-ADana (adana_slq; log-time moments, delta 8), same grid; alloc 3 water-filling, 4 water-filling v2
  --arm adana_base: plain ADana (kappa 0.85, gamma_3f in --g3f), lr grid
  spectral preconditioning of the hidden matrices (adana/spectral.py), LaProp order:
  --arm cmp_spd / cmp_soap_dana / cmp_muon_dana : Shampoo / SOAP-basis / msign preconditioning + DANA-SLQ momentum
                  (lr x s x alloc grid; --perheads 0,1 for cmp_spd)
  --arm soap / shampoo_m / muon : standard baselines (Adam in the Shampoo basis; Shampoo + heavy ball; Muon + Adam)
All N_eff / thresholds from the adaptive SLQ refresh.  Output runs/<e3|slq>/<task>_B<B>_sched_<arm>.npz
usage: python drivers/sched.py --task mlp --B 8 --nmax 30000 --arm base
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana import tasks, trainer
from adana.sim_ls import eval_points

KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s", "f", "C", "bk", "pt", "tau", "alloc", "T", "clip", "sched_mode", "gns", "gnsb",
        "sb", "seps", "rms", "mu", "perhead", "nh")
ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["modarith", "mlp", "plrf"], required=True)
ap.add_argument("--B", type=int, required=True)
ap.add_argument("--nmax", type=int, required=True)
ap.add_argument("--arm", choices=["base", "dana", "lpd", "dana_sweep", "lpd_sweep", "dana_oracle", "lpd_oracle", "dana_s", "lpd_s",
                                  "cmp_lpd", "cmp_adana", "adana_base", "cmp_spd", "cmp_soap_dana", "cmp_muon_dana",
                                  "soap", "shampoo_m", "muon"], required=True)
ap.add_argument("--perheads", type=str, default="0", help="cmp_spd: per-head output factor for q, k (0 and/or 1)")
ap.add_argument("--lrs", type=str, default="", help="cmp_*/adana_base: comma-separated peak lrs (default: the base Adam grid)")
ap.add_argument("--svals", type=str, default="0.25,0.5", help="cmp_*: budget fractions s")
ap.add_argument("--allocs", type=str, default="3,4", help="cmp_*: allocations (3 water-filling, 4 water-filling v2)")
ap.add_argument("--kp", type=float, default=32.0, help="cmp_*: signal-fraction constant k' (sequences)")
ap.add_argument("--g3f", type=str, default="1,2.5", help="adana_base: gamma_3f values")
ap.add_argument("--sched_mode", type=int, default=0, help="0: momentum ratio at the scheduled step; 1: ratio at the unscheduled "
                "step and the whole update scaled by sigma(n)")
ap.add_argument("--alpha", type=float, default=1.0, help="plrf only")
ap.add_argument("--beta", type=float, default=0.7, help="plrf only")
ap.add_argument("--d", type=int, default=4096, help="plrf only (v = 2d)")
ap.add_argument("--C", type=str, default="", help="dana_sweep/lpd_sweep: comma-separated saturation constants "
                "(s_eff = min(s, C/B)); replaces the alloc {0,3} axis by C (alloc 0); output suffix _C")
ap.add_argument("--gns", type=str, default="", help="dana_sweep/lpd_sweep: comma-separated signal-fraction constants k "
                "(budget x min{1, k B/B_noise}); replaces the alloc axis by k (alloc from --alloc); output suffix _gns")
ap.add_argument("--gnsb", type=str, default="", help="as --gns but batch-independent: multiplier min{1, k'/B_noise}; suffix _gnsb")
ap.add_argument("--alloc", type=float, default=0.0)
ap.add_argument("--tag", type=str, default="", help="extra output-name suffix (to split one sweep over several jobs)")
ap.add_argument("--lo", action="store_true", help="lpd_sweep only: extend the lr grid downward (logspace(-5.5,-3.5,5))")
args = ap.parse_args()
root = os.path.expanduser("~/dana-exp")
if args.task == "modarith":
    task = tasks.modarith(p=97, T=32, zipf=1.2, eta_sig=0.7); sub, S, nev, clip = "e3", 2, 40, 1.0
    agrid = list(np.logspace(-4, -1.5, 8))
elif args.task == "plrf":
    task = tasks.plrf(v=2 * args.d, d=args.d, alpha=args.alpha, beta=args.beta)
    sub, S, nev, clip = "slq", 3, 50, 0.0
    agrid = list(np.logspace(-5, -0.5, 10))
else:
    task = tasks.deep_mlp(v=256, width=128, depth=3, alpha=1.0, teacher_width=256, teacher_depth=3, n_eval=16384)
    sub, S, nev, clip = "slq", 3, 50, 0.0
    agrid = list(np.logspace(-4, -0.5, 8))
sgrid = 2.0 ** np.arange(-8, 5) if args.task == "plrf" else 2.0 ** np.arange(-4, 5)
out_dir = os.path.join(root, "runs", sub); os.makedirs(out_dir, exist_ok=True)
logf = open(os.path.join(out_dir, "log.txt"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
B, N = args.B, args.nmax
tag = f"{task['name']}_B{B}_sched_{args.arm}" + ("_m1" if args.sched_mode else "") + ("_beff" if args.task == "modarith" else "") + ("_lo" if args.lo else "") + ("_C" if args.C else "") + ("_gns" if args.gns else "") + ("_gnsb" if args.gnsb else "") + (f"_{args.tag}" if args.tag else "")
path = os.path.join(out_dir, tag + ".npz")
if os.path.exists(path):
    log(f"skip {tag}"); sys.exit()
if args.lo:
    agrid = list(np.logspace(-5.5, -3.5, 5))
ev = eval_points(N, nev)
if args.task == "modarith":   # sequence model: B_eff = B T_eff (online proxy), fixed b = 512 sequences, no rank-one de-biasing
    cfg = dict(adaptive=True, p=4, m_max=256, chunk=32, eps=0.05, b0=512, b_max=512, refresh0=True, map_configs=True,
               max_gap=2000, debias=False)
else:
    cfg = dict(adaptive=True, p=4, m_max=128, chunk=16, eps=0.05, b0=256, b_max=8192, refresh0=True, max_gap=1000)
common = dict(T=float(N), clip=clip, sched_mode=float(args.sched_mode))
t0 = time.time(); res = {}
def arm(name, opt, mode, rows, **kw):
    t = time.time()
    mr = 18 if args.task == "modarith" and "slq" not in kw else len(rows)      # transformer w/o SLQ: chunk the configs (memory)
    parts = [trainer.run(task, opt, mode, rows[i:i + mr], KEYS, S=S, B=B, n_max=N, eval_pts=ev, log=None, **kw)
             for i in range(0, len(rows), mr)]
    r = parts[0] if len(parts) == 1 else {k: (np.concatenate([np.asarray(p_[k]) for p_ in parts], axis=2)
                                             if np.ndim(parts[0][k]) >= 3 else parts[0][k]) for k in parts[0]}
    fin = r["risk"][-1].mean(0)
    res[name] = {k: v for k, v in r.items() if k != "refresh_info"}; res[name + "_hp"] = rows
    if "refresh_info" in r:
        res[name + "_info"] = [dict(n=x["n"], m=x["m"], b=x["b"], g2max=x.get("g2max", 0), teff=x.get("teff", 1)) for x in r["refresh_info"]]
    log(f"   {tag} {name:8s}: best {np.nanmin(np.where(np.isfinite(fin), fin, np.inf)):.4e} | "
        + " ".join(f"{fin[i]:.3g}" for i in range(len(rows))) + f" ({time.time()-t:.0f}s)")
if args.arm == "base":
    arm("sgd", "sgd", "adapt", [dict(common, lr=l) for l in sgrid])
    arm("adam", "adam", "adapt", [dict(common, lr=l, b1=0.9, b2=0.99, eps=1e-8) for l in agrid])
    arm("laprop", "laprop", "adapt", [dict(common, lr=l, b1=0.9, b2=0.99, eps=1e-8) for l in agrid])
elif args.arm in ("cmp_lpd", "cmp_adana", "adana_base"):
    lrs = [float(x) for x in args.lrs.split(",")] if args.lrs else agrid
    cm = dict(common, sched_mode=1.0, eps=1e-8, cap=1.0, tau=0.0)
    if args.arm == "adana_base":
        rows = [dict(cm, lr=l, delta=8.0, alloc=-1.0, c=float(c), kappa=0.85) for c in args.g3f.split(",") for l in lrs]
        arm(args.arm, "adana_slq", "slq", rows)
    else:
        grid = [dict(s=float(s_), alloc=float(a)) for a in args.allocs.split(",") for s_ in args.svals.split(",")]
        if args.arm == "cmp_lpd":
            rows = [dict(cm, lr=l, f=1e9, b2=0.99, delta=4.0, gnsb=args.kp, **x) for x in grid for l in lrs]
            arm(args.arm, "lpd2_slq", "slq", rows, slq=cfg)
        else:
            rows = [dict(cm, lr=l, delta=8.0, gnsb=args.kp, **x) for x in grid for l in lrs]
            arm(args.arm, "adana_slq", "slq", rows, slq=cfg)
elif args.arm in ("cmp_spd", "cmp_soap_dana", "cmp_muon_dana", "soap", "shampoo_m", "muon"):
    lrs = [float(x) for x in args.lrs.split(",")] if args.lrs else agrid
    sp = dict(common, sched_mode=1.0, eps=1e-8, b2=0.99, sb=0.95, seps=1e-6, rms=1.0, nh=4.0, cap=1.0, tau=0.0)
    if args.arm.startswith("cmp_"):
        opt = {"cmp_spd": "spd_slq", "cmp_soap_dana": "soap_dana_slq", "cmp_muon_dana": "muon_dana"}[args.arm]
        phs = [float(x) for x in args.perheads.split(",")] if args.arm == "cmp_spd" else [0.0]
        grid = [dict(s=float(s_), alloc=float(a), perhead=ph) for ph in phs for a in args.allocs.split(",") for s_ in args.svals.split(",")]
        rows = [dict(sp, lr=l, delta=4.0, gnsb=args.kp, **x) for x in grid for l in lrs]
        arm(args.arm, opt, "slq", rows, slq=cfg)
    else:
        extra = dict(b1=0.9, mu=0.95) if args.arm == "muon" else dict(b1=0.9)
        rows = [dict(sp, lr=l, **extra) for l in lrs]
        arm(args.arm, args.arm, "slq", rows)
elif args.arm == "dana":
    rows = [dict(common, lr=0.0, tau=t, s=0.25, delta=4.0, cap=1.0, alloc=a) for a in (0.0, 3.0) for t in (0.25, 0.5)]
    arm("dana", "dana_slq", "slq", rows, slq=cfg)
elif args.arm == "lpd":
    rows = [dict(common, lr=1e9, tau=t, s=0.25, f=0.5, b2=0.99, eps=1e-8, delta=4.0, cap=1.0, alloc=a)
            for a in (0.0, 3.0) for t in (0.125, 0.25, 0.5, 1.0)]
    arm("lpd", "lpd2_slq", "slq", rows, slq=cfg)
elif args.arm == "dana_sweep":      # gamma_2 swept like SGD (scheduled), gamma_3 from the SLQ rule
    ax = ([dict(alloc=0.0, C=float(c)) for c in args.C.split(",")] if args.C else
          [dict(alloc=args.alloc, gns=float(k)) for k in args.gns.split(",")] if args.gns else
          [dict(alloc=args.alloc, gnsb=float(k)) for k in args.gnsb.split(",")] if args.gnsb else [dict(alloc=a) for a in (0.0, 3.0)])
    rows = [dict(common, lr=l, tau=0.0, s=0.25, delta=4.0, cap=1.0, **x) for x in ax for l in sgrid]
    arm("dana_sweep", "dana_slq", "slq", rows, slq=cfg)
elif args.arm in ("dana_oracle", "lpd_oracle", "dana_s", "lpd_s"):
    # gamma_3 quality at the swept-optimal gamma_2 (best row of the dana/lpd sweep, water-filling), and gamma_2 x {1/2, 2}:
    #   *_oracle : fixed DANA-decaying schedule g3 = g2 min{c (1+n)^-kappa, 1} (whole update scaled by sigma(n)), (c, kappa) grid
    #   *_s      : the SLQ rule with budget s in {1/16 .. 2}
    lp = args.arm.startswith("lpd")
    sw = "lpd_sweep" if lp else "dana_sweep"
    sfx = "_beff" if args.task == "modarith" else ""
    best = (np.inf, None)
    for extra in ("", "_lo", "_m1", "_m1_lo"):
        f = os.path.join(out_dir, f"{task['name']}_B{B}_sched_{sw}{extra.replace('_lo', '')}{sfx}{'_lo' if '_lo' in extra else ''}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True); rows0 = json.loads(str(z[f"{sw}_hp"])); fin = z[f"{sw}_risk"][-1].mean(0)
        for r0, v in zip(rows0, fin):
            if np.isfinite(v) and v < best[0]:
                best = (v, r0)
    g2b = float(best[1]["lr"]); log(f"   {tag}: swept-optimal gamma_2 = {g2b:.4g} (risk {best[0]:.4g}, alloc {best[1]['alloc']:g})")
    g2s = [g2b, 4 * g2b] if args.task == "modarith" else [g2b / 2, g2b, 2 * g2b]
    lpx = dict(b2=0.99, eps=1e-8) if lp else {}
    if args.arm.endswith("_oracle"):
        cg = [(c, 0.0) for c in 2.0 ** np.arange(-10, 1, 2)] + [(c, k) for k in (0.25, 0.5, 0.75, 1.0) for c in 2.0 ** np.arange(-2, 13, 2)]
        rows = [dict(common, lr=g, delta=4.0, cap=1.0, c=c, kappa=k, **lpx) for g in g2s for c, k in cg]
        arm(args.arm, "laprop_dana" if lp else "dana", "oracle", rows)
    else:
        rows = [dict(common, lr=g, tau=0.0, s=s_, delta=4.0, cap=1.0, alloc=3.0, **(dict(lpx, f=1e9) if lp else {}))
                for g in g2s for s_ in 2.0 ** np.arange(-4, 2)]
        arm(args.arm, "lpd2_slq" if lp else "dana_slq", "slq", rows, slq=cfg)
else:                               # gamma_2 swept like LaProp (scheduled, no lambda_max cap), gamma_3 from the SLQ rule in z
    ax = ([dict(alloc=0.0, C=float(c)) for c in args.C.split(",")] if args.C else
          [dict(alloc=args.alloc, gns=float(k)) for k in args.gns.split(",")] if args.gns else
          [dict(alloc=args.alloc, gnsb=float(k)) for k in args.gnsb.split(",")] if args.gnsb else [dict(alloc=a) for a in (0.0, 3.0)])
    rows = [dict(common, lr=l, tau=0.0, s=0.25, f=1e9, b2=0.99, eps=1e-8, delta=4.0, cap=1.0, **x) for x in ax for l in agrid]
    arm("lpd_sweep", "lpd2_slq", "slq", rows, slq=cfg)
flat = dict(n=ev, meta=json.dumps(dict(task=task["name"], B=B, seeds=S, nmax=N, arm=args.arm, clip=clip, **cfg)))
for k, v in res.items():
    if isinstance(v, dict):
        for kk, vv in v.items():
            flat[f"{k}_{kk}"] = vv
    else:
        flat[k] = json.dumps(v)
np.savez_compressed(path, **flat)
log(f"done {tag} ({time.time()-t0:.0f}s)")
