"""Check of the adaptive SLQ estimator on PLRF spectra: Lanczos depth m stopped by the Gauss/Gauss-Radau bracket,
estimation batch b doubled until N(K_b)/b <= 1/2, deterministic-equivalent de-biasing.  For each target Delta:
m_used, b_used, bracket, raw and de-biased N_eff against the truth.  Output runs/slq/adaptive_<inst>.npz
usage: python drivers/slq_adaptive_check.py --alpha 1.0 --beta 0.7 [--v 8192]
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana.plrf import plrf_reduce
from adana import theory, slq

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--beta", type=float, required=True)
ap.add_argument("--d", type=int, default=4096)
ap.add_argument("--v", type=int, default=None)
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()
v = args.v or 2 * args.d
root = os.path.expanduser("~/dana-exp"); out_dir = os.path.join(root, "runs", "slq")
logf = open(os.path.join(out_dir, "log.txt"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
inst = plrf_reduce(v, args.d, args.alpha, args.beta, seed=0, cache_dir=os.path.join(root, "cache"))
lam = np.asarray(inst["lam"], np.float64); d = len(lam); itag = f"a{args.alpha}_b{args.beta}_d{args.d}_v{v}"
g2 = 0.25 * theory.g2_max(lam, 64)
Dg = np.logspace(-7, -1, 13)
Ntrue = np.array([np.sum(g2 * lam / (g2 * lam + D)) for D in Dg])


_cache = {}
def emp_factory(seed):
    def f(b):
        if (seed, b) not in _cache:
            _cache.clear()
            _cache[(seed, b)] = (np.random.default_rng(seed + b).standard_normal((b, d)) * np.sqrt(lam)).astype(np.float32)
        X = _cache[(seed, b)]
        return lambda u: (X.T @ (X @ u.astype(np.float32)) / b).astype(np.float64)
    return f


res = {k: np.zeros((2, args.reps, len(Dg))) for k in ("raw", "deb", "m", "b", "L", "U")}
t0 = time.time()
for e, kind in enumerate(("exact", "empirical")):
    for rep in range(args.reps):
        for i, D in enumerate(Dg):
            th, w, m, b, (L, U) = slq.adaptive_quadrature_np(
                emp_factory(1000 * rep), d, g2, D, eps=0.05, p=4, m_max=256, chunk=8, b0=256, b_max=16384,
                seed=100 * rep + i, exact_op=(lambda u: lam * u) if kind == "exact" else None)
            res["raw"][e, rep, i] = slq.fsum(th, w, g2, D)
            res["deb"][e, rep, i] = slq.neff_debiased(th, w, g2, D, b)
            res["m"][e, rep, i] = m; res["b"][e, rep, i] = b if np.isfinite(b) else 0
            res["L"][e, rep, i] = L; res["U"][e, rep, i] = U
        log(f"   adaptive {itag} {kind} rep {rep} ({time.time()-t0:.0f}s)")
# fixed-setting reference (m=64, b=1024, no de-biasing) for comparison
fixed = np.zeros((args.reps, len(Dg)))
for rep in range(args.reps):
    qn, qw = slq.quadrature_np(emp_factory(1000 * rep)(1024), d, m=64, probes=4, seed=rep)
    fixed[rep] = [slq.neff_from_quad(qn, qw, g2, D) for D in Dg]
np.savez_compressed(os.path.join(out_dir, f"adaptive_{itag}.npz"), Dg=Dg, Ntrue=Ntrue, g2=g2, fixed=fixed, **res)
for e, kind in enumerate(("exact", "empirical")):
    for i in range(0, len(Dg), 2):
        log(f"   {itag} {kind:9s} Delta={Dg[i]:.0e} N={Ntrue[i]:7.1f}: raw {np.median(res['raw'][e, :, i]) / Ntrue[i]:.3f} "
            f"deb {np.median(res['deb'][e, :, i]) / Ntrue[i]:.3f} | m {np.median(res['m'][e, :, i]):.0f} b {np.median(res['b'][e, :, i]):.0f} "
            f"| fixed(m64,b1024) {np.median(fixed[:, i]) / Ntrue[i]:.3f}")
log(f"adaptive check {itag} done ({time.time()-t0:.0f}s)")
