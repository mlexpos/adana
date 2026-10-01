"""Exact expected-risk dynamics of scheduled DANA on the PLRF (Gaussian least squares), per eigenmode.

For Gaussian features the minibatch gradient noise has diagonal covariance in the eigenbasis,
    q_j = (1/B) (lam_j^2 E r_j^2 + lam_j (sum_k lam_k E r_k^2 + P*)),
so the per-mode second moments A_j = E[(r_j, y_j)(r_j, y_j)^T] close exactly:
    A_j' = M_j A_j M_j^T + c c^T q_j,  M_j = [[1-(g2+g3) lam_j, -g3 (1-D)], [lam_j, 1-D]],  c = (-(g2+g3), 1),
with g2, g3 the scheduled steps (whole update x sigma(n), warmup 2% -> cosine to 0.1).  B = inf switches the noise off.
Purpose: separate noise (variance) from deterministic transient effects of a gamma_3 law below / above the line.
usage: python drivers/exact_moments.py --alpha 0.4 --B 1024
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana.plrf import plrf_reduce

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, default=0.4)
ap.add_argument("--beta", type=float, default=0.7)
ap.add_argument("--d", type=int, default=4096)
ap.add_argument("--B", type=int, default=1024)
ap.add_argument("--nmax", type=int, default=30000)
args = ap.parse_args()
inst = plrf_reduce(2 * args.d, args.d, args.alpha, args.beta, seed=0,
                   cache_dir=os.path.join(os.path.dirname(__file__), "..", "cache"))
lam = np.maximum(inst["lam"], 0.0); r0 = inst["r0"]; Ps = inst["Pstar"]
N = args.nmax; wu = max(0.02 * N, 1.0); delta = 4.0


def sigma(n):
    return (n + 1) / wu if n < wu else 0.1 + 0.45 * (1 + np.cos(np.pi * min(max((n - wu) / (N - wu), 0.0), 1.0)))


def run(cfgs, B):
    """cfgs: list of (name, g2, law) with law(n, D, g2) -> g3/g2 ratio (unscheduled). Returns final excess risk and traces."""
    G = len(cfgs)
    g2 = np.array([c[1] for c in cfgs])[:, None]
    Arr = np.tile(r0 ** 2, (G, 1)); Ary = np.zeros_like(Arr); Ayy = np.zeros_like(Arr)
    out = {c[0]: [] for c in cfgs}
    for n in range(N):
        D = delta / (delta + n); s = sigma(n)
        Ecur = (lam * Arr).sum(1); Gcur = (lam ** 2 * Arr).sum(1)
        ratio = np.array([c[2](n, D, c[1], Ecur[i], Gcur[i]) for i, c in enumerate(cfgs)])[:, None]
        a2 = s * g2; a3 = s * ratio * g2
        risk = (lam * Arr).sum(1, keepdims=True)
        q = 0.0 if not np.isfinite(B) else (lam ** 2 * Arr + lam * (risk + Ps)) / B
        m11 = 1 - (a2 + a3) * lam; m12 = -a3 * (1 - D); m21 = lam; m22 = 1 - D; c1 = -(a2 + a3)
        nrr = m11 ** 2 * Arr + 2 * m11 * m12 * Ary + m12 ** 2 * Ayy + c1 ** 2 * q
        nry = m11 * m21 * Arr + (m11 * m22 + m12 * m21) * Ary + m12 * m22 * Ayy + c1 * q
        nyy = m21 ** 2 * Arr + 2 * m21 * m22 * Ary + m22 ** 2 * Ayy + q
        Arr, Ary, Ayy = nrr, nry, nyy
        if n in (100, 1000, 3000, 10000, 20000, N - 1):
            rk = (lam * Arr).sum(1)
            for i, c in enumerate(cfgs):
                out[c[0]].append(rk[i])
    return out


def neff(D, g2):
    return np.sum(g2 * lam / (g2 * lam + D))


rule = lambda C: (lambda n, D, g2, E=0, Gs=0: min(0.25 * min(args.B, (C / 0.25) if C else np.inf) / neff(D, g2), 1.0))
oracle = lambda c, k: (lambda n, D, g2, E=0, Gs=0: min(c * (1.0 + n) ** (-k), 1.0))
# signal-fraction budget: F target = s * rho, rho = E / (E + P*)  (oracle knowledge of E and P* -- tests the principle only)
snr = lambda k: (lambda n, D, g2, E=0, Gs=0: min(0.25 * args.B * min(1.0, k * E / (E + Ps)) / neff(D, g2), 1.0))
# measurable proxy: B / B_noise, B_noise = trC / |grad|^2 = trK (E + P*) / sum lam^2 r^2  (split-batch estimable)
trK = lam.sum()
gns = lambda k: (lambda n, D, g2, E=0, Gs=0: min(0.25 * args.B * min(1.0, k * args.B * Gs / (trK * (E + Ps))) / neff(D, g2), 1.0))
if args.alpha == 0.4:
    cfgs = [("SGD g2=.5", 0.5, lambda n, D, g2, E=0, Gs=0: 0.0),
            ("rule nocap g2=2^-8", 2 ** -8, rule(0)), ("rule nocap g2=2^-4", 2 ** -4, rule(0)),
            ("rule C=1 g2=.25", 0.25, rule(1.0)),
            ("fixed c=256 k=1 g2=2^-7", 2 ** -7, oracle(256, 1.0)),
            ("Nesterov g2=2^-9", 2 ** -9, oracle(1, 0.0)),
            ("rule*rho g2=.25", 0.25, snr(1.0)), ("rule*4rho g2=.25", 0.25, snr(4.0)), ("rule*16rho g2=.25", 0.25, snr(16.0)),
            ("rule*4rho g2=2^-3", 2 ** -3, snr(4.0))] + [(f"rule*gns k={k:g} g2={g:g}", g, gns(k)) for k in (1.0, 4.0, 16.0) for g in (0.25, 0.125)]
else:
    cfgs = [("SGD g2=1", 1.0, lambda n, D, g2, E=0, Gs=0: 0.0),
            ("rule nocap g2=1", 1.0, rule(0)), ("rule C=4 g2=1", 1.0, rule(4.0)),
            ("fixed c=1024 k=.75 g2=1", 1.0, oracle(1024, 0.75)), ("Nesterov g2=1", 1.0, oracle(1, 0.0)),
            ("rule*4rho g2=1", 1.0, snr(4.0))] + [(f"rule*gns k={k:g} g2=1", 1.0, gns(k)) for k in (1.0, 4.0, 16.0)]
t0 = time.time()
for B in ((args.B,) if os.environ.get('NOINF') else (args.B, np.inf)):
    res = run(cfgs, B)
    print(f"alpha={args.alpha} B={B}: excess risk at n=100,1k,3k,10k,20k,{N} ({time.time()-t0:.0f}s)")
    for k, v in res.items():
        print(f"   {k:28s} " + " ".join(f"{x:.2e}" for x in v))
