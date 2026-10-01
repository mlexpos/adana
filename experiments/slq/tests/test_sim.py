"""Validation of the eigenbasis simulator.
1. chi^2 sampling trick == explicit Gaussian minibatch gradients (mean, second moment).
2. estimator expectations: E trC_hat = tr Cov(single-sample grad) ; E trF_hat = tr K.
3. reduced simulator == explicit PLRF (v x d, actual samples) for SGD and DANA (mean risk curves).
4. exact stability threshold (v3 Theorem 4.2) vs simulation for constant hyperparameters.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import jax, jax.numpy as jnp
from adana.sim_ls import _proj_sample, _chi2, simulate, make_configs
from adana.plrf import plrf_reduce, gaussian_ls
from adana import theory

rng = np.random.default_rng(0)


def test_trick():
    d, B, sigma = 4, 3, 0.7
    lam = np.array([1.0, 0.5, 0.2, 0.05]); r = np.array([0.3, -0.2, 0.5, 1.0])
    w = np.concatenate([np.sqrt(lam) * r, [-sigma]])
    N = 400000
    Z = rng.standard_normal((N, B, d + 1))
    ex = np.einsum("nbi,nbj,j->ni", Z, Z, w)[:, :d] * np.sqrt(lam) / B
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    chi = _chi2(k1, float(B), (N,))
    xi = jax.random.normal(k2, (N, d + 1))
    tr = np.asarray(_proj_sample(jnp.broadcast_to(jnp.asarray(w, jnp.float32), (N, d + 1)), chi, xi))[:, :d] * np.sqrt(lam) / B
    m_ex, m_tr = ex.mean(0), tr.mean(0)
    C_ex, C_tr = np.cov(ex.T), np.cov(tr.T)
    print("[trick] mean explicit", np.round(m_ex, 4), " trick", np.round(m_tr, 4), " exact", np.round(lam * r, 4))
    print("[trick] max |Cov diff| / max|Cov| =", np.abs(C_ex - C_tr).max() / np.abs(C_ex).max())
    # theory: Cov(single) = P K + K r r^T K ; batch: /B
    P = np.sum(lam * r ** 2) + sigma ** 2
    C_th = (P * np.diag(lam) + np.outer(lam * r, lam * r)) / B
    print("[trick] max |Cov trick - theory| / max =", np.abs(C_tr - C_th).max() / np.abs(C_th).max())


def test_reduced_vs_explicit():
    v, d, alpha, beta = 96, 48, 1.0, 0.6
    inst = plrf_reduce(v, d, alpha, beta, seed=3)
    # explicit PLRF simulation
    r2 = np.random.default_rng(3)
    W = r2.standard_normal((v, d)) / np.sqrt(d)
    j = np.arange(1, v + 1.0); sq = j ** (-alpha); b = j ** (-beta)
    B = 4; n = 3000; seeds = 400
    g2 = 0.25 * theory.g2_max(inst["lam"], B); delta = 4.0; c = 0.3; kap = 1 / (2 * alpha)
    def run_explicit(mode):
        th = np.zeros((seeds, d)); y = np.zeros((seeds, d)); out = []
        for t in range(n):
            X = r2.standard_normal((seeds, B, v)) * sq
            res = np.einsum("sbv,sv->sb", X, th @ W.T - b)
            g = np.einsum("sbv,sb->sv", X, res) @ W / B
            Dn = delta / (delta + t)
            y = (1 - Dn) * y + g
            g3 = 0.0 if mode == "sgd" else g2 * min(c * (1 + t) ** (-kap), 1.0)
            th = th - g2 * g - g3 * y
            if (t + 1) in (10, 100, 1000, 3000):
                P = np.sum(sq ** 2 * (th @ W.T - b) ** 2, -1)
                out.append(P.mean())
        return np.array(out)
    for mode in ("sgd", "oracle"):
        t0 = time.time(); ex = run_explicit(mode)
        hp = make_configs([dict(g2=g2, mode=mode, c=c, kappa=kap, delta=delta, seed=s) for s in range(seeds)])
        red = simulate(inst, hp, B, n, key=5, ev=[0, 10, 100, 1000, 3000])
        idx = [np.where(red["n"] == k)[0][0] for k in (10, 100, 1000, 3000)]
        Pred = (red["E"][idx] + red["Pstar"]).mean(1)
        print(f"[explicit vs reduced, {mode}] n=10,100,1000,3000: explicit {np.round(ex, 5)}  reduced {np.round(Pred, 5)}  ratio {np.round(Pred / ex, 3)}  ({time.time()-t0:.0f}s)")


def test_constant_delta():
    """Constant Delta is Delta_n = delta/(delta+n) with delta -> use tiny n range? Instead compare k_inf to growth
    using SGD (gamma3 = 0) where the threshold is exact: sum g lam/(2B-(B+1) g lam) < 1."""
    d = 200; lam = np.arange(1, d + 1.0) ** -1.0
    inst = gaussian_ls(lam, np.zeros(d), 1.0)
    for B in (2, 8):
        gm = theory.g2_max(lam, B)
        rows = [dict(g2=f * gm, mode="sgd", seed=s) for f in (0.8, 0.95, 1.05, 1.2) for s in range(4)]
        hp = make_configs(rows)
        out = simulate(inst, hp, B, 20000, key=7, n_pts=20, div_factor=1e6)
        Efin = out["E"][-1].reshape(4, 4).mean(1); alive = out["alive"][-1].reshape(4, 4).mean(1)
        print(f"[SGD threshold B={B}] g2/g2max = 0.8,0.95,1.05,1.2 -> final excess {np.round(Efin, 3)}, alive frac {alive}")


def test_buffer_identity():
    d = 400; lam = np.arange(1, d + 1.0) ** -2.0
    inst = gaussian_ls(lam, np.zeros(d), 1.0)
    B = 8; g2 = 0.25 * theory.g2_max(lam, B)
    # SGD with buffer tracked; Delta_n = delta/(delta+n) varies, compare NhB with true N_eff(Delta_n)/B
    hp = make_configs([dict(g2=g2, mode="sgd", delta=4.0, seed=s) for s in range(16)])
    out = simulate(inst, hp, B, 200000, key=11, n_pts=12)
    for i in range(0, len(out["n"]), 3):
        print(f"[buffer] n={out['n'][i]:7d}  Nhat/B={np.median(out['NhB'][i]):.3f}  N_eff/B={out['NtrueB'][i][0]:.3f}  T={np.median(out['T'][i]):.3f}")


if __name__ == "__main__":
    print(jax.devices())
    which = sys.argv[1:] or ["trick", "constant_delta", "buffer_identity", "reduced_vs_explicit"]
    for w in which:
        globals()["test_" + w]()
