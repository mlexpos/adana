"""Power-law random features (PLRF) instances, reduced to their exact Gaussian least-squares form.

Conditionally on the feature matrix W, the PLRF with x_j = j^{-alpha} z_j, features W^T x and target <x, b>
is exactly Gaussian least squares:
    features  x~ = W^T x ~ N(0, K),  K = W^T D W,  D = diag(j^{-2 alpha}),
    target    y  = <x~, theta*> + eps,  eps independent of x~, Var(eps) = P* (the irreducible risk).
In the eigenbasis (lam_j, V_j) of K, the iterate error theta - theta* has coordinates r_j, and the risk is
    P(theta) = sum_j lam_j r_j^2 + P*.
This module builds W (float64, numpy), computes (lam, r0 = V^T(theta0 - theta*), P*) and caches them.
"""
import os
import numpy as np


def plrf_reduce(v, d, alpha, beta, seed=0, cache_dir=None):
    tag = f"plrf_v{v}_d{d}_a{alpha}_b{beta}_s{seed}.npz"
    if cache_dir is not None:
        path = os.path.join(cache_dir, tag)
        if os.path.exists(path):
            z = np.load(path)
            return dict(lam=z["lam"], r0=z["r0"], Pstar=float(z["Pstar"]), P0=float(z["P0"]),
                        v=v, d=d, alpha=alpha, beta=beta, seed=seed)
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((v, d)) / np.sqrt(d)
    j = np.arange(1, v + 1, dtype=np.float64)
    Dg = j ** (-2 * alpha)
    b = j ** (-beta)
    K = (W * Dg[:, None]).T @ W
    lam, V = np.linalg.eigh(K)
    lam, V = lam[::-1].copy(), V[:, ::-1].copy()
    rhs = W.T @ (Dg * b)
    th_star = V @ ((V.T @ rhs) / lam)
    Pstar = float(np.sum(Dg * (W @ th_star - b) ** 2))
    r0 = V.T @ (np.zeros(d) - th_star)          # theta0 = 0
    P0 = float(np.sum(Dg * b ** 2))              # risk at theta = 0
    out = dict(lam=lam, r0=r0, Pstar=Pstar, P0=P0, v=v, d=d, alpha=alpha, beta=beta, seed=seed)
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(os.path.join(cache_dir, tag), lam=lam, r0=r0, Pstar=Pstar, P0=P0)
    return out


def gaussian_ls(lam, r0, sigma2):
    """A generic Gaussian least-squares instance with covariance eigenvalues lam, initial error r0,
    label-noise variance sigma2 (so P* = sigma2)."""
    lam = np.asarray(lam, dtype=np.float64)
    return dict(lam=lam, r0=np.asarray(r0, dtype=np.float64), Pstar=float(sigma2),
                P0=float(np.sum(lam * np.asarray(r0) ** 2) + sigma2), d=len(lam))
