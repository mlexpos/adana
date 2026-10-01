"""Closed-form quantities from dana_stability_v3.tex (exact discrete model, gamma1 = 1)."""
import numpy as np


def kappa(lam, g2, g3, D):
    """Per-mode homogenized noise mass kappa(lambda) of v3 eq. (4.1); +inf outside the Jury region."""
    lam = np.asarray(lam, dtype=np.float64)
    detM = (1 - D) * (1 - g2 * lam)
    J = (2 - D) * (2 - g2 * lam) - g3 * lam
    N = g2 ** 2 * lam * (1 - D) * (2 - D) + g2 * D * (2 - D) + g2 * g3 * lam * (1 - D) + g3 * (2 - D)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = lam * N / ((1 - detM) * J)
    return np.where((J > 0) & (1 - detM > 0), k, np.inf)


def k_inf(lam, g2, g3, D, B):
    """Exact kernel mass for constant hyperparameters (Theorem 4.2 of v3)."""
    k = kappa(lam, g2, g3, D) / B
    if np.any(~np.isfinite(k)) or np.any(k >= 1):
        return np.inf
    return float(np.sum(k / (1 - k)))


def sgd_mass(lam, g, B):
    den = 2 * B - (B + 1) * g * np.asarray(lam)
    if np.any(den <= 0):
        return np.inf
    return float(np.sum(g * lam / den))


def g2_max(lam, B, tol=1e-10):
    """Largest stable constant SGD step at batch B: root of sum g lam/(2B-(B+1) g lam) = 1."""
    lo, hi = 0.0, 2 * B / ((B + 1) * float(np.max(lam)))
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if sgd_mass(lam, m, B) < 1:
            lo = m
        else:
            hi = m
        if hi - lo < tol * hi:
            break
    return lo


def neff(lam, g2, D):
    lam = np.asarray(lam)
    return float(np.sum(g2 * lam / (g2 * lam + D)))


def g3_edge(lam, g2, D, B, target=1.0):
    """Largest constant gamma3 with k_inf < target (bisection)."""
    lo, hi = 0.0, float((2 - D) * (2 - g2 * np.max(lam)) / np.max(lam))
    if k_inf(lam, g2, 0.0, D, B) >= target:
        return 0.0
    for _ in range(100):
        m = 0.5 * (lo + hi)
        if k_inf(lam, g2, m, D, B) < target:
            lo = m
        else:
            hi = m
    return lo
