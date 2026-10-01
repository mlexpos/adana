"""Stochastic Lanczos quadrature (SLQ) estimate of the effective dimension

    N_eff(Delta) = tr[ g2 F (g2 F + Delta)^{-1} ] = sum_i g2 lam_i / (g2 lam_i + Delta).

One Lanczos run of m steps from a probe z returns a Gauss quadrature (nodes theta_k = Ritz values of F,
weights w_k = |z|^2 q_{1k}^2) for the spectral measure  z^T 1{F in .} z.  Averaged over probes this is an unbiased-in-
expectation estimate of the empirical spectral measure of F, so ONE refresh gives N_eff(Delta) for EVERY Delta (and
every g2) at the cost of a scalar sum:   N_hat(Delta) = sum_k w_k g2 theta_k / (g2 theta_k + Delta).

Per-block (per-tensor) version: with z supported on all blocks, the block Hutchinson estimate
    z_T^T f(F) z ~= |z| sum_k f(theta_k) q_{1k} sum_j q_{jk} <z_T, v_j>
is unbiased for tr[f(F)]_{TT} (off-block terms vanish in expectation), so the block weights
    w_{T,k} = |z| q_{1k} sum_j q_{jk} c_{T,j},    c_{T,j} = <z_T, v_j>
need only the m scalars c_{T,j} accumulated during the Lanczos run (no stored Lanczos vectors).  sum_T w_{T,k} = w_k.
"""
import numpy as np
import jax
import jax.numpy as jnp


# ------------------------------------------------------------------ numpy (validation)
def lanczos_np(matvec, z, m, reorth=False):
    """Returns (alpha, beta) of the m-step Lanczos tridiagonal and the Lanczos vectors if reorth."""
    n = z.shape[0]
    V = np.zeros((m, n)) if reorth else None
    v = z / np.linalg.norm(z)
    v_prev = np.zeros(n)
    b = 0.0
    al, be = [], []
    for j in range(m):
        if reorth:
            V[j] = v
        w = matvec(v) - b * v_prev
        a = float(v @ w)
        w = w - a * v
        if reorth:
            w = w - V[:j + 1].T @ (V[:j + 1] @ w)
        b = float(np.linalg.norm(w))
        al.append(a); be.append(b)
        if b < 1e-12 * max(1.0, abs(a)):
            break
        v_prev, v = v, w / b
    return np.array(al), np.array(be[:-1])


def lanczos_full_np(matvec, z, m):
    """m Lanczos steps; returns alpha (m,), beta (m,) with beta[m-1] the NEXT off-diagonal (needed for Gauss-Radau)."""
    v = z / np.linalg.norm(z); vp = np.zeros_like(v); b = 0.0; al, be = [], []
    for _ in range(m):
        w = matvec(v) - b * vp; a = float(v @ w); w = w - a * v; b = float(np.linalg.norm(w))
        al.append(a); be.append(b)
        if b < 1e-14:
            break
        vp, v = v, w / b
    return np.array(al), np.array(be)


def gauss_radau(al, be, m, a=0.0):
    """Gauss rule (m nodes) and Gauss-Radau rule with a prescribed node at a (m+1 nodes) from the Lanczos coefficients.
    Returns (thG, qG2), (thR, qR2): nodes and squared first components (weights / |z|^2), batched over leading dims.
    For f with f^(2m) < 0 < f^(2m+1) on [a, inf) (e.g. f(x) = x/(x+c)), Gauss >= z^T f(A) z >= Radau."""
    T = _tridiag(al[..., :m], be[..., :m - 1])
    thG, QG = np.linalg.eigh(T)
    rhs = np.zeros(al.shape[:-1] + (m,)); rhs[..., -1] = be[..., m - 1] ** 2
    # prescribed node slightly below 0 (still a lower bound for a PSD operator): keeps T - aI nonsingular when the
    # (rank-deficient) operator has Ritz values at 0 or Lanczos broke down
    scale = np.maximum(np.abs(thG).max(-1), 1e-30)
    a = np.minimum(a, -1e-8 * scale)[..., None, None] if np.ndim(a) == 0 else a
    dlt = np.linalg.lstsq(T - a * np.eye(m), rhs, rcond=None)[0] if T.ndim == 2 else \
        np.linalg.solve(T - a * np.eye(m) + 1e-30 * np.eye(m), rhs[..., None])[..., 0]
    a = a[..., 0, 0]
    TR = np.zeros(al.shape[:-1] + (m + 1, m + 1)); TR[..., :m, :m] = T
    TR[..., m, m - 1] = TR[..., m - 1, m] = be[..., m - 1]; TR[..., m, m] = a + dlt[..., -1]
    thR, QR = np.linalg.eigh(TR)
    return (thG, QG[..., 0, :] ** 2), (thR, QR[..., 0, :] ** 2)


def _tridiag(al, be):
    m = al.shape[-1]
    T = np.zeros(al.shape[:-1] + (m, m))
    i = np.arange(m); T[..., i, i] = al
    T[..., i[:-1], i[:-1] + 1] = be; T[..., i[:-1] + 1, i[:-1]] = be
    return T


def fsum(th, w, g2, D):
    t = np.maximum(th, 0.0)
    return np.sum(w * g2 * t / (g2 * t + D), -1)


def neff_debiased(th, w, g2, D, b, iters=30, rmax=0.9):
    """Deterministic-equivalent de-biasing of an estimate made on an empirical operator from b samples:
    N(F_b; D') ~= N(F; D) with D = D'/(1 - N(F_b; D')/b); solve D' = D (1 - N(F_b; D')/b) by fixed point.
    b = inf (or <= 0) returns the raw estimate.  Valid while N(F_b)/b <~ 1/2 (else grow b)."""
    if not np.isfinite(b) or b <= 0:
        return fsum(th, w, g2, D)
    d = D
    for _ in range(iters):
        d = D * (1 - np.minimum(fsum(th, w, g2, d) / b, rmax))
    return fsum(th, w, g2, d)


def debias_shift(th, w, g2, D, b, iters=30, rmax=0.9):
    """Fixed point D' = D (1 - N(F_b; D')/b) of the deterministic-equivalent correction (D' = D if b is inf)."""
    if not np.isfinite(b) or b <= 0:
        return D
    d = D
    for _ in range(iters):
        d = D * (1 - np.minimum(fsum(th, w, g2, d) / b, rmax))
    return d


def adaptive_quadrature_np(op_factory, n, g2, D_check, s_B=0.0, eps=0.05, p=4, m_max=512, chunk=8,
                           b0=256, b_max=65536, r_grow=0.5, seed=0, exact_op=None):
    """Adaptive SLQ.  For the current estimation batch b: Lanczos in chunks; after each chunk compute the de-biasing
    shift D' (fixed point, from the current Gauss rule) and stop when the Gauss/Gauss-Radau bracket AT D' has relative
    gap < eps for every probe (or the Gauss upper bound is <= s_B: cap binds).  Then, if N(F_b; D')/b > r_grow, double b
    and redo.  op_factory(b) -> matvec of the empirical operator; exact_op (if given) is used instead (b = inf).
    Returns nodes, weights (Gauss rule at the stopping m), m_used, b_used, (L, U) at D'."""
    rng = np.random.default_rng(seed)
    b = np.inf if exact_op is not None else b0
    while True:
        mv = exact_op if exact_op is not None else op_factory(int(b))
        Z = rng.standard_normal((p, n))
        coefs = [lanczos_full_np(mv, z, m_max) for z in Z]
        mm = min(len(c[0]) for c in coefs)
        al = np.stack([c[0][:mm] for c in coefs]); be = np.stack([c[1][:mm] for c in coefs]); nz2 = np.sum(Z * Z, 1)
        m_used = mm
        for m in range(chunk, mm + 1, chunk):
            (thG, qG), (thR, qR) = gauss_radau(al, be, m)
            th = thG.reshape(-1); w = (nz2[:, None] * qG).reshape(-1) / p
            Dp = debias_shift(th, w, g2, D_check, b)
            U = nz2 * fsum(thG, qG, g2, Dp); L = nz2 * fsum(thR, qR, g2, Dp)
            if np.all((U - L) <= eps * U) or np.mean(U) <= s_B:
                m_used = m; break
        (thG, qG), (thR, qR) = gauss_radau(al, be, m_used)
        th = thG.reshape(-1); w = (nz2[:, None] * qG).reshape(-1) / p
        Dp = debias_shift(th, w, g2, D_check, b)
        if exact_op is not None or fsum(th, w, g2, Dp) / b <= r_grow or b >= b_max:
            U = float(np.mean(nz2 * fsum(thG, qG, g2, Dp))); L = float(np.mean(nz2 * fsum(thR, qR, g2, Dp)))
            return th, w, m_used, b, (L, U)
        b *= 2


def quadrature_np(matvec, n, m=32, probes=4, seed=0, reorth=False, dist="gauss"):
    """Nodes (P*m,) and weights (P*m,) (weights already divided by #probes) of the SLQ spectral measure.
    dist='gauss' (rotation invariant: use when the operator is given in its eigenbasis, where Rademacher probes
    would be exact and hide the probe variance) or 'rademacher'."""
    rng = np.random.default_rng(seed)
    nodes, weights = [], []
    for _ in range(probes):
        z = rng.standard_normal(n) if dist == "gauss" else rng.choice([-1.0, 1.0], size=n)
        al, be = lanczos_np(matvec, z, m, reorth)
        T = np.diag(al) + np.diag(be, 1) + np.diag(be, -1)
        th, Q = np.linalg.eigh(T)
        nodes.append(th); weights.append((z @ z) * Q[0] ** 2 / probes)
    return np.concatenate(nodes), np.concatenate(weights)


def neff_from_quad(nodes, weights, g2, Delta):
    th = np.maximum(nodes, 0.0)
    return float(np.sum(weights * g2 * th / (g2 * th + Delta)))


# ------------------------------------------------------------------ JAX pytree (networks)
def _tdot(a, b):
    return jax.tree.reduce(lambda x, y: x + y, jax.tree.map(lambda u, v: jnp.sum(u * v), a, b))


def lanczos_tree(matvec, z, m):
    """m-step Lanczos (no reorthogonalization) on a pytree operator. Returns alpha (m,), beta (m-1,),
    C (m, L): C[j, T] = <z_T, v_j> for each leaf T, and |z|."""
    zn = jnp.sqrt(_tdot(z, z))
    leaves_z = jax.tree.leaves(z)
    L = len(leaves_z)
    v0 = jax.tree.map(lambda x: x / zn, z)
    zero = jax.tree.map(jnp.zeros_like, z)

    def blockdots(v):
        return jnp.stack([jnp.sum(a * b) for a, b in zip(leaves_z, jax.tree.leaves(v))])

    def body(j, carry):
        v, v_prev, b_prev, al, be, C = carry
        C = C.at[j].set(blockdots(v))
        w = matvec(v)
        w = jax.tree.map(lambda x, p: x - b_prev * p, w, v_prev)
        a = _tdot(v, w)
        w = jax.tree.map(lambda x, u: x - a * u, w, v)
        b = jnp.sqrt(_tdot(w, w))
        ok = b > 1e-20
        v_new = jax.tree.map(lambda x: jnp.where(ok, x / jnp.where(ok, b, 1.0), 0.0), w)
        al = al.at[j].set(a)
        be = be.at[j].set(jnp.where(ok, b, 0.0))
        return v_new, v, b, al, be, C

    init = (v0, zero, jnp.zeros(()), jnp.zeros((m,)), jnp.zeros((m,)), jnp.zeros((m, L)))
    _, _, _, al, be, C = jax.lax.fori_loop(0, m, body, init)
    return al, be[:-1], C, zn


def quadrature_tree(matvec, like, key, m=32, probes=4):
    """Rademacher probes shaped like `like`. Returns nodes (P*m,), global weights (P*m,), block weights (P*m, L)."""
    leaves, tdef = jax.tree.flatten(like)

    def one(k):
        ks = jax.random.split(k, len(leaves))
        z = jax.tree.unflatten(tdef, [jax.random.rademacher(kk, x.shape, jnp.float32) for kk, x in zip(ks, leaves)])
        al, be, C, zn = lanczos_tree(matvec, z, m)
        T = jnp.diag(al) + jnp.diag(be, 1) + jnp.diag(be, -1)
        th, Q = jnp.linalg.eigh(T)
        w = zn ** 2 * Q[0] ** 2
        wT = zn * Q[0][:, None] * (Q.T @ C)          # (m, L): |z| q_{1k} sum_j q_{jk} C[j, T]
        return th, w, wT

    th, w, wT = jax.vmap(one)(jax.random.split(key, probes))
    return th.reshape(-1), w.reshape(-1) / probes, wT.reshape(-1, wT.shape[-1]) / probes


def neff_quad(nodes, weights, g2, Delta):
    """weights (Q,) or (Q, L) -> scalar or (L,)"""
    th = jnp.maximum(nodes, 0.0)
    f = g2 * th / (g2 * th + Delta)
    return jnp.tensordot(f, weights, axes=(0, 0))


# ------------------------------------------------------------------ JAX pytree, chunked (early stopping)
def lanczos_state_tree(z):
    zn = jnp.sqrt(_tdot(z, z))
    return dict(z=z, v=jax.tree.map(lambda x: x / zn, z), vp=jax.tree.map(jnp.zeros_like, z), b=jnp.zeros(()), zn=zn)


def lanczos_chunk_tree(matvec, ls, al, be, C, j0, k):
    """Advance k Lanczos steps from index j0 (same for all vmapped instances). al, be (m_max,), C (m_max, L)."""
    leaves_z = jax.tree.leaves(ls["z"])

    def body(j, carry):
        v, vp, b_prev, al, be, C = carry
        C = C.at[j].set(jnp.stack([jnp.sum(a * c) for a, c in zip(leaves_z, jax.tree.leaves(v))]))
        w = matvec(v)
        w = jax.tree.map(lambda x, p: x - b_prev * p, w, vp)
        a = _tdot(v, w)
        w = jax.tree.map(lambda x, u: x - a * u, w, v)
        b = jnp.sqrt(_tdot(w, w))
        ok = b > 1e-20
        v_new = jax.tree.map(lambda x: jnp.where(ok, x / jnp.where(ok, b, 1.0), 0.0), w)
        return v_new, v, jnp.where(ok, b, 0.0), al.at[j].set(a), be.at[j].set(jnp.where(ok, b, 0.0)), C

    v, vp, b, al, be, C = jax.lax.fori_loop(j0, j0 + k, body, (ls["v"], ls["vp"], ls["b"], al, be, C))
    return dict(ls, v=v, vp=vp, b=b), al, be, C


def quad_from_coeffs_np(al, be, C, zn, m):
    """Host: Gauss rule of order m from batched Lanczos coefficients. al/be (..., m_max), C (..., m_max, L), zn (...).
    Returns nodes (..., m), weights (..., m) [= zn^2 q1k^2], block weights (..., m, L)."""
    T = _tridiag(al[..., :m], be[..., :m - 1])
    th, Q = np.linalg.eigh(T)
    q1 = Q[..., 0, :]
    w = zn[..., None] ** 2 * q1 ** 2
    wT = zn[..., None, None] * q1[..., :, None] * np.einsum("...jk,...jl->...kl", Q, C[..., :m, :])
    return th, w, wT


# ------------------------------------------------------------------ stability threshold / trace from the quadrature
def ritz_quadrature_np(al, be, zn2, m, conv_tol=1e-6, gap_fac=100.0, ghost_tol=1e-6):
    """Gauss rule of order m with DEFLATED weights for the isolated top of the spectrum.  Walking down from the largest
    Ritz value, a node is an eigenvalue if its residual beta_m |Q[m-1,k]| is < conv_tol*theta_max and < gap/gap_fac, with
    gap the distance to the next distinct Ritz value (ghost copies from lost orthogonality, within ghost_tol*theta_max,
    are merged).  Each such eigenvalue gets total weight exactly 1 (the expectation of the random probe weight
    (z^T u)^2), its ghosts 0; the walk stops at the first node that fails.  Removes the Hutchinson variance carried by
    isolated top eigenvalues at no extra cost.  al, be (..., m_max), zn2 (...).
    Returns nodes (..., m), deflated weights, raw weights, number of deflated eigenvalues (...)."""
    T = _tridiag(al[..., :m], be[..., :m - 1])
    th, Q = np.linalg.eigh(T)
    raw = zn2[..., None] * Q[..., 0, :] ** 2
    res = np.abs(be[..., m - 1:m]) * np.abs(Q[..., m - 1, :])
    wd = raw.copy()
    lead = th.shape[:-1]
    ndef = np.zeros(lead, int)
    for idx in np.ndindex(*lead):
        t, r, w = th[idx], res[idx], wd[idx]
        tmax = max(t[-1], 1e-30)
        k = m - 1
        while k >= 0:
            grp = [k]                                   # ghost copies of the same eigenvalue
            while grp[-1] - 1 >= 0 and t[k] - t[grp[-1] - 1] < ghost_tol * tmax:
                grp.append(grp[-1] - 1)
            nxt = grp[-1] - 1
            gap = t[k] - t[nxt] if nxt >= 0 else np.inf
            rk = r[grp].max()
            if not (rk < conv_tol * tmax and rk < gap / gap_fac):
                break
            w[grp] = 0.0; w[k] = 1.0; ndef[idx] += 1
            k = nxt
    return th, wd, raw, ndef


def g2max_quad(th, w, B, iters=80):
    """Largest stable constant SGD step at batch B from a quadrature of F (exact threshold for Gaussian least squares,
    dana_stability_v3 Thm 4.2):  sum_k w_k g th_k / (2B - (B+1) g th_k) = 1.  Vectorized over leading dims of th, w
    (last axis = nodes, already probe-averaged)."""
    th = np.maximum(th, 0.0)
    B = np.broadcast_to(np.asarray(B, np.float64), th.shape[:-1])          # scalar or per-instance (e.g. B_eff)
    hi = 2 * B / ((B + 1) * np.maximum(th.max(-1), 1e-30)); lo = np.zeros_like(hi)
    Bx = B[..., None]
    for _ in range(iters):
        g = 0.5 * (lo + hi)
        val = np.sum(w * g[..., None] * th / np.maximum(2 * Bx - (Bx + 1) * g[..., None] * th, 1e-30), -1)
        lo = np.where(val < 1, g, lo); hi = np.where(val < 1, hi, g)
    return lo
