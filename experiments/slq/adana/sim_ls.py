"""Exact-in-distribution simulator of minibatch Gen-Mom-SGD on Gaussian least squares, in the eigenbasis.

Model: features x ~ N(0, diag(lam)), label noise variance sigma2.  Error coordinates r = theta - theta*.
Averaged minibatch gradient of B fresh samples:
    g = (1/B) Lam^{1/2} [Z_aug^T Z_aug w_aug]_{1:d},  w_aug = (Lam^{1/2} r, -sigma),  Z_aug iid N(0,1) (B x (d+1)).
Exact sampling in O(d): Z^T Z w = |w| (chi * w_hat + sqrt(chi) * P_perp xi),  chi ~ chi^2_B, xi ~ N(0, I).
Two independent halves give the split-batch noise estimator.  A third draw with model-sampled residuals
eps~ ~ N(0,1) gives the Fisher (= K) trace estimator.  Risk P = sum lam r^2 + sigma2 = |w_aug|^2.

Algorithm (gamma1 = 1):
    y_n = (1 - Delta_n) y_{n-1} + g_n,   theta_{n+1} = theta_n - g2 g_n - g3_n y_n,   Delta_n = delta/(delta + n).
gamma3 modes:  0 SGD (g3 = 0, buffer still tracked for diagnostics)
               1 oracle DANA  g3 = g2 min{c (1+n)^-kappa, cap}
               2 adaptive     g3 = g2 min{s B / Nhat, cap},  Nhat/B = 2 g2 <|y|^2> <trF>/<trC>
               3 theory       g3 = g2 min{s B / N_eff(Delta_n), cap} with the true spectrum.
               4 adapt_mono   running minimum of the adaptive rule (monotone non-increasing g3).
               5 shadow_exact shadow buffer: u <- u - g2 (K u + zeta), y' <- (1-D) y' + K u + zeta, zeta ~ model-sampled
                              noise (cov K/B); Nhat/B = 2 g2 <|y'|^2> (signal-free; T = 1 by construction).
               6 shadow_batch same with the minibatch curvature product (Z^T Z / B) applied to u (chi^2 trick).
               7 slq          g3 = g2 min{s B / Nhat(Delta_n), cap}, Nhat from a per-config SLQ quadrature (nodes, weights)
                              of K (exact or empirical K_b), passed as quad=(qn (G,Q), qw (G,Q)); see adana/slq.py.
EMA rate rho_n = min(1, max(Delta_n, 10/(n+1))).
"""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

MODES = {"sgd": 0, "oracle": 1, "adapt": 2, "theory": 3, "adapt_mono": 4, "shadow_exact": 5, "shadow_batch": 6, "slq": 7}
HP_KEYS = ("g2", "mode", "c", "kappa", "s", "delta", "cap", "seed")


def make_configs(rows):
    """rows: list of dicts with keys in HP_KEYS (mode given as str or int). Returns dict of arrays."""
    out = {k: [] for k in HP_KEYS}
    for r in rows:
        for k in HP_KEYS:
            val = r.get(k, {"c": 0.0, "kappa": 0.0, "s": 0.0, "delta": 4.0, "cap": 1.0, "seed": 0}.get(k))
            if k == "mode" and isinstance(val, str):
                val = MODES[val]
            out[k].append(val)
    return {k: np.asarray(v, dtype=np.int32 if k in ("mode", "seed") else np.float32) for k, v in out.items()}


def init_state(inst, G):
    d = len(inst["lam"])
    r = jnp.broadcast_to(jnp.asarray(inst["r0"], jnp.float32), (G, d))
    z = jnp.zeros((G, d), jnp.float32)
    zg = jnp.zeros((G,), jnp.float32)
    return dict(r=r, y=z, gprev=z, eY=zg, eC=zg, eF=zg, alive=jnp.ones((G,), bool),
                g3=zg, T=zg, NhB=zg, u=z, ys=z, eS=zg)


def _chi2(key, dof, shape):
    return 2.0 * jax.random.gamma(key, dof / 2.0, shape)


def _proj_sample(w, chi, xi):
    """|w| (chi w_hat + sqrt(chi) P_perp xi) for batched w (G, d+1), chi (G,), xi (G, d+1)."""
    nw = jnp.sqrt(jnp.maximum(jnp.sum(w * w, -1, keepdims=True), 1e-30))
    wh = w / nw
    perp = xi - wh * jnp.sum(wh * xi, -1, keepdims=True)
    return nw * (chi[:, None] * wh + jnp.sqrt(chi)[:, None] * perp)


@partial(jax.jit, static_argnames=("B", "S", "lag"))
def advance(state, hp, lam, sigma, n0, n1, key, P_div, B, S, lag, qn, qw):
    sl = jnp.sqrt(lam)
    d = lam.shape[0]
    seed = hp["seed"]

    def body(n, st):
        k = jax.random.fold_in(key, n)
        k1, k2, k3, k4, k5, k6 = jax.random.split(k, 6)
        w = jnp.concatenate([sl * st["r"], jnp.full((st["r"].shape[0], 1), -sigma)], -1)   # (G, d+1)
        if lag:   # B == 1: one draw, noise from lag difference
            chi = _chi2(k1, float(B), (S,))[seed]
            xi = jax.random.normal(k4, (S, d + 1))[seed]
            g = sl * _proj_sample(w, chi, xi)[:, :d] / B
            trC = 0.5 * B * jnp.sum((g - st["gprev"]) ** 2, -1)
        else:
            m = B // 2
            c1 = _chi2(k1, float(m), (S,))[seed]
            c2 = _chi2(k2, float(m), (S,))[seed]
            x1 = jax.random.normal(k4, (S, d + 1))[seed]
            x2 = jax.random.normal(k5, (S, d + 1))[seed]
            g1 = sl * _proj_sample(w, c1, x1)[:, :d] / m
            g2h = sl * _proj_sample(w, c2, x2)[:, :d] / m
            g = 0.5 * (g1 + g2h)
            trC = 0.25 * B * jnp.sum((g1 - g2h) ** 2, -1)
        c3 = _chi2(k3, float(B), (S,))
        x3 = jax.random.normal(k6, (S, d))
        trF = (c3 * jnp.sum(lam * x3 * x3, -1) / B)[seed]          # B |g~|^2 with model-sampled residuals

        g2 = hp["g2"]
        Dn = hp["delta"] / (hp["delta"] + n)
        rho = jnp.minimum(1.0, jnp.maximum(Dn, 10.0 / (n + 1.0)))
        y = (1 - Dn)[:, None] * st["y"] + g
        eY = (1 - rho) * st["eY"] + rho * jnp.sum(y * y, -1)
        eC = (1 - rho) * st["eC"] + rho * trC
        eF = (1 - rho) * st["eF"] + rho * trF
        NhB = 2 * g2 * eY * eF / jnp.maximum(eC, 1e-30)             # estimate of N_eff / B
        neff = jnp.sum(g2[:, None] * lam / (g2[:, None] * lam + Dn[:, None]), -1)
        cap = hp["cap"]
        g3_or = g2 * jnp.minimum(hp["c"] * (1.0 + n) ** (-hp["kappa"]), cap)
        g3_ad = g2 * jnp.minimum(hp["s"] / jnp.maximum(NhB, 1e-30), cap)
        g3_th = g2 * jnp.minimum(hp["s"] * B / neff, cap)
        th = jnp.maximum(qn, 0.0)
        neff_q = jnp.sum(qw * g2[:, None] * th / (g2[:, None] * th + Dn[:, None]), -1)
        g3_q = g2 * jnp.minimum(hp["s"] * B / jnp.maximum(neff_q, 1e-30), cap)
        mode = hp["mode"]
        g3_mono = jnp.where(n == 0, g3_ad, jnp.minimum(st["g3"], g3_ad))
        # ---- shadow buffer (signal-free): synthetic noise with covariance K/B, curvature exact or minibatch
        k7, k8, k9 = jax.random.split(jax.random.fold_in(k, 7), 3)
        zeta = sl * jnp.sqrt(_chi2(k7, float(B), (S,)))[seed][:, None] * jax.random.normal(k8, (S, d))[seed] / B
        Ku_exact = lam * st["u"]
        wu = jnp.concatenate([sl * st["u"], jnp.zeros((st["u"].shape[0], 1))], -1)
        cb = _chi2(k9, float(B), (S,))[seed]
        xb = jax.random.normal(jax.random.fold_in(k9, 1), (S, d + 1))[seed]
        Ku_batch = sl * _proj_sample(wu, cb, xb)[:, :d] / B
        Ku = jnp.where((mode == 6)[:, None], Ku_batch, Ku_exact)
        sg = Ku + zeta
        ys = (1 - Dn)[:, None] * st["ys"] + sg
        u_new = st["u"] - g2[:, None] * sg
        eS = (1 - rho) * st["eS"] + rho * jnp.sum(ys * ys, -1)
        NhB_sh = 2 * g2 * eS
        g3_sh = g2 * jnp.minimum(hp["s"] / jnp.maximum(NhB_sh, 1e-30), cap)
        g3 = jnp.where(mode == 1, g3_or, jnp.where(mode == 2, g3_ad, jnp.where(mode == 3, g3_th,
                       jnp.where(mode == 4, g3_mono, jnp.where(mode == 7, g3_q, jnp.where(mode >= 5, g3_sh, 0.0))))))
        r = st["r"] - g2[:, None] * g - g3[:, None] * y
        P = jnp.sum(lam * r * r, -1) + sigma ** 2
        ok = st["alive"] & jnp.isfinite(P) & (P < P_div)
        keep = lambda new, old: jnp.where(ok[:, None], new, old)
        return dict(r=keep(r, st["r"]), y=keep(y, st["y"]), gprev=g,
                    eY=jnp.where(ok, eY, st["eY"]), eC=jnp.where(ok, eC, st["eC"]), eF=jnp.where(ok, eF, st["eF"]),
                    alive=ok, g3=g3, T=eC / jnp.maximum(eF, 1e-30),
                    NhB=jnp.where(mode == 7, neff_q / B, jnp.where(mode >= 5, NhB_sh, NhB)),
                    u=jnp.where(ok[:, None], u_new, st["u"]), ys=jnp.where(ok[:, None], ys, st["ys"]),
                    eS=jnp.where(ok, eS, st["eS"]))

    return jax.lax.fori_loop(n0, n1, body, state)


def eval_points(n_max, n_pts=80):
    pts = np.unique(np.round(np.geomspace(1, n_max, n_pts)).astype(np.int64))
    return np.concatenate([[0], pts])


def simulate(inst, hp, B, n_max, key=0, n_pts=80, S=None, div_factor=1e3, log=None, ev=None, quad=None):
    """Run all configs in hp for n_max steps. Returns dict of recorded arrays (n_eval, G)."""
    hp_j = {k: jnp.asarray(v) for k, v in hp.items()}
    G = len(hp["g2"])
    S = int(np.max(hp["seed"])) + 1 if S is None else S
    lam = jnp.asarray(inst["lam"], jnp.float32)
    sigma = float(np.sqrt(max(inst["Pstar"], 0.0)))
    st = init_state(inst, G)
    if quad is None:
        quad = (np.zeros((G, 1), np.float32), np.zeros((G, 1), np.float32))
    qn, qw = (jnp.asarray(q, jnp.float32) for q in quad)
    base = jax.random.PRNGKey(key)
    lag = B == 1
    ev = eval_points(n_max, n_pts) if ev is None else np.asarray(ev)
    rec = {k: [] for k in ("n", "E", "g3", "NhB", "NtrueB", "T", "alive")}
    lam_np = np.asarray(inst["lam"])
    P_div = float(div_factor * inst["P0"])
    prev = 0
    for n in ev:
        if n > prev:
            st = advance(st, hp_j, lam, sigma, int(prev), int(n), base, P_div, B=B, S=S, lag=lag, qn=qn, qw=qw)
            prev = int(n)
        E = np.asarray(jnp.sum(lam * st["r"] * st["r"], -1))
        Dn = np.asarray(hp["delta"]) / (np.asarray(hp["delta"]) + max(n - 1, 0))
        g2 = np.asarray(hp["g2"])
        Nt = (g2[:, None] * lam_np[None] / (g2[:, None] * lam_np[None] + Dn[:, None])).sum(-1) / B
        rec["n"].append(n); rec["E"].append(E); rec["g3"].append(np.asarray(st["g3"]))
        rec["NhB"].append(np.asarray(st["NhB"])); rec["NtrueB"].append(Nt); rec["T"].append(np.asarray(st["T"]))
        rec["alive"].append(np.asarray(st["alive"]))
        if log is not None:
            log(f"n={n} alive={int(np.asarray(st['alive']).sum())}/{G} minE={np.nanmin(np.where(np.asarray(st['alive']), E, np.nan)):.3e}")
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["Pstar"] = inst["Pstar"]
    return out
