"""Generic vmapped trainer: configs laid out as (S seeds) x (Gs configs per seed).
Data (and model-sampled label noise) are shared across configs of the same seed (common random numbers).
Every step computes two half-batch gradients (g = their mean) and one full-batch model-sampled gradient.
"""
from functools import partial
import time
import numpy as np
import jax
import jax.numpy as jnp
from . import optim_tree as O
from . import slq as SLQ
from . import spectral as SP


def stack_hp(rows, keys):
    return {k: jnp.asarray(np.array([r.get(k, 0.0) for r in rows], dtype=np.float32)) for k in keys}


def make_runner(task, opt, g3mode, B, nleaves):
    def one_step(params, st, hp, batch, kS):
        h = B // 2
        b1 = jax.tree.map(lambda a: a[:h], batch)
        b2 = jax.tree.map(lambda a: a[h:], batch)
        g1 = jax.grad(task["loss"])(params, b1)
        g2 = jax.grad(task["loss"])(params, b2)
        Fu = None
        if opt in ("dana", "laprop_dana", "dana_shadow", "lpd2"):      # (slq variants need no sampled gradient)
            gt = jax.grad(task["sampled_loss"])(params, batch, kS)
        else:
            gt = g1
        if opt == "dana_shadow":
            Fu = task["gnvp"](params, batch, st["u"])
        if opt == "lpd2":
            Fu = lambda vec: task["gnvp"](params, batch, vec)
        return O.step(opt, params, st, g1, g2, gt, hp, B, g3mode=g3mode, nleaves=nleaves, Fu=Fu)

    # inner vmap over configs of one seed (batch broadcast), outer vmap over seeds (batch per seed)
    inner = jax.vmap(one_step, in_axes=(0, 0, 0, None, None))
    outer = jax.vmap(inner, in_axes=(0, 0, 0, 0, 0))
    if opt in SP.EIG_OPTS:    # spectral preconditioners: eigen-refresh at n = 1 and every KEIG steps (unbatched branch)
        kinds = SP.leaf_kinds(task["init"](jax.random.PRNGKey(0)))
        eig_all = jax.vmap(jax.vmap(lambda s_, h_: SP.eig_refresh(s_, h_, kinds)))

    @partial(jax.jit, donate_argnums=(0, 1))
    def advance(params, st, hp, seed_keys, n0, n1):
        def body(n, carry):
            p, s = carry
            if opt in SP.EIG_OPTS:
                s = jax.lax.cond((n == 1) | ((n > 0) & (n % SP.KEIG == 0)), lambda s_: eig_all(s_, hp), lambda s_: s_, s)
            ks = jax.vmap(lambda k: jax.random.fold_in(k, n))(seed_keys)
            kd, kS = jax.vmap(lambda k: jax.random.split(k))(ks).transpose(1, 0, 2)
            batch = jax.vmap(lambda k: task["sample"](k, B))(kd)
            return outer(p, s, hp, batch, kS)
        return jax.lax.fori_loop(n0, n1, body, (params, st))

    ev = jax.jit(jax.vmap(jax.vmap(task["evaluate"])))
    return advance, ev


def make_refresh(task, opt, m, p, b):
    """SLQ refresh of the quadrature state for every (seed, config): GN matrix on a fresh estimation batch of size b
    (plain coordinates for dana_slq; z coordinates r GN r for lpd2_slq, r = (sqrt(v_hat)+eps)^-1/2)."""
    def one(params, st, hp, batch, key):
        if opt in SP.SLQ_SPECTRAL:
            Rz = lambda u: SP.apply_Rz(opt, st, hp, u)
            mv = lambda u: Rz(task["gnvp"](params, batch, Rz(u)))
        elif opt in ("lpd2_slq", "adana_slq"):
            bc2 = 1 - hp["b2"] ** jnp.maximum(st["n"].astype(jnp.float32), 1.0) if opt == "lpd2_slq" else 1.0
            r = jax.tree.map(lambda vv: (jnp.sqrt(vv / bc2) + hp["eps"]) ** -0.5, st["v"])
            mv = lambda u: jax.tree.map(lambda a, rr: a * rr, task["gnvp"](params, batch, jax.tree.map(lambda x, rr: x * rr, u, r)), r)
        else:
            mv = lambda u: task["gnvp"](params, batch, u)
        qn, qw, qT = SLQ.quadrature_tree(mv, params, key, m=m, probes=p)
        return dict(st, qn=qn, qw=qw, qT=qT, qlam=jnp.max(qn), qok=jnp.ones(()))
    inner = jax.vmap(one, in_axes=(0, 0, 0, None, None))
    outer = jax.vmap(inner, in_axes=(0, 0, 0, 0, 0))

    @jax.jit
    def refresh(params, st, hp, keys):
        batch = jax.vmap(lambda k: task["sample"](k, b))(keys)
        pk = jax.vmap(lambda k: jax.random.fold_in(k, 1))(keys)
        return outer(params, st, hp, batch, pk)
    return refresh


def make_refresh_adaptive(task, opt, P, m_max, chunk, eps, b0, b_max, r_grow=0.5, log=None, map_configs=False, debias=True):
    """Adaptive SLQ refresh (host-orchestrated): Lanczos in chunks of `chunk` steps for every (seed, config, probe),
    stopped when the Gauss/Gauss-Radau bracket of N_eff at D_check has relative gap < eps for all instances (or the
    Gauss upper bound is <= s_eff B, where the cap binds regardless); the shared estimation batch b is doubled and the
    refresh redone while max_config N(F_b; D_next)/b > r_grow.  Stores the Gauss quadrature and qb = b for de-biasing."""
    cache = {}

    def fns(b):
        if b in cache:
            return cache[b]

        def op(params, st, hp, batch):
            if opt in SP.SLQ_SPECTRAL:       # spectral z coordinates: P^{-1/2} GN P^{-1/2}
                Rz = lambda u: SP.apply_Rz(opt, st, hp, u)
                return lambda u: Rz(task["gnvp"](params, batch, Rz(u)))
            if opt in ("lpd2_slq", "adana_slq"):     # ADana: log-time v, no bias correction
                bc2 = 1 - hp["b2"] ** jnp.maximum(st["n"].astype(jnp.float32), 1.0) if opt == "lpd2_slq" else 1.0
                r = jax.tree.map(lambda vv: (jnp.sqrt(vv / bc2) + hp["eps"]) ** -0.5, st["v"])
                return lambda u: jax.tree.map(lambda a, rr: a * rr, task["gnvp"](params, batch, jax.tree.map(lambda x, rr: x * rr, u, r)), r)
            return lambda u: task["gnvp"](params, batch, u)

        def init_one(params, key):
            leaves, tdef = jax.tree.flatten(params)
            ks = jax.random.split(key, len(leaves))
            z = jax.tree.unflatten(tdef, [jax.random.rademacher(kk, x.shape, jnp.float32) for kk, x in zip(ks, leaves)])
            return SLQ.lanczos_state_tree(z)

        def chunk_one(params, st, hp, batch, ls, al, be, C, j0):
            return SLQ.lanczos_chunk_tree(op(params, st, hp, batch), ls, al, be, C, j0, chunk)

        # vmaps: probes (P) innermost, configs (G) share the batch, seeds (S) outermost
        init_p = jax.vmap(init_one, in_axes=(None, 0))
        init_g = jax.vmap(init_p, in_axes=(0, None))
        init_s = jax.jit(jax.vmap(init_g, in_axes=(0, 0)))
        ch_p = jax.vmap(chunk_one, in_axes=(None, None, None, None, 0, 0, 0, 0, None))
        if map_configs:     # sequential over configs (bounded memory for large models / estimation batches)
            def ch_g(params, st, hp, batch, ls, al, be, C, j0):
                return jax.lax.map(lambda x: ch_p(x[0], x[1], x[2], batch, x[3], x[4], x[5], x[6], j0), (params, st, hp, ls, al, be, C))
        else:
            ch_g = jax.vmap(ch_p, in_axes=(0, 0, 0, None, 0, 0, 0, 0, None))
        ch_s = jax.jit(jax.vmap(ch_g, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None)))
        samp = jax.jit(jax.vmap(lambda k: task["sample"](k, b)))
        if "teff" in task:        # T_eff proxy on (up to) 256 sequences of the estimation batch, sequential over configs
            nb = min(b, 256)
            te_g = lambda params, batch: jax.lax.map(lambda p_: task["teff"](p_, jax.tree.map(lambda a: a[:nb], batch)), params)
            te_s = jax.jit(jax.vmap(te_g, in_axes=(0, 0)))
        else:
            te_s = None
        cache[b] = (init_s, ch_s, samp, te_s)
        return cache[b]

    def refresh(params, st, hp, keys, b, D_check, D_next, g2_host, sB_host, B=None, n_now=0):
        """g2_host (S,G) step used in N_eff (for lpd2_slq: None -> derived from the Ritz values); sB_host (S,G) = s_eff B."""
        L = len(jax.tree.leaves(jax.tree.map(lambda a: a[0, 0], params)))
        Tt = np.asarray(hp["T"], np.float64) if "T" in hp else np.zeros(np.asarray(hp["lr"]).shape)
        wu = np.maximum(0.02 * Tt, 1.0)
        sig = np.where(Tt > 0, np.where(n_now < wu, (n_now + 1) / wu,
                       0.1 + 0.45 * (1 + np.cos(np.pi * np.clip((n_now - wu) / np.maximum(Tt - wu, 1.0), 0, 1)))), 1.0)
        tau_h = np.asarray(hp["tau"], np.float64) if "tau" in hp else np.zeros_like(sig)
        if "sched_mode" in hp:
            sig = np.where(np.asarray(hp["sched_mode"]) > 0.5, 1.0, sig)
        while True:
            init_s, ch_s, samp, te_s = fns(b)
            batch = samp(keys)
            S, G = np.asarray(hp["lr"]).shape
            teff = np.asarray(te_s(params, batch), np.float64) if te_s is not None else np.ones((S, G))
            Beff = None if B is None else B * np.maximum(teff, 1.0)
            pk = jax.vmap(lambda k: jax.random.split(jax.random.fold_in(k, 7), P))(keys)
            ls = init_s(params, pk)
            al = jnp.zeros((S, G, P, m_max)); be = jnp.zeros((S, G, P, m_max)); C = jnp.zeros((S, G, P, m_max, L))
            j = 0
            while j < m_max:
                ls, al, be, C = ch_s(params, st, hp, batch, ls, al, be, C, j)
                j += chunk
                alh, beh, zn = np.asarray(al, np.float64), np.asarray(be, np.float64), np.asarray(ls["zn"], np.float64)
                # a diverged config (NaN params) must not take the others down: empty quadrature for it
                bad = ~(np.isfinite(alh).all(-1) & np.isfinite(beh).all(-1) & np.isfinite(zn))
                alh = np.where(bad[..., None], 0.0, alh); beh = np.where(bad[..., None], 0.0, beh); zn = np.where(bad, 0.0, zn)
                (thG, qG), (thR, qR) = SLQ.gauss_radau(alh[..., :j], beh[..., :j], j)
                if opt == "lpd2_slq":
                    lam = thG.max(-1).max(-1)                                     # (S,G)
                    g2 = np.minimum(sig * np.asarray(hp["lr"]), np.asarray(hp["f"]) * 2.0 / np.maximum(lam, 1e-30))
                    if B is not None and (tau_h > 0).any():
                        thf = thG.reshape(S, G, -1); wf = (zn[..., None] ** 2 * qG).reshape(S, G, -1) / P
                        g2 = np.where(tau_h > 0, sig * tau_h * SLQ.g2max_quad(thf, wf, Beff), g2)
                else:
                    g2 = g2_host
                    tau = np.asarray(hp["tau"], np.float64) if "tau" in hp else np.zeros((S, G))
                    if B is not None and (tau > 0).any():            # auto step: g2 = tau * threshold (raw Gauss weights)
                        thf = thG.reshape(S, G, -1); wf = (zn[..., None] ** 2 * qG).reshape(S, G, -1) / P
                        g2 = np.where(tau > 0, sig * tau * SLQ.g2max_quad(thf, wf, Beff), sig * g2_host)
                # de-biasing shift D' at the target D_next (fixed point on the current Gauss rule, probe-averaged)
                g2e = g2[..., None, None]
                Nb_at = lambda dd: (zn ** 2 * SLQ.fsum(thG, qG, g2e, dd)).mean(-1)[..., None, None]
                Dp = D_next
                for _ in range(20 if debias else 0):
                    Dp = D_next * (1 - np.minimum(Nb_at(Dp) / b, 0.9))
                U = zn ** 2 * SLQ.fsum(thG, qG, g2e, Dp)
                Lw = zn ** 2 * SLQ.fsum(thR, qR, g2e, Dp)
                done = ((U - Lw) <= eps * U).all(-1) | (U.mean(-1) <= sB_host * np.maximum(teff, 1.0))
                if done.all():
                    break
            m = j
            th, w, wT = SLQ.quad_from_coeffs_np(alh, beh, np.asarray(C, np.float64), zn, m)
            Nb = Nb_at(Dp)[..., 0, 0]
            ratio = float((Nb / b).max())
            if (not debias) or ratio <= r_grow or b >= b_max:
                break
            if log:
                log(f"      refresh: N(F_b)/b = {ratio:.2f} > {r_grow} at b={b} -> b={2 * b}")
            b *= 2
        Q = P * m_max
        pad = lambda x, extra=(): np.concatenate([x, np.zeros(x.shape[:2] + (Q - x.shape[2],) + extra)], 2)
        qn = pad(th.reshape(S, G, P * m)); qw = pad((w / P).reshape(S, G, P * m)); qT = pad((wT / P).reshape(S, G, P * m, L), (L,))
        # SGD stability threshold at the training batch B from the DEFLATED quadrature (isolated top eigenvalues exact)
        g2m = np.zeros((S, G))
        if B is not None:
            thd, wd, _, _ = SLQ.ritz_quadrature_np(alh, beh, zn ** 2, m)
            g2m = SLQ.g2max_quad(thd.reshape(S, G, -1), wd.reshape(S, G, -1) / P, Beff)
        st = dict(st, qg2max=jnp.asarray(g2m, jnp.float32), qteff=jnp.asarray(teff, jnp.float32), qn=jnp.asarray(qn, jnp.float32), qw=jnp.asarray(qw, jnp.float32), qT=jnp.asarray(qT, jnp.float32),
                  qlam=jnp.asarray(th.max(-1).max(-1), jnp.float32), qok=jnp.ones((S, G), jnp.float32),
                  qb=jnp.full((S, G), float(b) if debias else 0.0, jnp.float32))
        return st, dict(m=m, b=b, ratio=ratio, g2max=float(np.median(g2m)), teff=float(np.median(teff)))
    return refresh


def refresh_schedule(n_max, first=1, ratio=1.25, max_gap=2000):
    pts, n = [], float(first)
    while n < n_max:
        pts.append(int(n)); n = min(n * ratio + 1, n + max_gap)
    return sorted(set(pts))


def run(task, opt, g3mode, hp_rows, hp_keys, S, B, n_max, eval_pts, seed0=0, log=print, slq=None):
    """hp_rows: list (len Gs) of dicts (same configs replicated over S seeds)."""
    Gs = len(hp_rows)
    hp = stack_hp(hp_rows, hp_keys)
    hp = jax.tree.map(lambda a: jnp.broadcast_to(a, (S, Gs)), hp)
    keys = jax.random.split(jax.random.PRNGKey(seed0), S)
    p0 = jax.vmap(task["init"])(keys)                         # (S, ...)
    params = jax.tree.map(lambda a: jnp.broadcast_to(a[:, None], (S, Gs) + a.shape[1:]).copy(), p0)
    nleaves = len(jax.tree.leaves(p0))
    adaptive = bool(slq and slq.get("adaptive"))
    Q = (slq["p"] * (slq["m_max"] if adaptive else slq["m"])) if slq else 1
    st = jax.vmap(jax.vmap(lambda p: O.init(opt, p, Q=Q)))(params)
    advance, ev = make_runner(task, opt, g3mode, B, nleaves)
    rpts = set()
    rinfo = []
    if adaptive:
        arefresh = make_refresh_adaptive(task, opt, slq["p"], slq["m_max"], slq.get("chunk", 8), slq.get("eps", 0.05),
                                         slq.get("b0", 256), slq.get("b_max", 16384), log=log, map_configs=slq.get("map_configs", False),
                                         debias=slq.get("debias", True))
        b_cur = slq.get("b0", 256)
    if slq:
        refresh = None if adaptive else make_refresh(task, opt, slq["m"], slq["p"], slq["b"])
        rpts = set(slq.get("sched") or refresh_schedule(n_max, max_gap=slq.get("max_gap", 2000)))
        if slq.get("refresh0"):
            rpts.add(0)
        rkeys = jax.random.split(jax.random.PRNGKey(seed0 + 4242), S)
    data_keys = jax.random.split(jax.random.PRNGKey(seed0 + 777), S)
    rec = dict(n=[], risk=[], g3=[], nhb=[], g2e=[], nres=[], eG=[], gmul=[])
    prev = 0
    t0 = time.time()
    evset = set(int(x) for x in eval_pts)
    for n in sorted(evset | {x for x in rpts if x <= int(max(eval_pts))}):
        if n > prev:
            params, st = advance(params, st, hp, data_keys, prev, int(n))
            prev = int(n)
        if n in rpts:
            rk = jax.vmap(lambda k: jax.random.fold_in(k, n))(rkeys)
            if adaptive:
                later = sorted(x for x in rpts if x > n)
                n_next = later[0] if later else n_max
                dl = np.asarray(hp["delta"], np.float64)
                D_next = dl / (dl + n_next)
                lr = np.asarray(hp["lr"], np.float64)
                sv = np.asarray(hp["s"], np.float64); Cv = np.asarray(hp.get("C", np.zeros_like(sv)), np.float64)
                s_eff = np.where(Cv > 0, np.minimum(sv, Cv / B), sv)
                st, info = arefresh(params, st, hp, rk, b_cur, D_next[..., None, None] / 2, D_next[..., None, None], lr, s_eff * B, B=B, n_now=n)
                b_cur = info["b"]; rinfo.append(dict(n=int(n), **info))
            else:
                st = refresh(params, st, hp, rk)
        if n not in evset:
            continue
        m = ev(params)
        rec["n"].append(n)
        rec["risk"].append(np.asarray(m["risk"] if "risk" in m else m["loss"]))
        if "g2e" in st:
            rec["g2e"].append(np.asarray(st["g2e"]))
        if "nres" in st:
            rec["nres"].append(np.asarray(st["nres"]))
        if "gmul" in st:
            rec["gmul"].append(np.asarray(st["gmul"]))
        if "eG" in st:
            rec["eG"].append(np.stack(jax.tree.leaves(jax.tree.map(np.asarray, st["eG"])), -1))
        if opt in ("dana", "laprop_dana", "dana_shadow", "lpd2", "dana_slq", "lpd2_slq", "adana_slq") + SP.SLQ_SPECTRAL:
            g3 = jax.tree.map(lambda a: np.asarray(a), st["g3"])
            rec["g3"].append(np.stack(jax.tree.leaves(g3), -1))
            nhb = jax.tree.map(lambda ey, ec, ef: np.asarray(2 * hp["lr"] * ey * ef / jnp.maximum(ec, 1e-30)),
                               st["eY"], st["eC"], st["eF"])
            rec["nhb"].append(np.stack(jax.tree.leaves(nhb), -1))
        if log and (len(rec["n"]) % 10 == 0):
            r = rec["risk"][-1]
            log(f"   n={n} risk min/median {np.nanmin(r):.4e}/{np.nanmedian(r):.4e} ({time.time()-t0:.0f}s)")
    out = {k: np.asarray(v) for k, v in rec.items() if len(v)}
    if slq and adaptive:
        out["refresh_info"] = rinfo
    return out
