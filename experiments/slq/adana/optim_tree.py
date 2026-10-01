"""Pytree optimizers (pure functions, vmap-able over hyperparameters).

All methods consume, per step, the two half-batch gradients g1, g2 and (for adaptive methods) a gradient gt
computed with model-sampled labels on the full batch.  g = (g1 + g2)/2 is the usual minibatch gradient.

Families (hp['opt'] is a static string):
  'sgd'          theta -= g2 * g
  'sgdm'         heavy ball with fixed beta1: m = b1 m + g ; theta -= g2 * m   (lr scaled so that effective step = g2/(1-b1))
  'dana'         Gen-Mom-SGD, gamma1 = 1: y = (1-D_n) y + u ; theta -= g2 u + g3_n y,  D_n = delta/(delta+n)
                 g3 mode (static hp['g3mode']): 'oracle': g2*min(c (1+n)^-kappa, cap)
                                                'adapt' : per leaf g2*min(s_leaf / NhB_leaf, cap),
                                                          NhB = 2 g2 <|y|^2><trF>/<trC>  (estimate of N_eff/B)
  'laprop'       v = b2 v + (1-b2) g^2 ; u = g/(sqrt(v_hat)+eps) ; m = b1 m + (1-b1) u ; theta -= lr m
  'adam'         standard Adam
  'laprop_dana'  u = g/(sqrt(v_hat)+eps) (b2 matched), then the 'dana' outer loop on u; traces are computed on the
                 preconditioned half gradients / sampled gradient.
  'lpd2'         LaProp-DANA v2 (consistent coordinates + stability-tracked step).  P = sqrt(v_hat)+eps, r = P^{-1/2}.
                 In z = P^{1/2} theta the step is plain SGD with gradient r*g, curvature H_z = r H r (GN), noise r C r.
                 lam_hat = top eigenvalue of the minibatch GN matrix in z (one power-iteration step per step, EMA);
                 g2_n = min(lr, f * 2/lam_hat)   [f = 1/2: half the stability limit; minibatch lam includes the tr/B term]
                 shadow buffer in z driven by r*gt with curvature r GN(r u) -> Nhat/B = 2 g2_n <|ys|^2>;
                 g3_n = g2_n * min{ min(s, C/B) / (Nhat/B), cap }  (saturating-batch law);  y_z = (1-D) y_z + r*g;
                 theta -= g2_n g/P + g3_n r*y_z.   Needs Fu = callable vec -> GN(params) vec.
  'dana_slq'     Gen-Mom-SGD; g3 from an SLQ quadrature (nodes qn, weights qw, per-leaf weights qT) of the GN matrix,
                 refreshed by the trainer on a schedule: Nhat(Delta_n) = sum_k qw_k g2 th_k/(g2 th_k + Delta_n);
                 global: g3 = g2 min{s_eff B/Nhat, cap};  per-tensor (pt=1): g3_T = g2 min{(s_eff/L) B/Nhat_T, cap}.
  'lpd2_slq'     LaProp-DANA v2.1 in z coordinates with the quadrature of r GN r (refreshed): g2_n = min(lr, f*2/qlam)
                 with qlam = largest Ritz value, g3 as in dana_slq with g2_n.  No per-step GN products.
                 alloc 4: water-filling with the v2 leftover rule (budget left once every tensor with positive signal is
                 at the cap is spread uniformly over the tensors without signal).
  'adana_slq'    SLQ-ADana, as in mlexpos/adana src/optim/adana_slq.py: log-time moments m, v with D_t = delta/(delta+t)
                 (t = n+1), step theta -= sigma(n) lr (g + alpha_T m)/(sqrt(v)+eps), alpha_T = r_T / D_t, quadrature of
                 r GN r with r = (sqrt(v)+eps)^-1/2, budget sum_T r_T N_T <= s B_eff mu (mu from gnsb = k').
                 alloc 0 global, 3 water-filling on the instantaneous buffer signal
                 G_T = |m_T|_z^2 - D^2 (N_T/B_eff) Temp/(2 lr), 4 water-filling v2 (leftover rule), -1 plain ADana:
                 alpha = c (1 + (1+t)^(1-kappa)) (c = gamma_3f).  r_T = 0 before the first refresh (SLQ modes).
Traces per leaf: trC = (B/4)|u1 - u2|^2 ,  trF = B |ut|^2.
"""
import jax
import jax.numpy as jnp

from . import spectral as SP

tmap = jax.tree.map


def _sq(t):
    return jax.tree.reduce(lambda a, b: a + b, tmap(lambda x: jnp.sum(x * x), t))


def leaf_sq(t):
    return tmap(lambda x: jnp.sum(x * x), t)


def init(opt, params, Q=1):
    if opt in SP.SPECTRAL:
        return SP.init(opt, params, Q=Q)
    z = tmap(jnp.zeros_like, params)
    zs = tmap(lambda x: jnp.zeros((), x.dtype), params)
    st = dict(n=jnp.zeros((), jnp.int32))
    if opt in ("sgdm",):
        st["m"] = z
    if opt in ("dana", "laprop_dana", "dana_shadow"):
        st.update(y=z, eY=zs, eC=zs, eF=zs, g3=zs)
    if opt == "dana_shadow":
        st.update(u=z, ys=z, eS=jnp.zeros((), jnp.float32), eSl=zs)
    if opt in ("laprop", "adam", "laprop_dana", "lpd2", "lpd2_slq", "adana_slq"):
        st["v"] = z
    if opt == "adana_slq":
        st.update(m=z, eDz=zs)
    if opt in ("dana_slq", "lpd2_slq", "adana_slq"):
        L = len(jax.tree.leaves(params))
        st.update(y=z, qn=jnp.zeros((Q,), jnp.float32), qw=jnp.zeros((Q,), jnp.float32), qT=jnp.zeros((Q, L), jnp.float32),
                  qlam=jnp.zeros((), jnp.float32), qok=jnp.zeros((), jnp.float32), qb=jnp.zeros((), jnp.float32),
                  qg2max=jnp.zeros((), jnp.float32), qteff=jnp.zeros((), jnp.float32), eG=zs, eD=zs,
                  eX=jnp.zeros((), jnp.float32), eDg=jnp.zeros((), jnp.float32), gmul=jnp.ones((), jnp.float32),
                  g2e=jnp.zeros((), jnp.float32),
                  g3=zs, eY=zs, eC=zs, eF=zs)
    if opt == "lpd2":
        leaves, tdef = jax.tree.flatten(params)
        ks = jax.random.split(jax.random.PRNGKey(123), len(leaves))
        w = jax.tree.unflatten(tdef, [jax.random.normal(k, x.shape, x.dtype) for k, x in zip(ks, leaves)])
        st.update(y=z, u=z, ys=z, w=w, eS=jnp.zeros((), jnp.float32), eL=jnp.zeros((), jnp.float32),
                  g2e=jnp.zeros((), jnp.float32), scale=jnp.ones((), jnp.float32), nres=jnp.zeros((), jnp.float32),
                  g3=zs, eY=zs, eC=zs, eF=zs, eSl=zs)
    if opt in ("laprop", "adam"):
        st["m"] = z
    return st


def step(opt, params, st, g1, g2h, gt, hp, B, g3mode="adapt", nleaves=1, Fu=None):
    n = st["n"]
    nf = n.astype(jnp.float32)
    g = tmap(lambda a, b: 0.5 * (a + b), g1, g2h)
    new = dict(st)
    new["n"] = n + 1
    # schedule multiplier (hp T > 0): linear warmup over 2% of T, cosine decay to 0.1 (applies to lr / tau)
    Tt = hp.get("T", 0.0)
    wu = jnp.maximum(0.02 * Tt, 1.0)
    cosd = 0.1 + 0.9 * 0.5 * (1 + jnp.cos(jnp.pi * jnp.clip((nf - wu) / jnp.maximum(Tt - wu, 1.0), 0.0, 1.0)))
    sig = jnp.where(Tt > 0, jnp.where(nf < wu, (nf + 1) / wu, cosd), 1.0)
    # clipping of the APPLIED gradient (hp clip > 0: global norm); estimators keep the unclipped g1, g2h, gt
    cl = hp.get("clip", 0.0)
    gnorm = jnp.sqrt(_sq(g))
    g = tmap(lambda x: x * jnp.where(cl > 0, jnp.minimum(1.0, cl / jnp.maximum(gnorm, 1e-30)), 1.0), g)
    lr = hp["lr"] * sig
    if opt in SP.SPECTRAL:
        return SP.step(opt, params, st, new, g1, g2h, g, hp, B, nf, n, sig)
    if opt == "sgd":
        return tmap(lambda p, gg: p - lr * gg, params, g), new
    if opt == "sgdm":
        m = tmap(lambda mm, gg: hp["b1"] * mm + gg, st["m"], g)
        new["m"] = m
        return tmap(lambda p, mm: p - lr * (1 - hp["b1"]) * mm, params, m), new
    if opt in ("laprop", "adam", "laprop_dana", "lpd2", "lpd2_slq"):
        b2 = hp["b2"]
        v = tmap(lambda vv, gg: b2 * vv + (1 - b2) * gg * gg, st["v"], g)
        new["v"] = v
        bc2 = 1 - b2 ** (nf + 1)
        den = tmap(lambda vv: jnp.sqrt(vv / bc2) + hp["eps"], v)
    if opt == "adam":
        b1 = hp["b1"]
        m = tmap(lambda mm, gg: b1 * mm + (1 - b1) * gg, st["m"], g)
        new["m"] = m
        bc1 = 1 - b1 ** (nf + 1)
        return tmap(lambda p, mm, dd: p - lr * (mm / bc1) / dd, params, m, den), new
    if opt == "laprop":
        b1 = hp["b1"]
        m = tmap(lambda mm, gg, dd: b1 * mm + (1 - b1) * gg / dd, st["m"], g, den)
        new["m"] = m
        bc1 = 1 - b1 ** (nf + 1)
        return tmap(lambda p, mm: p - lr * mm / bc1, params, m), new
    if opt == "lpd2":
        D = hp["delta"] / (hp["delta"] + nf)
        rho = jnp.minimum(1.0, jnp.maximum(D, 10.0 / (nf + 1.0)))
        r = tmap(lambda dd: dd ** -0.5, den)
        mul = lambda a, b: tmap(lambda x, y_: x * y_, a, b)
        # power iteration for the top eigenvalue of H_z = r GN r
        Hw = mul(r, Fu(mul(r, st["w"])))
        wn = jnp.sqrt(_sq(st["w"])) + 1e-30
        lam = jax.tree.reduce(lambda a, b: a + b, tmap(lambda a, b: jnp.sum(a * b), st["w"], Hw)) / wn ** 2
        hn = jnp.sqrt(_sq(Hw)) + 1e-30
        w = tmap(lambda x: x / hn, Hw)
        eL = jnp.where(n == 0, lam, (1 - rho) * st["eL"] + rho * lam)
        g2n = jnp.minimum(hp["lr"] * st["scale"], hp["f"] * 2.0 / jnp.maximum(eL, 1e-30))
        # shadow buffer in z coordinates
        sg = tmap(lambda a, b: a + b, mul(r, Fu(mul(r, st["u"]))), mul(r, gt))
        ys = tmap(lambda yy, x: (1 - D) * yy + x, st["ys"], sg)
        us = tmap(lambda uu, x: uu - g2n * x, st["u"], sg)
        # guard: a shadow blow-up (linearized dynamics noise-unstable, e.g. rare-token curvature spikes) resets the shadow,
        # holds the last estimate, and (bk < 1) backs the step off by bk
        sq = _sq(ys)
        bad = (~jnp.isfinite(sq)) | ((sq > 100.0 * st["eS"]) & (n > 100))
        us = tmap(lambda x: jnp.where(bad, 0.0, x), us)
        ys = tmap(lambda x: jnp.where(bad, 0.0, x), ys)
        eS = jnp.where(bad, st["eS"], (1 - rho) * st["eS"] + rho * sq)
        new.update(scale=jnp.where(bad, st["scale"] * hp["bk"], st["scale"]), nres=st["nres"] + bad)
        nhb = 2 * g2n * eS
        s_eff = jnp.minimum(hp["s"], hp["C"] / B)
        g3s = g2n * jnp.minimum(s_eff / jnp.maximum(nhb, 1e-30), hp["cap"])
        g3s = jnp.where(jnp.isfinite(g3s), g3s, 0.0)
        # per-tensor variant (hp pt=1): own shadow energy and gamma_3 per leaf, budget split s/L; step g2n stays global
        eSl = tmap(lambda e, x: jnp.where(bad, e, (1 - rho) * e + rho * jnp.sum(x * x)), st["eSl"], ys)
        nhbl = tmap(lambda e: 2 * g2n * e, eSl)
        g3l = tmap(lambda nb: g2n * jnp.minimum(s_eff / nleaves / jnp.maximum(nb, 1e-30), hp["cap"]), nhbl)
        g3 = tmap(lambda a: jnp.where(hp.get("pt", 0.0) > 0.5, jnp.where(jnp.isfinite(a), a, 0.0), g3s), g3l)
        gz = mul(r, g)
        y = tmap(lambda yy, x: (1 - D) * yy + x, st["y"], gz)
        new.update(w=w, eL=eL, g2e=g2n, u=us, ys=ys, eS=eS, eSl=eSl, y=y, g3=g3,
                   eY=tmap(lambda nb: jnp.where(hp.get("pt", 0.0) > 0.5, nb, nhb), nhbl), eC=tmap(lambda _: jnp.ones(()), st["eC"]),
                   eF=tmap(lambda _: jnp.ones(()) / (2 * hp["lr"]), st["eF"]))
        return tmap(lambda p, gg, dd, rr, yy, g3_: p - g2n * gg / dd - g3_ * rr * yy, params, g, den, r, y, g3), new
    if opt == "adana_slq":
        return _adana_slq_step(params, st, new, g, g1, g2h, hp, B, nf, n, sig)
    if opt in ("dana_slq", "lpd2_slq"):
        D = hp["delta"] / (hp["delta"] + nf)
        if opt == "lpd2_slq":
            r = tmap(lambda dd: dd ** -0.5, den)
            tau = hp.get("tau", 0.0)
            sm = hp.get("sched_mode", 0.0)
            g2u = jnp.where(tau > 0, tau * st["qg2max"] * st["qok"],
                            jnp.minimum(hp["lr"], hp["f"] * 2.0 / jnp.maximum(st["qlam"], 1e-30)))   # unscheduled step
            g2n = jnp.where(sm > 0.5, g2u, jnp.where(tau > 0, sig * g2u, jnp.minimum(lr, hp["f"] * 2.0 / jnp.maximum(st["qlam"], 1e-30))))
            gz = tmap(lambda a, b: a * b, r, g)
        else:
            # auto step (hp tau > 0): g2 = tau * (SGD stability threshold at batch B from the quadrature); 0 before 1st refresh
            tau = hp.get("tau", 0.0)
            sm = hp.get("sched_mode", 0.0)
            g2u = jnp.where(tau > 0, tau * st["qg2max"] * st["qok"], hp["lr"])        # unscheduled step
            g2n = jnp.where(sm > 0.5, g2u, jnp.where(tau > 0, sig * g2u, lr))
            gz = g
        th = jnp.maximum(st["qn"], 0.0)
        # deterministic-equivalent de-biasing for a quadrature built on an estimation batch of size qb (qb=0: none):
        # N(F; D) ~= N(F_b; D') with D' = D (1 - N(F_b; D')/qb)  (fixed point; same shift for every block)
        qb = st["qb"]
        def fp(_, dd):
            nb = jnp.dot(g2n * th / (g2n * th + dd), st["qw"])
            return jnp.where(qb > 0, D * (1 - jnp.minimum(nb / jnp.maximum(qb, 1.0), 0.9)), D)
        Dp = jax.lax.fori_loop(0, 12, fp, D)
        fq = g2n * th / (g2n * th + Dp)
        N = jnp.dot(fq, st["qw"])
        NT = fq @ st["qT"]                                       # (L,)
        # effective number of independent noise samples in the batch (sequence models: B * T_eff; else B)
        Bseq = B                                                  # samples (sequences) per batch, before T_eff
        B = B * jnp.maximum(st["qteff"], 1.0)
        C = hp.get("C", 0.0)
        s_eff = jnp.where(C > 0, jnp.minimum(hp["s"], C / B), hp["s"])      # C <= 0: no batch saturation
        # signal-fraction multiplier (hp gns = k > 0): the momentum-induced noise floor ~ F (E + P*) must stay below ~E/2,
        # E/(E+P*) proxied by B/B_noise = B |grad|^2 / trC = 4 <g1,g2> / |g1-g2|^2  (split halves; B cancels, so the
        # sequence split of a sequence model needs no T_eff).  budget s_eff B -> s_eff B min{1, k B/B_noise}
        # raw (unpreconditioned) gradients for every optimizer: the floor ratio E/(E+P*) is a property of the loss, not of the
        # coordinates; window ~ n/3 (the ratio moves on the time scale n; the per-step SNR of <g1,g2> is ~ B/B_noise << 1)
        rho_ = jnp.minimum(1.0, 3.0 / (nf + 1.0))
        h1, h2 = jax.tree.leaves(g1), jax.tree.leaves(g2h)
        xg = sum(jnp.sum(a * b) for a, b in zip(h1, h2)); dg = sum(jnp.sum((a - b) ** 2) for a, b in zip(h1, h2))
        eX = jnp.where(n == 0, xg, (1 - rho_) * st["eX"] + rho_ * xg)
        eDg = jnp.where(n == 0, dg, (1 - rho_) * st["eDg"] + rho_ * dg)
        kg = hp.get("gns", 0.0); kb = hp.get("gnsb", 0.0)
        bnr = 4 * jnp.maximum(eX, 0.0) / jnp.maximum(eDg, 1e-30)            # B / B_noise (B in samples/sequences)
        # gns = k: multiplier min{1, k B/B_noise};  gnsb = k' (batch-independent form): min{1, k'/B_noise} = min{1, (k'/B) B/B_noise}
        gmul = jnp.where(kg > 0, jnp.clip(kg * bnr, 0.0, 1.0), jnp.where(kb > 0, jnp.clip(kb * bnr / Bseq, 0.0, 1.0), 1.0))
        new.update(eX=eX, eDg=eDg, gmul=gmul)
        s_eff = s_eff * gmul
        g3g = g2n * jnp.minimum(s_eff * B / jnp.maximum(N, 1e-30), hp["cap"])
        leaves, tdef = jax.tree.flatten(params)
        L = len(leaves)
        g3T = g2n * jnp.minimum(s_eff / L * B / jnp.maximum(NT, 1e-2), hp["cap"])
        y = tmap(lambda yy, x: (1 - D) * yy + x, st["y"], gz)
        # ---- momentum-budget allocation across tensors (hp alloc): 0 global (a_T equal, i.e. budget prop. to N_T),
        # 1 equal split (s/L each; = pt=1), 2/3 water-filling a_T = min{1, nu G_T/N_T} with sum_T a_T N_T = s_eff B,
        # signal G_T from the split-batch inner product <g1_T, g2_T> (2) or from the buffer ||D y_T||^2 - (D/2) trC_T/B (3)
        rho = jnp.minimum(1.0, jnp.maximum(D, 10.0 / (nf + 1.0)))
        if opt == "lpd2_slq":      # allocation statistics in the z coordinates of the normalized step
            l1 = jax.tree.leaves(tmap(lambda a, b: a * b, r, g1)); l2 = jax.tree.leaves(tmap(lambda a, b: a * b, r, g2h))
        else:
            l1, l2 = jax.tree.leaves(g1), jax.tree.leaves(g2h)
        ly = jax.tree.leaves(y)
        cross = jnp.stack([jnp.sum(a * b) for a, b in zip(l1, l2)])
        dsq = jnp.stack([jnp.sum((a - b) ** 2) for a, b in zip(l1, l2)])          # E = 4 trC_T / B
        ysq = jnp.stack([jnp.sum(a * a) for a in ly])
        eG = jnp.stack(jax.tree.leaves(st["eG"])); eD = jnp.stack(jax.tree.leaves(st["eD"]))
        alloc = hp.get("alloc", 0.0)
        # buffer signal: subtract the dynamics-filtered noise energy of the buffer (buffer identity per tensor):
        # E||y_T,noise||^2 ~= (N_T/B) trC_T / (2 g2 trF_TT),  trC_T = B E||g1_T-g2_T||^2/4,  trF_TT = sum_k th_k qT_k
        # (global temperature trC/trF: block traces of small tensors are too noisy to divide by)
        temp = jnp.sum(B * eD / 4) / jnp.maximum(jnp.dot(th, st["qw"]), 1e-30)
        noiseT = (jnp.maximum(NT, 0.0) / B) * temp / (2 * g2n)
        Gsig = jnp.where(alloc > 2.5, (D ** 2) * (ysq - noiseT), cross)
        eG = jnp.where(n == 0, Gsig, (1 - rho) * eG + rho * Gsig)
        eD = jnp.where(n == 0, dsq, (1 - rho) * eD + rho * dsq)
        NTc = jnp.maximum(NT, 1e-2)
        c = jnp.maximum(eG, 0.0) / NTc
        budget = s_eff * B
        spent = lambda nu: jnp.sum(jnp.minimum(1.0, nu * c) * NTc)
        lo, hi = jnp.zeros(()), jnp.full((), 1e30)
        def bis(_, lh):
            lo_, hi_ = lh
            mid = jnp.sqrt(jnp.maximum(lo_, 1e-30) * hi_)
            ok = spent(mid) <= budget
            return jnp.where(ok, mid, lo_), jnp.where(ok, hi_, mid)
        lo, hi = jax.lax.fori_loop(0, 120, bis, (lo, hi))
        a_wf = jnp.minimum(1.0, lo * c)
        a_wf = jnp.where(jnp.sum(c) > 0, a_wf, jnp.minimum(budget / jnp.sum(NTc), 1.0))   # no signal yet: global
        a_wf = jnp.where(jnp.sum(NTc) <= budget, 1.0, a_wf)
        a_wf = jnp.where(alloc > 3.5, _leftover(a_wf, c, NTc, budget), a_wf)
        g3wf = g2n * jnp.minimum(a_wf, hp["cap"])
        g3sel = jnp.where(alloc > 1.5, g3wf, jnp.where((alloc > 0.5) | (hp.get("pt", 0.0) > 0.5), g3T, g3g * jnp.ones(L)))
        g3 = jax.tree.unflatten(tdef, [st["qok"] * g3sel[i] for i in range(L)])
        new.update(eG=jax.tree.unflatten(tdef, [eG[i] for i in range(L)]), eD=jax.tree.unflatten(tdef, [eD[i] for i in range(L)]))
        nhb = jax.tree.unflatten(tdef, [jnp.where(hp.get("pt", 0.0) > 0.5, NT[i], N) / B for i in range(L)])
        new.update(y=y, g3=g3, g2e=g2n, eY=nhb, eC=tmap(lambda _: jnp.ones(()), st["eC"]),
                   eF=tmap(lambda _: jnp.ones(()) / (2 * hp["lr"]), st["eF"]))
        tot = jnp.where(hp.get("sched_mode", 0.0) > 0.5, sig, 1.0)   # mode 1: whole update x sigma(n)
        new["g2e"] = tot * g2n
        if opt == "lpd2_slq":
            return tmap(lambda p, gg, dd, rr, yy, g3_: p - tot * (g2n * gg / dd + g3_ * rr * yy), params, g, den, r, y, g3), new
        return tmap(lambda p, gg, yy, g3_: p - tot * (g2n * gg + g3_ * yy), params, g, y, g3), new
    # ---- DANA outer loop on u (u = g for 'dana', preconditioned for 'laprop_dana')
    if opt == "dana_shadow":
        # shadow buffer driven by model-sampled noise gt (cov F/B), curvature F u from a GN-vector product
        D = hp["delta"] / (hp["delta"] + nf)
        rho = jnp.minimum(1.0, jnp.maximum(D, 10.0 / (nf + 1.0)))
        sg = tmap(lambda a, b: a + b, Fu, gt)
        ys = tmap(lambda yy, x: (1 - D) * yy + x, st["ys"], sg)
        us = tmap(lambda uu, x: uu - hp["lr"] * x, st["u"], sg)
        eS = (1 - rho) * st["eS"] + rho * _sq(ys)
        nhb = 2 * hp["lr"] * eS
        g3s = hp["lr"] * jnp.minimum(hp["s"] / jnp.maximum(nhb, 1e-30), hp["cap"])
        # per-tensor variant (hp pt=1): own shadow energy and gamma_3 per leaf, budget split s/L
        eSl = tmap(lambda e, x: (1 - rho) * e + rho * jnp.sum(x * x), st["eSl"], ys)
        nhbl = tmap(lambda e: 2 * hp["lr"] * e, eSl)
        g3 = tmap(lambda nb: jnp.where(hp.get("pt", 0.0) > 0.5, hp["lr"] * jnp.minimum(hp["s"] / nleaves / jnp.maximum(nb, 1e-30), hp["cap"]), g3s), nhbl)
        if g3mode.endswith("_mono"):
            g3 = tmap(lambda a, old: jnp.where(n == 0, a, jnp.minimum(a, old)), g3, st["g3"])
        y = tmap(lambda yy, uu: (1 - D) * yy + uu, st["y"], g)
        new.update(u=us, ys=ys, eS=eS, eSl=eSl, y=y, g3=g3,
                   eY=tmap(lambda nb: jnp.where(hp.get("pt", 0.0) > 0.5, nb, nhb), nhbl), eC=tmap(lambda _: jnp.ones(()), st["eC"]),
                   eF=tmap(lambda _: jnp.ones(()) / (2 * hp["lr"]), st["eF"]))
        return tmap(lambda p, gg, yy, g3_: p - hp["lr"] * gg - g3_ * yy, params, g, y, g3), new
    if opt == "laprop_dana":
        u, u1, u2, ut = (tmap(lambda a, dd: a / dd, x, den) for x in (g, g1, g2h, gt))
    else:
        u, u1, u2, ut = g, g1, g2h, gt
    D = hp["delta"] / (hp["delta"] + nf)
    rho = jnp.minimum(1.0, jnp.maximum(D, 10.0 / (nf + 1.0)))
    y = tmap(lambda yy, uu: (1 - D) * yy + uu, st["y"], u)
    trC = tmap(lambda a, b: 0.25 * B * jnp.sum((a - b) ** 2), u1, u2)
    trF = tmap(lambda a: B * jnp.sum(a * a), ut)
    eY = tmap(lambda e, yy: (1 - rho) * e + rho * jnp.sum(yy * yy), st["eY"], y)
    eC = tmap(lambda e, t: (1 - rho) * e + rho * t, st["eC"], trC)
    eF = tmap(lambda e, t: (1 - rho) * e + rho * t, st["eF"], trF)
    g2 = hp["lr"]
    cap = hp["cap"]
    if g3mode == "oracle":
        g3s = g2 * jnp.minimum(hp["c"] * (1.0 + nf) ** (-hp["kappa"]), cap)
        g3 = tmap(lambda _: g3s, eY)
    elif g3mode in ("adapt_global", "adapt_global_mono"):
        tot = lambda t: jax.tree.reduce(lambda a, b: a + b, t)
        nhb = 2 * g2 * tot(eY) * tot(eF) / jnp.maximum(tot(eC), 1e-30)
        g3s = g2 * jnp.minimum(hp["s"] / jnp.maximum(nhb, 1e-30), cap)
        g3 = tmap(lambda _: g3s, eY)
    else:   # 'adapt' or 'adapt_mono' (per leaf)
        s_leaf = hp["s"] / nleaves

        def rule(ey, ec, ef):
            nhb = 2 * g2 * ey * ef / jnp.maximum(ec, 1e-30)
            return g2 * jnp.minimum(s_leaf / jnp.maximum(nhb, 1e-30), cap)
        g3 = tmap(rule, eY, eC, eF)
    if g3mode.endswith("_mono"):
        g3 = tmap(lambda new_, old: jnp.where(n == 0, new_, jnp.minimum(old, new_)), g3, st["g3"])
    new.update(y=y, eY=eY, eC=eC, eF=eF, g3=g3)
    # schedule (hp T > 0): the whole update (instantaneous + momentum) is scaled by sigma(n); the g3/g2 law is unchanged
    return tmap(lambda p, uu, yy, gg3: p - sig * (g2 * uu + gg3 * yy), params, u, y, g3), new


def _leftover(a, c, NTc, budget):
    """v2 leftover rule: budget unspent once every tensor with positive signal is at the cap goes uniformly to the
    tensors without signal."""
    rem = budget - jnp.sum(a * NTc)
    z = c <= 0
    nz = jnp.sum(jnp.where(z, NTc, 0.0))
    fill = jnp.minimum(1.0, rem / jnp.maximum(nz, 1e-30))
    return jnp.where(z & (rem > 1e-9 * budget) & (nz > 0), fill, a)


def _waterfill(c, NTc, budget, iters=120):
    spent = lambda nu: jnp.sum(jnp.minimum(1.0, nu * c) * NTc)

    def bis(_, lh):
        lo_, hi_ = lh
        mid = jnp.sqrt(jnp.maximum(lo_, 1e-30) * hi_)
        ok = spent(mid) <= budget
        return jnp.where(ok, mid, lo_), jnp.where(ok, hi_, mid)
    lo, _ = jax.lax.fori_loop(0, iters, bis, (jnp.zeros(()), jnp.full((), 1e30)))
    a = jnp.minimum(1.0, lo * c)
    a = jnp.where(jnp.sum(c) > 0, a, jnp.minimum(budget / jnp.sum(NTc), 1.0))     # no signal anywhere: global
    return jnp.where(jnp.sum(NTc) <= budget, 1.0, a)


def _adana_slq_step(params, st, new, g, g1, g2h, hp, B, nf, n, sig):
    t = nf + 1.0
    D = hp["delta"] / (hp["delta"] + t)
    m = tmap(lambda mm, gg: (1 - D) * mm + D * gg, st["m"], g)
    v = tmap(lambda vv, gg: (1 - D) * vv + D * gg * gg, st["v"], g)
    den = tmap(lambda vv: jnp.sqrt(vv) + hp["eps"], v)
    g2n = hp["lr"]                                        # peak step: the schedule multiplies the whole update
    th = jnp.maximum(st["qn"], 0.0)
    fq = g2n * th / (g2n * th + D)
    N = jnp.dot(fq, st["qw"])
    NT = fq @ st["qT"]
    Bseq = B
    Be = B * jnp.maximum(st["qteff"], 1.0)
    # signal fraction from the split halves, raw coordinates, window ~ n/3
    rho_ = jnp.minimum(1.0, 3.0 / (nf + 1.0))
    h1, h2 = jax.tree.leaves(g1), jax.tree.leaves(g2h)
    xg = sum(jnp.sum(a * b) for a, b in zip(h1, h2)); dg = sum(jnp.sum((a - b) ** 2) for a, b in zip(h1, h2))
    eX = jnp.where(n == 0, xg, (1 - rho_) * st["eX"] + rho_ * xg)
    eDg = jnp.where(n == 0, dg, (1 - rho_) * st["eDg"] + rho_ * dg)
    kb = hp.get("gnsb", 0.0)
    bnr = 4 * jnp.maximum(eX, 0.0) / jnp.maximum(eDg, 1e-30)
    gmul = jnp.where(kb > 0, jnp.clip(kb * bnr / Bseq, 0.0, 1.0), 1.0)
    S = hp["s"] * Be * gmul
    # buffer signal (instantaneous buffer energy in z, global temperature from the z-weighted split noise)
    leaves, tdef = jax.tree.flatten(params)
    L = len(leaves)
    ld, lm = jax.tree.leaves(den), jax.tree.leaves(m)
    dz = jnp.stack([jnp.sum((a - b) ** 2 / dd) for a, b, dd in zip(h1, h2, ld)])
    eDz0 = jnp.stack(jax.tree.leaves(st["eDz"]))
    eDz = jnp.where(n == 0, dz, (1 - rho_) * eDz0 + rho_ * dz)
    mz2 = jnp.stack([jnp.sum(mm * mm / dd) for mm, dd in zip(lm, ld)])
    NTc = jnp.maximum(NT, 1e-2)
    temp = (Be * jnp.sum(eDz) / 4.0) / jnp.maximum(jnp.dot(th, st["qw"]), 1e-30)
    G = mz2 - D * D * (NTc / Be) * temp / (2.0 * g2n)
    alloc = hp.get("alloc", 0.0)
    c = jnp.maximum(G, 0.0) / NTc
    a_wf = _waterfill(c, NTc, S)
    a_wf = jnp.where(alloc > 3.5, _leftover(a_wf, c, NTc, S), a_wf)
    r_glob = jnp.minimum(S / jnp.maximum(N, 1e-12), hp["cap"]) * jnp.ones(L)
    ratio = jnp.where(alloc > 2.5, jnp.minimum(a_wf, hp["cap"]), r_glob) * st["qok"]
    alpha_adana = hp.get("c", 0.0) * (1.0 + (1.0 + t) ** (1.0 - hp.get("kappa", 0.0)))
    alpha = jnp.where(alloc < -0.5, alpha_adana * jnp.ones(L), ratio / D)
    ratio = jnp.where(alloc < -0.5, alpha * D, ratio)
    at = jax.tree.unflatten(tdef, [alpha[i] for i in range(L)])
    new.update(m=m, v=v, eX=eX, eDg=eDg, gmul=gmul, eDz=jax.tree.unflatten(tdef, [eDz[i] for i in range(L)]),
               eG=jax.tree.unflatten(tdef, [G[i] for i in range(L)]),
               g3=jax.tree.unflatten(tdef, [g2n * ratio[i] for i in range(L)]), g2e=sig * g2n,
               eY=jax.tree.unflatten(tdef, [N / Be for _ in range(L)]), eC=tmap(lambda _: jnp.ones(()), st["eC"]),
               eF=tmap(lambda _: jnp.ones(()) / (2 * hp["lr"]), st["eF"]))
    return tmap(lambda p, gg, mm, dd, a_: p - sig * g2n * (gg + a_ * mm) / dd, params, g, m, den, at), new
