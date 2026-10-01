"""Spectral (Shampoo-type) preconditioning of the hidden matrices in LaProp order, with DANA-SLQ momentum, and the
standard spectral baselines.  Plan: ~/.claude/plans/can-you-build-a-frolicking-tulip.md.

Hidden matrices (x @ W orientation, W in R^{d_in x d_out}): the leaves q, k, v, o, w1, w2 of every block.  Their
statistics are short EMAs  L = E[G G^T] (d_in x d_in) and R = E[G^T G] (d_out x d_out)  (beta = hp sb), bias-corrected
and eigendecomposed every KEIG steps by the trainer (eig_refresh, outside the vmapped step so that the refresh is a real
branch).  A step uses the stored eigendecomposition, i.e. statistics from before its own gradient.  Every other leaf
uses RMSProp (second moment v, beta2 = hp b2, bias-corrected), which is also the fallback before the first refresh.

  P^{-1} G   = c  L^{-1/4} G R^{-1/4}          spectral; c = rms sqrt(max(d_in, d_out)) gives RMSProp's unit RMS
  P^{-1/2} U = c^{1/2} L^{-1/8} U R^{-1/8}     z coordinates of the SLQ quadrature (R GN R)
  P U        = c^{-1} L^{1/4} U R^{1/4}         |y|_z^2 = <y, P y> for the buffer signal
  rotated-diagonal (SOAP basis): P^e U = Q_L [(Q_L^T U Q_R) (sqrt(v') + eps)^e] Q_R^T, v' the second moment in the basis
Eigenvalues are damped by seps * mean(eigenvalues).  Per-head (hp perhead = 1): the output-side factor R of q and k is
restricted to its hp nh diagonal head blocks.

Optimizers
  spd_slq        u = P^{-1} g (spectral); DANA sum buffer y = (1-D) y + u; theta -= sigma lr (u + r_T y), with r_T from
                 the SLQ budget sum_T r_T N_T <= s B_eff mu (alloc 0 global, 3 water-filling, 4 water-filling v2)
  soap_dana_slq  the same with the rotated-diagonal P
  muon_dana      u = c msign(g) (Newton-Schulz) on the hidden matrices, DANA momentum; budget from the spectral-P
                 quadrature (a heuristic arm: msign is not a linear preconditioner)
  soap           Adam in the eigenbasis of (L, R); first moment in the original coordinates
  shampoo_m      u = P^{-1} g (spectral), heavy-ball momentum in LaProp order m = b1 m + (1-b1) u, theta -= sigma lr m_hat
  muon           Muon on the hidden matrices (Nesterov momentum hp mu, msign, scale 0.2 rms sqrt(max dim)); Adam elsewhere
"""
import math

import jax
import jax.numpy as jnp

SPECTRAL = ("spd_slq", "soap_dana_slq", "muon_dana", "soap", "shampoo_m", "muon")
SLQ_SPECTRAL = ("spd_slq", "soap_dana_slq", "muon_dana")
EIG_OPTS = ("spd_slq", "soap_dana_slq", "muon_dana", "soap", "shampoo_m")
ROTATED = ("soap", "soap_dana_slq")
KEIG = 10
HIDDEN = ("q", "k", "v", "o", "w1", "w2")
PERHEAD = ("q", "k")


def leaf_kinds(tree):
    """[(is_hidden_matrix, per_head_eligible)] per leaf, from the leaf names."""
    out = []
    for path, x in jax.tree_util.tree_flatten_with_path(tree)[0]:
        name = str(getattr(path[-1], "key", path[-1]))
        out.append((x.ndim == 2 and name in HIDDEN, name in PERHEAD))
    return out


def _pw(d, e, seps):
    d = jnp.maximum(d, 0.0)
    return (d + seps * jnp.mean(d) + 1e-30) ** e


def sandwich(QL, aL, U, QR, aR):
    """Q_L diag(aL) Q_L^T U Q_R diag(aR) Q_R^T."""
    return (QL * aL) @ (QL.T @ U @ QR) @ (QR * aR).T


def msign(G, steps=5):
    """Orthogonalization by the quintic Newton-Schulz iteration of Muon (as zeropower_via_newtonschulz5)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    tr = G.shape[0] > G.shape[1]
    X = G.T if tr else G
    X = X / (jnp.linalg.norm(X) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


def pe(rotated, st, hp, mat, i, U, e, v_i, bc2):
    """P^e U for leaf i (e in {-1, -1/2, 1}); v_i the leaf's second moment (rotated for the SOAP basis)."""
    den = jnp.sqrt(v_i / bc2) + hp["eps"]
    if not mat:
        return U * den ** e
    if rotated:
        QL, QR = st["QL"][i], st["QR"][i]
        return QL @ ((QL.T @ U @ QR) * den ** e) @ QR.T
    c = hp["rms"] * math.sqrt(max(U.shape))
    sp = c ** (-e) * sandwich(st["QL"][i], _pw(st["dL"][i], e / 4, hp["seps"]), U, st["QR"][i], _pw(st["dR"][i], e / 4, hp["seps"]))
    return jnp.where(st["eig_ok"] > 0.5, sp, U * den ** e)


def init(opt, params, Q=1):
    z = jax.tree.map(jnp.zeros_like, params)
    zs = jax.tree.map(lambda x: jnp.zeros((), x.dtype), params)
    leaves = jax.tree.leaves(params)
    kinds = leaf_kinds(params)
    st = dict(n=jnp.zeros((), jnp.int32), v=z, eig_ok=jnp.zeros((), jnp.float32))
    if opt in EIG_OPTS:
        Ls, Rs, QL, dL, QR, dR = [], [], [], [], [], []
        for x, (mat, _) in zip(leaves, kinds):
            a, b = x.shape if mat else (1, 1)
            Ls.append(jnp.zeros((a, a))); Rs.append(jnp.zeros((b, b)))
            QL.append(jnp.eye(a)); dL.append(jnp.zeros((a,))); QR.append(jnp.eye(b)); dR.append(jnp.zeros((b,)))
        st.update(L=Ls, R=Rs, QL=QL, dL=dL, QR=QR, dR=dR)
    if opt in SLQ_SPECTRAL:
        L = len(leaves)
        f0 = lambda *s: jnp.zeros(s, jnp.float32)
        st.update(y=z, qn=f0(Q), qw=f0(Q), qT=f0(Q, L), qlam=f0(), qok=f0(), qb=f0(), qg2max=f0(), qteff=f0(),
                  eX=f0(), eDg=f0(), gmul=jnp.ones((), jnp.float32), g2e=f0(), g3=zs, eY=zs, eC=zs, eF=zs, eG=zs, eDz=zs)
    if opt in ("soap", "shampoo_m", "muon"):
        st["m"] = z
    if opt == "muon":
        st["mb"] = z
    return st


def eig_refresh(st, hp, kinds):
    """Eigendecompose the bias-corrected statistics (per-head output factor for q, k when hp perhead = 1)."""
    n = st["n"].astype(jnp.float32)
    bc = 1 - hp["sb"] ** jnp.maximum(n, 1.0)
    QL, dL, QR, dR = list(st["QL"]), list(st["dL"]), list(st["QR"]), list(st["dR"])
    for i, (mat, ph) in enumerate(kinds):
        if not mat:
            continue
        Lh, Rh = st["L"][i] / bc, st["R"][i] / bc
        if ph:
            dout = Rh.shape[0]
            hid = jnp.floor(jnp.arange(dout) / (dout / jnp.maximum(hp.get("nh", 4.0), 1.0)))
            Rh = jnp.where(hp.get("perhead", 0.0) > 0.5, Rh * (hid[:, None] == hid[None, :]), Rh)
        dL[i], QL[i] = jnp.linalg.eigh(Lh)
        dR[i], QR[i] = jnp.linalg.eigh(Rh)
    return dict(st, QL=QL, dL=dL, QR=QR, dR=dR, eig_ok=jnp.where(n > 0, 1.0, st["eig_ok"]))


def apply_Rz(opt, st, hp, U):
    """P^{-1/2} U leafwise (the z coordinates of the SLQ operator), with the state's current second moments."""
    leaves, tdef = jax.tree.flatten(U)
    kinds = leaf_kinds(U)
    vl = jax.tree.leaves(st["v"])
    bc2 = 1 - hp["b2"] ** jnp.maximum(st["n"].astype(jnp.float32), 1.0)
    rot = opt in ROTATED
    return jax.tree.unflatten(tdef, [pe(rot, st, hp, kinds[i][0], i, leaves[i], -0.5, vl[i], bc2) for i in range(len(leaves))])


def step(opt, params, st, new, g1, g2h, g, hp, B, nf, n, sig):
    from .optim_tree import _waterfill, _leftover
    pl, tdef = jax.tree.flatten(params)
    kinds = leaf_kinds(params)
    gl = jax.tree.leaves(g)
    Lf = len(pl)
    lr = hp["lr"]
    rot = opt in ROTATED
    # second moments: RMSProp (vectors; fallback for matrices), or the rotated second moment for the SOAP basis
    b2 = hp["b2"]
    vl = jax.tree.leaves(st["v"])
    gsq = lambda i: (lambda X: X * X)(st["QL"][i].T @ gl[i] @ st["QR"][i]) if (kinds[i][0] and rot) else gl[i] * gl[i]
    v = [b2 * vl[i] + (1 - b2) * gsq(i) for i in range(Lf)]
    new["v"] = jax.tree.unflatten(tdef, v)
    bc2 = 1 - b2 ** (nf + 1.0)
    if opt in EIG_OPTS:      # statistics for the NEXT eigen-refresh (the current step uses the stored decomposition)
        sb = hp["sb"]
        new["L"] = [sb * st["L"][i] + (1 - sb) * gl[i] @ gl[i].T if kinds[i][0] else st["L"][i] for i in range(Lf)]
        new["R"] = [sb * st["R"][i] + (1 - sb) * gl[i].T @ gl[i] if kinds[i][0] else st["R"][i] for i in range(Lf)]
    P = lambda i, U, e: pe(rot, st, hp, kinds[i][0], i, U, e, v[i], bc2)
    upd = lambda us: jax.tree.unflatten(tdef, [pl[i] - sig * lr * us[i] for i in range(Lf)])

    if opt == "soap":
        b1 = hp["b1"]; bc1 = 1 - b1 ** (nf + 1.0)
        ml = jax.tree.leaves(st["m"])
        m = [b1 * ml[i] + (1 - b1) * gl[i] for i in range(Lf)]
        new["m"] = jax.tree.unflatten(tdef, m)
        return upd([P(i, m[i] / bc1, -1.0) for i in range(Lf)]), new
    if opt == "shampoo_m":
        b1 = hp["b1"]; bc1 = 1 - b1 ** (nf + 1.0)
        ml = jax.tree.leaves(st["m"])
        m = [b1 * ml[i] + (1 - b1) * P(i, gl[i], -1.0) for i in range(Lf)]
        new["m"] = jax.tree.unflatten(tdef, m)
        return upd([m[i] / bc1 for i in range(Lf)]), new
    if opt == "muon":
        b1, mu = hp["b1"], hp["mu"]; bc1 = 1 - b1 ** (nf + 1.0)
        ml, bl = jax.tree.leaves(st["m"]), jax.tree.leaves(st["mb"])
        mb = [mu * bl[i] + gl[i] for i in range(Lf)]
        m = [b1 * ml[i] + (1 - b1) * gl[i] for i in range(Lf)]
        new["m"], new["mb"] = jax.tree.unflatten(tdef, m), jax.tree.unflatten(tdef, mb)
        us = [0.2 * hp["rms"] * math.sqrt(max(gl[i].shape)) * msign(gl[i] + mu * mb[i]) if kinds[i][0]
              else (m[i] / bc1) / (jnp.sqrt(v[i] / bc2) + hp["eps"]) for i in range(Lf)]
        return upd(us), new

    # ---- DANA-SLQ in LaProp order on the preconditioned gradient
    if opt == "muon_dana":
        u = [hp["rms"] * math.sqrt(max(gl[i].shape)) * msign(gl[i]) if kinds[i][0] else P(i, gl[i], -1.0) for i in range(Lf)]
    else:
        u = [P(i, gl[i], -1.0) for i in range(Lf)]
    D = hp["delta"] / (hp["delta"] + nf)
    yl = jax.tree.leaves(st["y"])
    y = [(1 - D) * yl[i] + u[i] for i in range(Lf)]
    g2n = lr
    th = jnp.maximum(st["qn"], 0.0)
    fq = g2n * th / (g2n * th + D)
    N = jnp.dot(fq, st["qw"])
    NT = fq @ st["qT"]
    Be = B * jnp.maximum(st["qteff"], 1.0)
    rho_ = jnp.minimum(1.0, 3.0 / (nf + 1.0))
    h1, h2 = jax.tree.leaves(g1), jax.tree.leaves(g2h)
    xg = sum(jnp.sum(a * b) for a, b in zip(h1, h2)); dg = sum(jnp.sum((a - b) ** 2) for a, b in zip(h1, h2))
    eX = jnp.where(n == 0, xg, (1 - rho_) * st["eX"] + rho_ * xg)
    eDg = jnp.where(n == 0, dg, (1 - rho_) * st["eDg"] + rho_ * dg)
    kb = hp.get("gnsb", 0.0)
    bnr = 4 * jnp.maximum(eX, 0.0) / jnp.maximum(eDg, 1e-30)
    gmul = jnp.where(kb > 0, jnp.clip(kb * bnr / B, 0.0, 1.0), 1.0)
    S = hp["s"] * Be * gmul
    # buffer signal in z coordinates: |y_T|_z^2 = <y, P y>, split noise <d, P^{-1} d>, global temperature
    dz = jnp.stack([jnp.sum((h1[i] - h2[i]) * P(i, h1[i] - h2[i], -1.0)) for i in range(Lf)])
    eDz0 = jnp.stack(jax.tree.leaves(st["eDz"]))
    eDz = jnp.where(n == 0, dz, (1 - rho_) * eDz0 + rho_ * dz)
    ysq = jnp.stack([jnp.sum(y[i] * P(i, y[i], 1.0)) for i in range(Lf)])
    NTc = jnp.maximum(NT, 1e-2)
    temp = (Be * jnp.sum(eDz) / 4.0) / jnp.maximum(jnp.dot(th, st["qw"]), 1e-30)
    G = D * D * (ysq - (NTc / Be) * temp / (2.0 * g2n))
    alloc = hp.get("alloc", 0.0)
    c = jnp.maximum(G, 0.0) / NTc
    a = _waterfill(c, NTc, S)
    a = jnp.where(alloc > 3.5, _leftover(a, c, NTc, S), a)
    r_glob = jnp.minimum(S / jnp.maximum(N, 1e-12), hp["cap"]) * jnp.ones(Lf)
    ratio = jnp.where(alloc > 2.5, jnp.minimum(a, hp["cap"]), r_glob) * st["qok"]
    un = lambda xs: jax.tree.unflatten(tdef, list(xs))
    new.update(y=un(y), eX=eX, eDg=eDg, gmul=gmul, eDz=un(eDz[i] for i in range(Lf)), eG=un(G[i] for i in range(Lf)),
               g3=un(g2n * ratio[i] for i in range(Lf)), g2e=sig * g2n, eY=un(N / Be for _ in range(Lf)),
               eC=jax.tree.map(lambda _: jnp.ones(()), st["eC"]), eF=jax.tree.map(lambda _: jnp.ones(()) / (2 * lr), st["eF"]))
    return upd([u[i] + ratio[i] * y[i] for i in range(Lf)]), new
