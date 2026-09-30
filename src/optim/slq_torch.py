"""Stochastic Lanczos quadrature (SLQ) of the preconditioned Gauss-Newton matrix, for the adana-slq momentum rule.

The quantity needed by the rule is the effective dimension of the preconditioned curvature at the buffer's time scale,
    N(Delta) = tr f(H_z),   f(x) = g x / (g x + Delta),   H_z = r GN r,   r = (sqrt(v) + eps)^{-1/2}
(z = r^{-1} theta are the coordinates in which an ADana step is plain SGD with step g), and its split over parameter
tensors N_T(Delta) = tr(P_T f(H_z)).  One Lanczos run per Rademacher probe z gives a Gauss quadrature
    z^T f(H) z ~= sum_k w_k f(theta_k),   w_k = |z|^2 q_{1k}^2,
and the per-tensor weights w_{T,k} = |z| q_{1k} sum_j q_{jk} <z_T, v_j>, accumulated during the recurrence (no basis is
stored).  All Delta and g are then evaluated from the stored (nodes, weights) at no cost.  A Gauss-Radau rule with a
node just below 0 gives a certified lower bound for f(x) = x/(x+c) (Gauss is an upper bound); the recurrence stops when
the bracket at the target Delta is within eps.

Gauss-Newton products use central finite differences of the logits (no forward-mode AD, so SDPA/flash kernels and
torch.compile'd training are unaffected):  J u ~= [f(theta + e u) - f(theta - e u)] / (2e)  in fp32, followed by the
softmax cross-entropy Hessian (diag(p) - p p^T per token, divided by the token count) and one backward pass.
"""
import math

import numpy as np
import torch
import torch.distributed as dist
from torch.func import functional_call


# ------------------------------------------------------------------ small list-of-tensors helpers
def _dot(a, b):
    return sum((x.double() * y.double()).sum() for x, y in zip(a, b))


def _axpy_(y, a, x):                       # y <- y + a x (in place)
    for yy, xx in zip(y, x):
        yy.add_(xx, alpha=a)


# ------------------------------------------------------------------ Gauss-Newton operator
class GaussNewtonOperator:
    """u (list of tensors shaped like params) -> r * GN(r * u), GN of the mean-over-tokens CE on a fixed batch.

    model: the raw (unwrapped, uncompiled) module; its forward(idx, targets, get_logits=True) returns dict(logits=...).
    names/params: the trainable parameters, in the optimizer's order.  r: list of tensors (or None for identity).
    Distributed: each rank uses its own batch; products are averaged across ranks (identical Lanczos on all ranks).
    """

    def __init__(self, model, names, params, x, y, r=None, fd_rel=1e-3, chunk=2, mode="jvp", cache=True):
        self.model, self.names, self.params = model, names, params
        self.x, self.y, self.r = x, y, r
        self.fd_rel, self.chunk = fd_rel, chunk
        self.mode, self.cache = mode, cache
        self._cache = {}
        self.dtype = torch.float64 if params[0].dtype == torch.float64 else torch.float32
        self.pnorm = math.sqrt(float(_dot([p.detach() for p in params], [p.detach() for p in params])))
        self.ntok = float((y >= 0).sum())
        self.world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if self.world > 1:
            t = torch.tensor([self.ntok], device=x.device, dtype=torch.float64)
            dist.all_reduce(t)
            self.ntok_global = float(t) / self.world
        else:
            self.ntok_global = self.ntok

    def _logits(self, pdict, xc, yc):
        return self.model_call(pdict, xc, yc)["logits"].to(self.dtype)

    def model_call(self, pdict, xc, yc):
        with torch.autocast(device_type=xc.device.type, enabled=False):
            if pdict is None:
                return self.model(xc, targets=yc, get_logits=True)
            return functional_call(self.model, pdict, (xc,), dict(targets=yc, get_logits=True))

    # ---- exact products: J u by forward-mode AD (math SDPA backend), J^T by a cached reverse-mode closure
    def _f(self, xc, yc):
        names, model = self.names, self.model

        def f(*ps):
            return functional_call(model, dict(zip(names, ps)), (xc,), dict(targets=yc, get_logits=True))["logits"].to(
                self.dtype)
        return f

    def _chunk_state(self, i, xc, yc):
        if i in self._cache:
            return self._cache[i]
        prim = tuple(p.detach() for p in self.params)
        with torch.enable_grad():
            logits, vjp_fn = torch.func.vjp(self._f(xc, yc), *prim)
        st = (torch.softmax(logits.detach(), dim=-1), vjp_fn)
        if self.cache:
            self._cache[i] = st
        return st

    def _vjp_ckpt(self, xc, yc, hv):
        """J^T hv by ordinary reverse mode with per-block activation checkpointing (no cached closure): the backward
        keeps only block inputs plus one block's residuals, instead of every layer's fp32 attention matrices."""
        cfg = getattr(self.model, "config", None)
        old = getattr(cfg, "activation_checkpointing", None)
        if cfg is not None:
            cfg.activation_checkpointing = True
        try:
            with torch.enable_grad():
                logits = self._f(xc, yc)(*self.params)
                return torch.autograd.grad(logits, self.params, grad_outputs=hv, allow_unused=True)
        finally:
            if cfg is not None:
                cfg.activation_checkpointing = old

    def _call_jvp(self, d):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        out = [torch.zeros_like(p) for p in self.params]
        prim = tuple(p.detach() for p in self.params)
        tang = tuple(di.to(p.dtype) for di, p in zip(d, self.params))
        B = self.x.shape[0]
        with torch.autocast(device_type=self.x.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
            for k, i in enumerate(range(0, B, self.chunk)):
                xc, yc = self.x[i:i + self.chunk], self.y[i:i + self.chunk]
                if self.cache:
                    p, vjp_fn = self._chunk_state(k, xc, yc)
                    with torch.no_grad():
                        _, ju = torch.func.jvp(self._f(xc, yc), prim, tang)
                else:
                    with torch.no_grad():
                        logits, ju = torch.func.jvp(self._f(xc, yc), prim, tang)
                    p = torch.softmax(logits, dim=-1); del logits
                hv = p * ju - p * (p * ju).sum(-1, keepdim=True)
                hv = hv * (yc >= 0).unsqueeze(-1).to(hv.dtype) / self.ntok
                gs = vjp_fn(hv) if self.cache else self._vjp_ckpt(xc, yc, hv)
                for o, gg in zip(out, gs):
                    if gg is not None:
                        o.add_(gg.to(o.dtype))
                del ju, hv, gs
        return out

    def __call__(self, u):
        d = [ui * ri for ui, ri in zip(u, self.r)] if self.r is not None else u
        if self.mode == "jvp":
            out = self._call_jvp(d)
            if self.world > 1:
                flat = torch.cat([o.reshape(-1) for o in out])
                dist.all_reduce(flat)
                flat /= self.world
                k = 0
                for o in out:
                    n = o.numel(); o.copy_(flat[k:k + n].view_as(o)); k += n
            return [o * ri for o, ri in zip(out, self.r)] if self.r is not None else out
        dn = math.sqrt(float(_dot(d, d)))
        out = [torch.zeros_like(p) for p in self.params]
        if dn == 0.0:
            return out
        e = self.fd_rel * max(self.pnorm, 1e-12) / dn
        plus = {n: (p.detach() + e * di) for n, p, di in zip(self.names, self.params, d)}
        minus = {n: (p.detach() - e * di) for n, p, di in zip(self.names, self.params, d)}
        B = self.x.shape[0]
        for i in range(0, B, self.chunk):
            xc, yc = self.x[i:i + self.chunk], self.y[i:i + self.chunk]
            with torch.no_grad():
                ju = (self._logits(plus, xc, yc) - self._logits(minus, xc, yc)) / (2 * e)
            with torch.enable_grad():
                logits = self.model_call(None, xc, yc)["logits"].to(self.dtype)
                p = torch.softmax(logits.detach(), dim=-1)
                hv = p * ju - p * (p * ju).sum(-1, keepdim=True)
                hv = hv * (yc >= 0).unsqueeze(-1).float() / self.ntok
                gs = torch.autograd.grad(logits, self.params, grad_outputs=hv, allow_unused=True)
            for o, gg in zip(out, gs):
                if gg is not None:
                    o.add_(gg.to(o.dtype))
            del ju, logits, p, hv, gs
        if self.world > 1:
            flat = torch.cat([o.reshape(-1) for o in out])
            dist.all_reduce(flat)
            flat /= self.world
            k = 0
            for o in out:
                n = o.numel(); o.copy_(flat[k:k + n].view_as(o)); k += n
        if self.r is not None:
            out = [o * ri for o, ri in zip(out, self.r)]
        return out


# ------------------------------------------------------------------ quadrature (host side, numpy)
def _tridiag(al, be):
    m = len(al)
    T = np.zeros((m, m))
    i = np.arange(m); T[i, i] = al
    T[i[:-1], i[:-1] + 1] = be; T[i[:-1] + 1, i[:-1]] = be
    return T


def gauss_rule(al, be, C, zn):
    """Gauss rule from Lanczos coefficients: nodes (m,), weights (m,) = zn^2 q1^2, block weights (m, L)."""
    m = len(al)
    th, Q = np.linalg.eigh(_tridiag(al, be[:m - 1]))
    q1 = Q[0, :]
    w = zn ** 2 * q1 ** 2
    wT = zn * q1[:, None] * (Q.T @ C[:m, :])
    return th, w, wT


def radau_lower(al, be, zn):
    """Gauss-Radau rule with a prescribed node just below 0 (lower bound for f(x) = x/(x+c) on a PSD operator)."""
    m = len(al)
    T = _tridiag(al, be[:m - 1])
    thG = np.linalg.eigvalsh(T)
    a = -1e-8 * max(np.abs(thG).max(), 1e-30)
    rhs = np.zeros(m); rhs[-1] = be[m - 1] ** 2
    dlt = np.linalg.lstsq(T - a * np.eye(m), rhs, rcond=None)[0]
    TR = np.zeros((m + 1, m + 1)); TR[:m, :m] = T
    TR[m, m - 1] = TR[m - 1, m] = be[m - 1]; TR[m, m] = a + dlt[-1]
    thR, QR = np.linalg.eigh(TR)
    return thR, zn ** 2 * QR[0, :] ** 2


def fsum(th, w, g, D):
    th = np.maximum(th, 0.0)
    return float(np.sum(w * g * th / (g * th + D)))


# ------------------------------------------------------------------ Lanczos driver
@torch.no_grad()
def slq(op, like, m_max=128, probes=1, chunk=8, eps=0.03, g=1.0, D_check=None, generator=None):
    """Run `probes` Lanczos recurrences (no reorthogonalization; Gauss quadrature of a smooth f is robust to it) of up to
    m_max steps on `op` (list-of-tensors -> list-of-tensors, symmetric PSD).  Stops a probe early when the Gauss / Gauss-
    Radau bracket on N(D) for every D in D_check is within relative eps.  For f(x) = g x / (g x + D) the Gauss rule is an
    upper bound on z^T f(H) z at EVERY m (all even derivatives of f are negative) and Gauss-Radau with a node at 0 is a
    lower bound, so a recurrence stopped at m_max still returns a certified (conservative) N.  Returns dict with nodes
    (P, m), weights (P, m), block weights (P, m, L) (per-probe arrays padded with zero weight), m used per probe, the final
    bracket gap, and the bracket itself: N_hi (Gauss) and N_lo (Radau), each (P, len(D_check))."""
    L = len(like)
    D_check = [] if D_check is None else list(D_check)
    out_th, out_w, out_wT, used, gaps, hists, corrs, convs = [], [], [], [], [], [], [], []
    n_hi, n_lo = [], []
    for _ in range(probes):
        z = [(torch.randint(0, 2, x.shape, device=x.device, generator=generator, dtype=torch.int8).to(x.dtype) * 2 - 1)
             for x in like]
        zn = math.sqrt(float(_dot(z, z)))
        v = [zi / zn for zi in z]
        vp = [torch.zeros_like(zi) for zi in z]
        bprev = 0.0
        al, be = np.zeros(m_max), np.zeros(m_max)
        C = np.zeros((m_max, L))
        m_used, gap = m_max, np.nan
        hist = []
        converged = not D_check
        for j in range(m_max):
            C[j] = [float((a.double() * c.double()).sum()) for a, c in zip(z, v)]
            w = op(v)
            if bprev > 0:
                _axpy_(w, -bprev, vp)
            a = float(_dot(v, w))
            _axpy_(w, -a, v)
            b = math.sqrt(max(float(_dot(w, w)), 0.0))
            al[j], be[j] = a, b
            if b < 1e-12 * max(abs(a), 1e-30):          # invariant subspace found (quadrature is exact)
                m_used = j + 1
                converged = True
                break
            vp, v, bprev = v, [wi / b for wi in w], b
            if D_check and (j + 1) % chunk == 0 and j + 1 < m_max:
                # stop when the Gauss estimate has converged in m (relative change over the last chunk < eps) or the
                # certified Gauss / Gauss-Radau bracket is within eps.  (The Radau lower bound is loose when the spectrum
                # spans many decades, so the bracket alone rarely closes on LM curvature.)
                m = j + 1
                thG, wG, _ = gauss_rule(al[:m], be[:m], C, zn)
                NG = [fsum(thG, wG, g, D) for D in D_check]
                hist.append((m, NG))
                thR, wR = radau_lower(al[:m], be[:m], zn)
                gap = max((a_ - fsum(thR, wR, g, D)) / max(a_, 1e-30) for a_, D in zip(NG, D_check))
                conv = len(hist) >= 2 and max(abs(a_ - b_) / max(a_, 1e-30) for a_, b_ in zip(NG, hist[-2][1])) < eps
                if gap < eps or conv:
                    m_used = m
                    converged = True
                    break
        th, w, wT = gauss_rule(al[:m_used], be[:m_used], C, zn)
        if not converged and D_check and hist:          # reached m_max: converged if the last chunk barely moved
            NGf = [fsum(th, w, g, D) for D in D_check]
            converged = max(abs(a_ - b_) / max(a_, 1e-30) for a_, b_ in zip(NGf, hist[-1][1])) < eps
        convs.append(converged)
        # Aitken extrapolation of the (monotonically decreasing, roughly geometric in m) Gauss estimates when the
        # recurrence stopped before converging: correction factor N_inf / N_m at each checked Delta, clipped to [0.25, 1]
        corr = []
        if D_check:
            NG = [fsum(th, w, g, D) for D in D_check]
            seq = [h for h in hist if h[0] < m_used] + [(m_used, NG)]
            for k in range(len(D_check)):
                if len(seq) >= 3:
                    n1, n2, n3 = seq[-3][1][k], seq[-2][1][k], seq[-1][1][k]
                    d1, d2 = n2 - n1, n3 - n2
                    if d1 < 0 and d2 < 0 and 0 < d2 / d1 < 1:
                        corr.append(float(np.clip((n3 - d2 * d2 / (d2 - d1)) / max(n3, 1e-30), 0.25, 1.0)))
                        continue
                corr.append(1.0)
        corrs.append(float(np.mean(corr)) if corr else 1.0)
        if D_check:
            thR, wR = radau_lower(al[:m_used], be[:m_used], zn)
            hi = [fsum(th, w, g, D) for D in D_check]
            lo = [fsum(thR, wR, g, D) for D in D_check]
            gap = max((h - l) / max(h, 1e-30) for h, l in zip(hi, lo))
            n_hi.append(hi); n_lo.append(lo)
        pad = m_max - m_used
        out_th.append(np.pad(th, (0, pad)))
        out_w.append(np.pad(w, (0, pad)))
        out_wT.append(np.pad(wT, ((0, pad), (0, 0))))
        used.append(m_used); gaps.append(gap); hists.append(hist)
    return dict(nodes=np.stack(out_th), weights=np.stack(out_w) / probes, block=np.stack(out_wT) / probes,
                m_used=np.array(used), gap=np.array(gaps), lam_max=float(np.max(np.stack(out_th))), hist=hists,
                aitken=np.array(corrs), converged=np.array(convs),
                N_hi=np.array(n_hi).reshape(len(n_hi), -1), N_lo=np.array(n_lo).reshape(len(n_lo), -1))


# ------------------------------------------------------------------ token correlation length (effective tokens / sequence)
def effective_tokens(model, params, x, y, generator=None):
    """T_eff = T' E_tok / E_seq, with E the squared distance between the gradients of two random halves of the batch,
    split by SEQUENCES (E_seq) or by TOKENS (E_tok, balanced random mask).  The signal cancels in both differences;
    E_seq / E_tok = Var(sum_t g_it) / (T' Var(g_it)) is the number of mutually correlated tokens, so independent tokens
    give T_eff = T' and fully correlated sequences give T_eff = 1."""
    B, T = y.shape
    valid = (y >= 0).float()

    def grad_w(wts):
        with torch.enable_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            logits = model(x, targets=y, get_logits=True)["logits"].float()
            ce = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1).clamp_min(0),
                                                   reduction="none").view(B, T)
            loss = (ce * wts).sum() / wts.sum().clamp_min(1.0)
            return torch.autograd.grad(loss, params, allow_unused=True)

    def sqdist(ga, gb):
        return sum(float(((a - b).double() ** 2).sum()) for a, b in zip(ga, gb) if a is not None and b is not None)

    half = B // 2
    wA = torch.zeros(B, T, device=x.device); wA[:half] = 1.0
    E_seq = sqdist(grad_w(wA * valid), grad_w((1 - wA) * valid))
    perm = torch.randperm(B * T, device=x.device, generator=generator)
    mA = torch.zeros(B * T, device=x.device); mA[perm[: B * T // 2]] = 1.0
    mA = mA.view(B, T)
    E_tok = sqdist(grad_w(mA * valid), grad_w((1 - mA) * valid))
    return float(T * E_tok / max(E_seq, 1e-30)), E_seq, E_tok
