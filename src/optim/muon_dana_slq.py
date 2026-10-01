"""Muon-DANA-SLQ: DANA-SLQ long momentum on momentum-free Muon (d-muon's update), with the update treated as the
stochastic gradient.

Update direction (as d-muon, src/optim/muon.py, without its momentum):
  * matrices with p.size(0) < 10000 (all hidden matrices; the fused QKV is one matrix; not wte / lm_head):
        u = c msign(g),  c = matched_rms sqrt(max(A, B)),  msign = 5 quintic Newton-Schulz steps in bf16
  * every other tensor:  u = g / (sqrt(v_hat) + eps),  v an EMA of g^2 (beta2; d-muon's AdamW without beta1)
Step (LaProp order, ADana's EMA convention m = Delta_t y):
    m <- (1 - Delta_t) m + Delta_t u,   theta <- theta - gamma(t) [u + (r_T / Delta_t) m]   (+ ADana weight decay)

Budget.  The DANA-SLQ rule sum_T r_T N_T <= s B_eff mu needs the curvature seen by the iteration.  Treating u(theta)
as the stochastic gradient, that is the Jacobian of the update map, J = A GN, with A = D u(g):
    hidden matrices  A = c D polar(G): with G = U S V^T (wide orientation), E = U X V^T + U Y V_perp^T,
                     A E = c (U [K o skew(X)] V^T + U S^-1 Y V_perp^T),  K_ij = 2/(s_i + s_j);
                     symmetric PSD, and zero on pure rescalings of the singular values (msign is scale-invariant).
                     Singular values are floored at polar_floor |G|_F (1/floor ~ 3.4445^5, the slope at 0 of the
                     Newton-Schulz map, which bounds its derivative).
    other tensors    A = diag (sqrt(v_hat) + eps)^-1
J is not symmetric, so the quadrature runs on the similar symmetric A^{1/2} GN A^{1/2} (same spectrum; Lanczos on J
itself is off by up to ~3x in the toy, drivers/jac_diag.py).  A is taken at the refresh step's gradient (the global
batch).  The dynamic range of A^{1/2} is at most ~(1/floor)^{1/2} ~ 22, so the refresh is numerically benign.
Allocation: 'global', 'typed_v2' or 'independent'.  The water-filling rules need the buffer signal G_T in the metric
of J, which msign does not provide (its output lies in the null space of its own derivative), so they are refused.
"""
import math

import torch

from .adana_slq import ADanaSLQ
from .muon import zeropower_via_newtonschulz5


def _polar_factors(G, floor):
    """SVD pieces of G (wide orientation; at least float32) for D polar(G); singular values floored at floor |G|_F."""
    tall = G.shape[0] > G.shape[1]
    Gw = G.T if tall else G
    Gw = Gw if Gw.dtype in (torch.float32, torch.float64) else Gw.float()
    U, s, Vh = torch.linalg.svd(Gw, full_matrices=False)
    return dict(U=U, s=s.clamp_min(floor * float(torch.linalg.norm(Gw))), Vh=Vh, tall=tall)


def _polar_pow(f, E, power):
    """A^power E for A = D polar(G), power 1 or 1/2 (see the module docstring)."""
    U, s, Vh = f["U"], f["s"], f["Vh"]
    Ew = (E.T if f["tall"] else E).to(U.dtype)
    EV = Ew @ Vh.T
    X = U.T @ EV
    Ep = Ew - EV @ Vh
    K = (2.0 / (s[:, None] + s[None, :])) ** power
    out = U @ (K * (0.5 * (X - X.T))) @ Vh + U @ (s.pow(-power)[:, None] * (U.T @ Ep))
    return out.T if f["tall"] else out


class MuonDanaSLQ(ADanaSLQ):
    def __init__(self, params, lr=1.0, delta=8.0, epsilon=1e-8, weight_decay=0.0, wd_decaying=False, wd_ts=1.0,
                 beta2=0.999, ns_steps=5, matched_rms=0.2, polar_floor=2e-3, **slq_kw):
        alloc = slq_kw.get("alloc", "global")
        if alloc not in ("global", "typed_v2", "independent"):
            raise ValueError(f"muon-dana-slq: allocation '{alloc}' needs the buffer signal; use global, typed_v2 or "
                             "independent")
        super().__init__(params, lr=lr, delta=delta, epsilon=epsilon, weight_decay=weight_decay, clipsnr=None,
                         wd_decaying=wd_decaying, wd_ts=wd_ts, **slq_kw)
        self.beta2, self.ns_steps = float(beta2), int(ns_steps)
        self.matched_rms, self.polar_floor = float(matched_rms), float(polar_floor)
        self.use_muon = [p.ndim >= 2 and p.size(0) < 10000 for p in self.plist]

    def _scale(self, p):
        return self.matched_rms * math.sqrt(max(p.shape[:2]))

    # ---------------------------------------------------------------- SLQ hooks
    def _slq_r(self):
        """R = A^{1/2}: c^{1/2} D polar(G)^{1/2} at the current gradient (muon matrices), (sqrt(v_hat)+eps)^{-1/2}
        elsewhere.  Called by refresh() after the backward pass, so p.grad is the global-batch gradient."""
        r = []
        for i, p in enumerate(self.plist):
            st = self.state.get(p, {})
            if self.use_muon[i] and p.grad is not None:
                f = _polar_factors(p.grad.detach(), self.polar_floor)
                c = self._scale(p)
                r.append(lambda U, f=f, c=c: (math.sqrt(c) * _polar_pow(f, U, 0.5)).to(U.dtype))
            elif "v" in st and st["step"] > 0:
                r.append((st["v"] / (1 - self.beta2 ** st["step"])).sqrt().add_(self.epsilon).rsqrt())
            else:
                r.append(torch.ones_like(p))
        return r

    # ---------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.t += 1
        self._compute_ratio(self.delta / (self.delta + self.t))
        mz2, mz2_idx = [], []
        for group in self.param_groups:
            g2 = group["lr"]
            schedule_factor = g2 / self.lr
            wd = group["weight_decay"]
            eps = group["epsilon"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                i = self.pidx[id(p)]
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    if not self.use_muon[i]:
                        st["v"] = torch.zeros_like(p)
                st["step"] += 1
                step = st["step"]
                g = p.grad
                if self.use_muon[i]:
                    u = self._scale(p) * zeropower_via_newtonschulz5(g.bfloat16(), steps=self.ns_steps).to(p.dtype)
                else:
                    st["v"].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                    u = g / (st["v"] / (1 - self.beta2 ** step)).sqrt().add_(eps)
                Dm = self.delta / (self.delta + step)
                st["m"].mul_(1 - Dm).add_(u, alpha=Dm)
                p.add_(u.add_(st["m"], alpha=float(self.A[i])), alpha=-g2)
                if self.wd_decaying:
                    wd_factor = -wd / (1 + step / self.wd_ts) * schedule_factor
                else:
                    wd_factor = -wd * schedule_factor
                p.mul_(1 + wd_factor)
                mz2.append((st["m"] * st["m"]).sum()); mz2_idx.append(i)
        if mz2:
            self.mz2[mz2_idx] = torch.stack(mz2).double().cpu().numpy()      # logged only (no water-filling)
        return loss
