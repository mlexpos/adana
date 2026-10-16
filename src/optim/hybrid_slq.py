"""hybrid-slq: SLQ-ADana (src/optim/adana_slq.py) with the plain term of the hidden matrices replaced by d-muon's step.

The ladder study (dana-lyapunov experiments/report/hybrid_ladder) found that, on a bias-dominated problem, the plain step
and the long momentum do different jobs: a small spectral (msign) step for the fast directions, and SLQ-ADana's
stability-budgeted long momentum, in Adam coordinates, for the slow ones.  Per tensor T, with ADana's moments m, v at the
log-time rate Delta_t = delta / (delta + t) and the LR schedule s(t) = group lr / lr:

  hidden matrices (p.ndim >= 2 and p.size(0) < 10000: every block matrix, not wte / lm_head):
      buf <- beta buf + g,   u = c msign(g + beta buf)       (d-muon's Nesterov buffer and Moonlight scale c = matched_rms
                                                              sqrt(max(A, B)); msign = 5 quintic Newton-Schulz steps, bf16)
      theta <- theta - s(t) [ eta_h u + gamma0 A_T m / (sqrt(v) + eps) ]
  every other tensor:
      theta <- theta - s(t) gamma0 [ va g + A_T m ] / (sqrt(v) + eps)
  (+ ADana's log-time weight decay on every tensor, as adana-slq)
  plain = 'adamw': the non-hidden tensors' plain term is d-muon's AdamW direction instead, va gamma0 a1_hat / (sqrt(a2_hat) + eps)
  with its own moments (betas 0.8 / 0.999, bias-corrected); va gamma0 = d-muon's lr gives d-muon's full plain step everywhere.

gamma0 = lr is SLQ-ADana's own peak LR: it sets both the momentum gain and the SLQ threshold, N_T(Delta) = tr P_T f(H_z),
f(x) = gamma0 x / (gamma0 x + Delta), H_z = r GN r, r = (sqrt(v) + eps)^{-1/2}.  A_T = ratio_T / Delta_t is ADanaSLQ's rule
(global, waterfill, ...) unchanged: the buffer m is the EMA of the raw gradient in both methods, so the estimator inputs
(split statistics, |m_T|_z^2, the quadrature) are those of adana-slq.  eta_h is the peak of the msign step (hyb_msign_lr);
va = 1 gives the Adam-treated tensors exactly SLQ-ADana's step.  With eta_h = 0 and va = 1 this is adana-slq on the
non-hidden tensors and pure long momentum on the hidden ones; with A_T = 0 it is d-muon's hidden step (without its AdamW
first moment elsewhere).

Cyclic waves (cyc = +1 / -1; the ladder study's deep-MLP result, user design): the momentum ratio of every large tensor is its OWN
noise cap r_T^rule = min{S / N_T, cap} (adana-slq's 'independent' allocation), small tensors (norms, biases, scalars) get none, and
the large tensors take turns in groups wte (+ wpe) -> block 0 -> ... -> block L-1 -> lm_head (cyc = -1: reversed).  Group k is
active on [k T/G, (k+1) T/G) (G = L + 2, T = cyc_T iterations): it runs at r^rule, switched on by a linear ramp over 10% of a stage;
when its stage ends at t_k it decays as min{r_end t_k / t, r^rule(t)} (r ~ 1/t holds its amplification r/Delta fixed); later groups
are off.  small_off = True (non-cyclic): every large tensor at its own cap, small tensors off.
"""
import math
import re

import numpy as np
import torch

from .adana_slq import ADanaSLQ
from .muon import zeropower_via_newtonschulz5


class HybridSLQ(ADanaSLQ):
    def __init__(self, params, lr=1.0, msign_lr=0.0, beta=0.95, va=1.0, delta=8.0, epsilon=1e-8, weight_decay=0.0,
                 wd_decaying=False, wd_ts=1.0, ns_steps=5, matched_rms=0.2, plain="rms", adam_betas=(0.8, 0.999),
                 cyc=0, cyc_T=0, small_off=False, **slq_kw):
        if cyc:
            slq_kw["alloc"] = "independent"                    # waves run every tensor at its own noise cap
            if cyc_T <= 0:
                raise ValueError("hybrid-slq: cyclic waves need cyc_T (the number of iterations)")
        super().__init__(params, lr=lr, delta=delta, epsilon=epsilon, weight_decay=weight_decay, clipsnr=None,
                         wd_decaying=wd_decaying, wd_ts=wd_ts, **slq_kw)
        if plain not in ("rms", "adamw"):
            raise ValueError(f"hybrid-slq: plain step '{plain}' (rms | adamw)")
        self.msign_lr, self.beta, self.va = float(msign_lr), float(beta), float(va)
        self.plain, self.adam_betas = plain, tuple(float(b) for b in adam_betas)
        self.cyc, self.cyc_T, self.small_off = int(cyc), int(cyc_T), bool(small_off)
        self.cgroup = self._cyc_groups()
        self.NG = int(self.cgroup.max()) + 1 if (self.cgroup >= 0).any() else 1
        self.rend = np.zeros(len(self.plist))
        self.ns_steps, self.matched_rms = int(ns_steps), float(matched_rms)
        self.use_muon = [p.ndim >= 2 and p.size(0) < 10000 for p in self.plist]

    def _cyc_groups(self):
        """Wave group of every tensor: 0 for wte / wpe, 1 + l for the matrices of block l, L + 1 for lm_head, -1 for small tensors."""
        names = [re.sub(r"^(module\.|_orig_mod\.)+", "", n) for n in self.pnames]
        layers = [int(m.group(1)) for n in names for m in [re.search(r"\.h\.(\d+)\.", n)] if m]
        L = max(layers) + 1 if layers else 0
        g = np.full(len(names), -1)
        for i, (n, p) in enumerate(zip(names, self.plist)):
            if p.ndim < 2:
                continue
            if "wte" in n or "wpe" in n:
                g[i] = 0
            elif "lm_head" in n:
                g[i] = L + 1
            else:
                m = re.search(r"\.h\.(\d+)\.", n)
                if m:
                    g[i] = 1 + int(m.group(1))
        return g

    def _compute_ratio(self, D):
        super()._compute_ratio(D)
        if self.q is None or not (self.cyc or self.small_off):
            return
        rule = self.ratio.copy()
        big = self.cgroup >= 0
        if self.cyc:
            stage = self.cyc_T / self.NG
            t = float(self.t)
            k = math.floor(t / stage)
            pos = self.cgroup if self.cyc > 0 else (self.NG - 1) - self.cgroup
            active = big & (pos == k)
            done = big & (pos < k)
            r_act = rule * np.clip((t - pos * stage) / max(0.1 * stage, 1.0), 0.0, 1.0)
            r_tail = np.minimum(self.rend * (pos + 1) * stage / max(t, 1.0), rule)
            self.ratio = np.where(active, r_act, np.where(done, r_tail, 0.0))
            self.rend = np.where(active, r_act, self.rend)
        else:
            self.ratio = np.where(big, rule, 0.0)
        self.A = self.ratio / D

    def _scale(self, p):
        return self.matched_rms * math.sqrt(max(p.shape[:2]))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.t += 1
        self._compute_ratio(self.delta / (self.delta + self.t))
        hmul = self.msign_lr / self.lr                     # msign peak relative to gamma0 (the group lr carries s(t) gamma0)
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
                    st["v"] = torch.zeros_like(p)
                    if self.use_muon[i]:
                        st["buf"] = torch.zeros_like(p)
                    elif self.plain == "adamw":
                        st["a1"] = torch.zeros_like(p); st["a2"] = torch.zeros_like(p)
                st["step"] += 1
                step = st["step"]
                g = p.grad
                Dm = self.delta / (self.delta + step)
                st["m"].lerp_(g, Dm)
                st["v"].mul_(1 - Dm).addcmul_(g, g, value=Dm)
                norm = torch.sqrt(st["v"]).add_(eps).reciprocal_()
                mom = st["m"] * norm
                st["mz2"] = (st["m"] * mom).sum()
                if self.use_muon[i]:
                    st["buf"].mul_(self.beta).add_(g)
                    u = zeropower_via_newtonschulz5(g.add(st["buf"], alpha=self.beta).bfloat16(), steps=self.ns_steps)
                    p.add_(u.to(p.dtype), alpha=-g2 * hmul * self._scale(p))
                elif self.plain == "adamw":           # d-muon's AdamW direction (bias-corrected), at va gamma0
                    b1, b2 = self.adam_betas
                    st["a1"].lerp_(g, 1 - b1); st["a2"].lerp_(g * g, 1 - b2)
                    sc = (1 - b1 ** step) / (1 - b2 ** step) ** 0.5
                    p.add_(st["a1"] / (st["a2"].sqrt() + eps), alpha=-g2 * self.va / sc)
                else:
                    p.add_(g * norm, alpha=-g2 * self.va)
                p.add_(mom, alpha=-g2 * float(self.A[i]))
                if self.wd_decaying:
                    wd_factor = -wd / (1 + step / self.wd_ts) * schedule_factor
                else:
                    wd_factor = -wd * schedule_factor
                p.mul_(1 + wd_factor)
        for i, p in enumerate(self.plist):
            if "mz2" in self.state.get(p, {}):
                self.mz2[i] = float(self.state[p]["mz2"])
        return loss

    def diagnostics(self):
        logs = super().diagnostics()
        logs["hyb/msign_lr"] = self.msign_lr
        logs["hyb/msign_over_gamma0"] = self.msign_lr / self.lr
        if self.cyc:
            logs["hyb/cyc_stage"] = math.floor(self.t / (self.cyc_T / self.NG))
        return logs
