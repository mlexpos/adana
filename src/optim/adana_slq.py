"""ADana-SLQ: ADana whose long-momentum coefficient is set by a measured stability budget instead of a hand schedule.

ADana (EMA form, log-time Delta_t = delta/(delta+t) for both moments):
    m <- (1-Delta_t) m + Delta_t g,   v <- (1-Delta_t) v + Delta_t g^2,
    theta <- theta - gamma(t) [ g + alpha(t) m ] / (sqrt(v) + eps)            (+ log-time weight decay)
with the hand-set alpha(t) = gamma_3_factor (1 + (1+t)^{1-kappa}).  In the sum-form of DANA (y = m / Delta_t),
alpha(t) = gamma_3 / (gamma_2 Delta_t) is the momentum AMPLIFICATION.  ADana-SLQ sets it per parameter tensor T:

    alpha_T(t) = min{ a_T S / N(Delta_t), cap } / Delta_t ,     S = s * B_eff * mu

  * N_T(Delta) = tr P_T f(H_z),  f(x) = g x/(g x + Delta): effective number of preconditioned curvature directions
    that the buffer integrates (H_z = r GN r, r = (sqrt(v)+eps)^{-1/2}, g = peak LR), from a stochastic Lanczos
    quadrature refreshed on a geometric schedule (slq_torch.py).  N = sum_T N_T.
  * s * B_eff: stochastic stability budget; B_eff = B_seq * T_eff (T_eff = effective independent tokens per sequence,
    measured by a sequence- vs token-split of the gradient noise).  The rule is (gamma_3/gamma_2) N <= s B_eff, a fixed
    fraction of the linear stability limit 2 B_eff.
  * mu = min{1, k'/B_noise}: signal-fraction multiplier.  The momentum-amplified noise floor is ~F (E + P*); keeping it
    below ~E/2 requires shrinking the budget with the signal fraction E/(E+P*), proxied by the gradient noise scale
    B_noise = tr C / |grad L|^2 = B |g1-g2|^2 / (4 <g1,g2>) from the two halves of each batch (sequence units).
    k' (sequences) is the open constant; k' <= 0 disables the multiplier.
  * a_T: allocation.  'global': a_T = 1 (same ratio for every tensor).  'waterfill': a_T = min{1, nu G_T / N_T} with
    sum_T a_T N_T = S, where G_T = |m_T|_z^2 - Delta^2 (N_T / B_eff) Temp / (2 g) is the buffer signal (buffer energy
    minus its noise part, Temp = tr C_z / tr H_z).  'independent': each tensor is its own stability problem with the
    FULL budget, ratio_T = min{S / N_T, cap} (no sharing; the total spent sum_T ratio_T N_T may exceed S and is logged
    against the linear limit 2 B_eff as slq/spent_over_limit).  s = inf gives every tensor the cap (Nesterov).
Everything else (moments, weight decay, optional per-element SNR clip `clipsnr`) is ADana's.  The LR schedule multiplies
the whole update, so the ratio is computed at the peak LR.
"""
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .adana import ADana
from .slq_torch import GaussNewtonOperator, effective_tokens, slq


def _alloc_group(name):
    """Tensor type for 'typed' allocation: attn (qkv + attention output), mlp (in + out), vocab (embedding +
    unembedding), tiny (layer norms, QK-norms, biases, scalars)."""
    n = re.sub(r"^(module\.|_orig_mod\.)+", "", name)
    if "wte" in n or "lm_head" in n:
        return "vocab"
    parts = n.split(".")
    module = parts[-2] if len(parts) >= 2 else n
    if re.search(r"(^ln_|norm)", module) or n.endswith(".bias"):
        return "tiny"
    if ".attn." in n:
        return "attn"
    if ".mlp." in n:
        return "mlp"
    return "tiny"


def _tensor_type(name):
    """'transformer.h.3.attn.c_attn.weight' -> ('attn.c_attn.weight', 3); non-block tensors -> (name, -1)."""
    mt = re.search(r"\.h\.(\d+)\.", name)
    if mt is None:
        return name.replace("_orig_mod.", "").replace("module.", ""), -1
    return name[mt.end():], int(mt.group(1))


class ADanaSLQ(ADana):
    def __init__(self, params, lr=1.0, delta=8.0, epsilon=1e-8, weight_decay=0.0, clipsnr=None, wd_decaying=False,
                 wd_ts=1.0, s=0.25, kprime=0.0, cap=1.0, alloc="global", m_max=128, probes=1, slq_batch=8,
                 slq_eps=0.03, refresh_ratio=2.0, max_gap=100, teff="auto", batch_seqs=1, log_dir=None, seed=0,
                 gn_mode="jvp", first_refresh=4, aitken=False, gn_chunk=2, gn_cache=True,
                 type_frac="attn=0.45,mlp=0.45,vocab=0.1,tiny=0", reject_gap=-1.0, leftover="uniform"):
        super().__init__(params, lr=lr, delta=delta, kappa=1.0, epsilon=epsilon, weight_decay=weight_decay,
                         clipsnr=clipsnr, wd_decaying=wd_decaying, wd_ts=wd_ts, gamma_3_factor=1.0, use_foreach=False)
        self.s, self.kprime, self.cap, self.alloc = float(s), float(kprime), float(cap), alloc
        self.m_max, self.probes, self.slq_batch, self.slq_eps = int(m_max), int(probes), int(slq_batch), float(slq_eps)
        self.refresh_ratio, self.max_gap = float(refresh_ratio), int(max_gap)
        self.teff_mode = teff
        self.gn_mode = gn_mode
        self.gn_chunk, self.gn_cache = int(gn_chunk), bool(gn_cache)
        self.aitken = aitken
        self.reject_gap = float(reject_gap)
        self.leftover = leftover
        self.n_reject = 0
        self.teff = 1.0 if teff == "auto" else float(teff)
        self.batch_seqs = int(batch_seqs)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.gen = None
        self.seed = seed
        # flat parameter bookkeeping (optimizer order)
        self.plist, self.pnames = [], []
        for g in self.param_groups:
            names = g.get("param_names", [None] * len(g["params"]))
            for p, n in zip(g["params"], names):
                self.plist.append(p); self.pnames.append(n if n is not None else f"param{len(self.pnames)}")
        L = len(self.plist)
        self.pidx = {id(p): i for i, p in enumerate(self.plist)}
        types = [_tensor_type(n) for n in self.pnames]
        self.ttype = [t for t, _ in types]
        self.tlayer = [l for _, l in types]
        # typed allocation: fixed budget fractions per tensor type (normalized over the listed types)
        fr = {k.strip(): float(v) for k, v in (kv.split("=") for kv in type_frac.split(",") if kv.strip())}
        tot = sum(fr.values())
        self.type_frac = {k: v / tot for k, v in fr.items()} if tot > 0 else {}
        self.agroup = np.array([_alloc_group(n) for n in self.pnames])
        self.t = 0                                  # completed optimizer steps
        self.next_refresh = max(1, int(first_refresh))
        self.q = None                               # quadrature: nodes (K,), w (K,), wT (K, L), trF, lam_max
        self.eX = np.zeros(L); self.eD = np.zeros(L); self.eDz = np.zeros(L); self.n_split = 0
        self.mz2 = np.zeros(L)                      # |m_T|^2 in z coordinates (last step)
        self.ratio = np.zeros(L); self.A = np.zeros(L); self.afrac = np.ones(L)
        self.NT = np.zeros(L); self.N = 0.0; self.S = 0.0; self.mu = 1.0; self.bnr = float("nan"); self.Delta = 1.0
        self.G = np.zeros(L); self.spent = 0.0
        self.n_refresh = 0; self.refresh_seconds = 0.0; self.last_refresh = {}
        self.history = []

    # ---------------------------------------------------------------- compiled update (ADana with alpha -> A_T)
    @staticmethod
    def _update_slq(p, grad, m, v, alpha, g2, A, schedule_factor, wd, step, epsilon: float, clipsnr: float,
                    wd_decaying: bool, wd_ts: float):
        m.lerp_(grad, alpha)
        v.mul_(1 - alpha).addcmul_(grad, grad, value=alpha)
        norm = 1.0 / torch.sqrt(v).add_(epsilon)
        if clipsnr is not None:
            mom = torch.sign(m) * torch.clamp(A * torch.abs(m) * norm, max=clipsnr)
        else:
            mom = A * m * norm
        p.add_((-g2) * (grad * norm + mom))
        if wd_decaying:
            wd_factor = -wd / (1 + step / wd_ts) * schedule_factor
        else:
            wd_factor = -wd * schedule_factor
        p.mul_(1 + wd_factor)
        mz2 = (m * m * norm).sum()
        return m, v, mz2

    def _get_compiled_fn(self, ndim):
        if ndim not in self._compiled_functions:
            self._compiled_functions[ndim] = torch.compile(self._update_slq, dynamic=False, fullgraph=False)
        return self._compiled_functions[ndim]

    # ---------------------------------------------------------------- the rule
    def _compute_ratio(self, D):
        L = len(self.plist)
        self.Delta = D
        if self.q is None:
            self.ratio[:] = 0.0; self.A[:] = 0.0; self.afrac[:] = 1.0
            return
        g = self.lr
        th = np.maximum(self.q["nodes"], 0.0)
        f = g * th / (g * th + D)
        self.N = max(float(f @ self.q["w"]), 1e-12)
        self.NT = f @ self.q["wT"]
        NTc = np.maximum(self.NT, 1e-2)
        B_eff = self.batch_seqs * max(self.teff, 1.0)
        if self.kprime > 0 and self.n_split > 0 and self.eD.sum() > 0:
            self.bnr = 4.0 * max(self.eX.sum(), 0.0) / self.eD.sum()          # B / B_noise (sequences)
            self.mu = float(min(1.0, self.kprime * self.bnr / self.batch_seqs))
        else:
            self.bnr = 4.0 * max(self.eX.sum(), 0.0) / self.eD.sum() if self.eD.sum() > 0 else float("nan")
            self.mu = 1.0
        self.S = self.s * B_eff * self.mu
        if self.n_split > 0:
            # buffer signal: buffer energy minus its predicted noise part (buffer identity, global temperature)
            temp = (B_eff * self.eDz.sum() / 4.0) / max(self.q["trF"], 1e-30)
            noise = (NTc / B_eff) * temp / (2.0 * g)
            self.G = self.mz2 - D * D * noise
        if self.alloc == "typed":
            # typed water-filling: each type g (attn / mlp / vocab / tiny) gets a FIXED fraction pi_g of the budget; within
            # a type the budget is water-filled on the buffer signal G_T (tensors compared only with tensors of the same
            # type). Conservative: a tensor with G_T <= 0 (noise on the order of the signal, or no split statistics yet)
            # gets NO long momentum; only tensors with positive signal share their type's budget.
            ratio = np.zeros(L)
            for gname, pi in self.type_frac.items():
                m = self.agroup == gname
                if not m.any() or pi <= 0 or self.n_split == 0:
                    continue
                c = np.maximum(self.G[m], 0.0) / NTc[m]
                ratio[m] = self._waterfill(c, NTc[m], pi * self.S, zero_without_signal=True)
            self.ratio = np.minimum(ratio, self.cap)
            self.afrac = self.ratio * NTc / max(self.S, 1e-30)        # budget share actually spent per tensor
        elif self.alloc == "waterfill" and self.n_split > 0:
            c = np.maximum(self.G, 0.0) / NTc
            a = self._waterfill(c, NTc, self.S, leftover=(self.leftover == "uniform"))
            self.afrac = a
            self.ratio = np.minimum(a, self.cap)
        elif self.alloc == "independent":
            self.ratio = np.minimum(self.S / NTc, self.cap)
            self.afrac = self.ratio * NTc / max(self.S, 1e-30)
        elif self.alloc != "typed":
            self.afrac = np.ones(L)
            self.ratio = np.full(L, min(self.S / self.N, self.cap))
        self.spent = float(self.ratio @ np.maximum(self.NT, 0.0)) / (2.0 * B_eff)   # fraction of the linear limit used
        self.A = self.ratio / D

    @staticmethod
    def _waterfill(c, NTc, budget, zero_without_signal=False, leftover=False):
        """a_T = min{1, nu c_T} with sum_T a_T NTc_T = budget (bisection on nu).
        zero_without_signal=False (original 'waterfill'): all-ones if the budget covers every tensor, one uniform ratio if
        no tensor has positive signal.  zero_without_signal=True ('typed'): tensors with c_T = 0 always get 0; the tensors
        with positive signal get 1 if the budget covers them all.
        leftover=True: when every tensor with positive signal is at the cap and budget remains, spread the remainder
        uniformly over the tensors without signal (so the budget is always spent, as in the global rule)."""
        if zero_without_signal:
            pos = c > 0
            a = np.zeros(len(NTc))
            if not pos.any():
                return a
            if NTc[pos].sum() <= budget:
                a[pos] = 1.0
                return a
        elif NTc.sum() <= budget:
            return np.ones(len(NTc))
        elif c.sum() <= 0:
            return np.full(len(NTc), min(budget / NTc.sum(), 1.0))
        lo, hi = 0.0, 1e30
        for _ in range(200):
            mid = math.sqrt(max(lo, 1e-300) * hi)
            if np.sum(np.minimum(1.0, mid * c) * NTc) <= budget:
                lo = mid
            else:
                hi = mid
        a = np.minimum(1.0, lo * c)
        if leftover:
            rem = budget - float(a @ NTc)
            z = c <= 0
            if rem > 1e-9 * budget and z.any():
                a[z] = min(1.0, rem / NTc[z].sum())
        return a

    # ---------------------------------------------------------------- estimator inputs from the training loop
    @torch.no_grad()
    def set_split_stats(self, half, full):
        """half / full: per-parameter gradients accumulated over the first half / all micro-steps (averaged over ranks).
        g1 = 2 half and g2 = 2 (full - half) are the mean gradients of the two halves of the (global) batch."""
        L = len(self.plist)
        xg, dg, dz = np.zeros(L), np.zeros(L), np.zeros(L)
        for i, (p, h, f) in enumerate(zip(self.plist, half, full)):
            if h is None or f is None:
                continue
            g1 = 2.0 * h.float(); g2 = 2.0 * (f.float() - h.float())
            d = g1 - g2
            xg[i] = float((g1 * g2).sum()); dg[i] = float((d * d).sum())
            st = self.state.get(p, {})
            if "v" in st:
                dz[i] = float((d * d / (torch.sqrt(st["v"]) + self.epsilon)).sum())
            else:
                dz[i] = dg[i]
        rho = min(1.0, 3.0 / (self.t + 1.0))                  # window ~ t/3
        if self.n_split == 0:
            self.eX, self.eD, self.eDz = xg, dg, dz
        else:
            self.eX = (1 - rho) * self.eX + rho * xg
            self.eD = (1 - rho) * self.eD + rho * dg
            self.eDz = (1 - rho) * self.eDz + rho * dz
        self.n_split += 1

    def needs_refresh(self):
        return self.t >= self.next_refresh

    @torch.no_grad()
    def refresh(self, model, x, y):
        """Refresh the quadrature (and T_eff) on the first `slq_batch` sequences of (x, y), using the raw module."""
        t0 = time.time()
        if self.gen is None:
            # same probes on every rank (the Lanczos is replicated; products are averaged across ranks)
            self.gen = torch.Generator(device=x.device); self.gen.manual_seed(1234)
        names_by_id = {id(p): n for n, p in model.named_parameters()}
        names = [names_by_id[id(p)] for p in self.plist]
        r = []
        for p in self.plist:
            st = self.state.get(p, {})
            r.append((torch.sqrt(st["v"]) + self.epsilon).rsqrt() if "v" in st else torch.ones_like(p))
        xb, yb = x[: self.slq_batch], y[: self.slq_batch]
        was_training = model.training
        model.eval()
        op = GaussNewtonOperator(model, names, self.plist, xb, yb, r=r, mode=self.gn_mode, chunk=self.gn_chunk,
                                 cache=self.gn_cache)
        D_now = self.delta / (self.delta + self.t + 1)
        D_next = self.delta / (self.delta + self._next_after(self.t) + 1)
        res = slq(op, self.plist, m_max=self.m_max, probes=self.probes, eps=self.slq_eps, g=self.lr,
                  D_check=[D_now, D_next], generator=self.gen)
        K = res["nodes"].size
        nodes = res["nodes"].reshape(K)
        w = res["weights"].reshape(K)
        wT = res["block"].reshape(K, -1)
        # default: N and every N_T are read off the Gauss rule, an upper bound on N at any m (conservative).  The Aitken
        # correction (opt-in) extrapolates below it.
        corr = float(np.mean(res["aitken"])) if self.aitken else 1.0
        # default (reject_gap < 0): always adopt the fresh quadrature.  Rejecting a non-converged refresh would replace
        # a certified bound on the current operator by the previous quadrature, which bounds nothing at the new point.
        accepted = (self.reject_gap < 0 or bool(np.all(res["converged"]))
                    or float(np.nanmax(res["gap"])) <= self.reject_gap)
        if accepted:
            self.q = dict(nodes=nodes, w=w * corr, wT=wT * corr, trF=float(np.sum(w * np.maximum(nodes, 0.0))),
                          lam_max=res["lam_max"])
        else:
            self.n_reject += 1
        te_new = float("nan")
        if self.teff_mode == "auto":
            with torch.enable_grad():
                te_new, _, _ = effective_tokens(model, self.plist, xb, yb, generator=self.gen)
            if dist.is_available() and dist.is_initialized():
                tt = torch.tensor([te_new], device=x.device, dtype=torch.float64); dist.all_reduce(tt)
                te_new = float(tt) / dist.get_world_size()
            te_new = float(np.clip(te_new, 1.0, yb.shape[1]))
            self.teff = te_new if self.n_refresh == 0 else math.sqrt(self.teff * te_new)
        if was_training:
            model.train()
        dt = time.time() - t0
        self.n_refresh += 1; self.refresh_seconds += dt
        self.last_refresh = dict(t=self.t, m_used=float(res["m_used"].mean()), gap=float(np.nanmax(res["gap"])),
                                 aitken=corr, accepted=float(accepted), rejects=float(self.n_reject),
                                 N_hi=float(np.mean(res["N_hi"][:, 0])) if res["N_hi"].size else float("nan"),
                                 N_lo=float(np.mean(res["N_lo"][:, 0])) if res["N_lo"].size else float("nan"),
                                 lam_max=res["lam_max"], trF=float(np.sum(w * np.maximum(nodes, 0.0))),
                                 teff_raw=te_new, seconds=dt)
        self.next_refresh = self._next_after(self.t)
        self._compute_ratio(self.delta / (self.delta + self.t + 1))
        if self.q is not None:
            self._save_diag()

    def _next_after(self, t):
        return max(t + 1, min(int(math.ceil(t * self.refresh_ratio)), t + self.max_gap))

    # ---------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.t += 1
        D = self.delta / (self.delta + self.t)
        self._compute_ratio(D)
        A_by_param = {id(p): self.A[i] for i, p in enumerate(self.plist)}
        for group in self.param_groups:
            g2 = group["lr"]
            schedule_factor = group["lr"] / self.lr
            wd = group["weight_decay"]
            eps = group["epsilon"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]
                dev = p.device
                fn = self._get_compiled_fn(p.shape)
                _, _, mz2 = fn(p, p.grad, state["m"], state["v"],
                               torch.tensor(self.delta / (self.delta + step), device=dev, dtype=torch.float32),
                               torch.tensor(g2, device=dev, dtype=torch.float32),
                               torch.tensor(A_by_param[id(p)], device=dev, dtype=torch.float32),
                               torch.tensor(schedule_factor, device=dev, dtype=torch.float32),
                               torch.tensor(wd, device=dev, dtype=torch.float32),
                               torch.tensor(step, device=dev, dtype=torch.float32),
                               eps, self.clipsnr, self.wd_decaying, self.wd_ts)
                state["mz2"] = mz2
        for i, p in enumerate(self.plist):
            if "mz2" in self.state.get(p, {}):
                self.mz2[i] = float(self.state[p]["mz2"])
        return loss

    # ---------------------------------------------------------------- logging
    def diagnostics(self):
        logs = {
            "slq/Delta": self.Delta, "slq/N": self.N, "slq/S_budget": self.S, "slq/mu": self.mu,
            "slq/B_over_Bnoise": self.bnr, "slq/T_eff": self.teff, "slq/B_eff": self.batch_seqs * max(self.teff, 1.0),
            "slq/ratio_global": min(self.S / max(self.N, 1e-12), self.cap) if self.q is not None else 0.0,
            "slq/A_mean": float(np.mean(self.A)), "slq/A_max": float(np.max(self.A)),
            "slq/ratio_mean": float(np.mean(self.ratio)), "slq/ratio_max": float(np.max(self.ratio)),
            "slq/spent_over_limit": self.spent,
            "slq/refreshes": self.n_refresh, "slq/refresh_seconds": self.refresh_seconds,
        }
        for k, v in self.last_refresh.items():
            logs[f"slq/last_{k}"] = v
        wN = np.maximum(self.NT, 0.0)
        for ty in sorted(set(self.ttype)):
            idx = [i for i, t in enumerate(self.ttype) if t == ty]
            key = ty.replace(".weight", "").replace("transformer.", "")
            logs[f"slq_type/ratio/{key}"] = float(np.mean(self.ratio[idx]))
            logs[f"slq_type/A/{key}"] = float(np.mean(self.A[idx]))
            logs[f"slq_type/afrac/{key}"] = float(np.mean(self.afrac[idx]))
            logs[f"slq_type/N_T/{key}"] = float(np.sum(wN[idx]))
            logs[f"slq_type/G_T/{key}"] = float(np.sum(self.G[idx]))
        logs["slq/refresh_rejects"] = self.n_reject
        for gname in ("attn", "mlp", "vocab", "tiny"):
            m = self.agroup == gname
            if m.any():
                logs[f"slq_group/ratio/{gname}"] = float(np.mean(self.ratio[m]))
                logs[f"slq_group/A/{gname}"] = float(np.mean(self.A[m]))
                logs[f"slq_group/N/{gname}"] = float(np.sum(wN[m]))
                logs[f"slq_group/budget_share/{gname}"] = float(np.sum(self.ratio[m] * wN[m]) / max(self.S, 1e-30))
        for layer in sorted(set(l for l in self.tlayer if l >= 0)):
            idx = [i for i, l in enumerate(self.tlayer) if l == layer]
            logs[f"slq_layer/ratio/{layer}"] = float(np.mean(self.ratio[idx]))
            logs[f"slq_layer/N_T/{layer}"] = float(np.sum(wN[idx]))
            logs[f"slq_layer/budget_share/{layer}"] = float(np.sum(self.ratio[idx] * wN[idx]) / max(self.S, 1e-30))
        return logs

    def _save_diag(self):
        if self.log_dir is None or (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0):
            return
        self.history.append(dict(t=self.t, NT=self.NT.copy(), ratio=self.ratio.copy(), A=self.A.copy(),
                                 afrac=self.afrac.copy(), G=self.G.copy(), N=self.N, S=self.S, spent=self.spent, mu=self.mu, bnr=self.bnr,
                                 teff=self.teff, nodes=self.q["nodes"].copy(), w=self.q["w"].copy(),
                                 lam_max=self.q["lam_max"], **{k: v for k, v in self.last_refresh.items() if k not in ("t", "lam_max")}))
        out = {"names": np.array(self.pnames)}
        for k in self.history[0]:
            vals = [h[k] for h in self.history]
            try:
                out[k] = np.array(vals)
            except ValueError:
                out[k] = np.array(vals, dtype=object)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.log_dir / "slq_diag.npz", **out)
        except Exception as e:  # never kill training over diagnostics
            print(f"[adana-slq] could not save diagnostics: {e}")
