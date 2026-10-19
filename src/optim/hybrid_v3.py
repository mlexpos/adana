"""hybrid-v3: the hybrid of the ladder study's version 3 (dana-lyapunov experiments/report/hybrid_ladder, hybrid_ladder-v3) on Enoki.

The update is hybrid-slq's (src/optim/hybrid_slq.py): d-muon's msign step on the hidden matrices, d-muon's AdamW on every other
tensor (plain = 'adamw', va gamma0 = d-muon's lr), and ADana's long momentum with gain kappa_t = s(t) gamma0 r_t / Delta_t.  The
hidden msign matrices carry the long momentum (mom = 'hidden'), the hidden matrices plus wte and lm_head (mom = 'hidden+terminal'),
or every tensor (mom = 'all': also the norm gains, a third block under the SLQ kernel, exact for their AdamW step).  What changes is the momentum floor rule:

    r_t = min{1, s_b B_eff rho-hat / K(Delta_t)}        (one global ratio)

  * rho-hat: the measured excess fraction E_delta / tau-hat, an EMA over refreshes (rate r_rho), with
        E_delta = <z_A, (H_z + delta_z)^-1 z_B>,   delta_z = m Delta_t / gamma0  (m = mdamp, 1 by default on Enoki),
    z_A, z_B = R g_A, R g_B the two halves of the training batch in the Adam metric (R = (sqrt v + eps)^-1/2), and H_z = R GN R on a
    curvature batch of `cbatch` sequences drawn from a separate stream (the trainer passes `curv`).  One Lanczos recurrence from z_B
    on the full curvature batch and one on its first half (Richardson, E = 2 E_c - E_{c/2}: the curvature-sampling bias falls like
    1/c, bench_enoki_rho.py), no stored basis (c_j = <z_A, v_j> accumulated on the fly), stopped when both change by less than
    lz_eps over 16 steps, at most k_lz steps.
  * tau-hat = tr(R C R) / tr(H_z) in effective samples: tr(R C R) = (n_eff / 2) |z_A - z_B|^2 with n_eff = (sequences per half) T_eff,
    tr(H_z) from the SLQ quadrature of the refresh; tau / P smoothed over refreshes (rate r_tau; P the training loss).
  * K: the kernel of the hidden block, from the SLQ quadrature's per-tensor weights: 'slq' N = sum_j w_j g th_j / (g th_j + Delta),
    'msign' N_M = sum_j w_j g th_j / (Delta + nu_t sqrt(th_j)), nu_t = eta_t c_W sqrt(B_beta s-bar / (N_W tau-hat)),
    B_beta = B_eff S1^2 / S2 (the Nesterov buffer's effective batch), c_W the parameter-weighted Moonlight scale, N_W the smallest
    width of the hidden matrices, s-bar the mean sqrt(v) over the hidden block.  g = gamma0.
  * B_eff = sequences per step x T_eff.
Before step meas_from (32) only the SLQ refreshes run and rho-hat = 1: early on E_delta / tau >> 1 is clipped anyway, and the solve
is the most expensive there (lambda_max ~1e7 near initialization).
GN products use forward-mode AD in TF32 when slq_tf32 (bench_enoki_gn.py: within 2-3% of float64 on a trained 12h checkpoint).
"""
import math
import time

import numpy as np
import torch
import torch.distributed as dist

from .hybrid_slq import HybridSLQ
from .slq_torch import GaussNewtonOperator, _dot


class HybridV3(HybridSLQ):
    def __init__(self, params, kernel="slq", mdamp=1.0, cbatch=64, k_lz=128, lz_eps=0.01, r_rho=0.5, r_tau=0.3, meas_from=32, mom="hidden", split="global", **kw):
        kw["alloc"] = "global"; kw["plain"] = kw.get("plain", "adamw")
        super().__init__(params, **kw)
        if kernel not in ("slq", "msign"):
            raise ValueError(f"hybrid-v3: kernel '{kernel}' (slq | msign)")
        self.kernel, self.mdamp, self.cbatch, self.k_lz, self.lz_eps = kernel, float(mdamp), int(cbatch), int(k_lz), float(lz_eps)
        self.r_rho, self.r_tau = float(r_rho), float(r_tau)
        self.meas_from = int(meas_from)                     # first step with the excess form; before it rho-hat = 1 (where it is clipped early)
        self.hid = np.array(self.use_muon, bool)
        # tensors with the long momentum: the hidden msign matrices, with mom = 'hidden+terminal' also wte and lm_head (AdamW plain
        # step), with mom = 'all' also every remaining tensor (the norm gains).  The kernel adds over blocks, each at its own plain
        # step's relaxation rate.
        if mom not in ("hidden", "hidden+terminal", "all"):
            raise ValueError(f"hybrid-v3: mom '{mom}' (hidden | hidden+terminal | all)")
        import re as _re
        base = [_re.sub(r"^(module\.|_orig_mod\.)+", "", n) for n in self.pnames]
        self.term = np.array([(("wte" in n) or ("lm_head" in n)) and p.ndim >= 2 for n, p in zip(base, self.plist)], bool)
        self.mom_mode = mom
        if split not in ("global", "independent"):
            raise ValueError(f"hybrid-v3: split '{split}' (global | independent)")
        self.split = split          # 'independent': the hidden block, wte + lm_head and the gains each take min{1, S / K_block} (same S, rho-hat)
        self.gain = ~self.hid & ~self.term
        self.r_hid = self.r_term = self.r_gain = 0.0
        self.mom = {"hidden": self.hid.copy(), "hidden+terminal": self.hid | self.term, "all": np.ones(len(self.plist), bool)}[mom]
        self.K_hid = self.K_term = self.K_gain = float("nan")
        hp = [p for p, h in zip(self.plist, self.hid) if h]
        self.N_W = min(min(p.shape[:2]) for p in hp)
        self.c_W = sum(p.numel() * self._scale(p) for p in hp) / sum(p.numel() for p in hp)
        b = self.beta; self.bb_fac = (1 / (1 - b)) ** 2 / ((1 + b) ** 2 + b ** 4 / (1 - b ** 2))
        self.gA = self.gB = None; self.last_loss = float("nan"); self.sfac = 0.0; self.sbar = 1.0
        self.rho = None; self.tq = None; self.tau = float("nan"); self.meas = {}

    # ---------------------------------------------------------------- the halves of the batch at a refresh step
    @torch.no_grad()
    def set_split_stats(self, half, full):
        super().set_split_stats(half, full)
        if self.needs_refresh():
            self.gA = [None if h is None else 2.0 * h.float() for h in half]
            self.gB = [None if (h is None or f is None) else 2.0 * (f.float() - h.float()) for h, f in zip(half, full)]

    def refresh(self, model, x, y, curv=None):
        for k in self.V3KEYS:
            self.last_refresh.pop(k, None)
        super().refresh(model, x, y)                         # SLQ quadrature (kernel nodes, tr H_z) and T_eff
        if self.t < self.meas_from:
            # early refreshes: the SLQ kernel only.  The excess form is expensive there (lambda_max ~1e7 near initialization) and
            # rho-hat is clipped at 1 anyway (bench: Enoki 12h, t = 4, 8 cost 11 min each), so the rule runs at rho-hat = 1.
            self.rho = 1.0 if self.rho is None else self.rho
        elif curv is not None and self.gA is not None and self.q is not None:
            prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = self.slq_tf32
            try:
                self._measure(model, curv)
            finally:
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
        self.gA = self.gB = None
        self._compute_ratio(self.delta / (self.delta + self.t + 1))
        if self.meas.get("t") == self.t and self.history:
            # the SLQ refresh saved its record before the excess form existed: replace it with one that carries the v3 measurement,
            # the kernel and the ratio after it (slq_diag.npz: one row per refresh)
            self.last_refresh.update({f"v3_{k}": v for k, v in self.meas.items() if k != "t"})
            self.last_refresh.update(v3_K=float(self.N), v3_S=float(self.S), v3_r=self.r_hid,
                                     v3_nu=float(getattr(self, "nu", float("nan"))), v3_sbar=self.sbar,
                                     v3_K_hid=self.K_hid, v3_K_term=self.K_term, v3_bnr=float(self.bnr), v3_mu_k64=float(self.mu),
                                     v3_r_hid=self.r_hid, v3_r_term=self.r_term,
                                     v3_K_gain=self.K_gain, v3_r_gain=self.r_gain)
            self.history.pop()
            self._save_diag()

    # ---------------------------------------------------------------- the excess form, Richardson over the curvature batch
    @torch.no_grad()
    def _measure(self, model, curv):
        t0 = time.time()
        names_by_id = {id(p): n for n, p in model.named_parameters()}
        names = [names_by_id[id(p)] for p in self.plist]
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        xc, yc = curv(max(2, self.cbatch // world))
        R = self._slq_r()
        z = lambda g: [torch.zeros_like(p) if gi is None else r * gi for p, r, gi in zip(self.plist, R, g)]
        zA, zB = z(self.gA), z(self.gB)
        self.gA = self.gB = None                             # the halves are not needed past z (memory: two parameter-sized vectors)
        h = xc.shape[0] // 2
        was = model.training; model.eval()
        mk = lambda a, b: GaussNewtonOperator(model, names, self.plist, xc[a:b], yc[a:b], r=R, mode=self.gn_mode, chunk=self.gn_chunk,
                                              cache=False)
        op1, op2 = mk(0, h), mk(h, xc.shape[0])
        def Hfull(v):
            out = op1(v)
            for a_, b_ in zip(out, op2(v)):
                a_.add_(b_).mul_(0.5)
            return out
        D = self.delta / (self.delta + self.t + 1); g0 = self.lr * self.g0_mult
        dz = self.mdamp * D / g0
        E, m = self._bilinear([Hfull, op1], zA, zB, dz)
        if was:
            model.train()
        Ec, Eh = E
        Ed = 2 * Ec - Eh                                     # Richardson: bias ~ 1/c
        teff = max(self.teff, 1.0); neff = 0.5 * self.batch_seqs * teff
        d2 = float(_dot([a_ - b_ for a_, b_ in zip(zA, zB)], [a_ - b_ for a_, b_ in zip(zA, zB)]))
        trH = max(self.q["trF"], 1e-30)
        tau_raw = neff / 2 * d2 / trH
        P = self.last_loss if math.isfinite(self.last_loss) else tau_raw
        tq = tau_raw / max(P, 1e-30)
        self.tq = tq if self.tq is None else (1 - self.r_tau) * self.tq + self.r_tau * tq
        self.tau = self.tq * P
        raw = float(np.clip(Ed / max(self.tau, 1e-30), 0.0, 1.0))
        self.rho = raw if self.rho is None else (1 - self.r_rho) * self.rho + self.r_rho * raw
        hv = [torch.sqrt(self.state[p]["v"]).mean() for p, hd in zip(self.plist, self.hid) if hd and "v" in self.state.get(p, {})]
        hn = [p.numel() for p, hd in zip(self.plist, self.hid) if hd and "v" in self.state.get(p, {})]
        self.sbar = float(sum(float(a_) * n for a_, n in zip(hv, hn)) / max(sum(hn), 1)) + self.epsilon
        self.meas = dict(t=self.t, E_c=Ec, E_half=Eh, E_delta=Ed, rich_gap=abs(Ec - Eh) / max(abs(Ed), 1e-30), lz_m=m, tau_raw=tau_raw,
                         tau=self.tau, tau_over_P=self.tq, rho_raw=raw, rho=self.rho, delta_z=dz, P=P, seconds=time.time() - t0)

    def _bilinear(self, ops, za, zb, dz):
        """<za, (H + dz)^-1 zb> for each operator H in ops, by a Lanczos recurrence from zb per operator (no reorthogonalization,
        c_j = <za, v_j> accumulated on the fly); each stops when its estimate changes by < lz_eps over 16 steps.  The recurrences
        run one after the other, so only one holds its vectors at a time; returns the estimates and the largest step count."""
        E, ms = [], []
        for op in ops:
            e, m = self._bilinear1(op, za, zb, dz)
            E.append(e[0]); ms.append(m)
        return E, max(ms)

    def _bilinear1(self, op, za, zb, dz):
        bn = math.sqrt(max(float(_dot(zb, zb)), 1e-300))
        st = [dict(v=[x / bn for x in zb], vp=None, bp=0.0, al=[], be=[], c=[], hist=[])]
        ops = [op]

        def est(s):
            T = np.diag(s["al"]) + np.diag(s["be"][:-1], 1) + np.diag(s["be"][:-1], -1)
            th, Q = np.linalg.eigh(T)
            return bn * float(np.array(s["c"]) @ (Q @ (Q[0] / np.maximum(th + dz, 1e-30))))
        m = 0
        for m in range(1, self.k_lz + 1):
            for op, s in zip(ops, st):
                s["c"].append(float(_dot(za, s["v"])))
                w = op(s["v"])
                if s["vp"] is not None:
                    for wi, vi in zip(w, s["vp"]):
                        wi.add_(vi, alpha=-s["bp"])
                a = float(_dot(s["v"], w))
                for wi, vi in zip(w, s["v"]):
                    wi.add_(vi, alpha=-a)
                b = math.sqrt(max(float(_dot(w, w)), 0.0))
                s["al"].append(a); s["be"].append(b)
                s["vp"], s["v"], s["bp"] = s["v"], [wi / max(b, 1e-30) for wi in w], b
            if m % 16 == 0:
                done = True
                for s in st:
                    s["hist"].append(est(s))
                    if len(s["hist"]) < 2 or abs(s["hist"][-1] - s["hist"][-2]) > self.lz_eps * abs(s["hist"][-1]):
                        done = False
                if done:
                    break
        return [est(s) for s in st], m

    V3KEYS = ("v3_E_c", "v3_E_half", "v3_E_delta", "v3_rich_gap", "v3_lz_m", "v3_tau_raw", "v3_tau", "v3_tau_over_P", "v3_rho_raw", "v3_rho",
              "v3_delta_z", "v3_P", "v3_seconds", "v3_K", "v3_S", "v3_r", "v3_nu", "v3_sbar", "v3_K_hid", "v3_K_term", "v3_bnr",
              "v3_mu_k64", "v3_r_hid", "v3_r_term", "v3_K_gain", "v3_r_gain")

    def _save_diag(self):
        # every row of slq_diag.npz carries the same v3 fields (NaN at refreshes without the excess form): the saver takes its keys
        # from the first row
        for k in self.V3KEYS:
            self.last_refresh.setdefault(k, float("nan"))
        super()._save_diag()

    # ---------------------------------------------------------------- the rule
    def _kernel(self, D):
        """K = K_hidden + K_terminal.  Hidden msign matrices: the msign kernel N_M (or the SLQ kernel with kernel = 'slq').  wte and
        lm_head (mom = 'hidden+terminal'): the SLQ kernel, exact for their AdamW plain step.  Block weights from the one quadrature."""
        th = np.maximum(self.q["nodes"], 0.0); wT = self.q["wT"]
        g = self.lr * self.g0_mult
        slq = lambda w: float(np.sum(w * g * th / (g * th + D)))
        wh = wT[:, self.hid].sum(axis=1)
        if self.kernel == "slq":
            self.K_hid = slq(wh)
        else:
            Be = self.batch_seqs * max(self.teff, 1.0)
            nu = self.msign_lr * self.sfac * self.c_W * math.sqrt(self.bb_fac * Be * self.sbar / (self.N_W * max(self.tau, 1e-30)))
            self.nu = nu
            self.K_hid = float(np.sum(wh * g * th / (D + nu * np.sqrt(th))))
        tm, gm = self.mom & self.term, self.mom & self.gain
        self.K_term = slq(wT[:, tm].sum(axis=1)) if tm.any() else 0.0
        self.K_gain = slq(wT[:, gm].sum(axis=1)) if gm.any() else 0.0
        return self.K_hid + self.K_term + self.K_gain

    def _compute_ratio(self, D):
        self.Delta = D
        L = len(self.plist)
        # version 1's cheap signal, recorded for comparison (not used by the rule): B / B_noise = 4 <g_A, g_B> / |g_A - g_B|^2 from the
        # running half-batch sums (sequence units) and mu = min{1, k' (B / B_noise) / B} at k' = 64 (version 1's Enoki setting)
        if self.n_split > 0 and self.eD.sum() > 0:
            self.bnr = 4.0 * max(float(self.eX.sum()), 0.0) / float(self.eD.sum())
            self.mu = float(min(1.0, 64.0 * self.bnr / self.batch_seqs))
        if self.q is None or self.rho is None:
            self.ratio = np.zeros(L); self.A = np.zeros(L); return
        K = max(self._kernel(D), 1e-12)
        Be = self.batch_seqs * max(self.teff, 1.0)
        self.S = self.s * Be * self.rho; self.N = K
        tm, gm = self.mom & self.term, self.mom & self.gain
        if self.split == "independent" and (tm.any() or gm.any()):
            # each block against the same budget: the hidden block's momentum is not throttled by the large terminal kernel
            rb = lambda Kb: min(self.cap, self.S / max(Kb, 1e-12))
            self.r_hid = rb(self.K_hid); self.r_term = rb(self.K_term) if tm.any() else 0.0
            self.r_gain = rb(self.K_gain) if gm.any() else 0.0
            self.ratio = np.where(self.hid, self.r_hid, np.where(tm, self.r_term, np.where(gm, self.r_gain, 0.0)))
        else:
            r = min(self.cap, self.S / K); self.r_hid = r
            self.r_term = r if tm.any() else 0.0; self.r_gain = r if gm.any() else 0.0
            self.ratio = np.where(self.mom, r, 0.0)
        self.A = self.ratio / D

    @torch.no_grad()
    def step(self, closure=None):
        self.sfac = self.param_groups[0]["lr"] / self.lr        # s(t): the schedule factor of this step (msign rate nu_t)
        return super().step(closure)

    def refresh_message(self):
        q = self.meas
        if not q:
            return f"[hybrid-v3] refresh t={self.t}: SLQ only (no halves or curvature batch yet)"
        return (f"[hybrid-v3] refresh t={q['t']}: E_c {q['E_c']:.4g} E_half {q['E_half']:.4g} -> E_delta {q['E_delta']:.4g} (rich gap "
                f"{q['rich_gap']:.2f}, lz m {q['lz_m']}) tau/P {q['tau_over_P']:.3f} rho {q['rho_raw']:.4f} -> {q['rho']:.4f} "
                f"T_eff {self.teff:.0f} K {self.N:.4g} S {self.S:.4g} r {self.r_hid:.4g} (term {self.r_term:.3g}, gain {self.r_gain:.3g}) "
                f"delta_z {q['delta_z']:.3g} ({q['seconds']:.0f}s + SLQ {self.last_refresh.get('seconds', 0):.0f}s)")

    def diagnostics(self):
        logs = super().diagnostics()
        for k, v in self.meas.items():
            if isinstance(v, (int, float)):
                logs[f"v3/{k}"] = float(v)
        logs["v3/r"] = self.r_hid
        logs["v3/K"] = float(self.N); logs["v3/S"] = float(self.S); logs["v3/K_hid"] = self.K_hid; logs["v3/K_term"] = self.K_term
        logs["v3/r_hid"] = self.r_hid; logs["v3/r_term"] = self.r_term; logs["v3/r_gain"] = self.r_gain; logs["v3/K_gain"] = self.K_gain
        logs["v3/bnr"] = float(self.bnr); logs["v3/mu_k64"] = float(self.mu)
        if self.kernel == "msign":
            logs["v3/nu"] = float(getattr(self, "nu", float("nan")))
        return logs
