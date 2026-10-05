"""Muon-DANA-F2g: d-muon with its hidden-matrix momentum replaced by DANA momentum whose ratio r_t is set per matrix by
the F2g rule (dana-excess.tex, Section "A basis-free rule for sign-SVD: drift power", the log-time kernel F2g).

Update (hidden matrices: p.ndim >= 2 and p.size(0) < 10000, as d-muon; Newton-Schulz on the gradient, no Muon momentum):
    u = NS(g),   m <- (1 - Delta_t) m + Delta_t u,   p <- p (1 - lr wd) - lr c_p [u + (r_t / Delta_t) m],
    c_p = matched_rms sqrt(max(rows, cols)),  Delta_t = delta / (delta + t)            (d-muon's c_p and weight decay)
Every other tensor: d-muon's AdamW (betas, eps, p <- p (1 - lr wd)), copied from DistributedMuon.

Rule, per hidden matrix:  r_t = sup{r in [0, 1]: K(r) <= K(0) + c (E - K(0))_+}.
  * E, the drift power of the two half-batch updates: x_t = <NS(g1), NS(g2)> (g1, g2 the mean gradients of the two
    halves of the global batch, from set_split_stats), running mean xbar and second moment Vbar at rate
    rho_t = max(Delta_t, 1/t), E = xbar - nse * sqrt(rho Vbar / (2 - rho)).
  * K(r) = 1/2 sum_j tau_j kappa(a0 sqrt(w_j); r, Delta_t)  (qbar = 1/2; kappa of Lemma 2 with gamma_2 = lr_t c_p,
    gamma_3 = gamma_2 r / Delta_t), a0 = Nbar / S_L, S_L = sum_j tau_j sqrt(w_j).
  * Numerator, every R steps (refresh): N = <u', D NS(G)[H_seq u']>, with u' the previous step's update (independent of
    the current batch), G the current global-batch gradient, D NS the forward-mode derivative of the same Newton-Schulz
    map (or of the exact polar factor, polar="svd"), and H_seq the per-sequence sampled-label Gauss-Newton estimate on
    the first `seqs` sequences of the last micro-batch (an exchangeable subset of the batch):
        H_seq U = (1 / N_tok) sum_a <S_a, U> S_a,   S_a = sum_{t in a} b_t x_t^T,
    b_t the gradient of the summed sampled-label loss at the layer output, x_t the layer input (forward hooks).  This is
    unbiased for the Gauss-Newton matrix of the mean token loss.  Nbar: log-time average at rate
    max(1 - (1 - Delta_t)^R, 1/(n + 1)) (window ~ t / delta).
  * Nodes w_j, weights tau_j: k Lanczos steps (full reorthogonalization) from u' on the per-token estimate
        H_tok U = (1 / N_tok) sum_t (b_t^T U x_t) b_t x_t^T
    at geometrically spaced times (first refresh, then the first refresh with t >= geo * t_last).  Only the shape of this
    spectrum matters: a0 = Nbar / S_L makes sum_j tau_j h_j = Nbar whatever the scale of H_tok.
  r_t = 0 before step `warm` and before the first numerator.
State per hidden matrix: the DANA buffer, u' and 2k + 5 scalars (in self.state[p], so checkpoints carry it).
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from .muon import adjust_lr_wd_for_muon, zeropower_via_newtonschulz5
from .muon_dana_slq import _polar_factors, _polar_pow


# ------------------------------------------------------------------ kernel and rule (host side, vectorized over matrices)
def kappa(h, D, eps, r):
    """Per-mode noise mass kappa(h; r, Delta) of Lemma 2 (dana-excess.tex); +inf outside the stable region."""
    g1, g2, g3 = D, eps, eps * r / D
    Nn = g2 ** 2 * h * (1 - D) * (2 - D) + g2 * D * (2 - D) + g1 * g2 * g3 * h * (1 - D) + g1 * g3 * (2 - D)
    d1 = D + g2 * h * (1 - D)
    d2 = (2 - D) * (2 - g2 * h) - g1 * g3 * h
    ok = (d1 > 0) & (d2 > 0)
    return np.where(ok, h * Nn / np.where(ok, d1 * d2, 1.0), np.inf)


def rule_r(w, tau, a0, D, eps, E, c, n_bis=30, rmax=1.0):
    """r per matrix (searched in [0, rmax]) and K(0).  w, tau: (L, k); a0, eps, E: (L,)."""
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        h = a0[:, None] * np.sqrt(np.maximum(w, 0.0))
        on = tau > 0

        def K(r):
            return 0.5 * np.sum(np.where(on, tau * kappa(h, D, eps[:, None], r[:, None]), 0.0), axis=1)

        L = len(a0)
        K0 = K(np.zeros(L))
        tgt = K0 + c * np.maximum(E - K0, 0.0)
        lo, hi = np.zeros(L), np.full(L, float(rmax))
        for _ in range(n_bis):
            mid = 0.5 * (lo + hi)
            ok = K(mid) <= tgt
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        r = np.where(K(np.full(L, float(rmax))) <= tgt, float(rmax), lo)
        r = np.where(np.isfinite(K0) & (E > K0), r, 0.0)
    return r, K0


# ------------------------------------------------------------------ Gauss-Newton products from captured layer vectors
def h_tok(X, B, U, ntok):
    """Per-token sampled GN of a linear layer: (1/N) sum_t (b_t^T U x_t) b_t x_t^T.  X (N, n), B (N, m), U (m, n)."""
    s = ((B @ U) * X).sum(1)
    return B.T @ (s[:, None] * X) / ntok


def h_seq(X, B, seq, nseq, U, ntok):
    """Per-sequence sampled GN: (1/N) sum_a <S_a, U> S_a, S_a = sum_{t in a} b_t x_t^T; seq (N,) sequence index."""
    s = ((B @ U) * X).sum(1)
    c = torch.zeros(nseq, dtype=s.dtype, device=s.device).index_add_(0, seq, s)
    return B.T @ (c[seq][:, None] * X) / ntok


def dns(G, E, steps):
    """D NS(G)[E]: forward-mode derivative of the Newton-Schulz map (fp32, or fp64 for fp64 inputs)."""
    dt = torch.float64 if G.dtype == torch.float64 else torch.float32
    f = lambda Z: zeropower_via_newtonschulz5(Z, steps=steps)
    return torch.func.jvp(f, (G.to(dt),), (E.to(dt),))[1]


def lanczos(op, v0, k):
    """k steps of Lanczos with full reorthogonalization on a symmetric PSD op (matrix -> matrix).  Returns Gauss nodes and
    weights of the spectral measure of v0: <v0, f(H) v0> ~ sum_j tau_j f(w_j)."""
    shape, d = v0.shape, v0.numel()
    nb = float(v0.norm())
    Q = torch.zeros(k, d, dtype=v0.dtype, device=v0.device)
    al, be = np.zeros(k), np.zeros(k)
    q = v0.reshape(-1) / max(nb, 1e-30)
    qprev, bprev, m = torch.zeros_like(q), 0.0, k
    for j in range(k):
        Q[j] = q
        w = op(q.view(shape)).reshape(-1)
        a = float(q @ w)
        w = w - a * q - bprev * qprev
        for _ in range(2):
            w = w - Q[:j + 1].T @ (Q[:j + 1] @ w)
        b = float(w.norm())
        al[j], be[j] = a, b
        if b < 1e-7 * (abs(a) + bprev + 1e-30):
            m = j + 1
            break
        qprev, q, bprev = q, w / b, b
    T = np.diag(al[:m]) + np.diag(be[:m - 1], 1) + np.diag(be[:m - 1], -1)
    th, S = np.linalg.eigh(T)
    tau = nb ** 2 * S[0] ** 2
    tau = np.where(th > 1e-9 * max(th.max(), 1e-30), tau, 0.0)
    return th, tau, m


def _world():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _mean_across_ranks_(t):
    if _world() > 1:
        dist.all_reduce(t)
        t /= _world()
    return t


class MuonDanaF2G(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, delta=8.0, weight_decay=0.0, adamw_betas=(0.8, 0.999), adamw_eps=1e-8,
                 ns_steps=5, matched_rms=0.2, c=0.2, R=25, geo=1.25, k=40, seqs=16, chunk=4, nse=2.0, warm=50,
                 polar="ns", polar_floor=2e-3, store="fp32", hidden_lr_mult=1.0, fixed=None, wd_decaying=False,
                 wd_ts=1.0, rmax=1.0, log_dir=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, adamw_betas=adamw_betas, adamw_eps=adamw_eps)
        super().__init__(params, defaults)
        self.delta, self.ns_steps, self.matched_rms = float(delta), int(ns_steps), float(matched_rms)
        self.c, self.R, self.geo, self.k = float(c), int(R), float(geo), int(k)
        self.seqs, self.chunk, self.nse, self.warm = int(seqs), int(chunk), float(nse), int(warm)
        if polar not in ("ns", "svd"):
            raise ValueError(f"muon-dana-f2g: polar must be 'ns' or 'svd', got {polar}")
        self.polar, self.polar_floor = polar, float(polar_floor)
        if store not in ("fp32", "bf16"):
            raise ValueError(f"muon-dana-f2g: store must be 'fp32' or 'bf16', got {store}")
        self.store = store                          # dtype of the captured layer vectors (products always in fp32/fp64)
        self.hmult = float(hidden_lr_mult)          # hidden-matrix step = lr * hmult * c_p (...); AdamW and weight decay use lr
        # weight decay: d-muon's p <- p (1 - lr_t wd) (default), or ADana's log-time form
        # p <- p (1 - wd s(t) / (1 + t / wd_ts)), s(t) = lr_t / peak lr (independent of the peak lr and of hmult)
        self.lr0, self.wd_decaying, self.wd_ts = float(lr), bool(wd_decaying), float(wd_ts)
        # fixed schedule control: r_t = min(rmax, c0 (1 + t/delta)^-kappa) from the first step; the rule still runs (shadow)
        self.fixed = None if fixed is None else (float(fixed[0]), float(fixed[1]))
        self.rmax = float(rmax)                     # cap on r for the rule and the fixed schedule (1 = Nesterov's g1 g3 = g2)
        self.r_rule = None
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.plist, self.pnames = [], []
        for g in self.param_groups:
            names = g.get("param_names", [None] * len(g["params"]))
            for p, n in zip(g["params"], names):
                self.plist.append(p); self.pnames.append(n if n is not None else f"param{len(self.pnames)}")
        self.pidx = {id(p): i for i, p in enumerate(self.plist)}
        self.use_muon = [p.ndim >= 2 and p.size(0) < 10000 and "wte" not in n and "lm_head" not in n
                         for p, n in zip(self.plist, self.pnames)]
        self.fidx = [i for i, u in enumerate(self.use_muon) if u]        # hidden matrices (the rule acts on these)
        self.ftype = [self._type(self.pnames[i]) for i in self.fidx]
        g0 = self.param_groups[0]
        g0.setdefault("f2g_t", 0); g0.setdefault("f2g_n", 0); g0.setdefault("f2g_tL", 0.0)
        g0.setdefault("f2g_nL", 0); g0.setdefault("f2g_nref", 0)
        L = len(self.fidx)
        self.r = np.zeros(L); self.K0 = np.zeros(L); self.E = np.zeros(L); self.a0 = np.zeros(L)
        self.last_N = np.full(L, np.nan)
        self.refresh_seconds, self.last_refresh = 0.0, {}
        self.history = []
        self._mods = None

    # ---------------------------------------------------------------- bookkeeping
    @staticmethod
    def _type(name):
        parts = name.split(".")
        for i, s in enumerate(parts):
            if s in ("attn", "mlp") and i + 1 < len(parts):
                return f"{s}.{parts[i + 1]}"
        return name.replace(".weight", "")

    @property
    def t(self):
        return int(self.param_groups[0]["f2g_t"])

    def _scale(self, p):
        return adjust_lr_wd_for_muon(1.0, self.matched_rms, p.shape)

    def _lr_of(self, p):
        for g in self.param_groups:
            for q in g["params"]:
                if q is p:
                    return g["lr"], g["weight_decay"]
        raise KeyError

    def _fstate(self, p):
        st = self.state[p]
        if "f2g" not in st:
            # xbar, Vbar, Nbar, S_L; nodes and weights of the latest Lanczos run
            st["f2g"] = torch.zeros(4, dtype=torch.float64)
            st["f2g_w"] = torch.zeros(self.k, dtype=torch.float64)
            st["f2g_tau"] = torch.zeros(self.k, dtype=torch.float64)
        return st

    # ---------------------------------------------------------------- drift power (called by train() before refresh/step)
    @torch.no_grad()
    def set_split_stats(self, half, full):
        tn = self.t + 1
        D = self.delta / (self.delta + tn)
        rho = max(D, 1.0 / tn)
        xs = []
        for i in self.fidx:
            h, f = half[i], full[i]
            if h is None or f is None:
                xs.append(torch.zeros((), device=self.plist[i].device)); continue
            g1, g2 = 2.0 * h, 2.0 * (f - h)
            u1 = zeropower_via_newtonschulz5(g1.bfloat16(), steps=self.ns_steps).float()
            u2 = zeropower_via_newtonschulz5(g2.bfloat16(), steps=self.ns_steps).float()
            xs.append((u1 * u2).sum())
        x = torch.stack(xs).double().cpu().numpy()
        for j, i in enumerate(self.fidx):
            st = self._fstate(self.plist[i])["f2g"]
            st[0] = (1 - rho) * float(st[0]) + rho * x[j]
            st[1] = (1 - rho) * float(st[1]) + rho * x[j] ** 2

    # ---------------------------------------------------------------- refresh: numerator every R steps, Lanczos at geometric times
    def needs_refresh(self):
        tn = self.t + 1
        return tn >= self.R and tn % self.R == 0 and all("u_prev" in self.state[self.plist[i]] for i in self.fidx)

    def _modules(self, model):
        if self._mods is None:
            ids = {id(self.plist[i]): j for j, i in enumerate(self.fidx)}
            mods = [None] * len(self.fidx)
            for _, mod in model.named_modules():
                if isinstance(mod, nn.Linear) and id(mod.weight) in ids:
                    mods[ids[id(mod.weight)]] = mod
            self._mods = mods
        return self._mods

    def _capture(self, model, x, y):
        """Layer inputs X (N, n) and sampled-label output gradients B (N, m) on the first `seqs` sequences, per hidden
        matrix; seq (N,) the sequence index of each token; N the number of valid tokens."""
        mods = self._modules(model)
        ns = min(self.seqs, x.shape[0])
        Xs, Bs = [[] for _ in mods], [[] for _ in mods]
        seqs, ntok = [], 0.0
        self._last_labels = []
        for c0 in range(0, ns, self.chunk):
            xc, yc = x[c0:min(ns, c0 + self.chunk)], y[c0:min(ns, c0 + self.chunk)]
            cap = {}
            handles = []
            for j, mod in enumerate(mods):
                if mod is None:
                    continue
                def hook(m_, inp, out, j=j):
                    cap[j] = (inp[0].detach(), out)
                handles.append(mod.register_forward_hook(hook))
            try:
                with torch.enable_grad(), torch.autocast(device_type=xc.device.type, enabled=False):
                    logits = model(xc, targets=yc, get_logits=True)["logits"]
                    logits = logits if logits.dtype == torch.float64 else logits.float()
                    ys = torch.distributions.Categorical(logits=logits).sample()
                    self._last_labels.append(ys.detach())
                    valid = (yc >= 0).to(logits.dtype)
                    ce = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), ys.reshape(-1), reduction="none")
                    loss = (ce.view_as(valid) * valid).sum()
                    js = sorted(cap)
                    gr = torch.autograd.grad(loss, [cap[j][1] for j in js])
            finally:
                for hd in handles:
                    hd.remove()
            for j, gj in zip(js, gr):
                dt_ = torch.bfloat16 if self.store == "bf16" else (torch.float64 if gj.dtype == torch.float64 else torch.float32)
                Xs[j].append(cap[j][0].reshape(-1, cap[j][0].shape[-1]).to(dt_))
                Bs[j].append(gj.reshape(-1, gj.shape[-1]).to(dt_))
            b, T = yc.shape
            seqs.append(torch.arange(c0, c0 + b, device=xc.device).repeat_interleave(T))
            ntok += float(valid.sum())
            del logits, ys, ce, loss, gr, cap
        X = [torch.cat(v) if v else None for v in Xs]
        B = [torch.cat(v) if v else None for v in Bs]
        # products are normalized per rank (rank-local token count) and then averaged across ranks
        return X, B, torch.cat(seqs), ns, ntok

    @torch.no_grad()
    def refresh(self, model, x, y):
        prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        t0 = time.perf_counter()
        try:
            self._refresh(model, x, y)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
        sec = time.perf_counter() - t0
        self.refresh_seconds += sec
        self.last_refresh["seconds"] = sec
        self._save_diag()

    def _refresh(self, model, x, y):
        g0 = self.param_groups[0]
        tn = self.t + 1
        D = self.delta / (self.delta + tn)
        X, B, seq, nseq, ntok = self._capture(model, x, y)
        do_L = tn >= g0["f2g_tL"]
        Ns = torch.zeros(len(self.fidx), dtype=torch.float64)
        lanc = []
        for j, i in enumerate(self.fidx):
            p = self.plist[i]
            st = self._fstate(p)
            if X[j] is None:
                continue
            Xj = X[j].float() if X[j].dtype == torch.bfloat16 else X[j]
            Bj = B[j].float() if B[j].dtype == torch.bfloat16 else B[j]
            X[j] = B[j] = None                              # free the stored copy as we go
            u = st["u_prev"].to(Xj.dtype)
            Hu = h_seq(Xj, Bj, seq, nseq, u, ntok)
            G = p.grad.to(Xj.dtype)
            if self.polar == "ns":
                DH = dns(G, Hu, self.ns_steps)
            else:
                DH = _polar_pow(_polar_factors(G, self.polar_floor), Hu, 1.0)
            Ns[j] = float((u * DH.to(u.dtype)).sum())
            if do_L:
                op = lambda V, Xj=Xj, Bj=Bj: _mean_across_ranks_(h_tok(Xj, Bj, V, ntok))
                th, tau, m = lanczos(op, u, self.k)
                lanc.append(m)
                st["f2g_w"].zero_(); st["f2g_tau"].zero_()
                st["f2g_w"][:len(th)] = torch.from_numpy(np.maximum(th, 0.0))
                st["f2g_tau"][:len(tau)] = torch.from_numpy(tau)
                st["f2g"][3] = float(np.sum(tau * np.sqrt(np.maximum(th, 0.0))))
                del op
            del Xj, Bj, Hu, DH
        Ns = _mean_across_ranks_(Ns.to(x.device)).cpu() if _world() > 1 else Ns
        rho = max(1.0 - (1.0 - D) ** self.R, 1.0 / (g0["f2g_n"] + 1))
        for j, i in enumerate(self.fidx):
            st = self._fstate(self.plist[i])["f2g"]
            st[2] = (1 - rho) * float(st[2]) + rho * float(Ns[j])
        self.last_N = Ns.numpy().copy()
        g0["f2g_n"] += 1
        g0["f2g_nref"] += 1
        if do_L:
            g0["f2g_tL"] = self.geo * tn
            g0["f2g_nL"] += 1
        self.last_refresh = dict(t=tn, lanczos=bool(do_L), lanczos_m=float(np.mean(lanc)) if lanc else float("nan"),
                                 ntok=ntok, N_median=float(np.median(Ns.numpy())))
        del X, B, seq

    def refresh_message(self):
        lr_ = self.last_refresh
        if not lr_:
            return None
        Nb = np.array([float(self._fstate(self.plist[i])["f2g"][2]) for i in self.fidx])
        return (f"[f2g] refresh t={lr_['t']}: N med {lr_['N_median']:.3g} (Nbar med {np.median(Nb):.3g})"
                + (f", Lanczos m={lr_['lanczos_m']:.0f}" if lr_["lanczos"] else "")
                + f"; r med {np.median(self.r):.3f} max {np.max(self.r) if len(self.r) else 0:.3f} ({lr_.get('seconds', 0):.1f}s)")

    # ---------------------------------------------------------------- step
    def _rule(self, D):
        g0 = self.param_groups[0]
        L = len(self.fidx)
        if L == 0:
            return np.zeros(0)
        W = np.stack([self._fstate(self.plist[i])["f2g_w"].numpy() for i in self.fidx])
        TAU = np.stack([self._fstate(self.plist[i])["f2g_tau"].numpy() for i in self.fidx])
        S = np.array([float(self._fstate(self.plist[i])["f2g"][3]) for i in self.fidx])
        Nb = np.array([float(self._fstate(self.plist[i])["f2g"][2]) for i in self.fidx])
        xb = np.array([float(self._fstate(self.plist[i])["f2g"][0]) for i in self.fidx])
        Vb = np.array([float(self._fstate(self.plist[i])["f2g"][1]) for i in self.fidx])
        tn = self.t + 1
        rho = max(D, 1.0 / tn)
        E = xb - self.nse * np.sqrt(np.maximum(Vb, 0.0) * rho / (2.0 - rho))
        eps = np.array([self._lr_of(self.plist[i])[0] * self.hmult * self._scale(self.plist[i]) for i in self.fidx])
        a0 = np.where(S > 0, np.maximum(Nb, 0.0) / np.maximum(S, 1e-300), 0.0)
        r, K0 = rule_r(W, TAU, a0, D, eps, E, self.c, rmax=self.rmax)
        ready = tn >= self.warm and g0["f2g_n"] > 0 and g0["f2g_nL"] > 0
        self.E, self.K0, self.a0 = E, K0, a0
        r = np.where((Nb > 0) & (S > 0), r, 0.0)                 # no positive numerator yet: no floor known, no momentum
        return r if ready else np.zeros(L)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        g0 = self.param_groups[0]
        tn = self.t + 1
        D = self.delta / (self.delta + tn)
        self.r = self._rule(D)
        if self.fixed is not None:
            c0, kap = self.fixed
            self.r_rule = self.r
            self.r = np.full(len(self.fidx), min(self.rmax, c0 * (1.0 + tn / self.delta) ** (-kap)))
        rmap = {self.fidx[j]: float(self.r[j]) for j in range(len(self.fidx))}
        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
            lr, wd = group["lr"], group["weight_decay"]
            decay = (wd * (lr / self.lr0) / (1.0 + tn / self.wd_ts)) if self.wd_decaying else lr * wd
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                i = self.pidx[id(p)]
                st = self.state[p]
                g = p.grad
                if self.use_muon[i]:
                    if "m" not in st:
                        st["m"] = torch.zeros_like(p)
                    u = zeropower_via_newtonschulz5(g.bfloat16(), steps=self.ns_steps).to(p.dtype)
                    st["u_prev"] = (u.clone() if u.dtype == torch.float64 else u.float().clone())
                    st["m"].mul_(1 - D).add_(u, alpha=D)
                    p.mul_(1 - decay)
                    p.add_(u.add_(st["m"], alpha=rmap.get(i, 0.0) / D), alpha=-lr * self.hmult * self._scale(p))
                else:
                    if "adamw_exp_avg" not in st:
                        st["adamw_exp_avg"] = torch.zeros_like(g)
                        st["adamw_exp_avg_sq"] = torch.zeros_like(g)
                    b1, b2 = st["adamw_exp_avg"], st["adamw_exp_avg_sq"]
                    b1.lerp_(g, 1 - beta1)
                    b2.lerp_(g.square(), 1 - beta2)
                    step = group["step"]
                    scale = (1 - beta1 ** step) / (1 - beta2 ** step) ** 0.5
                    p.mul_(1 - decay)
                    p.add_(b1 / (eps + b2.sqrt()), alpha=-lr / scale)
        g0["f2g_t"] = tn
        return loss

    # ---------------------------------------------------------------- logging
    def diagnostics(self):
        logs = {"f2g/r_mean": float(np.mean(self.r)) if len(self.r) else 0.0,
                "f2g/r_max": float(np.max(self.r)) if len(self.r) else 0.0,
                "f2g/refreshes": int(self.param_groups[0]["f2g_nref"]),
                "f2g/lanczos_runs": int(self.param_groups[0]["f2g_nL"]),
                "f2g/refresh_seconds": self.refresh_seconds}
        if self.r_rule is not None and len(self.r_rule):
            logs["f2g/r_rule_shadow_mean"] = float(np.mean(self.r_rule))
            logs["f2g/r_rule_shadow_max"] = float(np.max(self.r_rule))
        D = self.delta / (self.delta + max(self.t, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            red = np.where(self.K0 > 0, self.E / self.K0 - 1.0, np.nan)
        Nb = np.array([float(self._fstate(self.plist[i])["f2g"][2]) for i in self.fidx]) if self.fidx else np.zeros(0)
        xb = np.array([float(self._fstate(self.plist[i])["f2g"][0]) for i in self.fidx]) if self.fidx else np.zeros(0)
        for ty in sorted(set(self.ftype)):
            idx = [j for j, t_ in enumerate(self.ftype) if t_ == ty]
            logs[f"f2g_type/r/{ty}"] = float(np.mean(self.r[idx]))
            logs[f"f2g_type/A/{ty}"] = float(np.mean(self.r[idx]) / D)
            logs[f"f2g_type/reducible/{ty}"] = float(np.nanmedian(red[idx])) if np.isfinite(red[idx]).any() else 0.0
            logs[f"f2g_type/a0/{ty}"] = float(np.mean(self.a0[idx]))
            logs[f"f2g_type/Nbar/{ty}"] = float(np.mean(Nb[idx]))
            logs[f"f2g_type/xbar/{ty}"] = float(np.mean(xb[idx]))
        return logs

    def _save_diag(self):
        if self.log_dir is None or (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0):
            return
        F = [self._fstate(self.plist[i]) for i in self.fidx]
        self.history.append(dict(t=self.t + 1, r=self.r.copy(), E=self.E.copy(), K0=self.K0.copy(), a0=self.a0.copy(),
                                 N=self.last_N.copy(), Nbar=np.array([float(f["f2g"][2]) for f in F]),
                                 xbar=np.array([float(f["f2g"][0]) for f in F]), S=np.array([float(f["f2g"][3]) for f in F]),
                                 w=np.stack([f["f2g_w"].numpy() for f in F]), tau=np.stack([f["f2g_tau"].numpy() for f in F]),
                                 lanczos=self.last_refresh.get("lanczos", False), seconds=self.last_refresh.get("seconds", 0.0),
                                 r_rule=(self.r_rule.copy() if self.r_rule is not None else self.r.copy())))
        out = {"names": np.array([self.pnames[i] for i in self.fidx])}
        for key in self.history[0]:
            out[key] = np.array([h_[key] for h_ in self.history])
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.log_dir / "f2g_diag.npz", **out)
        except Exception as e:  # never kill training over diagnostics
            print(f"[f2g] could not save diagnostics: {e}")
