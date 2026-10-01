"""SOAP-DANA-SLQ: DANA-SLQ long momentum on SOAP's preconditioner, in LaProp order (precondition first, then momentum).

Preconditioner (SOAP's, src/optim/soap.py).  For a matrix block G (out x in) the statistics GL = E[G G^T] and
GR = E[G^T G] are short EMAs (shampoo_beta); their eigenbases Q_L, Q_R come from eigh at the first step and are refreshed
by one power iteration + QR every precondition_frequency steps.  A dimension above max_precond_dim keeps the identity
(wte, lm_head: one-sided).  The second moment v lives in the rotated basis (EMA beta2, bias-corrected):
    P^{-1} G = Q_L [ (Q_L^T G Q_R) / (sqrt(v_hat) + eps) ] Q_R^T.
Vectors (norms, biases) use the same rule with Q = identity (RMSProp).  With split_qkv the fused QKV matrix is split into
its q, k, v rows; each is a block with its own statistics and its own share of the momentum budget.

Step (ADana's EMA convention m = Delta_t y for the DANA sum buffer y):
    u = P^{-1} g,   m <- (1 - Delta_t) m + Delta_t u,   theta <- theta - gamma(t) [u + (r_T / Delta_t) m]   (+ wd)
In z = P^{1/2} theta this is SGD with DANA momentum on H_z = P^{-1/2} GN P^{-1/2}, so ADanaSLQ's rule applies unchanged:
r_T from the budget sum_T r_T N_T <= s B_eff mu (any --slq_alloc), the quadrature run with R = P^{-1/2} (rotate, scale,
rotate back), and the buffer signal from |m_T|_z^2 = <m, P m> and the split-half noise <d, P^{-1} d>.
The momentum is stored in the rotated basis (m' = Q_L^T m Q_R) and re-rotated exactly when the basis is refreshed, so a
step costs SOAP's two projections; the split-half statistics add one.  Weight decay is ADana's (independent of the peak
LR, optionally log-time: wd / (1 + t / wd_ts)).  As in SOAP, the first step only initialises the preconditioner, so a
projection never uses its own gradient.
"""
import re

import torch

from .adana_slq import ADanaSLQ


def _rot(X, QL, QR):
    """Q_L^T X Q_R (None: identity on that side)."""
    if QL is not None:
        X = QL.T @ X
    if QR is not None:
        X = X @ QR
    return X


def _unrot(X, QL, QR):
    """Q_L X Q_R^T."""
    if QL is not None:
        X = QL @ X
    if QR is not None:
        X = X @ QR.T
    return X


def _eigvecs(M):
    """Eigenvectors of a symmetric PSD matrix, by decreasing eigenvalue (as SOAP.get_orthogonal_matrix)."""
    I = torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
    try:
        _, Q = torch.linalg.eigh(M + 1e-30 * I)
    except Exception:
        _, Q = torch.linalg.eigh(M.double() + 1e-30 * I.double())
        Q = Q.to(M.dtype)
    return torch.flip(Q, [1])


def _is_qkv(name):
    return re.sub(r"^(module\.|_orig_mod\.)+", "", name).endswith("attn.c_attn.weight")


class SOAPDanaSLQ(ADanaSLQ):
    def __init__(self, params, lr=1.0, delta=8.0, epsilon=1e-8, weight_decay=0.0, wd_decaying=False, wd_ts=1.0,
                 beta2=0.95, shampoo_beta=-1.0, precondition_frequency=10, max_precond_dim=10000, split_qkv=True,
                 **slq_kw):
        super().__init__(params, lr=lr, delta=delta, epsilon=epsilon, weight_decay=weight_decay, clipsnr=None,
                         wd_decaying=wd_decaying, wd_ts=wd_ts, **slq_kw)
        self.beta2 = float(beta2)
        self.sb = self.beta2 if shampoo_beta < 0 else float(shampoo_beta)
        self.freq, self.maxd, self.split_qkv = int(precondition_frequency), int(max_precond_dim), bool(split_qkv)
        self.nsplit = [3 if self.split_qkv and _is_qkv(n) and p.dim() == 2 and p.shape[0] % 3 == 0 else 1
                       for p, n in zip(self.plist, self.pnames)]
        bnames, self.bidx = [], []
        for n, k in zip(self.pnames, self.nsplit):
            self.bidx.append(list(range(len(bnames), len(bnames) + k)))
            bnames += [n] if k == 1 else [n.replace("c_attn.weight", f"c_attn_{c}.weight") for c in "qkv"]
        self._setup_blocks(bnames)
        self.n_basis = 0                            # basis refreshes (summed over blocks)

    # ---------------------------------------------------------------- preconditioner state
    def _chunks(self, i, X):
        k = self.nsplit[i]
        return X.chunk(k, 0) if k > 1 else (X,)

    def _init_state(self, i, p, g):
        blocks = []
        for gb in self._chunks(i, g):
            b = dict(v=torch.zeros_like(gb), m=torch.zeros_like(gb), GL=None, GR=None, QL=None, QR=None)
            if gb.dim() == 2:
                o, n = gb.shape
                if o <= self.maxd:
                    b["GL"] = (1 - self.sb) * (gb @ gb.T); b["QL"] = _eigvecs(b["GL"])
                if n <= self.maxd:
                    b["GR"] = (1 - self.sb) * (gb.T @ gb); b["QR"] = _eigvecs(b["GR"])
            blocks.append(b)
        self.state[p]["blocks"] = blocks
        self.state[p]["step"] = 0

    def _refresh_basis(self, b):
        """One power iteration + QR per side (SOAP.get_orthogonal_matrix_QR).  v is permuted with the basis and kept
        (as in SOAP); the momentum is re-rotated exactly."""
        m_par = _unrot(b["m"], b["QL"], b["QR"])
        v = b["v"]
        for side, (Gk, Qk) in enumerate((("GL", "QL"), ("GR", "QR"))):
            if b[Gk] is None:
                continue
            G, Q = b[Gk], b[Qk]
            idx = torch.argsort(torch.diag(Q.T @ G @ Q), descending=True)
            v = v.index_select(side, idx)
            b[Qk], _ = torch.linalg.qr(G @ Q[:, idx])
        b["v"] = v
        b["m"] = _rot(m_par, b["QL"], b["QR"])
        self.n_basis += 1

    def _den(self, b, step, eps):
        return (b["v"] / (1 - self.beta2 ** step)).sqrt_().add_(eps)

    # ---------------------------------------------------------------- SLQ hooks
    def _slq_splits(self):
        return self.nsplit

    def _slq_r(self):
        """R = P^{-1/2}: Q_L [(Q_L^T U Q_R) (sqrt(v_hat) + eps)^{-1/2}] Q_R^T per block (diagonal for vectors)."""
        r = []
        for p in self.plist:
            st = self.state.get(p, {})
            if "blocks" not in st or st["step"] == 0:
                r.append(torch.ones_like(p))
                continue
            blocks = st["blocks"]
            dens = [self._den(b, st["step"], self.epsilon).rsqrt_() for b in blocks]
            if len(blocks) == 1 and blocks[0]["QL"] is None and blocks[0]["QR"] is None:
                r.append(dens[0])
                continue

            def rz(U, blocks=blocks, dens=dens):
                k = len(blocks)
                out = [_unrot(_rot(Ub.to(d.dtype), b["QL"], b["QR"]) * d, b["QL"], b["QR"]).to(U.dtype)
                       for Ub, b, d in zip(U.chunk(k, 0) if k > 1 else (U,), blocks, dens)]
                return torch.cat(out, 0) if k > 1 else out[0]
            r.append(rz)
        return r

    @torch.no_grad()
    def set_split_stats(self, half, full):
        """Per-block <g1, g2>, |d|^2 and |d|^2_{P^{-1}} = sum (Q_L^T d Q_R)^2 / (sqrt(v_hat) + eps), d = g1 - g2."""
        vals, idx = [], []
        for i, (p, h, f) in enumerate(zip(self.plist, half, full)):
            if h is None or f is None:
                continue
            g1 = 2.0 * h.float(); g2 = 2.0 * (f.float() - h.float())
            d = g1 - g2
            st = self.state.get(p, {})
            ready = "blocks" in st and st["step"] > 0
            for j, (a, c, dd) in enumerate(zip(self._chunks(i, g1), self._chunks(i, g2), self._chunks(i, d))):
                dg = (dd * dd).sum()
                if ready:
                    b = st["blocks"][j]
                    dr = _rot(dd.to(b["v"].dtype), b["QL"], b["QR"])
                    dz = (dr * dr / self._den(b, st["step"], self.epsilon)).sum()
                else:
                    dz = dg
                vals.append(torch.stack([(a * c).sum(), dg, dz.float()]))
                idx.append(self.bidx[i][j])
        L = len(self.bnames)
        xg, dg, dz = [torch.zeros(L, dtype=torch.float64) for _ in range(3)]
        if vals:
            V = torch.stack(vals).double().cpu()
            xg[idx], dg[idx], dz[idx] = V[:, 0], V[:, 1], V[:, 2]
        self._accumulate_split(xg.numpy(), dg.numpy(), dz.numpy())

    # ---------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        fresh = set()
        for i, p in enumerate(self.plist):
            if p.grad is not None and "blocks" not in self.state[p]:
                self._init_state(i, p, p.grad)
                fresh.add(id(p))
        if fresh and self.t == 0:
            return loss                             # first call: preconditioner initialised, no update (as SOAP)
        self.t += 1
        self._compute_ratio(self.delta / (self.delta + self.t))
        mz2, mz2_idx = [], []
        for group in self.param_groups:
            g2 = group["lr"]
            schedule_factor = g2 / self.lr
            wd = group["weight_decay"]
            eps = group["epsilon"]
            for p in group["params"]:
                if p.grad is None or id(p) in fresh:
                    continue
                i = self.pidx[id(p)]
                st = self.state[p]
                st["step"] += 1
                step = st["step"]
                Dm = self.delta / (self.delta + step)
                outs = []
                for j, (b, gb) in enumerate(zip(st["blocks"], self._chunks(i, p.grad))):
                    bi = self.bidx[i][j]
                    gr = _rot(gb, b["QL"], b["QR"])
                    b["v"].mul_(self.beta2).addcmul_(gr, gr, value=1 - self.beta2)
                    den = self._den(b, step, eps)
                    ur = gr / den
                    b["m"].mul_(1 - Dm).add_(ur, alpha=Dm)
                    outs.append(_unrot(ur.add_(b["m"], alpha=float(self.A[bi])), b["QL"], b["QR"]))
                    mz2.append((b["m"] * b["m"] * den).sum()); mz2_idx.append(bi)
                    # statistics after the projection (the next projections never use this gradient)
                    if b["GL"] is not None:
                        b["GL"].lerp_(gb @ gb.T, 1 - self.sb)
                    if b["GR"] is not None:
                        b["GR"].lerp_(gb.T @ gb, 1 - self.sb)
                    if step % self.freq == 0 and (b["GL"] is not None or b["GR"] is not None):
                        self._refresh_basis(b)
                p.add_(torch.cat(outs, 0) if len(outs) > 1 else outs[0], alpha=-g2)
                if self.wd_decaying:
                    wd_factor = -wd / (1 + step / self.wd_ts) * schedule_factor
                else:
                    wd_factor = -wd * schedule_factor
                p.mul_(1 + wd_factor)
        if mz2:
            self.mz2[mz2_idx] = torch.stack(mz2).double().cpu().numpy()
        return loss

    def diagnostics(self):
        logs = super().diagnostics()
        logs["slq/basis_refreshes"] = self.n_basis
        return logs
