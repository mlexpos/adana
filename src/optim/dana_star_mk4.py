import math
import torch
from torch.optim import Optimizer
from typing import Union, Callable, Iterable


class DANA_STAR_MK4(Optimizer):

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1.0,
        delta: float = 8.0,
        kappa: float = 1.0,
        mk4A: float = 0.0,
        mk4B: float = 0.0,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        clipsnr: float = 1.0,
        weight_time: bool = False,
        wd_decaying: bool = False,
        wd_ts: float = 1.0,
        ):
        """
        DANA-STAR MK4 optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            delta: Delta parameter for EMA coefficient.
            kappa: Kappa parameter for effective time scaling.
            mk4A: Must be 0.0 (kept for compatibility).
            mk4B: Must be 0.0 (kept for compatibility).
            epsilon: Small constant for numerical stability.
            weight_decay: Weight decay parameter.
            clipsnr: SNR clipping parameter.
            weight_time: Whether to weight time by lr / lr_max factor.
            wd_decaying: Whether to decay weight decay over time.
            wd_ts: Timescale for weight decay decay.
        """
        # Enforce mk4A and mk4B are 0
        assert mk4A == 0.0, f"mk4A must be 0.0, got {mk4A}"
        assert mk4B == 0.0, f"mk4B must be 0.0, got {mk4B}"

        defaults = dict(
            lr=lr, delta=delta, clipsnr=clipsnr, epsilon=epsilon, weight_decay=weight_decay, weighted_step_count=0)
        self.lr = lr
        self.delta = delta
        self.kappa = kappa
        self.mk4A = 0.0  # Hardcoded
        self.mk4B = 0.0  # Hardcoded
        self.clipsnr = clipsnr
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.weight_time = weight_time
        self.wd_decaying = wd_decaying
        self.wd_ts = wd_ts

        super(DANA_STAR_MK4, self).__init__(params, defaults)

    def _make_schedule(self, value: Union[float, Callable[[int], float]]) -> Callable[[int], float]:
        """Convert scalar or schedule to callable function."""
        if callable(value):
            return value
        else:
            return lambda step: value

    def _clip_to_half(self, tau: torch.Tensor) -> torch.Tensor:
        """Clip tau values to at most 0.5."""
        return torch.clamp(tau, max=0.5)

    def _tau_reg(self, tau: torch.Tensor, step: int) -> torch.Tensor:
        """
        Tau regularization: converts tau/(1-tau) to p estimate and clips to prevent
        poor estimation when p << 1/t.
        """
        clipped_tau = self._clip_to_half(tau)
        p_estimate = clipped_tau / (1.0 - clipped_tau)
        min_p = torch.full_like(tau, 1.0 / (1.0 + step))
        return torch.maximum(p_estimate, min_p)

    def _root_tau_reg(self, tau: torch.Tensor, step: int) -> torch.Tensor:
        """Square root of tau regularization."""
        return torch.sqrt(self._tau_reg(tau, step))

    def _effective_time(self, tau: torch.Tensor, step: int) -> torch.Tensor:
        """Compute effective time for tau regularization."""
        return torch.maximum(tau * step, torch.ones_like(tau))

    def _tau_updater(
        self,
        g: torch.Tensor,
        v: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """Update tau estimate based on gradient and second moment."""
        return torch.abs(g) / (torch.abs(g) + torch.sqrt(v) + epsilon)

    def _norm_term(
        self,
        v: torch.Tensor,
        tau: torch.Tensor,
        step: int,
        epsilon: float
    ) -> torch.Tensor:
        """Compute normalization term for parameter updates."""
        root_tau_reg = self._root_tau_reg(tau, step)
        return root_tau_reg / (torch.sqrt(v) + epsilon)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Extract hyperparameters
            g2 = group['lr']
            g3 = group['lr']
            lr = group['lr']
            time_factor = group['lr'] / self.lr
            group['weighted_step_count'] += time_factor
            delta = group['delta']
            wd = group['weight_decay']
            epsilon = group['epsilon']
            clipsnr = group['clipsnr']

            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['tau'] = torch.zeros_like(p)

                m, v, tau = state['m'], state['v'], state['tau']
                state['step'] += 1

                # Compute EMA coefficient: alpha = delta / (delta + t)
                if self.weight_time:
                    step = group['weighted_step_count']
                    alpha = delta / (delta + step) * time_factor
                else:
                    step = state['step']
                    alpha = delta / (delta + step)

                # Update first moment (EMA of gradient)
                m.mul_(1 - alpha).add_(grad, alpha=alpha)

                # Update second moment (EMA of gradient squared)
                v.mul_(1 - alpha).addcmul_(grad, grad, value=alpha)

                # Update tau estimate
                tau_update = self._tau_updater(grad, v, epsilon)
                tau.mul_(1 - alpha).add_(tau_update, alpha=alpha)

                # Compute effective time
                effective_time = self._effective_time(tau, step)

                # Store diagnostics
                state["current_alpha"] = alpha

                # Compute normalization term
                norm_term = self._norm_term(v, tau, step, epsilon)

                # Compute momentum factor (mfac) and alpha factor
                # With mk4A = mk4B = 0, this simplifies significantly
                mfac = (norm_term * torch.abs(m) / self._tau_reg(tau, step))
                alpha_factor = torch.clamp(
                    (effective_time ** (1 - self.kappa)) * mfac,
                    max=clipsnr
                )

                # Compute g3 term (momentum-based update)
                g3_term = g3 * (self._tau_reg(tau, step) * torch.sign(m) * alpha_factor + 1.0 * m * norm_term)

                # Store additional diagnostics
                state["current_alpha"] = (alpha_factor).mean().detach()
                state["gradient_norm"] = grad.norm().detach()
                state["auto_factor_mean"] = mfac.mean().detach()
                state["current_kappa_factor"] = state["current_alpha"] / state["auto_factor_mean"]
                state["m_norm"] = m.norm().detach()

                # Compute g2 term (gradient-based update)
                g2_term = g2 * grad * norm_term

                # Combine updates
                update = -(g2_term + g3_term)

                # Apply decoupled weight decay (AdamW-style)
                if wd != 0:
                    if self.wd_decaying:
                        p.add_(p, alpha=-wd / (1 + step / self.wd_ts) * lr)
                    else:
                        p.add_(p, alpha=-wd * lr)

                # Apply parameter update
                p.add_(update)

        return loss
