import math
import torch
from torch.optim import Optimizer
from typing import Union, Callable, Iterable

torch._dynamo.config.cache_size_limit = 64

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

        # Create compiled versions of the update function, one per tensor rank
        # This prevents excessive recompilation when mixing 1D (biases) and 2D (weights) tensors
        # We use a cache to lazily compile for each unique rank encountered
        self._compiled_functions = {}

    def _get_compiled_fn(self, ndim):
        """Get or create compiled function for given tensor rank."""
        if ndim not in self._compiled_functions:
            self._compiled_functions[ndim] = torch.compile(
                self._update_param_compiled,
                dynamic=False,  # Handle different sizes within same rank
                fullgraph=False
            )
        return self._compiled_functions[ndim]

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

    @staticmethod
    def _update_param_compiled(
        p: torch.Tensor,
        grad: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
        tau: torch.Tensor,
        step: torch.Tensor,
        alpha: torch.Tensor,
        g2: torch.Tensor,
        g3: torch.Tensor,
        lr: torch.Tensor,
        wd: torch.Tensor,
        epsilon: float,
        clipsnr: float,
        kappa: float,
        wd_decaying: bool,
        wd_ts: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Compiled function for updating a single parameter.

        Note: Dynamic hyperparameters (step, alpha, g2, g3, lr, wd) are passed
        as 0-D tensors to avoid recompilation when values change.
        Static hyperparameters (epsilon, clipsnr, kappa, wd_ts) remain as scalars.

        Returns:
            Updated (m, v, tau) and diagnostics dict
        """
        # Update first moment (EMA of gradient)
        m = m.mul(1 - alpha).add(grad, alpha=alpha)

        # Update second moment (EMA of gradient squared)
        v = v.mul(1 - alpha).addcmul(grad, grad, value=alpha)

        # Update tau estimate
        tau_update = torch.abs(grad) / (torch.abs(grad) + torch.sqrt(v) + epsilon)
        tau = tau.mul(1 - alpha).add(tau_update, alpha=alpha)

        # Compute tau regularization
        clipped_tau = torch.clamp(tau, max=0.5)
        p_estimate = clipped_tau / (1.0 - clipped_tau)
        min_p = 1.0 / (1.0 + step)
        tau_reg = torch.clamp(p_estimate, min=min_p)

        # Compute effective time
        effective_time = torch.clamp(tau * step, min=1.0)

        # Compute normalization term
        root_tau_reg = torch.sqrt(tau_reg)
        norm_term = root_tau_reg / (torch.sqrt(v) + epsilon)

        # Compute momentum factor and alpha factor
        mfac = (norm_term * torch.abs(m) / tau_reg)
        alpha_factor = torch.clamp(
            (effective_time ** (1 - kappa)) * mfac,
            max=clipsnr
        )

        # Compute g3 term (momentum-based update)
        g3_term = g3 * (tau_reg * torch.sign(m) * alpha_factor + 1.0 * m * norm_term)

        # Compute g2 term (gradient-based update)
        g2_term = g2 * grad * norm_term

        # Combine updates
        update = -(g2_term + g3_term)

        # Apply parameter update
        p = p.add(update)

        # Apply decoupled weight decay (AdamW-style)
        # Use tensor operations to avoid .item() in compiled code
        if wd_decaying:
            wd_factor = -wd / (1 + step / wd_ts) * lr
        else:
            wd_factor = -wd * lr
        p = p.add(p, alpha=wd_factor)

        # Compute diagnostics
        diagnostics = {
            'current_alpha': alpha_factor.mean(),
            'gradient_norm': grad.norm(),
            'auto_factor_mean': mfac.mean(),
            'm_norm': m.norm(),
        }

        return m, v, tau, diagnostics

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

                # Convert dynamic scalars to 0-D tensors to avoid recompilation
                # when values change. Static hyperparameters remain as Python scalars.
                device = p.device
                step_t = torch.tensor(step, device=device, dtype=torch.float32)
                alpha_t = torch.tensor(alpha, device=device, dtype=torch.float32)
                g2_t = torch.tensor(g2, device=device, dtype=torch.float32)
                g3_t = torch.tensor(g3, device=device, dtype=torch.float32)
                lr_t = torch.tensor(lr, device=device, dtype=torch.float32)
                wd_t = torch.tensor(wd, device=device, dtype=torch.float32)

                # Get compiled function for this tensor's shape
                update_fn = self._get_compiled_fn(p.shape)

                # Call rank-specific compiled update function
                m_new, v_new, tau_new, diagnostics = update_fn(
                    p, grad, m, v, tau,
                    step_t, alpha_t, g2_t, g3_t, lr_t, wd_t,
                    epsilon, clipsnr, self.kappa, self.wd_decaying, self.wd_ts
                )

                # Update state tensors in-place
                m.copy_(m_new)
                v.copy_(v_new)
                tau.copy_(tau_new)

                # Store diagnostics
                state["current_alpha"] = diagnostics['current_alpha'].detach()
                state["gradient_norm"] = diagnostics['gradient_norm'].detach()
                state["auto_factor_mean"] = diagnostics['auto_factor_mean'].detach()
                state["current_kappa_factor"] = state["current_alpha"] / state["auto_factor_mean"]
                state["m_norm"] = diagnostics['m_norm'].detach()

        return loss
