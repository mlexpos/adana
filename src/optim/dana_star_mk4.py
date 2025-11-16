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
        use_foreach: bool = True,
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
            use_foreach: Whether to use fused foreach operations (default: True).
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
        self.use_foreach = use_foreach

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
        tau_reg = torch.maximum(p_estimate, torch.full_like(tau, min_p))

        # Compute effective time
        effective_time = torch.maximum(tau * step, torch.ones_like(tau))

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

            # Use foreach path if enabled
            if self.use_foreach:
                self._foreach_step_group(
                    group, g2, g3, lr, time_factor, delta, wd, epsilon, clipsnr
                )
                continue

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

    def _foreach_step_group(
        self,
        group,
        g2: float,
        g3: float,
        lr: float,
        time_factor: float,
        delta: float,
        wd: float,
        epsilon: float,
        clipsnr: float,
    ) -> None:
        """
        Highly optimized foreach update with maximum fusion.

        This implementation minimizes kernel launches by:
        - Using lerp for EMA updates
        - Using addcmul/addcdiv for fused operations
        - Reusing intermediate results (sqrt_vs)
        - Eliminating Python list comprehensions
        - Using foreach operations wherever possible
        """
        # Collect parameters and state
        params, grads, ms, vs, taus = [], [], [], [], []
        alpha_tensors, step_tensors = [], []
        states = []

        for p in group['params']:
            grad = p.grad
            if grad is None:
                continue

            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['m'] = torch.zeros_like(p)
                state['v'] = torch.zeros_like(p)
                state['tau'] = torch.zeros_like(p)

            state['step'] += 1

            if self.weight_time:
                step_value = group['weighted_step_count']
                alpha = delta / (delta + step_value) * time_factor
            else:
                step_value = state['step']
                alpha = delta / (delta + step_value)

            device = p.device
            alpha_tensors.append(torch.as_tensor(alpha, device=device, dtype=p.dtype))
            step_tensors.append(torch.as_tensor(step_value, device=device, dtype=p.dtype))

            params.append(p)
            grads.append(grad)
            ms.append(state['m'])
            vs.append(state['v'])
            taus.append(state['tau'])
            states.append(state)

        if not params:
            return

        # =====================================================================
        # OPTIMIZED FOREACH OPERATIONS - MAXIMUM FUSION
        # =====================================================================

        # Step 1: Update first moment using lerp (linear interpolation)
        # m = lerp(m, grad, alpha) = m + alpha * (grad - m) = m * (1-alpha) + grad * alpha
        # This is more efficient than separate mul + add
        torch._foreach_lerp_(ms, grads, alpha_tensors)

        # Step 2: Update second moment using addcmul
        # v = v * (1 - alpha) + alpha * grad^2
        # First scale v by (1 - alpha)
        one_minus_alphas = torch._foreach_neg(alpha_tensors)
        torch._foreach_add_(one_minus_alphas, 1.0)
        torch._foreach_mul_(vs, one_minus_alphas)
        # Then add alpha * grad^2 using addcmul
        torch._foreach_addcmul_(vs, grads, grads, value=alpha_tensors)

        # Step 3: Update tau estimate
        # tau_update = |grad| / (|grad| + sqrt(v) + epsilon)
        abs_grads = torch._foreach_abs(grads)
        sqrt_vs = torch._foreach_sqrt(vs)  # SAVE THIS - reuse later!

        # Compute denominator: |grad| + sqrt(v) + epsilon
        denom = torch._foreach_add(abs_grads, sqrt_vs)
        torch._foreach_add_(denom, epsilon)

        # tau_update = |grad| / denominator
        tau_updates = torch._foreach_div(abs_grads, denom)

        # Update tau using lerp
        torch._foreach_lerp_(taus, tau_updates, alpha_tensors)

        # Step 4: Tau regularization
        # clipped_tau = clamp(tau, max=0.5)
        # p_estimate = clipped_tau / (1 - clipped_tau)
        clipped_tau = torch._foreach_clamp(taus, max=0.5)

        # Compute (1 - clipped_tau) using foreach ops (avoid list comprehension!)
        one_minus_clipped = torch._foreach_neg(clipped_tau)
        torch._foreach_add_(one_minus_clipped, 1.0)

        p_estimate = torch._foreach_div(clipped_tau, one_minus_clipped)

        # tau_reg = max(p_estimate, 1/(1+step))
        # This is equivalent to clamp(p_estimate, min=1/(1+step)), but since min varies per param,
        # we use _foreach_maximum with computed min_p values
        one_plus_step = torch._foreach_add(step_tensors, 1.0)
        min_p = torch._foreach_reciprocal(one_plus_step)
        tau_reg = torch._foreach_maximum(p_estimate, min_p)

        # Step 5: Effective time
        # effective_time = max(tau * step, 1) = clamp(tau * step, min=1)
        effective_time = torch._foreach_mul(taus, step_tensors)
        effective_time = torch._foreach_clamp(effective_time, min=1.0)

        # Step 6: Normalization term
        # norm_term = sqrt(tau_reg) / (sqrt(v) + epsilon)
        # REUSE sqrt_vs from Step 3!
        sqrt_tau_reg = torch._foreach_sqrt(tau_reg)
        norm_den = torch._foreach_add(sqrt_vs, epsilon)
        norm_term = torch._foreach_div(sqrt_tau_reg, norm_den)

        # Step 7: Momentum factor
        # mfac = norm_term * |m| / tau_reg
        m_abs = torch._foreach_abs(ms)
        m_over_tau = torch._foreach_div(m_abs, tau_reg)
        mfac = torch._foreach_mul(norm_term, m_over_tau)

        # Step 8: Alpha factor with clipping
        # alpha_factor = clamp((effective_time^(1-kappa)) * mfac, max=clipsnr)
        eff_pow = torch._foreach_pow(effective_time, 1.0 - self.kappa)
        alpha_factor = torch._foreach_mul(eff_pow, mfac)
        alpha_factor = torch._foreach_clamp(alpha_factor, max=clipsnr)

        # Step 9: Compute g3 term
        # g3_term = g3 * (tau_reg * sign(m) * alpha_factor + m * norm_term)
        sign_m = torch._foreach_sign(ms)

        # First part: tau_reg * sign(m) * alpha_factor
        # Use temporary to avoid extra allocations
        tau_sign = torch._foreach_mul(tau_reg, sign_m)
        tau_sign_alpha = torch._foreach_mul(tau_sign, alpha_factor)

        # Second part: m * norm_term
        m_norm_term = torch._foreach_mul(ms, norm_term)

        # Combine: tau_sign_alpha + m_norm_term
        g3_inner = torch._foreach_add(tau_sign_alpha, m_norm_term)
        g3_term = torch._foreach_mul(g3_inner, g3)

        # Step 10: Compute g2 term
        # g2_term = g2 * grad * norm_term
        grad_norm = torch._foreach_mul(grads, norm_term)
        g2_term = torch._foreach_mul(grad_norm, g2)

        # Step 11: Combine updates and apply
        # update = -(g2_term + g3_term)
        update = torch._foreach_add(g2_term, g3_term)
        torch._foreach_neg_(update)  # Negate in place
        torch._foreach_add_(params, update)

        # Step 12: Weight decay
        if wd != 0:
            if self.wd_decaying:
                # wd_factor = -wd / (1 + step / wd_ts) * lr
                # Use foreach operations to compute factors
                step_over_ts = torch._foreach_div(step_tensors, self.wd_ts)
                one_plus_ratio = torch._foreach_add(step_over_ts, 1.0)
                wd_decay = torch._foreach_reciprocal(one_plus_ratio)
                wd_factors = torch._foreach_mul(wd_decay, -wd * lr)
            else:
                # All factors are the same, but still use foreach for consistency
                wd_factors = [torch.tensor(-wd * lr, device=p.device, dtype=p.dtype) for p in params]

            # Apply weight decay: p = p + p * wd_factor
            wd_updates = torch._foreach_mul(params, wd_factors)
            torch._foreach_add_(params, wd_updates)

        # Step 13: Store diagnostics (minimal Python loop)
        for state, a_factor, g, m_tensor, m_fac in zip(states, alpha_factor, grads, ms, mfac):
            state["current_alpha"] = a_factor.mean().detach()
            state["gradient_norm"] = g.norm().detach()
            state["auto_factor_mean"] = m_fac.mean().detach()
            state["current_kappa_factor"] = state["current_alpha"] / state["auto_factor_mean"]
            state["m_norm"] = m_tensor.norm().detach()
