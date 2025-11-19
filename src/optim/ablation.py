import torch
from torch.optim import Optimizer
import math
from typing import Union, Callable, Iterable


class AdamWDecayingWD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        wd_decaying: bool = False,
        wd_ts: float = 1.0,
    ):
        """
        AdamW optimizer with optional decaying weight decay.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            betas: Coefficients used for computing running averages of gradient and its square.
            epsilon: Term added to denominator for numerical stability.
            weight_decay: Weight decay parameter.
            wd_decaying: Whether to decay weight decay over time.
            wd_ts: Timescale for weight decay decay.
        """
        defaults = dict(lr=lr, betas=betas, epsilon=epsilon, weight_decay=weight_decay)
        self.wd_decaying = wd_decaying
        self.wd_ts = wd_ts
        super(AdamWDecayingWD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            epsilon = group['epsilon']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected moments
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1
                bias_correction2_sqrt = bias_correction2 ** 0.5

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(epsilon)

                # Update parameters
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Apply decoupled weight decay (AdamW-style)
                if weight_decay != 0:
                    if self.wd_decaying:
                        wd_factor = -weight_decay / (1 + step / self.wd_ts) * lr
                    else:
                        wd_factor = -weight_decay * lr
                    p.mul_(1 + wd_factor)

        return loss

torch._dynamo.config.cache_size_limit = 64

class DANA_MK4(Optimizer):

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
        use_foreach: bool = False,
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
            weight_time: Must be False (kept for compatibility).
            wd_decaying: Whether to decay weight decay over time.
            wd_ts: Timescale for weight decay decay.
            use_foreach: Whether to use fused foreach operations (default: False).
        """
        # Enforce mk4A, mk4B, and weight_time constraints
        assert mk4A == 0.0, f"mk4A must be 0.0, got {mk4A}"
        assert mk4B == 0.0, f"mk4B must be 0.0, got {mk4B}"
        assert weight_time == False, f"weight_time must be False, got {weight_time}"

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
        self.weight_time = False  # Hardcoded
        self.wd_decaying = wd_decaying
        self.wd_ts = wd_ts
        self.use_foreach = use_foreach

        super(DANA_MK4, self).__init__(params, defaults)

        # Create compiled versions of the update function, one per tensor rank
        # This prevents excessive recompilation when mixing 1D (biases) and 2D (weights) tensors
        # We use a cache to lazily compile for each unique rank encountered
        self._compiled_functions = {}

        # Compile the foreach computation kernel if using foreach
        if self.use_foreach:
            self._foreach_compute_kernel = torch.compile(
                self._foreach_compute_kernel_impl,
                fullgraph=False,
                dynamic=True
            )
        else:
            self._foreach_compute_kernel = None

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

    @staticmethod
    def _update_param_compiled(
        p: torch.Tensor,
        grad: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
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
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compiled function for updating a single parameter.

        Note: Dynamic hyperparameters (step, alpha, g2, g3, lr, wd) are passed
        as 0-D tensors to avoid recompilation when values change.
        Static hyperparameters (epsilon, clipsnr, kappa, wd_ts) remain as scalars.

        Returns:
            Updated (m, v) and diagnostics dict
        """
        # Update first moment (EMA of gradient) using in-place lerp
        # m = m * (1-alpha) + grad * alpha = lerp(m, grad, alpha)
        m.lerp_(grad, alpha)

        # Update second moment (EMA of gradient squared) using in-place ops
        v.mul_(1 - alpha).addcmul_(grad, grad, value=alpha)

        # Compute sqrt(v) + epsilon once and reuse
        sqrt_v_eps = torch.sqrt(v).add_(epsilon)

        # Compute normalization term directly (no tau)
        norm_term = 1.0 / sqrt_v_eps
        m_norm_term = torch.abs(m) * norm_term
        
        # Compute momentum factor and alpha factor using step-based time
        effective_time = 1.0 + step
        mfac = m_norm_term
        alpha_factor = torch.clamp(
            (effective_time ** (1 - kappa)) * mfac,
            max=clipsnr
        )

        # Compute g3 term (momentum-based update, no tau_reg)
        g3_term = (-g3) * (torch.sign(m) * alpha_factor + m * norm_term)

        # Compute g2 term (gradient-based update)
        g2_term = (-g2) * grad * norm_term

        # Combine updates
        update = g2_term + g3_term

        # Apply parameter update (in-place)
        p.add_(update)

        # Apply decoupled weight decay (AdamW-style)
        # Use tensor operations to avoid .item() in compiled code
        # Formula: p = p + wd_factor * p = p * (1 + wd_factor)
        if wd_decaying:
            wd_factor = -wd / (1 + step / wd_ts) * lr
        else:
            wd_factor = -wd * lr
        p.mul_(1 + wd_factor)

        # Compute diagnostics
        diagnostics = {
            'current_alpha': alpha_factor.mean(),
            'gradient_norm': grad.norm(),
            'auto_factor_mean': mfac.mean(),
            'm_norm': m.norm(),
        }

        return m, v, diagnostics

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
                # Convert scalars to tensors to avoid recompilation
                # Use CPU tensors since they'll be broadcast to each parameter's device
                g2_t = torch.tensor(g2, dtype=torch.float32)
                g3_t = torch.tensor(g3, dtype=torch.float32)
                lr_t = torch.tensor(lr, dtype=torch.float32)
                delta_t = torch.tensor(delta, dtype=torch.float32)
                wd_t = torch.tensor(wd, dtype=torch.float32)

                # Call the foreach step (with state extraction outside compiled region)
                self._foreach_step_group(
                    group, g2_t, g3_t, lr_t, delta_t, wd_t, epsilon, clipsnr
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

                m, v = state['m'], state['v']
                state['step'] += 1

                # Compute EMA coefficient: alpha = delta / (delta + t)
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
                m_new, v_new, diagnostics = update_fn(
                    p, grad, m, v,
                    step_t, alpha_t, g2_t, g3_t, lr_t, wd_t,
                    epsilon, clipsnr, self.kappa, self.wd_decaying, self.wd_ts
                )

                # Update state tensors in-place
                m.copy_(m_new)
                v.copy_(v_new)

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
        g2: torch.Tensor,
        g3: torch.Tensor,
        lr: torch.Tensor,
        delta: torch.Tensor,
        wd: torch.Tensor,
        epsilon: float,
        clipsnr: float,
    ) -> None:
        """
        Foreach update with state extraction (uncompiled) + computation kernel (compiled).

        This method extracts state and prepares tensors (uncompiled, can access self.state),
        then calls a compiled computation kernel for the actual math operations.
        """
        # ===================================================================
        # STATE EXTRACTION (UNCOMPILED - accesses self.state and group)
        # ===================================================================
        params, grads, ms, vs = [], [], [], []
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

            state['step'] += 1

            device = p.device

            step_value = state['step']
            step_t = torch.tensor(step_value, device=device, dtype=p.dtype)
            alpha_t = delta / (delta + step_t)
            alpha_tensors.append(alpha_t.to(device))

            step_tensors.append(step_t)

            params.append(p)
            grads.append(grad)
            ms.append(state['m'])
            vs.append(state['v'])
            states.append(state)

        if not params:
            return

        # ===================================================================
        # CALL COMPILED COMPUTATION KERNEL
        # ===================================================================
        if self._foreach_compute_kernel is not None:
            alpha_factors, mfacs = self._foreach_compute_kernel(
                params, grads, ms, vs, alpha_tensors, step_tensors,
                g2, g3, lr, wd, epsilon, clipsnr, self.kappa, self.wd_decaying, self.wd_ts
            )
        else:
            alpha_factors, mfacs = self._foreach_compute_kernel_impl(
                params, grads, ms, vs, alpha_tensors, step_tensors,
                g2, g3, lr, wd, epsilon, clipsnr, self.kappa, self.wd_decaying, self.wd_ts
            )

        # ===================================================================
        # STORE DIAGNOSTICS (UNCOMPILED - accesses self.state)
        # ===================================================================
        for state, a_factor, g, m_tensor, m_fac in zip(states, alpha_factors, grads, ms, mfacs):
            state["current_alpha"] = a_factor.mean().detach()
            state["gradient_norm"] = g.norm().detach()
            state["auto_factor_mean"] = m_fac.mean().detach()
            state["current_kappa_factor"] = state["current_alpha"] / state["auto_factor_mean"]
            state["m_norm"] = m_tensor.norm().detach()

    def _foreach_compute_kernel_impl(
        self,
        params,
        grads,
        ms,
        vs,
        alpha_tensors,
        step_tensors,
        g2: torch.Tensor,
        g3: torch.Tensor,
        lr: torch.Tensor,
        wd: torch.Tensor,
        epsilon: float,
        clipsnr: float,
        kappa: float,
        wd_decaying: bool,
        wd_ts: float,
    ):
        """
        Pure computation kernel (compilable) - no access to self.state or group.

        This function contains only tensor operations and can be compiled without
        triggering guards on self.state or group['params'].

        Returns:
            alpha_factors, mfacs: For diagnostics
        """

        # =====================================================================
        # OPTIMIZED FOREACH OPERATIONS - MAXIMUM FUSION (NO TAU)
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
        # Note: _foreach_addcmul_ expects scalars tuple, not keyword arg
        grad_sq_scaled = torch._foreach_mul(grads, grads)
        grad_sq_scaled = torch._foreach_mul(grad_sq_scaled, alpha_tensors)
        torch._foreach_add_(vs, grad_sq_scaled)

        # Step 3: Compute normalization term (no tau)
        # norm_term = 1 / (sqrt(v) + epsilon)
        sqrt_vs = torch._foreach_sqrt(vs)
        norm_den = torch._foreach_add(sqrt_vs, epsilon)
        norm_term = torch._foreach_reciprocal(norm_den)

        # Step 4: Momentum factor (no tau_reg)
        # mfac = |m| * norm_term
        m_abs = torch._foreach_abs(ms)
        mfac = torch._foreach_mul(m_abs, norm_term)

        # Step 5: Effective time (using step directly, no tau)
        # effective_time = 1 + step
        effective_time = torch._foreach_add(step_tensors, 1.0)

        # Step 6: Alpha factor with clipping
        # alpha_factor = clamp((effective_time^(1-kappa)) * mfac, max=clipsnr)
        eff_pow = torch._foreach_pow(effective_time, 1.0 - kappa)
        alpha_factor = torch._foreach_mul(eff_pow, mfac)
        alpha_factor = [torch.clamp(af, max=clipsnr) for af in alpha_factor]

        # Step 7: Compute g3 term (no tau_reg)
        # g3_term = g3 * (sign(m) * alpha_factor + m * norm_term)
        sign_m = torch._foreach_sign(ms)

        # First part: sign(m) * alpha_factor
        sign_alpha = torch._foreach_mul(sign_m, alpha_factor)

        # Second part: m * norm_term
        m_norm_term = torch._foreach_mul(ms, norm_term)

        # Combine: sign_alpha + m_norm_term
        g3_inner = torch._foreach_add(sign_alpha, m_norm_term)
        g3_term = torch._foreach_mul(g3_inner, g3)

        # Step 8: Compute g2 term
        # g2_term = g2 * grad * norm_term
        grad_norm = torch._foreach_mul(grads, norm_term)
        g2_term = torch._foreach_mul(grad_norm, g2)

        # Step 9: Combine updates and apply
        # update = -(g2_term + g3_term)
        update = torch._foreach_add(g2_term, g3_term)
        torch._foreach_neg_(update)  # Negate in place
        torch._foreach_add_(params, update)

        # Step 10: Weight decay
        # Apply weight decay unconditionally (if wd=0, the update is zero anyway)
        if wd_decaying:
            # wd_factor = -wd / (1 + step / wd_ts) * lr
            # Use foreach operations to compute factors
            step_over_ts = torch._foreach_div(step_tensors, wd_ts)
            one_plus_ratio = torch._foreach_add(step_over_ts, 1.0)
            wd_decay = torch._foreach_reciprocal(one_plus_ratio)
            # Compute -wd * lr as a scalar tensor operation
            neg_wd_lr = -wd * lr
            wd_factors = torch._foreach_mul(wd_decay, neg_wd_lr)
        else:
            # All factors are the same
            # Compute -wd * lr and broadcast to each parameter
            neg_wd_lr = -wd * lr
            wd_factors = [neg_wd_lr.to(device=p.device, dtype=p.dtype) for p in params]

        # Apply weight decay: p = p + p * wd_factor
        # If wd=0, this is a no-op (adds zero), but keeps graph intact
        wd_updates = torch._foreach_mul(params, wd_factors)
        torch._foreach_add_(params, wd_updates)

        # Return diagnostics for storage outside compiled region
        return alpha_factor, mfac

