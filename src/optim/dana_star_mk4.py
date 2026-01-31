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
        use_foreach: bool = False,
        ):
        """
        DANA-STAR MK4 optimizer.

        With tau probability estimator and optional SNR clipping. Two modes:
        - clipsnr=None: Dana-Star (tau + no SNR clipping)
        - clipsnr=float: Dana-Star-MK4 (tau + SNR clipping)

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            delta: Delta parameter for EMA coefficient.
            kappa: Kappa parameter for effective time scaling.
            mk4A: Must be 0.0 (kept for compatibility).
            mk4B: Must be 0.0 (kept for compatibility).
            epsilon: Small constant for numerical stability.
            weight_decay: Weight decay parameter.
            clipsnr: SNR clipping parameter. None disables clipping.
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
            lr=lr, delta=delta, epsilon=epsilon, weight_decay=weight_decay, weighted_step_count=0)
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

        super(DANA_STAR_MK4, self).__init__(params, defaults)

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
        tau: torch.Tensor,
        step: torch.Tensor,
        alpha: torch.Tensor,
        g2: torch.Tensor,
        g3: torch.Tensor,
        schedule_factor: torch.Tensor,
        wd: torch.Tensor,
        epsilon: float,
        clipsnr: float,
        kappa: float,
        wd_decaying: bool,
        wd_ts: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Compiled function for updating a single parameter.

        Note: Dynamic hyperparameters (step, alpha, g2, g3, schedule_factor, wd) are
        passed as 0-D tensors to avoid recompilation when values change.
        Static hyperparameters (epsilon, clipsnr, kappa, wd_ts) remain as scalars.

        Returns:
            Updated (m, v, tau) and diagnostics dict
        """
        # Update first moment (EMA of gradient) using in-place lerp
        # m = m * (1-alpha) + grad * alpha = lerp(m, grad, alpha)
        m.lerp_(grad, alpha)

        # Update second moment (EMA of gradient squared) using in-place ops
        v.mul_(1 - alpha).addcmul_(grad, grad, value=alpha)

        # Compute sqrt(v) + epsilon once and reuse
        sqrt_v_eps = torch.sqrt(v).add_(epsilon)

        # Update tau estimate
        # tau_update = |grad| / (|grad| + sqrt_v_eps)
        abs_grad = torch.abs(grad)
        tau_update = abs_grad / (abs_grad + sqrt_v_eps)
        tau.lerp_(tau_update, alpha)

        # Compute tau regularization
        clipped_tau = torch.clamp(tau, max=0.5)
        p_estimate = clipped_tau / (1.0 - clipped_tau)
        min_p = 1.0 / (1.0 + step)
        # Use relu-based max to avoid DTensor/tensor mixing issues with FSDP
        tau_reg = min_p + torch.relu(p_estimate - min_p)

        # Compute effective time (clamping is not needed since tau_reg is already clamped)
        effective_time = torch.clamp(tau * step, min=1.0)
        #effective_time = tau_reg * (1.0 + step)

        # Compute normalization term (reusing sqrt_v_eps)
        root_tau_reg = torch.sqrt(tau_reg)
        norm_term = root_tau_reg / sqrt_v_eps
        m_norm_term = torch.abs(m) * norm_term
        # Compute momentum factor and alpha factor
        mfac = (m_norm_term / tau_reg)
        if clipsnr is not None:
            alpha_factor = torch.clamp(
                (effective_time ** (1 - kappa)) * mfac,
                max=clipsnr
            )
        else:
            alpha_factor = (effective_time ** (1 - kappa)) * mfac

        # Compute g3 term (momentum-based update)
        g3_term = (-g3) * (torch.sign(m) * (tau_reg * alpha_factor + m_norm_term))

        # Compute g2 term (gradient-based update)
        g2_term = (-g2) * grad * norm_term

        # Combine updates
        update = g2_term + g3_term

        # Apply parameter update (in-place)
        p.add_(update)

        # Apply independent weight decay (paper convention):
        # WD is multiplied by schedule γ(t) but NOT by peak LR γ*
        # Formula: p = p + wd_factor * p = p * (1 + wd_factor)
        if wd_decaying:
            wd_factor = -wd / (1 + step / wd_ts) * schedule_factor
        else:
            wd_factor = -wd * schedule_factor
        p.mul_(1 + wd_factor)

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
            schedule_factor = group['lr'] / self.lr  # γ(t) without peak LR
            time_factor = schedule_factor
            group['weighted_step_count'] += time_factor
            delta = group['delta']
            wd = group['weight_decay']
            epsilon = group['epsilon']
            clipsnr = self.clipsnr

            # Use foreach path if enabled
            if self.use_foreach:
                # Convert scalars to tensors to avoid recompilation
                # Use CPU tensors since they'll be broadcast to each parameter's device
                g2_t = torch.tensor(g2, dtype=torch.float32)
                g3_t = torch.tensor(g3, dtype=torch.float32)
                sf_t = torch.tensor(schedule_factor, dtype=torch.float32)
                delta_t = torch.tensor(delta, dtype=torch.float32)
                wd_t = torch.tensor(wd, dtype=torch.float32)

                # Call the foreach step (with state extraction outside compiled region)
                self._foreach_step_group(
                    group, g2_t, g3_t, sf_t, delta_t, wd_t, epsilon, clipsnr
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
                step = state['step']
                alpha = delta / (delta + step)

                # Convert dynamic scalars to 0-D tensors to avoid recompilation
                # when values change. Static hyperparameters remain as Python scalars.
                device = p.device
                step_t = torch.tensor(step, device=device, dtype=torch.float32)
                alpha_t = torch.tensor(alpha, device=device, dtype=torch.float32)
                g2_t = torch.tensor(g2, device=device, dtype=torch.float32)
                g3_t = torch.tensor(g3, device=device, dtype=torch.float32)
                sf_t = torch.tensor(schedule_factor, device=device, dtype=torch.float32)
                wd_t = torch.tensor(wd, device=device, dtype=torch.float32)

                # Get compiled function for this tensor's shape
                update_fn = self._get_compiled_fn(p.shape)

                # Call rank-specific compiled update function
                m_new, v_new, tau_new, diagnostics = update_fn(
                    p, grad, m, v, tau,
                    step_t, alpha_t, g2_t, g3_t, sf_t, wd_t,
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
        g2: torch.Tensor,
        g3: torch.Tensor,
        schedule_factor: torch.Tensor,
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
            taus.append(state['tau'])
            states.append(state)

        if not params:
            return

        # ===================================================================
        # CALL COMPILED COMPUTATION KERNEL
        # ===================================================================
        if self._foreach_compute_kernel is not None:
            alpha_factors, mfacs = self._foreach_compute_kernel(
                params, grads, ms, vs, taus, alpha_tensors, step_tensors,
                g2, g3, schedule_factor, wd, epsilon, clipsnr, self.kappa, self.wd_decaying, self.wd_ts
            )
        else:
            alpha_factors, mfacs = self._foreach_compute_kernel_impl(
                params, grads, ms, vs, taus, alpha_tensors, step_tensors,
                g2, g3, schedule_factor, wd, epsilon, clipsnr, self.kappa, self.wd_decaying, self.wd_ts
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
        taus,
        alpha_tensors,
        step_tensors,
        g2: torch.Tensor,
        g3: torch.Tensor,
        schedule_factor: torch.Tensor,
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
        # Note: _foreach_addcmul_ expects scalars tuple, not keyword arg
        grad_sq_scaled = torch._foreach_mul(grads, grads)
        grad_sq_scaled = torch._foreach_mul(grad_sq_scaled, alpha_tensors)
        torch._foreach_add_(vs, grad_sq_scaled)

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
        clipped_tau = [torch.clamp(tau, max=0.5) for tau in taus]

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
        ones_like_eff = [torch.ones_like(e) for e in effective_time]
        effective_time = torch._foreach_maximum(effective_time, ones_like_eff)

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
        eff_pow = torch._foreach_pow(effective_time, 1.0 - kappa)
        alpha_factor = torch._foreach_mul(eff_pow, mfac)
        if clipsnr is not None:
            alpha_factor = [torch.clamp(af, max=clipsnr) for af in alpha_factor]

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

        # Step 12: Independent weight decay (paper convention)
        # WD is multiplied by schedule γ(t) but NOT by peak LR γ*
        if wd_decaying:
            step_over_ts = torch._foreach_div(step_tensors, wd_ts)
            one_plus_ratio = torch._foreach_add(step_over_ts, 1.0)
            wd_decay = torch._foreach_reciprocal(one_plus_ratio)
            neg_wd_sf = -wd * schedule_factor
            wd_factors = torch._foreach_mul(wd_decay, neg_wd_sf)
        else:
            neg_wd_sf = -wd * schedule_factor
            wd_factors = [neg_wd_sf.to(device=p.device, dtype=p.dtype) for p in params]

        # Apply weight decay: p = p + p * wd_factor
        # If wd=0, this is a no-op (adds zero), but keeps graph intact
        wd_updates = torch._foreach_mul(params, wd_factors)
        torch._foreach_add_(params, wd_updates)

        # Return diagnostics for storage outside compiled region
        return alpha_factor, mfac
