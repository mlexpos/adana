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


def linear_warmup_scheduler(step, alpha_end, alpha_start=0, warmup=1):
    if step < warmup:
        a = step / float(warmup)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end


def linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):
    def f(beta, eps=1e-8):
        return math.log(0.5) / math.log(beta + eps) - 1

    def f_inv(t):
        return math.pow(0.5, 1 / (t + 1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))
    return beta_end


class AdEMAMix_DecayingWD(torch.optim.Optimizer):
    r"""Implements the AdEMAMix_DecayingWD algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999, 0.9999))
            corresponding to beta_1, beta_2, beta_3 in AdEMAMix
        alpha (float): AdEMAMix alpha coeficient mixing the slow and fast EMAs (default: 2)
        beta3_warmup (int, optional): number of warmup steps used to increase beta3 (default: None)
        alpha_warmup: (int, optional): number of warmup steps used to increase alpha (default: None)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay as in AdamW (default: 0)
        wd_decaying: Whether to decay weight decay over time.
        wd_ts: Timescale for weight decay decay.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999, 0.9999),
        alpha=2.0,
        beta3_warmup=None,
        alpha_warmup=None,
        eps=1e-8,
        weight_decay=0,
        gamma_3_factor=1.0,
        wd_decaying=False,
        wd_ts=1.0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            alpha=alpha,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
            weight_decay=weight_decay,
            gamma_3_factor=gamma_3_factor,
            wd_decaying=wd_decaying,
            wd_ts=wd_ts,
        )
        super(AdEMAMix_DecayingWD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdEMAMix_DecayingWD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]
            gamma_3_factor = group["gamma_3_factor"]
            wd_decaying = group["wd_decaying"]
            wd_ts = group["wd_ts"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    if beta1 != 0.0:  # save memory in case beta1 is 0.0
                        state["exp_avg_fast"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    else:
                        state["exp_avg_fast"] = None
                    state["exp_avg_slow"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg_fast, exp_avg_slow, exp_avg_sq = (
                    state["exp_avg_fast"],
                    state["exp_avg_slow"],
                    state["exp_avg_sq"],
                )

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute the effective alpha and beta3 in case warmup is used
                if alpha_warmup is not None:
                    alpha = linear_warmup_scheduler(
                        state["step"],
                        alpha_end=alpha_final,
                        alpha_start=0,
                        warmup=alpha_warmup,
                    )
                else:
                    alpha = alpha_final

                if beta3_warmup is not None:
                    beta3 = linear_hl_warmup_scheduler(
                        state["step"],
                        beta_end=beta3_final,
                        beta_start=beta1,
                        warmup=beta3_warmup,
                    )
                else:
                    beta3 = beta3_final

                # Store current alpha and beta_3 values for logging
                state["current_alpha"] = alpha
                state["current_beta3"] = beta3
                state["current_one_minus_beta3"] = 1 - beta3

                # Decay the first and second moment running average coefficient
                if beta1 != 0.0:
                    exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
                else:
                    exp_avg_fast = grad
                exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                update = (
                    exp_avg_fast.div(bias_correction1) + alpha * exp_avg_slow * gamma_3_factor
                ) / denom

                # decay weight decay
                if wd_decaying:
                    wd_factor = -wd / (1 + state["step"] / wd_ts) * lr
                else:
                    wd_factor = -wd * lr
                p.mul_(1 + wd_factor)

                p.add_(-lr * update)
                
        return loss

# class AdEMAMix_DecayingBETA2_DecayingWD(torch.optim.Optimizer):
#     r"""Implements the AdEMAMix_DecayingBETA2_DecayingWD algorithm.

#     Arguments:
#         params (iterable): iterable of parameters to optimize or dicts defining
#             parameter groups
#         lr (float, optional): learning rate (default: 1e-3)
#         betas (Tuple[float, float, float], optional): coefficients used for computing
#             running averages of gradient and its square (default: (0.9, 0.999, 0.9999))
#             corresponding to beta_1, beta_2, beta_3 in AdEMAMix
#         alpha (float): AdEMAMix alpha coeficient mixing the slow and fast EMAs (default: 2)
#         beta3_warmup (int, optional): number of warmup steps used to increase beta3 (default: None)
#         alpha_warmup: (int, optional): number of warmup steps used to increase alpha (default: None)
#         eps (float, optional): term added to the denominator to improve
#             numerical stability (default: 1e-8)
#         weight_decay (float, optional): weight decay as in AdamW (default: 0)
#         wd_decaying: Whether to decay weight decay over time.
#         wd_ts: Timescale for weight decay decay.
#     """

#     def __init__(
#         self,
#         params,
#         lr=1e-3,
#         betas=(0.9, 0.999, 0.9999), # beta_1, beta_2, beta_3 in AdEMAMix. beta_2 is unused.
#         alpha=2.0,
#         beta3_warmup=None,
#         alpha_warmup=None,
#         delta=8.0,
#         eps=1e-8,
#         weight_decay=0,
#         gamma_3_factor=1.0,
#         wd_decaying=False,
#         wd_ts=1.0,
#     ):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
#         if not 0.0 <= betas[2] < 1.0:
#             raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
#         if not 0.0 <= weight_decay:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#         if not 0.0 <= alpha:
#             raise ValueError("Invalid alpha value: {}".format(alpha))
#         defaults = dict(
#             lr=lr,
#             betas=betas,
#             eps=eps,
#             alpha=alpha,
#             beta3_warmup=beta3_warmup,
#             alpha_warmup=alpha_warmup,
#             delta=delta,
#             weight_decay=weight_decay,
#             gamma_3_factor=gamma_3_factor,
#             wd_decaying=wd_decaying,
#             wd_ts=wd_ts,
#         )
#         super(AdEMAMix_DecayingBETA2_DecayingWD, self).__init__(params, defaults)

#     def __setstate__(self, state):
#         super(AdEMAMix_DecayingBETA2_DecayingWD, self).__setstate__(state)

#     @torch.no_grad()
#     def step(self, closure=None):
#         """Performs a single optimization step.

#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             lr = group["lr"]
#             wd = group["weight_decay"]
#             eps = group["eps"]
#             beta1, beta2, beta3_final = group["betas"]
#             beta3_warmup = group["beta3_warmup"]
#             alpha_final = group["alpha"]
#             alpha_warmup = group["alpha_warmup"]
#             delta = group["delta"]
#             gamma_3_factor = group["gamma_3_factor"]
#             wd_decaying = group["wd_decaying"]
#             wd_ts = group["wd_ts"]

#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad
#                 if grad.is_sparse:
#                     raise RuntimeError("AdEMAMix does not support sparse gradients.")

#                 state = self.state[p]

#                 # State initialization
#                 if len(state) == 0:
#                     state["step"] = 0
#                     # Exponential moving average of gradient values
#                     if beta1 != 0.0:  # save memory in case beta1 is 0.0
#                         state["exp_avg_fast"] = torch.zeros_like(
#                             p, memory_format=torch.preserve_format 
#                         )
#                     else:
#                         state["exp_avg_fast"] = None
#                     state["exp_avg_slow"] = torch.zeros_like(
#                         p, memory_format=torch.preserve_format
#                     )
#                     # Exponential moving average of squared gradient values
#                     state["exp_avg_sq"] = torch.zeros_like(
#                         p, memory_format=torch.preserve_format
#                     )

#                 exp_avg_fast, exp_avg_slow, exp_avg_sq = (
#                     state["exp_avg_fast"],
#                     state["exp_avg_slow"],
#                     state["exp_avg_sq"],
#                 )

#                 state["step"] += 1
#                 #bias_correction1 = 1 - beta1 ** state["step"]
#                 #bias_correction2 = 1 - beta2 ** state["step"]

#                 # Compute the effective alpha and beta3 in case warmup is used
#                 if alpha_warmup is not None:
#                     alpha = linear_warmup_scheduler(
#                         state["step"],
#                         alpha_end=alpha_final,
#                         alpha_start=0,
#                         warmup=alpha_warmup,
#                     )
#                 else:
#                     alpha = alpha_final

#                 if beta3_warmup is not None:
#                     beta3 = linear_hl_warmup_scheduler(
#                         state["step"],
#                         beta_end=beta3_final,
#                         beta_start=beta1,
#                         warmup=beta3_warmup,
#                     )
#                 else:
#                     beta3 = beta3_final

#                 # Store current alpha and beta_3 values for logging
#                 state["current_alpha"] = alpha
#                 state["current_beta3"] = beta3
#                 state["current_one_minus_beta3"] = 1 - beta3

#                 # Decay the first and second moment running average coefficient
#                 if beta1 != 0.0:
#                     exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
#                 else:
#                     exp_avg_fast = grad
#                 exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                
#                 # Compute decaying beta2 (increases from low to high, approaching 1.0)
#                 # Similar to DANA's alpha formula but inverted: starts at ~0.5, increases toward 1.0
#                 beta2_decaying = 1 - (delta / (delta + state["step"]))
#                 # Clamp to ensure beta2 is in valid range [0, 1)
#                 exp_avg_sq.mul_(beta2_decaying).addcmul_(grad, grad, value=1 - beta2_decaying)

                
#                 denom = (exp_avg_sq.sqrt()).add_(eps)
                
#                 update = (
#                     exp_avg_fast + alpha * exp_avg_slow * gamma_3_factor
#                 ) / denom

#                 # decay weight decay
#                 if wd_decaying:
#                     wd_factor = -wd / (1 + state["step"] / wd_ts) * lr
#                 else:
#                     wd_factor = -wd * lr
#                 p.mul_(1 + wd_factor)

#                 p.add_(-lr * update)
                
#         return loss

class DANA_STAR_NO_TAU(Optimizer):
    
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1.0, # learning rate for debugging
        # g2: float = 1e-4,
        # g3: float = 1e-5,
        delta: float = 8.0,
        kappa: float = 1.0,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        clipsnr: float = 1.0,
        weight_time: bool = False,
        wd_decaying: bool = False,
        wd_ts: float = 1.0,
        ):

        """
        DANA-STAR NO TAU optimizer.
        
        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate. In the code, g2=g3 are taken equal to lr.
            delta: Delta parameter.
            kappa: Kappa parameter.
            epsilon: Epsilon parameter.
            weight_decay: Weight decay parameter.
            clipsnr: Clipsnr parameter.
            weight_time: Whether to use a weighting of the time by a factor lr / lr_max when using scheduler to better handle annealing (doesn't work currently).
            wd_decaying: Whether to decay the wd parameter along training by a (1 + t) factor.
        """

        defaults = dict(
            lr=lr, delta=delta, clipsnr=clipsnr, epsilon=epsilon, kappa=kappa, weight_decay=weight_decay, weighted_step_count=0)
        self.lr = lr
        self.delta = delta
        self.clipsnr = clipsnr
        self.epsilon = epsilon
        self.kappa = kappa
        self.weight_decay = weight_decay
        self.weight_time = weight_time
        self.wd_decaying = wd_decaying
        self.wd_ts = wd_ts

        super(DANA_STAR_NO_TAU, self).__init__(params, defaults)
        
        # Global step counter
        self._step_count = 0
    
    def _make_schedule(self, value: Union[float, Callable[[int], float]]) -> Callable[[int], float]:
        """Convert scalar or schedule to callable function."""
        if callable(value):
            return value
        else:
            return lambda step: value    
    
    #@torch.compile
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step_count += 1
        
        for group in self.param_groups:
            # Get schedule functions
            g2 = group['lr']
            g3 = group['lr']
            lr = group['lr']
            time_factor = group['lr'] / self.lr
            group['weighted_step_count'] += time_factor
            delta = group['delta']
            kappa = group['kappa']
            wd = group['weight_decay']
            epsilon = group['epsilon']
            clipsnr = group['clipsnr']
            
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)  # First moment
                    state['v'] = torch.zeros_like(p)  # Second moment
                    #state['tau'] = torch.zeros_like(p)  # Tau estimates
                
                m, v = state['m'], state['v']
                state['step'] += 1
                
                # Stable EMA coefficient in (0,1): alpha = delta / (delta + t)
                if self.weight_time: # potentially take one during warmup
                    step = group['weighted_step_count']
                    alpha = delta / (delta + step) * time_factor
                else:
                    step = state['step']
                    alpha = delta / (delta + step)
                
                # Update first moment
                m.mul_(1 - alpha).add_(grad, alpha=alpha)
                # Update second moment
                v.mul_(1 - alpha).addcmul_(grad, grad, value=alpha)
                
                # Update tau using the specified tau updater
                #tau_update = self._tau_updater(grad, v, epsilon)
                #tau.mul_(1 - alpha).add_(tau_update, alpha=alpha)
                # Compute effective time
                #effective_time = self._effective_time(tau, step)
                
                # Store current alpha and kappa-based factor for logging
                state["current_alpha"] = alpha
                state["current_kappa_factor"] = (1 + step)**(1-kappa)
                # Compute momentum terms
                #norm_term = self._norm_term(v, tau, step, epsilon)
                #clip_g2_term = torch.clamp(clipsnr * torch.sqrt(v) / (self._root_tau_reg(tau, step) * torch.abs(grad) + epsilon), max=1.0)
                
                # Compute parameter updates using effective time for g2 and g3 scheduling
                g2_term = g2 * grad / (torch.sqrt(v) + epsilon)
                g3_term = g3 * (1 + step)**(1-kappa) * m / (torch.sqrt(v) + epsilon)
                
                # Apply the main update
                update = -(g2_term + g3_term)

                # Decoupled weight decay (AdamW-style)
                if wd != 0:
                    if self.wd_decaying:
                        p.add_(p, alpha= - wd / (1 + step / self.wd_ts) * lr)
                    else:
                        p.add_(p, alpha= - wd * lr)
                
                # Apply update to parameters with scheduled LR
                p.add_(update)
        
        return loss

class DANA_STAR_NO_TAU_KAPPA_0_8(Optimizer):
    
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1.0, # learning rate for debugging
        # g2: float = 1e-4,
        # g3: float = 1e-5,
        delta: float = 8.0,
        kappa: float = 0.8,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        clipsnr: float = 1.0,
        weight_time: bool = False,
        wd_decaying: bool = False,
        wd_ts: float = 1.0,
        ):

        """
        DANA-STAR NO TAU optimizer with kappa = 0.8.
        
        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate. In the code, g2=g3 are taken equal to lr.
            delta: Delta parameter.
            kappa: Kappa parameter.
            epsilon: Epsilon parameter.
            weight_decay: Weight decay parameter.
            clipsnr: Clipsnr parameter.
            weight_time: Whether to use a weighting of the time by a factor lr / lr_max when using scheduler to better handle annealing (doesn't work currently).
            wd_decaying: Whether to decay the wd parameter along training by a (1 + t) factor.
        """

        defaults = dict(
            lr=lr, delta=delta, clipsnr=clipsnr, epsilon=epsilon, kappa=kappa, weight_decay=weight_decay, weighted_step_count=0)
        self.lr = lr
        self.delta = delta
        self.clipsnr = clipsnr
        self.epsilon = epsilon
        self.kappa = kappa
        self.weight_decay = weight_decay
        self.weight_time = weight_time
        self.wd_decaying = wd_decaying
        self.wd_ts = wd_ts

        super(DANA_STAR_NO_TAU_KAPPA_0_8, self).__init__(params, defaults)
        
        # Global step counter
        self._step_count = 0
    
    def _make_schedule(self, value: Union[float, Callable[[int], float]]) -> Callable[[int], float]:
        """Convert scalar or schedule to callable function."""
        if callable(value):
            return value
        else:
            return lambda step: value    
    
    #@torch.compile
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step_count += 1
        
        for group in self.param_groups:
            # Get schedule functions
            g2 = group['lr']
            g3 = group['lr']
            lr = group['lr']
            time_factor = group['lr'] / self.lr
            group['weighted_step_count'] += time_factor
            delta = group['delta']
            kappa = group['kappa']
            wd = group['weight_decay']
            epsilon = group['epsilon']
            clipsnr = group['clipsnr']
            
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)  # First moment
                    state['v'] = torch.zeros_like(p)  # Second moment
                    #state['tau'] = torch.zeros_like(p)  # Tau estimates
                
                m, v = state['m'], state['v']
                state['step'] += 1
                
                # Stable EMA coefficient in (0,1): alpha = delta / (delta + t)
                if self.weight_time: # potentially take one during warmup
                    step = group['weighted_step_count']
                    alpha = delta / (delta + step) * time_factor
                else:
                    step = state['step']
                    alpha = delta / (delta + step)
                
                # Update first moment
                m.mul_(1 - alpha).add_(grad, alpha=alpha)
                # Update second moment
                v.mul_(1 - alpha).addcmul_(grad, grad, value=alpha)
                
                # Update tau using the specified tau updater
                #tau_update = self._tau_updater(grad, v, epsilon)
                #tau.mul_(1 - alpha).add_(tau_update, alpha=alpha)
                # Compute effective time
                #effective_time = self._effective_time(tau, step)
                
                # Store current alpha and kappa-based factor for logging
                state["current_alpha"] = alpha
                state["current_kappa_factor"] = (1 + step)**(1-kappa)
                # Compute momentum terms
                #norm_term = self._norm_term(v, tau, step, epsilon)
                #clip_g2_term = torch.clamp(clipsnr * torch.sqrt(v) / (self._root_tau_reg(tau, step) * torch.abs(grad) + epsilon), max=1.0)
                
                # Compute parameter updates using effective time for g2 and g3 scheduling
                g2_term = g2 * grad / (torch.sqrt(v) + epsilon)
                g3_term = g3 * (1 + step)**(1-kappa) * m / (torch.sqrt(v) + epsilon)
                
                # Apply the main update
                update = -(g2_term + g3_term)

                # Decoupled weight decay (AdamW-style)
                if wd != 0:
                    if self.wd_decaying:
                        p.add_(p, alpha= - wd / (1 + step / self.wd_ts) * lr)
                    else:
                        p.add_(p, alpha= - wd * lr)
                
                # Apply update to parameters with scheduled LR
                p.add_(update)
        
        return loss

class DANA_STAR_NO_TAU_KAPPA_0_85(Optimizer):
    
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1.0, # learning rate for debugging
        # g2: float = 1e-4,
        # g3: float = 1e-5,
        delta: float = 8.0,
        kappa: float = 0.85,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        clipsnr: float = 1.0,
        weight_time: bool = False,
        wd_decaying: bool = False,
        wd_ts: float = 1.0,
        ):

        """
        DANA-STAR NO TAU optimizer with kappa = 0.85.
        
        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate. In the code, g2=g3 are taken equal to lr.
            delta: Delta parameter.
            kappa: Kappa parameter.
            epsilon: Epsilon parameter.
            weight_decay: Weight decay parameter.
            clipsnr: Clipsnr parameter.
            weight_time: Whether to use a weighting of the time by a factor lr / lr_max when using scheduler to better handle annealing (doesn't work currently).
            wd_decaying: Whether to decay the wd parameter along training by a (1 + t) factor.
        """

        defaults = dict(
            lr=lr, delta=delta, clipsnr=clipsnr, epsilon=epsilon, kappa=kappa, weight_decay=weight_decay, weighted_step_count=0)
        self.lr = lr
        self.delta = delta
        self.clipsnr = clipsnr
        self.epsilon = epsilon
        self.kappa = kappa
        self.weight_decay = weight_decay
        self.weight_time = weight_time
        self.wd_decaying = wd_decaying
        self.wd_ts = wd_ts

        super(DANA_STAR_NO_TAU_KAPPA_0_85, self).__init__(params, defaults)
        
        # Global step counter
        self._step_count = 0
    
    def _make_schedule(self, value: Union[float, Callable[[int], float]]) -> Callable[[int], float]:
        """Convert scalar or schedule to callable function."""
        if callable(value):
            return value
        else:
            return lambda step: value    
    
    #@torch.compile
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step_count += 1
        
        for group in self.param_groups:
            # Get schedule functions
            g2 = group['lr']
            g3 = group['lr']
            lr = group['lr']
            time_factor = group['lr'] / self.lr
            group['weighted_step_count'] += time_factor
            delta = group['delta']
            kappa = group['kappa']
            wd = group['weight_decay']
            epsilon = group['epsilon']
            clipsnr = group['clipsnr']
            
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)  # First moment
                    state['v'] = torch.zeros_like(p)  # Second moment
                    #state['tau'] = torch.zeros_like(p)  # Tau estimates
                
                m, v = state['m'], state['v']
                state['step'] += 1
                
                # Stable EMA coefficient in (0,1): alpha = delta / (delta + t)
                if self.weight_time: # potentially take one during warmup
                    step = group['weighted_step_count']
                    alpha = delta / (delta + step) * time_factor
                else:
                    step = state['step']
                    alpha = delta / (delta + step)
                
                # Update first moment
                m.mul_(1 - alpha).add_(grad, alpha=alpha)
                # Update second moment
                v.mul_(1 - alpha).addcmul_(grad, grad, value=alpha)
                
                # Update tau using the specified tau updater
                #tau_update = self._tau_updater(grad, v, epsilon)
                #tau.mul_(1 - alpha).add_(tau_update, alpha=alpha)
                # Compute effective time
                #effective_time = self._effective_time(tau, step)
                
                # Store current alpha and kappa-based factor for logging
                state["current_alpha"] = alpha
                state["current_kappa_factor"] = (1 + step)**(1-kappa)
                # Compute momentum terms
                #norm_term = self._norm_term(v, tau, step, epsilon)
                #clip_g2_term = torch.clamp(clipsnr * torch.sqrt(v) / (self._root_tau_reg(tau, step) * torch.abs(grad) + epsilon), max=1.0)
                
                # Compute parameter updates using effective time for g2 and g3 scheduling
                g2_term = g2 * grad / (torch.sqrt(v) + epsilon)
                g3_term = g3 * (1 + step)**(1-kappa) * m / (torch.sqrt(v) + epsilon)
                
                # Apply the main update
                update = -(g2_term + g3_term)

                # Decoupled weight decay (AdamW-style)
                if wd != 0:
                    if self.wd_decaying:
                        p.add_(p, alpha= - wd / (1 + step / self.wd_ts) * lr)
                    else:
                        p.add_(p, alpha= - wd * lr)
                
                # Apply update to parameters with scheduled LR
                p.add_(update)
        
        return loss

class DANA_STAR_NO_TAU_KAPPA_0_9(Optimizer):
    
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1.0, # learning rate for debugging
        # g2: float = 1e-4,
        # g3: float = 1e-5,
        delta: float = 8.0,
        kappa: float = 0.9,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        clipsnr: float = 1.0,
        weight_time: bool = False,
        wd_decaying: bool = False,
        wd_ts: float = 1.0,
        ):

        """
        DANA-STAR NO TAU optimizer with kappa = 0.9.
        
        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate. In the code, g2=g3 are taken equal to lr.
            delta: Delta parameter.
            kappa: Kappa parameter.
            epsilon: Epsilon parameter.
            weight_decay: Weight decay parameter.
            clipsnr: Clipsnr parameter.
            weight_time: Whether to use a weighting of the time by a factor lr / lr_max when using scheduler to better handle annealing (doesn't work currently).
            wd_decaying: Whether to decay the wd parameter along training by a (1 + t) factor.
        """

        defaults = dict(
            lr=lr, delta=delta, clipsnr=clipsnr, epsilon=epsilon, kappa=kappa, weight_decay=weight_decay, weighted_step_count=0)
        self.lr = lr
        self.delta = delta
        self.clipsnr = clipsnr
        self.epsilon = epsilon
        self.kappa = kappa
        self.weight_decay = weight_decay
        self.weight_time = weight_time
        self.wd_decaying = wd_decaying
        self.wd_ts = wd_ts

        super(DANA_STAR_NO_TAU_KAPPA_0_9, self).__init__(params, defaults)
        
        # Global step counter
        self._step_count = 0
    
    def _make_schedule(self, value: Union[float, Callable[[int], float]]) -> Callable[[int], float]:
        """Convert scalar or schedule to callable function."""
        if callable(value):
            return value
        else:
            return lambda step: value    
    
    #@torch.compile
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step_count += 1
        
        for group in self.param_groups:
            # Get schedule functions
            g2 = group['lr']
            g3 = group['lr']
            lr = group['lr']
            time_factor = group['lr'] / self.lr
            group['weighted_step_count'] += time_factor
            delta = group['delta']
            kappa = group['kappa']
            wd = group['weight_decay']
            epsilon = group['epsilon']
            clipsnr = group['clipsnr']
            
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)  # First moment
                    state['v'] = torch.zeros_like(p)  # Second moment
                    #state['tau'] = torch.zeros_like(p)  # Tau estimates
                
                m, v = state['m'], state['v']
                state['step'] += 1
                
                # Stable EMA coefficient in (0,1): alpha = delta / (delta + t)
                if self.weight_time: # potentially take one during warmup
                    step = group['weighted_step_count']
                    alpha = delta / (delta + step) * time_factor
                else:
                    step = state['step']
                    alpha = delta / (delta + step)
                
                # Update first moment
                m.mul_(1 - alpha).add_(grad, alpha=alpha)
                # Update second moment
                v.mul_(1 - alpha).addcmul_(grad, grad, value=alpha)
                
                # Update tau using the specified tau updater
                #tau_update = self._tau_updater(grad, v, epsilon)
                #tau.mul_(1 - alpha).add_(tau_update, alpha=alpha)
                # Compute effective time
                #effective_time = self._effective_time(tau, step)
                
                # Store current alpha and kappa-based factor for logging
                state["current_alpha"] = alpha
                state["current_kappa_factor"] = (1 + step)**(1-kappa)
                # Compute momentum terms
                #norm_term = self._norm_term(v, tau, step, epsilon)
                #clip_g2_term = torch.clamp(clipsnr * torch.sqrt(v) / (self._root_tau_reg(tau, step) * torch.abs(grad) + epsilon), max=1.0)
                
                # Compute parameter updates using effective time for g2 and g3 scheduling
                g2_term = g2 * grad / (torch.sqrt(v) + epsilon)
                g3_term = g3 * (1 + step)**(1-kappa) * m / (torch.sqrt(v) + epsilon)
                
                # Apply the main update
                update = -(g2_term + g3_term)

                # Decoupled weight decay (AdamW-style)
                if wd != 0:
                    if self.wd_decaying:
                        p.add_(p, alpha= - wd / (1 + step / self.wd_ts) * lr)
                    else:
                        p.add_(p, alpha= - wd * lr)
                
                # Apply update to parameters with scheduled LR
                p.add_(update)
        
        return loss