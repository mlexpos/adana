import math
import torch
from torch.optim import Optimizer
from typing import Union, Callable, Iterable

torch._dynamo.config.cache_size_limit = 64


class ADana(Optimizer):
    """
    ADana (Adaptive Damped Nesterov Acceleration) optimizer.

    Log-time scheduling of momentum and weight decay without tau probability estimator.
    Uses compiled update kernels for performance. Two modes:
    - clipsnr=None: No SNR clipping (base ADana)
    - clipsnr=float: SNR clipping enabled (Dana-MK4)

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        delta: Delta parameter for EMA coefficient (default: 8.0).
        kappa: Kappa parameter for effective time scaling (default: 1.0).
        epsilon: Small constant for numerical stability (default: 1e-8).
        weight_decay: Weight decay parameter (default: 0.0).
        clipsnr: SNR clipping parameter. None disables clipping (default: None).
        wd_decaying: Whether to decay weight decay over time (default: False).
        wd_ts: Timescale for weight decay decay (default: 1.0).
        gamma_3_factor: Scaling factor for the g3 (long-momentum) term (default: 1.0).
        use_foreach: Whether to use fused foreach operations (default: False).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1.0,
        delta: float = 8.0,
        kappa: float = 1.0,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        clipsnr: float = None,
        wd_decaying: bool = False,
        wd_ts: float = 1.0,
        gamma_3_factor: float = 1.0,
        use_foreach: bool = False,
    ):
        defaults = dict(
            lr=lr, delta=delta, epsilon=epsilon, weight_decay=weight_decay, weighted_step_count=0)
        self.lr = lr
        self.delta = delta
        self.kappa = kappa
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.clipsnr = clipsnr
        self.wd_decaying = wd_decaying
        self.wd_ts = wd_ts
        self.gamma_3_factor = gamma_3_factor
        self.use_foreach = use_foreach

        super(ADana, self).__init__(params, defaults)

        self._compiled_functions = {}

        if self.use_foreach:
            self._foreach_compute_kernel = torch.compile(
                self._foreach_compute_kernel_impl,
                fullgraph=False,
                dynamic=True
            )
        else:
            self._foreach_compute_kernel = None

    def _get_compiled_fn(self, ndim):
        if ndim not in self._compiled_functions:
            self._compiled_functions[ndim] = torch.compile(
                self._update_param_compiled,
                dynamic=False,
                fullgraph=False
            )
        return self._compiled_functions[ndim]

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
        schedule_factor: torch.Tensor,
        wd: torch.Tensor,
        epsilon: float,
        clipsnr: float,
        kappa: float,
        wd_decaying: bool,
        wd_ts: float,
        gamma_3_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # Update first moment (EMA of gradient)
        m.lerp_(grad, alpha)

        # Update second moment (EMA of gradient squared)
        v.mul_(1 - alpha).addcmul_(grad, grad, value=alpha)

        # Compute sqrt(v) + epsilon once and reuse
        sqrt_v_eps = torch.sqrt(v).add_(epsilon)

        # Compute normalization term directly (no tau)
        norm_term = 1.0 / sqrt_v_eps
        m_norm_term = torch.abs(m) * norm_term

        # Compute momentum factor and alpha factor using step-based time
        effective_time = 1.0 + step
        mfac = m_norm_term

        if clipsnr is not None:
            alpha_factor = torch.clamp(
                (effective_time ** (1 - kappa)) * mfac,
                max=clipsnr
            )
        else:
            alpha_factor = (effective_time ** (1 - kappa)) * mfac

        # Compute g3 term (momentum-based update, no tau_reg), scaled by gamma_3_factor
        g3_term = (-g3) * gamma_3_factor * (torch.sign(m) * alpha_factor + m * norm_term)

        # Compute g2 term (gradient-based update)
        g2_term = (-g2) * grad * norm_term

        # Combine updates
        update = g2_term + g3_term

        # Apply parameter update (in-place)
        p.add_(update)

        # Apply independent weight decay (paper convention):
        # WD is multiplied by schedule γ(t) but NOT by peak LR γ*
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

        return m, v, diagnostics

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            g2 = group['lr']
            g3 = group['lr']
            schedule_factor = group['lr'] / self.lr  # γ(t) without peak LR
            time_factor = schedule_factor
            group['weighted_step_count'] += time_factor
            delta = group['delta']
            wd = group['weight_decay']
            epsilon = group['epsilon']

            if self.use_foreach:
                g2_t = torch.tensor(g2, dtype=torch.float32)
                g3_t = torch.tensor(g3, dtype=torch.float32)
                sf_t = torch.tensor(schedule_factor, dtype=torch.float32)
                delta_t = torch.tensor(delta, dtype=torch.float32)
                wd_t = torch.tensor(wd, dtype=torch.float32)

                self._foreach_step_group(
                    group, g2_t, g3_t, sf_t, delta_t, wd_t, epsilon
                )
                continue

            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

                m, v = state['m'], state['v']
                state['step'] += 1

                step = state['step']
                alpha = delta / (delta + step)

                device = p.device
                step_t = torch.tensor(step, device=device, dtype=torch.float32)
                alpha_t = torch.tensor(alpha, device=device, dtype=torch.float32)
                g2_t = torch.tensor(g2, device=device, dtype=torch.float32)
                g3_t = torch.tensor(g3, device=device, dtype=torch.float32)
                sf_t = torch.tensor(schedule_factor, device=device, dtype=torch.float32)
                wd_t = torch.tensor(wd, device=device, dtype=torch.float32)

                update_fn = self._get_compiled_fn(p.shape)

                m_new, v_new, diagnostics = update_fn(
                    p, grad, m, v,
                    step_t, alpha_t, g2_t, g3_t, sf_t, wd_t,
                    epsilon, self.clipsnr, self.kappa, self.wd_decaying, self.wd_ts, self.gamma_3_factor
                )

                m.copy_(m_new)
                v.copy_(v_new)

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
    ) -> None:
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

        if self._foreach_compute_kernel is not None:
            alpha_factors, mfacs = self._foreach_compute_kernel(
                params, grads, ms, vs, alpha_tensors, step_tensors,
                g2, g3, schedule_factor, wd, epsilon, self.clipsnr, self.kappa, self.wd_decaying, self.wd_ts, self.gamma_3_factor
            )
        else:
            alpha_factors, mfacs = self._foreach_compute_kernel_impl(
                params, grads, ms, vs, alpha_tensors, step_tensors,
                g2, g3, schedule_factor, wd, epsilon, self.clipsnr, self.kappa, self.wd_decaying, self.wd_ts, self.gamma_3_factor
            )

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
        schedule_factor: torch.Tensor,
        wd: torch.Tensor,
        epsilon: float,
        clipsnr: float,
        kappa: float,
        wd_decaying: bool,
        wd_ts: float,
        gamma_3_factor: float,
    ):
        # Step 1: Update first moment using lerp
        torch._foreach_lerp_(ms, grads, alpha_tensors)

        # Step 2: Update second moment
        one_minus_alphas = torch._foreach_neg(alpha_tensors)
        torch._foreach_add_(one_minus_alphas, 1.0)
        torch._foreach_mul_(vs, one_minus_alphas)
        grad_sq_scaled = torch._foreach_mul(grads, grads)
        grad_sq_scaled = torch._foreach_mul(grad_sq_scaled, alpha_tensors)
        torch._foreach_add_(vs, grad_sq_scaled)

        # Step 3: Compute normalization term (no tau)
        sqrt_vs = torch._foreach_sqrt(vs)
        norm_den = torch._foreach_add(sqrt_vs, epsilon)
        norm_term = torch._foreach_reciprocal(norm_den)

        # Step 4: Momentum factor (no tau_reg)
        m_abs = torch._foreach_abs(ms)
        mfac = torch._foreach_mul(m_abs, norm_term)

        # Step 5: Effective time
        effective_time = torch._foreach_add(step_tensors, 1.0)

        # Step 6: Alpha factor with optional clipping
        eff_pow = torch._foreach_pow(effective_time, 1.0 - kappa)
        alpha_factor = torch._foreach_mul(eff_pow, mfac)
        if clipsnr is not None:
            alpha_factor = [torch.clamp(af, max=clipsnr) for af in alpha_factor]

        # Step 7: Compute g3 term (no tau_reg), scaled by gamma_3_factor
        sign_m = torch._foreach_sign(ms)
        sign_alpha = torch._foreach_mul(sign_m, alpha_factor)
        m_norm_term = torch._foreach_mul(ms, norm_term)
        g3_inner = torch._foreach_add(sign_alpha, m_norm_term)
        torch._foreach_mul_(g3_inner, gamma_3_factor)
        g3_term = torch._foreach_mul(g3_inner, g3)

        # Step 8: Compute g2 term
        grad_norm = torch._foreach_mul(grads, norm_term)
        g2_term = torch._foreach_mul(grad_norm, g2)

        # Step 9: Combine updates and apply
        update = torch._foreach_add(g2_term, g3_term)
        torch._foreach_neg_(update)
        torch._foreach_add_(params, update)

        # Step 10: Independent weight decay (paper convention)
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

        wd_updates = torch._foreach_mul(params, wd_factors)
        torch._foreach_add_(params, wd_updates)

        return alpha_factor, mfac
