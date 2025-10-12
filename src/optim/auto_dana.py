import torch
from torch.optim import Optimizer


class AUTO_DANA(Optimizer):
    """
    AUTO_DANA optimizer: like DANA but with an automatic per-parameter
    schedule on the g3 (momentum) term based on the ratio u^2 / (m^2 + eps),
    capped at 1.0. Here, u is the current gradient (or its EMA if beta1>0)
    and m is the momentum (EMA of gradients).
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        delta=8.0,
        kappa=1.0,
        eps=1e-8,
        weight_decay=0.0,
        weight_time=False,
        use_v_ema=False,
        v_ema_beta=0.999,
        gamma_3_factor=1.0,
        beta1=0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= delta:
            raise ValueError("Invalid delta value: {}".format(delta))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= v_ema_beta < 1.0:
            raise ValueError("Invalid v_ema_beta value: {}".format(v_ema_beta))
        
        defaults = dict(
            lr=lr,
            delta=delta,
            kappa=kappa,
            eps=eps,
            weight_decay=weight_decay,
            weight_time=weight_time,
            beta1=beta1,
            use_v_ema=use_v_ema,
            v_ema_beta=v_ema_beta,
            weighted_step_count=0,
            gamma_3_factor=gamma_3_factor
        )
        super(AUTO_DANA, self).__init__(params, defaults)
        
        # Global step counter for weight_time
        self._step_count = 0

    def __setstate__(self, state):
        super(AUTO_DANA, self).__setstate__(state)

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

        self._step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            delta = group["delta"]
            kappa = group["kappa"]
            weight_time = group["weight_time"]
            use_v_ema = group["use_v_ema"]
            v_ema_beta = group["v_ema_beta"]
            gamma_3_factor = group["gamma_3_factor"]
            beta1 = group["beta1"]
            # Update weighted step count for weight_time feature
            if weight_time:
                time_factor = lr / getattr(self, 'initial_lr', lr)  # Fallback to current lr if initial_lr not set
                group['weighted_step_count'] += time_factor

            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AUTO_DANA does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # First moment estimate (momentum)
                    state["m"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Second moment estimate (RMSprop-style)
                    state["v"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    
                    state["grad_ema"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                m, v = state["m"], state["v"]
                state["step"] += 1
                
                # Determine which step count to use for alpha calculation
                if weight_time:
                    step_for_alpha = group['weighted_step_count']
                else:
                    step_for_alpha = state["step"]
                
                # DANA's delta-based EMA coefficient: alpha = delta / (delta + t)
                alpha = delta / (delta + step_for_alpha)

                # Store current alpha and kappa-based factor for logging
                state["current_alpha"] = alpha
                state["current_kappa_factor"] = (1 + state["step"])**(-kappa)

                # Update first moment (momentum) - DANA style
                m.mul_(1 - alpha).add_(grad, alpha=alpha)
                
                # Update second moment (squared gradients)
                if use_v_ema:
                    # Use traditional EMA for second moment with v_ema_beta
                    v.mul_(v_ema_beta).addcmul_(grad, grad, value=1 - v_ema_beta)
                else:
                    # Use DANA style with delta-based alpha
                    v.mul_(1 - alpha).addcmul_(grad, grad, value=alpha)

                
                grad_ema = state["grad_ema"]
                grad_ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                g2_gradient = grad_ema

                # Compute normalized factor (denominator)
                normalized_factor = v.sqrt() + eps
                
                # DANA's parameter update terms:
                # g2_term: gradient (current or EMA) scaled by learning rate
                g2_term = lr * g2_gradient / normalized_factor
                
                # Compute norms for auto_factor and logging
                g2_gradient_norm = g2_gradient.norm()
                m_norm = m.norm()
                
                #auto_factor = torch.clamp(g2_gradient_norm**2 / (m_norm**2 * (1 + state["step"])**2 + eps), max=1.0) * gamma_3_factor
                auto_factor = g2_gradient**2 / (m_norm**2 + eps) * gamma_3_factor
                # g3_term: momentum with kappa-based time scaling
                g3_term = lr * (1 + state["step"]) * m / normalized_factor * auto_factor
                
                # Combine update terms (note: negative because we're doing gradient descent)
                update = -(g2_term + g3_term)
                
                # Store metrics for logging
                state["auto_factor_mean"] = auto_factor.mean().detach()
                state["g2_gradient_norm"] = g2_gradient_norm.detach()
                state["m_norm"] = m_norm.detach()

                # Decoupled weight decay (AdamW-style) - applied before main update
                if wd != 0:
                    p.add_(p, alpha=-wd * lr)
                
                # Apply the main parameter update
                p.add_(update)

        return loss


