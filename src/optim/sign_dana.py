import math
import torch
from torch.optim import Optimizer
from typing import Union, Callable, Iterable

class sign_DANA(Optimizer):
    """
    sign_DANA optimizer: like DANA but with sign of the gradient instead of dividing by second moment average
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        delta=8.0,
        kappa=1.0,
        weight_decay=0.0,
        weight_time=False,
        beta1=0.0,
        gamma_3_factor=1.0,
        norm_type="linf",
        eps=1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= delta:
            raise ValueError("Invalid delta value: {}".format(delta))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(
            lr=lr,
            delta=delta,
            kappa=kappa,
            weight_decay=weight_decay,
            weight_time=weight_time,
            beta1=beta1,
            weighted_step_count=0,
            gamma_3_factor=gamma_3_factor,
            norm_type=norm_type,
            eps=eps
        )
        super(sign_DANA, self).__init__(params, defaults)
        print(f"sign_DANA initialized with beta1={beta1}, gamma_3_factor={gamma_3_factor}, norm_type={norm_type}, eps={eps}")
        # Global step counter for weight_time
        self._step_count = 0

    def __setstate__(self, state):
        super(sign_DANA, self).__setstate__(state)

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
            delta = group["delta"]
            kappa = group["kappa"]
            weight_time = group["weight_time"]
            gamma_3_factor = group["gamma_3_factor"]
            norm_type = group["norm_type"]
            eps = group["eps"]
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
                    raise RuntimeError("sign_DANA does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # First moment estimate (momentum)
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["grad_ema"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
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
                state["current_kappa_factor"] = (1 + state["step"])**(1-kappa)

                # Update first moment (momentum) - DANA style
                exp_avg.mul_(1 - alpha).add_(grad, alpha=alpha)
                
                # No second moment update needed for sign_DANA

                
                grad_ema = state["grad_ema"]
                grad_ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                g2_gradient = grad_ema
                
                # g3_term: momentum with kappa-based time scaling
                g3_term = (1 + state["step"])**(1-kappa) * exp_avg * gamma_3_factor
                
                # Combine update terms (note: negative because we're doing gradient descent)
                update = g2_gradient + g3_term

                # Decoupled weight decay (AdamW-style) - applied before main update
                if wd != 0:
                    p.add_(p, alpha=-wd * lr)
                
                # Apply the main parameter update
                if norm_type == "linf":
                    p.add_(update.sign_(), alpha=-lr)
                elif norm_type == "l2":
                    update_norm = update.norm(p=2)
                    p.add_(update.div_(update_norm + eps), alpha=-lr)

        return loss