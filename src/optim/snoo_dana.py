import math
import torch
from torch.optim import Optimizer
from typing import Union, Callable, Iterable

class snoo_DANA(Optimizer):
    """
    SNOO-DANA optimizer: Combines AdamW steps with periodic DANA corrections.
    
    At each iteration:
    - Performs a standard AdamW step
    - Every k iterations, performs a DANA step using outer loop buffer
    
    Uses DANA's mathematical formulation with:
    - Delta-based EMA coefficient: alpha = delta / (delta + t)
    - DANA's specific moment updates and parameter updates
    - Kappa-based momentum scaling: (1 + step)**(1-kappa)
    - Optional gradient EMA for g2 term
    - Optional traditional EMA for second moment v with configurable beta
    """
    
    def __init__(
        self,
        params,
        lr_inner=1e-3,
        lr_outer=0.5,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        delta=8.0,
        kappa=1.0,
        gamma_3_factor=1.0,
        k=1,
    ):
        if not 0.0 <= lr_inner:
            raise ValueError("Invalid learning rate: {}".format(lr_inner))
        if not 0.0 <= lr_outer:
            raise ValueError("Invalid learning rate: {}".format(lr_outer))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= delta:
            raise ValueError("Invalid delta value: {}".format(delta))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1 value: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2 value: {}".format(beta2))
        
        defaults = dict(
            lr_inner=lr_inner,
            lr_outer=lr_outer,
            delta=delta,
            kappa=kappa,
            eps=eps,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            gamma_3_factor=gamma_3_factor,
            k=k,
        )
        super(snoo_DANA, self).__init__(params, defaults)
        
        # Global step counter
        self._step_count = 0
        
        # Store model params and outer buffer for periodic DANA steps
        self.model_params = []
        self.outer_buf = []
        
        for group in self.param_groups:
            for p in group['params']:
                self.model_params.append(p)
                self.outer_buf.append(p.clone().detach())

    def __setstate__(self, state):
        super(snoo_DANA, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step combining AdamW and periodic DANA corrections.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        
        # Step 1: Perform AdamW step
        for group in self.param_groups:
            lr_inner = group["lr_inner"]
            wd = group["weight_decay"]
            eps = group["eps"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("snoo_DANA does not support sparse gradients.")

                state = self.state[p]

                # State initialization for AdamW
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["y"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # DANA-specific states
                    state["dana_step"] = 0
                    state["dana_y"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                m = state["m"]
                v = state["v"]
                y = state["y"]
                state["step"] += 1

                # AdamW update
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                step_size = lr_inner / bias_correction1
                bias_correction_sqrt = (bias_correction2 ** 0.5)

                denom = (v.sqrt() / bias_correction_sqrt).add_(eps)

                # Decoupled weight decay (AdamW-style)
                if wd != 0:
                    p.add_(p, alpha=-wd * lr_inner)
                
                # Apply AdamW parameter update
                p.addcdiv_(m, denom, value=-step_size)

        # Step 2: Every k iterations, perform DANA correction step
        k = self.param_groups[0]['k']
        if self._step_count % k == 0:
            # Perform DANA step using difference from inner loop
            for group in self.param_groups:
                lr_outer = group["lr_outer"]
                wd = group["weight_decay"]
                eps = group["eps"]
                delta = group["delta"]
                kappa = group["kappa"]
                gamma_3_factor = group["gamma_3_factor"]

                for p, p_old in zip(group["params"], self.outer_buf):
                    
                    # Compute difference from inner loop (outer buffer - current params)
                    p_diff = p_old.data - p.data
                    
                    state = self.state[p]
                    
                    dana_y = state["dana_y"]
                    state["dana_step"] += 1
                    
                    # DANA's delta-based EMA coefficient: alpha = delta / (delta + t)
                    alpha = delta / (delta + state["dana_step"])

                    # Store current alpha and kappa-based factor for logging
                    state["current_alpha"] = alpha
                    state["current_kappa_factor"] = (1 + state["dana_step"])**(1-kappa)

                    # Update first moment (momentum) - DANA style
                    # p_diff represents the accumulated change from k inner steps
                    dana_y.mul_(1 - alpha).add_(p_diff, alpha=alpha)
                    
                    # Combine update terms (note: negative because we're doing gradient descent)
                    update = -lr_outer * (p_diff + gamma_3_factor * (1 + state["dana_step"])**(1-kappa) * dana_y)
                    
                    # Apply the DANA parameter update
                    p.add_(update)
            
            # Update outer buffer with new parameters
            for p_new, p_old in zip(self.model_params, self.outer_buf):
                p_old.copy_(p_new, non_blocking=True)

        return loss