import math
import torch
import torch.distributed as dist
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
        lr=1e-3,
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
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
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
            lr=lr,
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
        
        # Only print from rank 0 (GPU 0) in distributed setting
        is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        
        if is_rank0:
            print(f"[SNOO-DANA] Global step: {self._step_count}")
        
        # Step 1: Perform AdamW step
        if is_rank0:
            print(f"[SNOO-DANA] Starting inner loop (AdamW step)")
        for group in self.param_groups:
            lr = group["lr"]
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
                    # DANA-specific states
                    state["dana_step"] = 0
                    state["dana_y"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                m = state["m"]
                v = state["v"]
                state["step"] += 1

                # AdamW update
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                step_size = lr / bias_correction1
                bias_correction_sqrt = (bias_correction2 ** 0.5)

                denom = (v.sqrt() / bias_correction_sqrt).add_(eps)

                # Decoupled weight decay (AdamW-style)
                if wd != 0:
                    p.add_(p, alpha=-wd * lr)
                
                # Apply AdamW parameter update
                p.addcdiv_(m, denom, value=-step_size)

        # Step 2: Every k iterations, perform DANA correction step
        k = self.param_groups[0]['k']
        if is_rank0:
            print(f"[SNOO-DANA] Checking if outer loop needed: step {self._step_count} % k={k} = {self._step_count % k}")
        if self._step_count % k == 0:
            if is_rank0:
                print(f"[SNOO-DANA] *** OUTER LOOP TRIGGERED at step {self._step_count} (every {k} steps) ***")
            # Perform DANA step using difference from inner loop
            buf_idx = 0
            for group in self.param_groups:
                lr_outer = group["lr_outer"]
                wd = group["weight_decay"]
                eps = group["eps"]
                delta = group["delta"]
                kappa = group["kappa"]
                gamma_3_factor = group["gamma_3_factor"]

                for p in group["params"]:
                    p_old = self.outer_buf[buf_idx]
                    buf_idx += 1
                    
                    # Compute pseudo-gradient from inner loop drift (current - old)
                    # This represents the direction AdamW moved, which is in the descent direction
                    grad_outer = p.data - p_old.data
                    
                    state = self.state[p]
                    
                    dana_y = state["dana_y"]
                    state["dana_step"] += 1
                    
                    # DANA's delta-based EMA coefficient: alpha = delta / (delta + t)
                    alpha = delta / (delta + state["dana_step"])

                    # Store current alpha and kappa-based factor for logging
                    state["current_alpha"] = alpha
                    state["current_kappa_factor"] = (1 + state["dana_step"])**(1-kappa)

                    # Update first moment (momentum) - DANA style
                    # grad_outer represents the accumulated change from k inner steps
                    dana_y.mul_(1 - alpha).add_(grad_outer, alpha=alpha)
                    
                    # Combine update terms and apply (using the drift direction with momentum)
                    # grad_outer is already in descent direction, so we scale and apply directly
                    update = lr_outer * (grad_outer + gamma_3_factor * (1 + state["dana_step"])**(1-kappa) * dana_y)
                    # Apply the DANA parameter update to the old checkpoint
                    p_old.add_(update)
                    
                    # Copy the corrected parameters back to the model
                    p.data.copy_(p_old.data)
            
            # Update outer buffer with new parameters
            if is_rank0:
                print(f"[SNOO-DANA] Updating outer buffer with new parameters")
            for p_new, p_old in zip(self.model_params, self.outer_buf):
                p_old.copy_(p_new, non_blocking=True)
            if is_rank0:
                print(f"[SNOO-DANA] Outer loop complete. Next outer loop at step {self._step_count + k}")

        return loss

class snoo(Optimizer):
    """
    SNOO optimizer: Combines AdamW steps with periodic Nesterov corrections.
    
    At each iteration:
    - Performs a standard AdamW step
    - Every k iterations, performs a Nesterov step using outer loop buffer
    
    Uses Nesterov's mathematical formulation with:
    - Nesterov's specific moment updates and parameter updates
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        lr_outer=0.5,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        mu=0.9,
        k=1,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_outer:
            raise ValueError("Invalid learning rate: {}".format(lr_outer))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1 value: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2 value: {}".format(beta2))
        if not 0.0 <= mu < 1.0:
            raise ValueError("Invalid mu value: {}".format(mu))
        
        defaults = dict(
            lr=lr,
            lr_outer=lr_outer,
            eps=eps,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            mu=mu,
            k=k,
        )
        super(snoo, self).__init__(params, defaults)
        
        # Global step counter
        self._step_count = 0
        
        # Store model params and outer buffer for periodic Nesterov steps
        self.model_params = []
        self.outer_buf = []
        
        for group in self.param_groups:
            for p in group['params']:
                self.model_params.append(p)
                self.outer_buf.append(p.clone().detach())

    def __setstate__(self, state):
        super(snoo, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step combining AdamW and periodic Nesterov corrections.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        
        # Only print from rank 0 (GPU 0) in distributed setting
        is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        
        if is_rank0:
            print(f"[SNOO] Global step: {self._step_count}")
        
        # Step 1: Perform AdamW step
        if is_rank0:
            print(f"[SNOO] Starting inner loop (AdamW step)")
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("snoo does not support sparse gradients.")

                state = self.state[p]

                # State initialization for AdamW
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Nesterov-specific states
                    state["nesterov_step"] = 0
                    state["nesterov_momentum"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                m = state["m"]
                v = state["v"]
                state["step"] += 1

                # AdamW update
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                step_size = lr / bias_correction1
                bias_correction_sqrt = (bias_correction2 ** 0.5)

                denom = (v.sqrt() / bias_correction_sqrt).add_(eps)

                # Decoupled weight decay (AdamW-style)
                if wd != 0:
                    p.add_(p, alpha=-wd * lr)
                
                # Apply AdamW parameter update
                p.addcdiv_(m, denom, value=-step_size)

        # Step 2: Every k iterations, perform Nesterov correction step
        k = self.param_groups[0]['k']
        if is_rank0:
            print(f"[SNOO] Checking if outer loop needed: step {self._step_count} % k={k} = {self._step_count % k}")
        if self._step_count % k == 0:
            if is_rank0:
                print(f"[SNOO] *** OUTER LOOP TRIGGERED at step {self._step_count} (every {k} steps) ***")
            # Perform Nesterov step using difference from inner loop
            buf_idx = 0
            for group in self.param_groups:
                lr_outer = group["lr_outer"]
                mu = group["mu"]

                for p in group["params"]:
                    p_old = self.outer_buf[buf_idx]
                    buf_idx += 1
                    
                    # Compute pseudo-gradient from inner loop drift (current - old)
                    # This represents the direction AdamW moved, which is in the descent direction
                    grad_outer = p.data - p_old.data
                    
                    state = self.state[p]
                    nesterov_momentum = state["nesterov_momentum"]
                    state["nesterov_step"] += 1

                    # Nesterov momentum update: v_t = mu * v_{t-1} + grad_outer
                    nesterov_momentum.mul_(mu).add_(grad_outer)
                    
                    # Nesterov step: update = lr_outer * (grad_outer + mu * v_t)
                    # This is the standard Nesterov momentum formulation
                    update = lr_outer * (grad_outer + mu * nesterov_momentum)
                    
                    # Apply the Nesterov parameter update to the old checkpoint
                    p_old.add_(update)
                    
                    # Copy the corrected parameters back to the model
                    p.data.copy_(p_old.data)
            
            # Update outer buffer with new parameters
            if is_rank0:
                print(f"[SNOO] Updating outer buffer with new parameters")
            for p_new, p_old in zip(self.model_params, self.outer_buf):
                p_old.copy_(p_new, non_blocking=True)
            if is_rank0:
                print(f"[SNOO] Outer loop complete. Next outer loop at step {self._step_count + k}")

        return loss