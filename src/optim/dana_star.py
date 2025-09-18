import math
import torch
from torch.optim import Optimizer
from typing import Union, Callable, Iterable

class DANA_STAR(Optimizer):
    
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
        DANA-STAR optimizer.
        
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

        super(DANA_STAR, self).__init__(params, defaults)
        
        # Global step counter
        self._step_count = 0
    
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
        Tau regularization function that:
        1. Ensures tau is not a poor estimate when p << 1/t
        2. Converts tau/(1-tau) form to proper p estimate  
        3. Clips to prevent numerical issues
        """
        clipped_tau = self._clip_to_half(tau)
        p_estimate = clipped_tau / (1.0 - clipped_tau)
        min_p = torch.full_like(tau, 1.0 / (1.0 + step))
        return torch.maximum(p_estimate, min_p)
    
    def _root_tau_reg(self, tau: torch.Tensor, step: int) -> torch.Tensor:
        """Square root of tau regularization."""
        return torch.sqrt(self._tau_reg(tau, step))
    
    # def _quarter_root_tau_reg(self, tau: torch.Tensor, step: int) -> torch.Tensor:
    #     """Quarter root of tau regularization."""
    #     return torch.pow(self._tau_reg(tau, step), 0.25)
    
    def _effective_time(self, tau: torch.Tensor, step: int) -> torch.Tensor:
        """Compute effective time for tau regularization."""
        return torch.maximum(tau * step, torch.ones_like(tau))
    
    def _tau_updater(
        self, 
        # tau: torch.Tensor, 
        g: torch.Tensor, # gradient
        v: torch.Tensor, # second moment
        epsilon: float
    ) -> torch.Tensor:
        return torch.abs(g) / (torch.abs(g) + torch.sqrt(v) + epsilon)
        
    
    def _norm_term(
        self, 
        # g : torch.Tensor, # gradient
        # md: torch.Tensor, 
        v: torch.Tensor, 
        tau: torch.Tensor, 
        step: int, 
        #clipsnr: float, 
        epsilon: float
    ) -> torch.Tensor:
    
        root_tau_reg = self._root_tau_reg(tau, step)
        return root_tau_reg / (torch.sqrt(v) + epsilon)
    
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
                    state['tau'] = torch.zeros_like(p)  # Tau estimates
                
                m, v, tau = state['m'], state['v'], state['tau']
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
                tau_update = self._tau_updater(grad, v, epsilon)
                tau.mul_(1 - alpha).add_(tau_update, alpha=alpha)
                # Compute effective time
                effective_time = self._effective_time(tau, step)
                # Compute momentum terms
                norm_term = self._norm_term(v, tau, step, epsilon)
                clip_g2_term = torch.clamp(clipsnr * torch.sqrt(v) / (self._root_tau_reg(tau, step) * torch.abs(grad) + epsilon), max=1.0)
                
                # Compute parameter updates using effective time for g2 and g3 scheduling
                g2_term = g2 * grad * norm_term * clip_g2_term
                g3_term = g3 * (1 + effective_time)**(1-kappa) * m * norm_term
                
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

# class DANA(Optimizer):
    
#     def __init__(
#         self,
#         params: Iterable[torch.Tensor],
#         lr: float = 1.0, # learning rate for debugging
#         # g2: float = 1e-4,
#         # g3: float = 1e-5,
#         delta: float = 8.0,
#         kappa: float = 1.0,
#         epsilon: float = 1e-8,
#         weight_decay: float = 0.0,
#         weight_time: bool = False,
#         ):

#         """
#         DANA optimizer: same as DANA-STAR but without the effective time term.
        
#         Args:
#             params: Iterable of parameters to optimize.
#             lr: Learning rate. In the code, g2=g3 are taken equal to lr.
#             delta: Delta parameter.
#             kappa: Kappa parameter.
#             epsilon: Epsilon parameter.
#             weight_decay: Weight decay parameter.
#         """

#         defaults = dict(
#             lr=lr, delta=delta, epsilon=epsilon, kappa=kappa, weight_decay=weight_decay, weighted_step_count=0)
#         self.lr = lr
#         self.delta = delta
#         self.epsilon = epsilon
#         self.kappa = kappa
#         self.weight_decay = weight_decay
#         self.weight_time = weight_time

#         super(DANA, self).__init__(params, defaults)
        
#         # Global step counter
#         self._step_count = 0
    
#     def _make_schedule(self, value: Union[float, Callable[[int], float]]) -> Callable[[int], float]:
#         """Convert scalar or schedule to callable function."""
#         if callable(value):
#             return value
#         else:
#             return lambda step: value

#     #@torch.compile
#     @torch.no_grad()
#     def step(self, closure=None):
#         """Perform a single optimization step."""
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()
        
#         self._step_count += 1
        
#         for group in self.param_groups:
#             # Get schedule functions
#             g2 = group['lr']
#             g3 = group['lr']
#             lr = group['lr']
#             time_factor = group['lr'] / self.lr
#             group['weighted_step_count'] += time_factor
#             delta = group['delta']
#             kappa = group['kappa']
#             wd = group['weight_decay']
#             epsilon = group['epsilon']
            
#             for p in group['params']:
#                 grad = p.grad
#                 if grad is None:
#                     continue
                
#                 state = self.state[p]
                
#                 # State initialization
#                 if len(state) == 0:
#                     state['step'] = 0
#                     state['m'] = torch.zeros_like(p)  # First moment
#                     state['v'] = torch.zeros_like(p)  # Second moment
                
#                 m, v = state['m'], state['v']
#                 state['step'] += 1
                
#                 # Stable EMA coefficient in (0,1): alpha = delta / (delta + t)
#                 if self.weight_time: # potentially take one during warmup
#                     step = group['weighted_step_count']
#                     alpha = delta / (delta + step) * time_factor
#                 else:
#                     step = state['step']
#                     alpha = delta / (delta + step)              
                
#                 # Update first moment
#                 m.mul_(1 - alpha).add_(grad, alpha=alpha)
#                 # Update second moment
#                 v.mul_(1 - alpha).addcmul_(grad, grad, value=alpha)
                
#                 normalized_factor = torch.sqrt(v) + epsilon
#                 # Compute parameter updates using effective time for g2 and g3 scheduling
#                 g2_term = g2 * grad / normalized_factor
#                 g3_term = g3 * (1 + step)**(1-kappa) * m / normalized_factor
                
#                 # Apply the main update
#                 update = -(g2_term + g3_term)

#                 # Decoupled weight decay (AdamW-style)
#                 if wd != 0:
#                     p.add_(p, alpha= - wd * lr)
#                 # print(f"g2: {g2}, g3: {g3}, lr: {lr}, wd: {wd}")
#                 # Apply update to parameters with scheduled LR
#                 p.add_(update)
        
#         return loss


class DANA(Optimizer):
    """
    DANA2 optimizer: DANA algorithm implemented with AdEMAMix-style structure.
    
    This implementation takes the clean, well-structured code from AdEMAMix
    but uses DANA's mathematical formulation with:
    - Delta-based EMA coefficient: alpha = delta / (delta + t)
    - DANA's specific moment updates and parameter updates
    - Kappa-based momentum scaling: (1 + step)**(1-kappa)
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
        use_grad_ema_for_g2=False,
        grad_ema_beta=0.9,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= delta:
            raise ValueError("Invalid delta value: {}".format(delta))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= grad_ema_beta < 1.0:
            raise ValueError("Invalid grad_ema_beta value: {}".format(grad_ema_beta))
        
        defaults = dict(
            lr=lr,
            delta=delta,
            kappa=kappa,
            eps=eps,
            weight_decay=weight_decay,
            weight_time=weight_time,
            use_grad_ema_for_g2=use_grad_ema_for_g2,
            grad_ema_beta=grad_ema_beta,
            weighted_step_count=0
        )
        super(DANA2, self).__init__(params, defaults)
        
        # Global step counter for weight_time
        self._step_count = 0

    def __setstate__(self, state):
        super(DANA2, self).__setstate__(state)

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
            use_grad_ema_for_g2 = group["use_grad_ema_for_g2"]
            grad_ema_beta = group["grad_ema_beta"]
            
            # Update weighted step count for weight_time feature
            if weight_time:
                time_factor = lr / getattr(self, 'initial_lr', lr)  # Fallback to current lr if initial_lr not set
                group['weighted_step_count'] += time_factor

            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("DANA2 does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # First moment estimate (momentum)
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Second moment estimate (RMSprop-style)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Optional gradient EMA for g2 term
                    if use_grad_ema_for_g2:
                        state["grad_ema"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                
                # Determine which step count to use for alpha calculation
                if weight_time:
                    step_for_alpha = group['weighted_step_count']
                else:
                    step_for_alpha = state["step"]
                
                # DANA's delta-based EMA coefficient: alpha = delta / (delta + t)
                alpha = delta / (delta + step_for_alpha)

                # Update first moment (momentum) - DANA style
                exp_avg.mul_(1 - alpha).add_(grad, alpha=alpha)
                
                # Update second moment (squared gradients) - DANA style  
                exp_avg_sq.mul_(1 - alpha).addcmul_(grad, grad, value=alpha)

                # Update gradient EMA if enabled for g2 term
                if use_grad_ema_for_g2:
                    grad_ema = state["grad_ema"]
                    grad_ema.mul_(grad_ema_beta).add_(grad, alpha=1 - grad_ema_beta)
                    g2_gradient = grad_ema
                else:
                    g2_gradient = grad

                # Compute normalized factor (denominator)
                normalized_factor = exp_avg_sq.sqrt() + eps
                
                # DANA's parameter update terms:
                # g2_term: gradient (current or EMA) scaled by learning rate
                g2_term = lr * g2_gradient / normalized_factor
                
                # g3_term: momentum with kappa-based time scaling
                g3_term = lr * (1 + state["step"])**(1-kappa) * exp_avg / normalized_factor
                
                # Combine update terms (note: negative because we're doing gradient descent)
                update = -(g2_term + g3_term)

                # Decoupled weight decay (AdamW-style) - applied before main update
                if wd != 0:
                    p.add_(p, alpha=-wd * lr)
                
                # Apply the main parameter update
                p.add_(update)

        return loss