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
                        p.add_(p, alpha= - wd / (1 + step) * lr)
                    else:
                        p.add_(p, alpha= - wd * lr)
                
                # Apply update to parameters with scheduled LR
                p.add_(update)
        
        return loss