import torch
from torch.optim import Optimizer


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
        self.initial_lr = lr  # Peak LR for computing schedule factor
        self.wd_decaying = wd_decaying
        self.wd_ts = wd_ts
        super(AdamWDecayingWD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            schedule_factor = group['lr'] / self.initial_lr  # γ(t) without peak LR
            beta1, beta2 = group['betas']
            epsilon = group['epsilon']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1
                bias_correction2_sqrt = bias_correction2 ** 0.5

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(epsilon)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Independent weight decay (paper convention):
                # WD is multiplied by schedule γ(t) but NOT by peak LR γ*
                if weight_decay != 0:
                    if self.wd_decaying:
                        wd_factor = -weight_decay / (1 + step / self.wd_ts) * schedule_factor
                    else:
                        wd_factor = -weight_decay * schedule_factor
                    p.mul_(1 + wd_factor)

        return loss
