import math
import torch


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
    r"""AdEMAMix with optional decaying weight decay.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999, 0.9999))
        alpha (float): mixing coefficient (default: 2)
        beta3_warmup (int, optional): warmup steps for beta3 (default: None)
        alpha_warmup (int, optional): warmup steps for alpha (default: None)
        eps (float, optional): numerical stability term (default: 1e-8)
        weight_decay (float, optional): weight decay (default: 0)
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

                if len(state) == 0:
                    state["step"] = 0
                    if beta1 != 0.0:
                        state["exp_avg_fast"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    else:
                        state["exp_avg_fast"] = None
                    state["exp_avg_slow"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
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

                state["current_alpha"] = alpha
                state["current_beta3"] = beta3
                state["current_one_minus_beta3"] = 1 - beta3

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

                if wd_decaying:
                    wd_factor = -wd / (1 + state["step"] / wd_ts) * lr
                else:
                    wd_factor = -wd * lr
                p.mul_(1 + wd_factor)

                p.add_(-lr * update)

        return loss
