import math

import numpy as np


def powerlaw_schedule_with_warmup(
    n_iterations,
    n_warmup,
    init_value=1.0,
    saturation_value=0.0,
    power=-0.5,
    time_scale=1.0,
):
    """Power-law schedule with linear warmup.

    Args:
        n_iterations: total number of iterations
        n_warmup: number of warmup iterations
        init_value: initial value after warmup
        saturation_value: minimum/maximum value (depending on power sign)
        power: exponent for power law (negative for decay, positive for growth)
        time_scale: scales the rate of change

    Returns:
        schedule: a function that takes the current iteration and
                 returns the multiplicative factor for the learning rate

    Examples:
        # Decay schedule (like 1/sqrt(t))
        schedule = powerlaw_schedule_with_warmup(
            n_iterations=10000,
            n_warmup=1000,
            init_value=1.0,
            saturation_value=0.1,
            power=-0.5,
            time_scale=1000.0
        )

        # Growth schedule
        schedule = powerlaw_schedule_with_warmup(
            n_iterations=10000,
            n_warmup=1000,
            init_value=0.1,
            saturation_value=1.0,
            power=0.5,
            time_scale=1000.0
        )
    """
    def schedule(step):
        if step < n_warmup:
            # Linear warmup from 0 to init_value
            return (step / n_warmup) * init_value
        else:
            # Power-law schedule starting from warmup end
            t = step - n_warmup  # Time since warmup ended
            frac = 1.0 + t / time_scale
            value = init_value * (frac ** power)

            # Clamp to saturation value (works for both growth and decay)
            if power < 0:
                # Decay: take maximum (don't go below saturation)
                return max(value, saturation_value)
            else:
                # Growth: take minimum (don't go above saturation)
                return min(value, saturation_value)

    return schedule


def cos_inf_schedule(n_iterations, n_warmup, div_factor, final_div_factor, n_inf):
    """Cosine annealing with warmup and _constant_ final_lr after cycle ended.
    Args:
        n_iterations: total number of iterations
        n_warmup: number of warmup iterations
        div_factor: initial division factor for warmup
        final_div_factor: final division factor for final lr
        n_inf: number of iterations for the final lr (constant lr after cycle ended)
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    max_lr = 1.0
    base_lr = max_lr / div_factor
    final_lr = base_lr / final_div_factor

    n_anneal_steps = n_iterations - n_inf

    def schedule(step):
        if step < n_warmup:
            return (step / n_warmup) + (1 - step / n_warmup) / div_factor
        elif step < n_anneal_steps:
            t = (step - n_warmup) / (n_anneal_steps - n_warmup)
            lr = final_lr + 0.5 * (max_lr - final_lr) * (1 + np.cos(np.pi * t))
            return lr
        else:
            return final_lr

    return schedule


def wsd_schedule(
    n_iterations,
    final_lr_factor=0.0,
    n_warmup=1000,
    init_div_factor=100,
    fract_decay=0.1,
    decay_type="linear",
):
    """Warmup, hold, and decay schedule.
    Args:
        n_iterations: total number of iterations
        final_lr_factor: factor by which to reduce max_lr at the end
        warmup_fract: fraction of iterations used for warmup
        init_div_factor: initial division factor for warmup
        fract_decay: fraction of iterations used for decay
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    n_anneal_steps = int(fract_decay * n_iterations)
    n_hold = n_iterations - n_anneal_steps

    def schedule(step):
        if step < n_warmup:
            return (step / n_warmup) + (1 - step / n_warmup) / init_div_factor
        elif step < n_hold:
            return 1.0
        elif step < n_iterations:
            if decay_type == "linear":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
            elif decay_type == "exp":
                return final_lr_factor ** ((step - n_hold) / n_anneal_steps)
            elif decay_type == "cosine":
                return (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
            elif decay_type == "miror_cosine":
                cosine_value = (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
                linear_value = final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
                return linear_value * 2 - cosine_value
            elif decay_type == "square":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - ((step - n_hold) / n_anneal_steps) ** 2
                )

            elif decay_type == "sqrt":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - math.sqrt((step - n_hold) / n_anneal_steps)
                )

            else:
                raise ValueError(
                    f"decay type {decay_type} is not in ['cosine','miror_cosine','linear','exp','square','sqrt']"
                )

        else:
            return final_lr_factor

    return schedule
