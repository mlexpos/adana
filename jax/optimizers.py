from typing import NamedTuple, Optional, Callable, Union, Any

import jax
import jax.numpy as jnp
import chex

import optax
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src import utils

def powerlaw_schedule(
    init_value: chex.Scalar,
    saturation_value: chex.Scalar,
    power: chex.Scalar,
    time_scale: chex.Scalar,
) -> base.Schedule:
  """Constructs power-law schedule.

  This function decays (or grows) the learning rate, until it is below
  the saturation_value, at which time it is held. The formula is given by
  :math:`max{ I*(1+t / time_scale) ^ {power}, saturation_value}`

  where :math:`I` is the initial value, :math:`t` is the current iteration,
   :math:`time_scale` is the time scale of the power law,
   :math:`power` is the power, and :math:`saturation_value` is the value
   at which the power law is saturated.

  Args:
    init_value: initial value for the scalar to be annealed.
    saturation_value: end value of the scalar to be annealed.
    power: the power of the power law.
    time_scale: number of steps over which the power law takes place.
      The scalar starts changing at ``transition_begin`` steps and completes
      the transition by ``transition_begin + transition_steps`` steps.
      If ``transition_steps <= 0``, then the entire annealing process is
      disabled and the value is held fixed at ``init_value``.
    transition_begin: must be positive. After how many steps to start annealing
      (before this many steps the scalar value is held fixed at ``init_value``).

  Returns:
    schedule
      A function that maps step counts to values.

  Examples:
    >>> schedule_fn = optax.powerlaw_schedule(
    ...    init_value=1.0, saturation_value=0.01, time_scale=100, power=2)
    >>> schedule_fn(0)  # learning rate on the first iteration
    Array(1., dtype=float32, weak_type=True)
    >>> schedule_fn(100)  # learning rate on the last iteration
    Array(0.01, dtype=float32, weak_type=True)
  """

  def schedule(count):
    frac = 1 + count / time_scale
    return jnp.maximum((init_value) * (frac**power),saturation_value)

  return schedule


class GalaxyOptimizerState(NamedTuple):
  """State for the Galaxy algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  m: base.Updates
  v: base.Updates
  tau: base.Updates

def get_dana_star(
    g2_constant: float,
    g3_constant: float,
    kappa_exponent: float,
    learning_rate: base.ScalarOrSchedule, #outer learning rate
    epsilon: float = 1e-8,
    y_dtype: Optional[chex.ArrayDType] = None
):
    g2 = powerlaw_schedule(g2_constant, 0.0, 0.0, 1.0)
    g3 = powerlaw_schedule(g3_constant, 0.0, (1.0-kappa_exponent), 1.0)
    beta_m=powerlaw_schedule(1.0, 0.0, -1.0, 6.0)

    optimizer = optax.chain(
        galaxy_optimizer(g2, g3, beta_m, epsilon=epsilon, y_dtype=y_dtype),
        optax.scale_by_learning_rate(learning_rate, flip_sign = False)
    )
    return optimizer

class DanaMK4OptimizerState(NamedTuple):
    """State for the Dana MK4 algorithm (no tau tracking)."""
    count: chex.Array  # shape=(), dtype=jnp.int32.
    m: base.Updates
    v: base.Updates


def adana_optimizer(
    g2: base.ScalarOrSchedule,
    g3: base.ScalarOrSchedule,
    Delta: base.ScalarOrSchedule,
    epsilon: float = 1e-8,
    kappa: float = 1.0,
    wd: Optional[base.ScalarOrSchedule] = None,
    *,
    y_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
    """ADana optimizer (no SNR clipping).

    This is a simplified version of dana_mk4_optimizer without the SNR clipping
    on alpha_factor. Matches PyTorch ADana with clipsnr=None.

    Args:
        g2: A scalar or schedule determining the gradient coefficient.
        g3: A scalar or schedule determining the momentum coefficient.
        Delta: A scalar or schedule determining the EMA decay (alpha = Delta/(Delta+t)).
        epsilon: Small constant for numerical stability.
        kappa: Exponent for effective time scaling (default: 1.0).
        wd: Optional scalar or schedule for weight decay.
        y_dtype: Optional `dtype` to be used for the momentum accumulator.

    Returns:
        A :class:`optax.GradientTransformation` object.
    """
    y_dtype = utils.canonicalize_dtype(y_dtype)

    if wd is None:
        wd = lambda _: 0.0
    elif not callable(wd):
        wd = lambda _: wd

    def init_fn(params):
        m = otu.tree_zeros_like(params, dtype=y_dtype)  # First moment
        v = otu.tree_zeros_like(params, dtype=y_dtype)  # Second moment
        return DanaMK4OptimizerState(count=jnp.zeros([], jnp.int32), m=m, v=v)

    def update_fn(updates, state, params):
        new_wd = wd(state.count)
        newDelta = Delta(state.count)
        alpha = newDelta

        # Update second moment: v = v*(1-alpha) + alpha*u^2
        new_v = jax.tree.map(
            lambda v, u: None if v is None else v * (1 - alpha) + alpha * (u ** 2),
            state.v,
            updates,
            is_leaf=lambda x: x is None,
        )

        # Update first moment: m = m*(1-alpha) + alpha*u
        new_m = jax.tree.map(
            lambda m, u: None if m is None else m * (1 - alpha) + alpha * u,
            state.m,
            updates,
            is_leaf=lambda x: x is None,
        )

        # Compute updates using ADana formula (no SNR clipping)
        updates = jax.tree.map(
            lambda m, u, v: -1.0 * g2(state.count) * u
            if m is None
            else adana_update(m, u, v, state.count),
            new_m,
            updates,
            new_v,
            is_leaf=lambda x: x is None,
        )

        # Apply weight decay
        if params is not None:
            updates = jax.tree.map(
                lambda u, p: u + (-1.0 * new_wd) * p,
                updates,
                params,
                is_leaf=lambda x: x is None,
            )

        new_m = otu.tree_cast(new_m, y_dtype)
        new_v = otu.tree_cast(new_v, y_dtype)
        count_inc = numerics.safe_increment(state.count)

        return updates, DanaMK4OptimizerState(count=count_inc, m=new_m, v=new_v)

    def adana_update(m, u, v, t):
        """ADana update rule (no SNR clipping)."""
        sqrt_v_eps = jnp.sqrt(v) + epsilon
        norm_term = 1.0 / sqrt_v_eps

        # m_norm_term = |m| / (sqrt(v) + epsilon)
        m_norm_term = jnp.abs(m) * norm_term

        # Momentum factor
        mfac = m_norm_term

        # Effective time (for t >= 1): eff_time = t
        eff_time = jnp.maximum(t, 1.0)

        # Alpha factor without clipping (unlike dana_mk4 which clips to clipsnr)
        alpha_factor = jnp.power(eff_time, 1.0 - kappa) * mfac

        # g3 term: -g3 * (sign(m) * (alpha_factor + m_norm_term))
        g3_term = (-g3(eff_time)) * (jnp.sign(m) * (alpha_factor + m_norm_term))

        # g2 term: -g2 * u * norm_term
        g2_term = (-g2(eff_time)) * u * norm_term

        return g2_term + g3_term

    return base.GradientTransformation(init_fn, update_fn)


def get_adana(
    g2_constant: float,
    g3_constant: float,
    learning_rate: base.ScalarOrSchedule,
    epsilon: float = 1e-8,
    kappa: float = 1.0,
    delta: float = 8.0,
    y_dtype: Optional[chex.ArrayDType] = None
):
    """Get ADana optimizer with standard hyperparameters (no SNR clipping).

    Args:
        g2_constant: Constant value for g2 (gradient coefficient).
        g3_constant: Constant value for g3 (momentum coefficient).
        learning_rate: Outer learning rate schedule.
        epsilon: Small constant for numerical stability (default: 1e-8).
        kappa: Exponent for effective time scaling (default: 1.0).
        delta: Delta parameter for EMA coefficient (default: 8.0).
        y_dtype: Optional dtype for momentum accumulators.

    Returns:
        Configured optimizer chain.
    """
    g2 = powerlaw_schedule(g2_constant, 0.0, 0.0, 1.0)
    g3 = powerlaw_schedule(g3_constant, 0.0, 0.0, 1.0)
    Delta = powerlaw_schedule(1.0, 0.0, -1.0, delta)

    optimizer = optax.chain(
        adana_optimizer(g2, g3, Delta, epsilon=epsilon, kappa=kappa, y_dtype=y_dtype),
        optax.scale_by_learning_rate(learning_rate, flip_sign=False)
    )
    return optimizer


def dana_mk4_optimizer(
    g2: base.ScalarOrSchedule,
    g3: base.ScalarOrSchedule,
    Delta: base.ScalarOrSchedule,
    epsilon: float = 1e-8,
    kappa: float = 1.0,
    clipsnr: float = 2.0,
    wd: Optional[base.ScalarOrSchedule] = None,
    *,
    y_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
    """Dana MK4 optimizer (tau fixed to 1.0).

    This is a simplified version of dana-star-mk4 where tau is always 1.0.
    This removes all tau tracking and regularization, simplifying the update rule.

    Key features:
    - No tau tracking (tau = 1 everywhere)
    - Kappa-based effective time scaling: t^(1-kappa)
    - Alpha factor with SNR clipping
    - Uses |m|/(v+epsilon) for the g3 momentum term

    Args:
        g2: A scalar or schedule determining the gradient coefficient.
        g3: A scalar or schedule determining the momentum coefficient.
        Delta: A scalar or schedule determining the EMA decay (alpha = Delta/(Delta+t)).
        epsilon: Small constant for numerical stability.
        kappa: Exponent for effective time scaling (default: 1.0).
        clipsnr: Clipping factor for alpha_factor (default: 2.0).
        wd: Optional scalar or schedule for weight decay.
        y_dtype: Optional `dtype` to be used for the momentum accumulator.

    Returns:
        A :class:`optax.GradientTransformation` object.
    """
    y_dtype = utils.canonicalize_dtype(y_dtype)

    if wd is None:
        wd = lambda _: 0.0
    elif not callable(wd):
        wd = lambda _: wd

    def init_fn(params):
        m = otu.tree_zeros_like(params, dtype=y_dtype)  # First moment
        v = otu.tree_zeros_like(params, dtype=y_dtype)  # Second moment
        return DanaMK4OptimizerState(count=jnp.zeros([], jnp.int32), m=m, v=v)

    def update_fn(updates, state, params):
        new_wd = wd(state.count)
        newDelta = Delta(state.count)
        alpha = newDelta

        # Update second moment: v = v*(1-alpha) + alpha*u^2
        new_v = jax.tree.map(
            lambda v, u: None if v is None else v * (1 - alpha) + alpha * (u ** 2),
            state.v,
            updates,
            is_leaf=lambda x: x is None,
        )

        # Update first moment: m = m*(1-alpha) + alpha*u
        new_m = jax.tree.map(
            lambda m, u: None if m is None else m * (1 - alpha) + alpha * u,
            state.m,
            updates,
            is_leaf=lambda x: x is None,
        )

        # Compute updates using MK4 formula (tau = 1)
        updates = jax.tree.map(
            lambda m, u, v: -1.0 * g2(state.count) * u
            if m is None
            else mk4_update_tau1(m, u, v, state.count),
            new_m,
            updates,
            new_v,
            is_leaf=lambda x: x is None,
        )

        # Apply weight decay
        if params is not None:
            updates = jax.tree.map(
                lambda u, p: u + (-1.0 * new_wd) * p,
                updates,
                params,
                is_leaf=lambda x: x is None,
            )

        new_m = otu.tree_cast(new_m, y_dtype)
        new_v = otu.tree_cast(new_v, y_dtype)
        count_inc = numerics.safe_increment(state.count)

        return updates, DanaMK4OptimizerState(count=count_inc, m=new_m, v=new_v)

    def mk4_update_tau1(m, u, v, t):
        """MK4 update rule with tau fixed to 1.0.

        When tau = 1:
        - tau_reg = 1
        - sqrt_tau_reg = 1
        - effective_time = max(1 * t, 1) = max(t, 1) = t (for t >= 1)
        """
        # With tau_reg = 1, norm_term = 1 / (sqrt(v) + epsilon)
        sqrt_v_eps = jnp.sqrt(v) + epsilon
        norm_term = 1.0 / sqrt_v_eps

        # m_norm_term = |m| / (sqrt(v) + epsilon)
        m_norm_term = jnp.abs(m) * norm_term

        # Momentum factor: mfac = m_norm_term / tau_reg = m_norm_term / 1 = m_norm_term
        mfac = m_norm_term

        # Effective time (for t >= 1): eff_time = t
        eff_time = jnp.maximum(t, 1.0)

        # Alpha factor with kappa scaling and clipping
        # alpha_factor = clamp((eff_time^(1-kappa)) * mfac, max=clipsnr)
        alpha_factor = jnp.minimum(
            jnp.power(eff_time, 1.0 - kappa) * mfac,
            clipsnr
        )

        # g3 term: -g3 * (sign(m) * (tau_reg * alpha_factor + m_norm_term))
        # With tau_reg = 1: -g3 * sign(m) * (alpha_factor + m_norm_term)
        g3_term = (-g3(eff_time)) * (jnp.sign(m) * (alpha_factor + m_norm_term))

        # g2 term: -g2 * u * norm_term
        g2_term = (-g2(eff_time)) * u * norm_term

        # Combined update
        return g2_term + g3_term

    return base.GradientTransformation(init_fn, update_fn)


def get_dana_mk4(
    g2_constant: float,
    g3_constant: float,
    learning_rate: base.ScalarOrSchedule,
    epsilon: float = 1e-8,
    kappa: float = 1.0,
    clipsnr: float = 2.0,
    delta: float = 8.0,
    y_dtype: Optional[chex.ArrayDType] = None
):
    """Get Dana MK4 optimizer with standard hyperparameters (tau fixed to 1).

    Args:
        g2_constant: Constant value for g2 (gradient coefficient).
        g3_constant: Constant value for g3 (momentum coefficient).
        learning_rate: Outer learning rate schedule.
        epsilon: Small constant for numerical stability (default: 1e-8).
        kappa: Exponent for effective time scaling (default: 1.0).
        clipsnr: Clipping factor for alpha_factor (default: 2.0).
        delta: Delta parameter for EMA coefficient (default: 8.0).
        y_dtype: Optional dtype for momentum accumulators.

    Returns:
        Configured optimizer chain.
    """
    g2 = powerlaw_schedule(g2_constant, 0.0, 0.0, 1.0)
    g3 = powerlaw_schedule(g3_constant, 0.0, 0.0, 1.0)
    Delta = powerlaw_schedule(1.0, 0.0, -1.0, delta)

    optimizer = optax.chain(
        dana_mk4_optimizer(g2, g3, Delta, epsilon=epsilon, kappa=kappa, clipsnr=clipsnr, y_dtype=y_dtype),
        optax.scale_by_learning_rate(learning_rate, flip_sign=False)
    )
    return optimizer


def dana_star_mk4_optimizer(
    g2: base.ScalarOrSchedule,
    g3: base.ScalarOrSchedule,
    Delta: base.ScalarOrSchedule,
    epsilon: float = 1e-8,
    kappa: float = 1.0,
    clipsnr: float = 2.0,
    wd: Optional[base.ScalarOrSchedule] = None,
    *,
    y_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
    """Dana-Star MK4 optimizer (PyTorch-equivalent implementation).

    This implementation follows the PyTorch dana_star_mk4.py logic exactly.
    Key differences from other optimizers:
    - Uses |m|/(v+epsilon) for the g3 momentum term (mk4 formula)
    - Includes kappa-based effective time scaling: (effective_time)^(1-kappa)
    - Alpha factor with SNR clipping
    - Simplified momentum update (no complex tau regularization in g3 term)

    Args:
        g2: A scalar or schedule determining the gradient coefficient.
        g3: A scalar or schedule determining the momentum coefficient.
        Delta: A scalar or schedule determining the EMA decay (alpha = Delta/(Delta+t)).
        epsilon: Small constant for numerical stability.
        kappa: Exponent for effective time scaling (default: 1.0).
        clipsnr: Clipping factor for alpha_factor (default: 2.0).
        wd: Optional scalar or schedule for weight decay.
        y_dtype: Optional `dtype` to be used for the momentum accumulator.

    Returns:
        A :class:`optax.GradientTransformation` object.
    """
    y_dtype = utils.canonicalize_dtype(y_dtype)

    if wd is None:
        wd = lambda _: 0.0
    elif not callable(wd):
        wd = lambda _: wd

    # Helper functions matching PyTorch implementation
    clip_tohalf = lambda tau: jnp.minimum(tau, 0.5)

    # Tau regularization: converts tau estimate to probability estimate
    # tau_reg = max(tau/(1-tau), 1/(1+t))
    tau_reg = lambda tau, t: jnp.maximum(
        clip_tohalf(tau) / (1.0 - clip_tohalf(tau)),
        jnp.power(1.0 + t, -1.0)
    )
    root_tau_reg = lambda tau, t: jnp.sqrt(tau_reg(tau, t))

    # Effective time: max(tau * t, 1.0)
    effective_time = lambda tau, t: jnp.maximum(tau * t, 1.0)

    # Tau updater: tau_update = |u| / (|u| + sqrt(v) + epsilon)
    tau_updater = lambda tau, u, v, t: jnp.abs(u) / (jnp.abs(u) + jnp.sqrt(v) + epsilon)

    def init_fn(params):
        m = otu.tree_zeros_like(params, dtype=y_dtype)  # First moment
        v = otu.tree_zeros_like(params, dtype=y_dtype)  # Second moment
        tau = otu.tree_zeros_like(params, dtype=y_dtype)  # Tau estimate
        return GalaxyOptimizerState(count=jnp.zeros([], jnp.int32), m=m, v=v, tau=tau)

    def update_fn(updates, state, params):
        new_wd = wd(state.count)
        newDelta = Delta(state.count)
        alpha = newDelta

        # Update second moment: v = v*(1-alpha) + alpha*u^2
        new_v = jax.tree.map(
            lambda v, u: None if v is None else v * (1 - alpha) + alpha * (u ** 2),
            state.v,
            updates,
            is_leaf=lambda x: x is None,
        )

        # Update tau estimate
        new_tau = jax.tree.map(
            lambda tau, u, v: None if tau is None else tau * (1 - alpha) + alpha * tau_updater(tau, u, v, state.count),
            state.tau,
            updates,
            new_v,
            is_leaf=lambda x: x is None,
        )

        # Update first moment: m = m*(1-alpha) + alpha*u
        new_m = jax.tree.map(
            lambda m, u: None if m is None else m * (1 - alpha) + alpha * u,
            state.m,
            updates,
            is_leaf=lambda x: x is None,
        )

        # Compute updates using MK4 formula
        updates = jax.tree.map(
            lambda m, u, v, tau: -1.0 * g2(effective_time(tau, state.count)) * u
            if m is None
            else mk4_update(m, u, v, tau, state.count),
            new_m,
            updates,
            new_v,
            new_tau,
            is_leaf=lambda x: x is None,
        )

        # Apply weight decay
        if params is not None:
            updates = jax.tree.map(
                lambda u, p: u + (-1.0 * new_wd) * p,
                updates,
                params,
                is_leaf=lambda x: x is None,
            )

        new_m = otu.tree_cast(new_m, y_dtype)
        new_v = otu.tree_cast(new_v, y_dtype)
        new_tau = otu.tree_cast(new_tau, y_dtype)
        count_inc = numerics.safe_increment(state.count)

        return updates, GalaxyOptimizerState(count=count_inc, m=new_m, v=new_v, tau=new_tau)

    def mk4_update(m, u, v, tau, t):
        """MK4 update rule matching PyTorch implementation exactly."""
        # Compute tau regularization
        tau_r = tau_reg(tau, t)
        sqrt_tau_r = root_tau_reg(tau, t)
        eff_time = effective_time(tau, t)

        # Compute sqrt(v) + epsilon (reused)
        sqrt_v_eps = jnp.sqrt(v) + epsilon

        # Normalization term: sqrt(tau_reg) / (sqrt(v) + epsilon)
        norm_term = sqrt_tau_r / sqrt_v_eps

        # m_norm_term = |m| * norm_term
        m_norm_term = jnp.abs(m) * norm_term

        # Momentum factor: mfac = m_norm_term / tau_reg
        mfac = m_norm_term / tau_r

        # Alpha factor with kappa scaling and clipping
        # alpha_factor = clamp((eff_time^(1-kappa)) * mfac, max=clipsnr)
        alpha_factor = jnp.minimum(
            jnp.power(eff_time, 1.0 - kappa) * mfac,
            clipsnr
        )

        # g3 term: -g3 * (sign(m) * (tau_reg * alpha_factor + m_norm_term))
        g3_term = (-g3(eff_time)) * (jnp.sign(m) * (tau_r * alpha_factor + m_norm_term))

        # g2 term: -g2 * u * norm_term
        g2_term = (-g2(eff_time)) * u * norm_term

        # Combined update
        return g2_term + g3_term

    return base.GradientTransformation(init_fn, update_fn)


def get_dana_star_mk4(
    g2_constant: float,
    g3_constant: float,
    learning_rate: base.ScalarOrSchedule,
    epsilon: float = 1e-8,
    kappa: float = 1.0,
    clipsnr: float = 2.0,
    delta: float = 8.0,
    y_dtype: Optional[chex.ArrayDType] = None
):
    """Get Dana-Star MK4 optimizer with standard hyperparameters.

    Args:
        g2_constant: Constant value for g2 (gradient coefficient).
        g3_constant: Constant value for g3 (momentum coefficient).
        learning_rate: Outer learning rate schedule.
        epsilon: Small constant for numerical stability (default: 1e-8).
        kappa: Exponent for effective time scaling (default: 1.0).
        clipsnr: Clipping factor for alpha_factor (default: 2.0).
        delta: Delta parameter for EMA coefficient (default: 8.0).
        y_dtype: Optional dtype for momentum accumulators.

    Returns:
        Configured optimizer chain.
    """
    g2 = powerlaw_schedule(g2_constant, 0.0, 0.0, 1.0)
    g3 = powerlaw_schedule(g3_constant, 0.0, 0.0, 1.0)
    Delta = powerlaw_schedule(1.0, 0.0, -1.0, delta)

    optimizer = optax.chain(
        dana_star_mk4_optimizer(g2, g3, Delta, epsilon=epsilon, kappa=kappa, clipsnr=clipsnr, y_dtype=y_dtype),
        optax.scale_by_learning_rate(learning_rate, flip_sign=False)
    )
    return optimizer

def galaxy_optimizer(
    g2: base.ScalarOrSchedule,
    g3: base.ScalarOrSchedule,
    beta_m : base.ScalarOrSchedule,
    epsilon: float = 1e-8,
    beta_v : Optional[base.ScalarOrSchedule] = None,
    beta_tau : Optional[base.ScalarOrSchedule] = None,
    #g1: Optional[base.ScalarOrSchedule] = None,
    #beta_v : Optional[base.ScalarOrSchedule] = None,
    magic_tau : float = 1.0,
    wd : Optional[base.ScalarOrSchedule] = None,
    #momentum_flavor: str = "effective-clip",
    #tau_flavor: str = "second-moment",
    clipsnr: float = 2.0,
    *,
    y_dtype: Optional[chex.ArrayDType] = None,
  ) -> base.GradientTransformation:
    """Galaxy optimizer.

    Args:
        g2: A scalar or schedule determining the second gradient coefficient.
        g3: A scalar or schedule determining the third gradient coefficient.
        beta1: A scalar or schedule determining the momentum hits both terms (Delta & g1)
        beta2: A scalar or schedule determining the momentum hits only the first term (Delta)
        epsilon: Small constant for numerical stability.
        eta: A scalar or schedule determining the learning rate hits all terms (g2, g3), used for WSD
        beta_m: Optional scalar or schedule for momentum decay rate.
        g1: Optional scalar or schedule for first gradient coefficient.
        beta_v: Optional scalar or schedule for second momentum decay rate.
        magic_tau: Scaling factor for tau updates.
        wd: Optional scalar or schedule for weight decay.
        momentum_flavor: Type of momentum term for g3. Options are "effective-clip" (default) or "theory".
        clipsnr: Clipping factor for signal-to-noise ratio in g2 momentum term (default: 2.0).
        y_dtype: Optional `dtype` to be used for the momentum accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    y_dtype = utils.canonicalize_dtype(y_dtype)
    #if beta_m is None:
    #    beta_m = Delta
    #elif not callable(beta_m):
    #    beta_m = lambda _: beta_m
    if beta_v is None:
        beta_v = beta_m
    elif not callable(beta_v):
        beta_v = lambda _: beta_v
    if beta_tau is None:
        beta_tau = beta_m
    elif not callable(beta_tau):
        beta_tau = lambda _: beta_tau
    #if g1 is None:
    #     g1 = lambda _: 1.0
    #elif not callable(g1):
    #    g1 = lambda _: g1
    elif not callable(eta):
        eta = lambda _: eta
    if wd is None:
        wd = lambda _: 0.0
    elif not callable(wd):
        wd = lambda _: wd


    ## Helper functions
    clip_tohalf = lambda tau : jnp.minimum(tau,0.5)

    ##  This function acocomplishes three things:
    ## 1. tau is not a faithful estimate of p if p << 1/t.  So this function never ouputs less than 1/t.
    ## 2. The tau-updater will converge to p/(1+p) in an idealized environment.  The formula will output p instead of p/(1+p).
    ## 3. Since the inverse of this function is tau/(1-tau), which is correct only when tau < 0.5, we clip to tau < 0.5 first.
    tau_reg = lambda tau, t : jnp.maximum(clip_tohalf(tau)/(1.0-clip_tohalf(tau)), jnp.pow(1.0+t,-1.0))
    root_tau_reg = lambda tau, t : jnp.sqrt(tau_reg(tau, t))
    effective_time = lambda tau, t: jnp.maximum(tau*t,1.0)
    #quarter_root_tau_reg = lambda tau, t : jnp.power(tau_reg(tau, t),0.25)


    #In an idealized environment, this will lead to tau storing p/(1+p)
    tau_updater = lambda tau, u, v, t : (jnp.abs(u))/ ( (jnp.abs(u)) + jnp.sqrt(v) + epsilon) #lambda tau,u,v,t : (u**2)/ ( (u**2) + v + epsilon**2)
    g2_momentum_term = lambda u, v, tau, t: (root_tau_reg(tau, t)/((jnp.sqrt(v)+epsilon)))*jnp.minimum(1.0,clipsnr * jnp.sqrt(v)/(root_tau_reg(tau, t)*jnp.abs(u)+epsilon))

    g3_momentum_term = lambda v, tau, t: (root_tau_reg(tau, t)/((jnp.sqrt(v)+epsilon)))

    def init_fn(params):

        m = otu.tree_zeros_like(params, dtype=y_dtype)  #First-Momentum
        v = otu.tree_zeros_like(params, dtype=y_dtype)  #Second-Momentum
        tau = otu.tree_zeros_like(params, dtype=y_dtype)  #Tau
        return GalaxyOptimizerState(count=jnp.zeros([], jnp.int32), m=m, v=v, tau=tau)

    def update_fn(updates, state, params):

        new_wd = wd(state.count)
        #newDelta = Delta(state.count)
        #newg1 = g1(state.count)
        new_beta_m = beta_m(state.count)
        new_beta_v = beta_v(state.count)
        new_beta_tau = beta_tau(state.count)

        new_v = jax.tree.map(
            lambda v,u : None if v is None else v*(1-new_beta_v) + new_beta_v*(u**2),
            state.v,
            updates,
            is_leaf=lambda x: x is None,
        )

        new_tau = jax.tree.map(
            lambda tau,u,v : None if tau is None else tau*(1-new_beta_tau) + new_beta_tau*tau_updater(tau, u, v, state.count),
            state.tau,
            updates,
            new_v,
            is_leaf=lambda x: x is None,
        )

        new_m = jax.tree.map(
            lambda m,u : None if m is None else m*(1-new_beta_m) + new_beta_m*u,
            state.m,
            updates,
            is_leaf=lambda x: x is None,
        )

        # This was used in an attempt to stabilize training without clipping.
        #else -1.0*(g2(effective_time(tau, state.count))*u*root_tau_reg(tau, state.count))/(jnp.sqrt(u**2 * tau_reg(tau, state.count)+v)+epsilon)-(g3(effective_time(tau, state.count))*m*g3_momentum_term(u, v, tau, state.count)),
        updates = jax.tree.map(
            lambda m,u,v,tau : -1.0*g2(effective_time(tau, state.count))*u*root_tau_reg(tau, state.count)
            if m is None
            else -1.0*(g2(effective_time(tau, state.count))*u*g2_momentum_term(u, v, tau, state.count))-(g3(effective_time(tau, state.count))*m*g3_momentum_term(v, tau, state.count)),
            #else -1.0*(g2(effective_time(tau, state.count))*u*root_tau_reg(tau, state.count))/(jnp.sqrt(v)+epsilon)-(g3(effective_time(tau, state.count))*m*g3_momentum_term(u, v, tau, state.count)),
            new_m,
            updates,
            new_v,
            new_tau,
            #jnp.maximum(new_tau, 1.0/(1.0+state.count)),
            is_leaf=lambda x: x is None,
        )

        #Apply weight decay
        updates = jax.tree.map(
            lambda u,p : u+(-1.0*new_wd)*p,
            updates,
            params,
            is_leaf=lambda x: x is None,
        )
        new_m = otu.tree_cast(new_m, y_dtype)
        new_v = otu.tree_cast(new_v, y_dtype)
        new_tau = otu.tree_cast(new_tau, y_dtype)
        count_inc = numerics.safe_increment(state.count)

        return updates, GalaxyOptimizerState(count=count_inc, m=new_m, v=new_v, tau=new_tau)

    return base.GradientTransformation(init_fn, update_fn)


class AdEMAMixOptimizerState(NamedTuple):
    """State for the AdEMAMix algorithm."""
    count: chex.Array  # shape=(), dtype=jnp.int32.
    exp_avg_fast: base.Updates  # Fast EMA (beta1)
    exp_avg_slow: base.Updates  # Slow EMA (beta3)
    exp_avg_sq: base.Updates    # Second moment (beta2)


def ademamix_optimizer(
    beta1: base.ScalarOrSchedule = 0.9,
    beta2: base.ScalarOrSchedule = 0.999,
    beta3: base.ScalarOrSchedule = 0.9999,
    alpha: base.ScalarOrSchedule = 2.0,
    gamma_3_factor: base.ScalarOrSchedule = 1.0,
    epsilon: float = 1e-8,
    wd: Optional[base.ScalarOrSchedule] = None,
    *,
    y_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
    """AdEMAMix optimizer (JAX implementation).

    Based on the PyTorch implementation from Apple ML:
    https://github.com/apple/ml-ademamix

    AdEMAMix maintains two exponential moving averages of gradients:
    - Fast EMA with decay rate beta1
    - Slow EMA with decay rate beta3
    The update combines both EMAs with the slow EMA scaled by alpha and gamma_3_factor.

    Args:
        beta1: Decay rate for fast EMA (scalar or schedule, default: 0.9)
        beta2: Decay rate for second moment estimate (scalar or schedule, default: 0.999)
        beta3: Decay rate for slow EMA (scalar or schedule, default: 0.9999)
        alpha: Weight for slow EMA in the update (scalar or schedule, default: 2.0)
        gamma_3_factor: Additional scaling factor for slow EMA (scalar or schedule, default: 1.0)
        epsilon: Small constant for numerical stability (default: 1e-8)
        wd: Optional weight decay schedule or constant
        y_dtype: Optional dtype for momentum accumulators

    Returns:
        A :class:`optax.GradientTransformation` object.
    """
    y_dtype = utils.canonicalize_dtype(y_dtype)

    # Convert scalars to callables
    if wd is None:
        wd = lambda _: 0.0
    elif not callable(wd):
        wd_val = wd
        wd = lambda _: wd_val

    if not callable(beta1):
        beta1_val = beta1
        beta1 = lambda _: beta1_val

    if not callable(beta2):
        beta2_val = beta2
        beta2 = lambda _: beta2_val

    if not callable(beta3):
        beta3_val = beta3
        beta3 = lambda _: beta3_val

    if not callable(alpha):
        alpha_val = alpha
        alpha = lambda _: alpha_val

    if not callable(gamma_3_factor):
        gamma_3_factor_val = gamma_3_factor
        gamma_3_factor = lambda _: gamma_3_factor_val

    def init_fn(params):
        # Initialize fast and slow EMAs, and second moment
        # Check if beta1(0) is 0 for initialization
        init_beta1 = beta1(0)
        exp_avg_fast = otu.tree_zeros_like(params, dtype=y_dtype) if init_beta1 != 0.0 else None
        exp_avg_slow = otu.tree_zeros_like(params, dtype=y_dtype)
        exp_avg_sq = otu.tree_zeros_like(params, dtype=y_dtype)
        return AdEMAMixOptimizerState(
            count=jnp.zeros([], jnp.int32),
            exp_avg_fast=exp_avg_fast,
            exp_avg_slow=exp_avg_slow,
            exp_avg_sq=exp_avg_sq
        )

    def update_fn(updates, state, params):
        new_wd = wd(state.count)
        count = state.count + 1

        # Get current values from schedules
        current_beta1 = beta1(state.count)
        current_beta2 = beta2(state.count)
        current_beta3 = beta3(state.count)
        current_alpha = alpha(state.count)
        current_gamma_3_factor = gamma_3_factor(state.count)

        # Compute bias corrections
        bias_correction1 = 1.0 - current_beta1 ** count
        bias_correction2 = 1.0 - current_beta2 ** count

        # Update fast EMA (if beta1 != 0)
        if state.exp_avg_fast is not None:
            new_exp_avg_fast = jax.tree.map(
                lambda m, u: None if m is None else m * current_beta1 + u * (1.0 - current_beta1),
                state.exp_avg_fast,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            # If beta1 = 0, fast EMA is just the gradient
            new_exp_avg_fast = updates

        # Update slow EMA
        new_exp_avg_slow = jax.tree.map(
            lambda m, u: None if m is None else m * current_beta3 + u * (1.0 - current_beta3),
            state.exp_avg_slow,
            updates,
            is_leaf=lambda x: x is None,
        )

        # Update second moment
        new_exp_avg_sq = jax.tree.map(
            lambda v, u: None if v is None else v * current_beta2 + (u ** 2) * (1.0 - current_beta2),
            state.exp_avg_sq,
            updates,
            is_leaf=lambda x: x is None,
        )

        # Compute denominator (with bias correction for second moment)
        # denom = (sqrt(v) / sqrt(bias_correction2)) + epsilon
        sqrt_bias_correction2 = jnp.sqrt(bias_correction2)

        # Compute update: (m_fast / bias_correction1 + alpha * m_slow * gamma_3_factor) / denom
        if state.exp_avg_fast is not None:
            updates = jax.tree.map(
                lambda m_fast, m_slow, v, p: None if m_fast is None else compute_update(
                    m_fast, m_slow, v, p, bias_correction1, sqrt_bias_correction2, new_wd, current_alpha, current_gamma_3_factor
                ),
                new_exp_avg_fast,
                new_exp_avg_slow,
                new_exp_avg_sq,
                params if params is not None else otu.tree_zeros_like(updates),
                is_leaf=lambda x: x is None,
            )
        else:
            # If beta1 = 0, fast EMA is just the gradient (no bias correction needed)
            updates = jax.tree.map(
                lambda m_fast, m_slow, v, p: None if m_fast is None else compute_update_beta1_zero(
                    m_fast, m_slow, v, p, sqrt_bias_correction2, new_wd, current_alpha, current_gamma_3_factor
                ),
                new_exp_avg_fast,
                new_exp_avg_slow,
                new_exp_avg_sq,
                params if params is not None else otu.tree_zeros_like(updates),
                is_leaf=lambda x: x is None,
            )

        new_exp_avg_fast = otu.tree_cast(new_exp_avg_fast, y_dtype)
        new_exp_avg_slow = otu.tree_cast(new_exp_avg_slow, y_dtype)
        new_exp_avg_sq = otu.tree_cast(new_exp_avg_sq, y_dtype)
        count_inc = numerics.safe_increment(state.count)

        return updates, AdEMAMixOptimizerState(
            count=count_inc,
            exp_avg_fast=new_exp_avg_fast,
            exp_avg_slow=new_exp_avg_slow,
            exp_avg_sq=new_exp_avg_sq
        )

    def compute_update(m_fast, m_slow, v, p, bias_correction1, sqrt_bias_correction2, wd_val, current_alpha, current_gamma_3_factor):
        """Compute update with bias correction for beta1."""
        denom = (jnp.sqrt(v) / sqrt_bias_correction2) + epsilon
        numerator = (m_fast / bias_correction1) + current_alpha * m_slow * current_gamma_3_factor
        update = numerator / denom
        # Add weight decay
        update = update + wd_val * p
        return -update  # Negative for gradient descent

    def compute_update_beta1_zero(m_fast, m_slow, v, p, sqrt_bias_correction2, wd_val, current_alpha, current_gamma_3_factor):
        """Compute update when beta1 = 0 (no bias correction for fast EMA)."""
        denom = (jnp.sqrt(v) / sqrt_bias_correction2) + epsilon
        numerator = m_fast + current_alpha * m_slow * current_gamma_3_factor
        update = numerator / denom
        # Add weight decay
        update = update + wd_val * p
        return -update  # Negative for gradient descent

    return base.GradientTransformation(init_fn, update_fn)


def get_ademamix(
    learning_rate: base.ScalarOrSchedule,
    beta1: base.ScalarOrSchedule = 0.9,
    beta2: base.ScalarOrSchedule = 0.999,
    beta3: base.ScalarOrSchedule = 0.9999,
    alpha: base.ScalarOrSchedule = 2.0,
    gamma_3_factor: base.ScalarOrSchedule = 1.0,
    epsilon: float = 1e-8,
    weight_decay: float = 0.0,
    y_dtype: Optional[chex.ArrayDType] = None
):
    """Get AdEMAMix optimizer with standard hyperparameters.

    Args:
        learning_rate: Outer learning rate schedule or constant.
        beta1: Decay rate for fast EMA (scalar or schedule, default: 0.9)
        beta2: Decay rate for second moment estimate (scalar or schedule, default: 0.999)
        beta3: Decay rate for slow EMA (scalar or schedule, default: 0.9999)
        alpha: Weight for slow EMA in the update (scalar or schedule, default: 2.0)
        gamma_3_factor: Additional scaling factor for slow EMA (scalar or schedule, default: 1.0).
                       This can be used to implement kappa-based scaling.
        epsilon: Small constant for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.0)
        y_dtype: Optional dtype for momentum accumulators.

    Returns:
        Configured optimizer chain.
    """
    optimizer = optax.chain(
        ademamix_optimizer(
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            alpha=alpha,
            gamma_3_factor=gamma_3_factor,
            epsilon=epsilon,
            wd=weight_decay,
            y_dtype=y_dtype
        ),
        optax.scale_by_learning_rate(learning_rate, flip_sign=False)
    )
    return optimizer
