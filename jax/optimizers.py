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


class DanaOptimizerState(NamedTuple):
  """State for the Dana algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  y: base.Updates
  dimensions: base.Updates

def dana_optimizer(
    g1: base.ScalarOrSchedule,
    g2: base.ScalarOrSchedule,
    g3: base.ScalarOrSchedule,
    Delta: base.ScalarOrSchedule,
    *,
    y_dtype: Optional[chex.ArrayDType] = None,
  ) -> base.GradientTransformation:
    """DANA optimizer.

    Args:
        g1: A scalar or schedule determining the first gradient coefficient.
        g2: A scalar or schedule determining the second gradient coefficient.
        g3: A scalar or schedule determining the third gradient coefficient.
        Delta: A scalar or schedule determining the momentum decay rate.
        y_dtype: Optional `dtype` to be used for the momentum accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    y_dtype = utils.canonicalize_dtype(y_dtype)

    def init_fn(params):
        y = otu.tree_zeros_like(params, dtype=y_dtype)  # Momentum
        return DanaOptimizerState(count=jnp.zeros([], jnp.int32), y=y, dimensions=y)

    def update_fn(updates, state, params=None):
        del params
        newDelta = Delta(state.count)
        newg1 = g1(state.count)
        newg2 = g2(state.count)
        newg3 = g3(state.count)

        y = jax.tree.map(
            lambda m,u : None if m is None else m*(1-newDelta) + newg1*u,
            state.y,
            updates,
            is_leaf=lambda x: x is None,
        )
        updates = jax.tree.map(
            lambda m,u : newg2*u if m is None else -1.0*(newg2*u + newg3*m),
            y,
            updates,
            is_leaf=lambda x: x is None,
        )
        y = otu.tree_cast(y, y_dtype)
        count_inc = numerics.safe_increment(state.count)

        return updates, DanaOptimizerState(count=count_inc, y=y, dimensions=state.dimensions)

    return base.GradientTransformation(init_fn, update_fn)

def dana_optimizer_layerwise(
    g1: base.ScalarOrSchedule,
    g2: base.ScalarOrSchedule,
    g3: base.ScalarOrSchedule,
    Delta: base.ScalarOrSchedule,
    *,
    y_dtype: Optional[chex.ArrayDType] = None,
    ) -> base.GradientTransformation:
    """DANA optimizer, with layerwise dimension scaling.

    This differs from the decaying momentum version, in that each layer is scaled by its dimension.
    Args:
        g1: A scalar or schedule determining the first gradient coefficient.
        g2: A scalar or schedule determining the second gradient coefficient.
        g3: A scalar or schedule determining the third gradient coefficient.
        Delta: A scalar or schedule determining the momentum decay rate.
        y_dtype: Optional `dtype` to be used for the momentum accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    y_dtype = utils.canonicalize_dtype(y_dtype)

    def init_fn(params):
        y = otu.tree_zeros_like(params, dtype=y_dtype)  # Momentum
        dimensions = jax.tree.map(lambda x: jnp.float32(len(jnp.reshape(x,-1))), params)
        return DanaOptimizerState(count=jnp.zeros([], jnp.int32), y=y, dimensions=dimensions)

    def update_fn(updates, state, params=None):
        del params
        newDelta = Delta(state.count)
        newg1 = g1(state.count)
        newg2 = g2(state.count)
        newg3 = g3(state.count)
        

        y = jax.tree.map(
            lambda m,u : None if m is None else m*(1-newDelta) + newg1*u,
            state.y,
            updates,
            is_leaf=lambda x: x is None,
        )
        updates = jax.tree.map(
            lambda m,u,d : newg2*u if m is None else -1.0*(newg2*u + newg3*m/d),
            y,
            updates,
            state.dimensions,
            is_leaf=lambda x: x is None,
        )
        y = otu.tree_cast(y, y_dtype)
        count_inc = numerics.safe_increment(state.count)

        return updates, DanaOptimizerState(count=count_inc, y=y, dimensions=state.dimensions)

    return base.GradientTransformation(init_fn, update_fn)


class LongAdamNesterovOptimizerState(NamedTuple):
  """State for the Galaxy algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  m: base.Updates
  v: base.Updates

def get_long_adam_nesterov(
    learning_rate: base.ScalarOrSchedule, #outer learning rate
    beta_m_constant: float = 6.0,
    epsilon: float = 1e-8,
    y_dtype: Optional[chex.ArrayDType] = None
):
    g2=powerlaw_schedule(1.0, 0.0, 0.0, 1.0)

    optimizer = optax.chain(
        long_adam_nesterov_optimizer(g2, beta_m_constant=beta_m_constant, epsilon=epsilon, y_dtype=y_dtype),
        optax.scale_by_learning_rate(learning_rate, flip_sign = False)
    )
    return optimizer

def get_long_adam(
    learning_rate: base.ScalarOrSchedule, #outer learning rate
    beta_m_constant: float = 6.0,
    epsilon: float = 1e-8,
    y_dtype: Optional[chex.ArrayDType] = None
):
    g2=powerlaw_schedule(0.0, 0.0, 0.0, 1.0)

    optimizer = optax.chain(
        long_adam_nesterov_optimizer(g2, beta_m_constant=beta_m_constant, epsilon=epsilon, y_dtype=y_dtype),
        optax.scale_by_learning_rate(learning_rate, flip_sign = False)
    )
    return optimizer


def long_adam_nesterov_optimizer(
    g2: base.ScalarOrSchedule,
    beta_m_constant : float = 6.0,
    epsilon: float = 1e-8,
    wd : Optional[base.ScalarOrSchedule] = None,
    *,
    y_dtype: Optional[chex.ArrayDType] = None,
  ) -> base.GradientTransformation:
    """Long-Adam optimizer.

    Args:
        g2: A scalar or schedule
        beta_m_constant: constant used in the beta_m = beta_m_constant / (1+t), if not choosen then set to 6
        epsilon: Small constant for numerical stability.
        wd: Optional scalar or schedule for weight decay.
        y_dtype: Optional `dtype` to be used for the momentum accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    y_dtype = utils.canonicalize_dtype(y_dtype)
    if wd is None:
        wd = lambda _: 0.0
    elif not callable(wd):
        wd = lambda _: wd

    update_rule = lambda v: (1.0/((jnp.sqrt(v)+epsilon))) 

    #Learning rate schedules
    beta_m=powerlaw_schedule(1.0, 0.0, -1.0, beta_m_constant)
    beta_v = beta_m

    def init_fn(params):
        m = otu.tree_zeros_like(params, dtype=y_dtype)  #First-Momentum
        v = otu.tree_zeros_like(params, dtype=y_dtype)  #Second-Momentum
        return LongAdamNesterovOptimizerState(count=jnp.zeros([], jnp.int32), m=m, v=v)

    def update_fn(updates, state, params):

        new_wd = wd(state.count)
        new_beta_m = beta_m(state.count)
        new_beta_v = beta_v(state.count)

        new_v = jax.tree.map(
            lambda v,u : None if v is None else v*(1-new_beta_v) + new_beta_v*(u**2),
            state.v,
            updates,
            is_leaf=lambda x: x is None,
        )

        new_m = jax.tree.map(
            lambda m,u : None if m is None else m*(1-new_beta_m) + new_beta_m*u,
            state.m,
            updates,
            is_leaf=lambda x: x is None,
        )

        updates = jax.tree.map(
            lambda m,u,v : -1.0*u*g2(state.count)
            if m is None 
            else -1.0*g2(state.count)*u*update_rule(v)-(m*update_rule(v)),
            new_m,
            updates,
            new_v,
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
        count_inc = numerics.safe_increment(state.count)

        return updates, LongAdamNesterovOptimizerState(count=count_inc, m=new_m, v=new_v)

    return base.GradientTransformation(init_fn, update_fn)




class GalaxyOptimizerState(NamedTuple):
  """State for the Galaxy algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  m: base.Updates
  v: base.Updates
  tau: base.Updates

def get_adam_star(
    learning_rate: base.ScalarOrSchedule, #outer learning rate
    epsilon: float = 1e-8,
    y_dtype: Optional[chex.ArrayDType] = None
):
    g2=powerlaw_schedule(0.0, 0.0, 0.0, 1.0)
    g3=powerlaw_schedule(1.0, 0.0, 0.0, 1.0)
    beta_m=powerlaw_schedule(1.0, 0.0, -1.0, 6.0)

    optimizer = optax.chain(
        galaxy_optimizer(g2, g3, beta_m, epsilon=epsilon, y_dtype=y_dtype),
        optax.scale_by_learning_rate(learning_rate, flip_sign = False)
    )
    return optimizer

def get_adam_nesterov_star(
    learning_rate: base.ScalarOrSchedule, #outer learning rate
    epsilon: float = 1e-8,
    y_dtype: Optional[chex.ArrayDType] = None
):
    g2=powerlaw_schedule(1.0, 0.0, 0.0, 1.0)
    g3=powerlaw_schedule(1.0, 0.0, 0.0, 1.0)
    beta_m=powerlaw_schedule(1.0, 0.0, -1.0, 6.0)

    optimizer = optax.chain(
        galaxy_optimizer(g2, g3, beta_m, epsilon=epsilon, y_dtype=y_dtype),
        optax.scale_by_learning_rate(learning_rate, flip_sign = False)
    )
    return optimizer

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
        return TaneaOptimizerState(count=jnp.zeros([], jnp.int32), m=m, v=v, tau=tau)

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

        return updates, TaneaOptimizerState(count=count_inc, m=new_m, v=new_v, tau=new_tau)

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




class TaneaOptimizerState(NamedTuple):
  """State for the Tanea algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  m: base.Updates
  v: base.Updates
  tau: base.Updates

def tanea_optimizer(
    g2: base.ScalarOrSchedule,
    g3: base.ScalarOrSchedule,
    Delta: base.ScalarOrSchedule,
    epsilon: float = 1e-8,
    beta_m : Optional[base.ScalarOrSchedule] = None,
    g1: Optional[base.ScalarOrSchedule] = None,
    beta_v : Optional[base.ScalarOrSchedule] = None,
    magic_tau : float = 1.0,
    wd : Optional[base.ScalarOrSchedule] = None,
    momentum_flavor: str = "effective-clip",
    tau_flavor: str = "second-moment",
    clipsnr: float = 2.0,
    *,
    y_dtype: Optional[chex.ArrayDType] = None,
  ) -> base.GradientTransformation:
    """Tanea optimizer.

    Args:
        g2: A scalar or schedule determining the second gradient coefficient.
        g3: A scalar or schedule determining the third gradient coefficient.
        Delta: A scalar or schedule determining the momentum decay rate.
        epsilon: Small constant for numerical stability.
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
    if beta_m is None:
        beta_m = Delta
    elif not callable(beta_m):
        beta_m = lambda _: beta_m
    if beta_v is None:
        beta_v = Delta
    elif not callable(beta_v):
        beta_v = lambda _: beta_v
    if g1 is None:
        g1 = lambda _: 1.0
    elif not callable(g1):
        g1 = lambda _: g1
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
    quarter_root_tau_reg = lambda tau, t : jnp.power(tau_reg(tau, t),0.25)


    #In an idealized environment, this will lead to tau storing p/(1+p)
    tau_updater = lambda tau,u,v,t : (u**2)/ ( (u**2) + v + epsilon**2)
    if tau_flavor == "second-moment":
        tau_updater = lambda tau,u,v,t : (u**2)/ ( (u**2) + v + epsilon**2)
    elif tau_flavor == "first-moment":
        tau_updater = lambda tau,u,v,t : (jnp.abs(u))/ ( (jnp.abs(u)) + jnp.sqrt(v) + epsilon)
    elif tau_flavor == "second-moment-massive":
        tau_updater = lambda tau,u,v,t : (u**2)*(root_tau_reg(tau,t)*magic_tau) / ( (u**2)*(root_tau_reg(tau, t)*magic_tau) + v + epsilon**2)
    elif tau_flavor == "first-moment-massive":
        tau_updater = lambda tau,u,v,t : jnp.abs(u)*(quarter_root_tau_reg(tau,t)*magic_tau) / ( jnp.abs(u*(quarter_root_tau_reg(tau, t)*magic_tau)) + jnp.sqrt(v) + epsilon)
    else:
        raise ValueError(f"Unknown tau_flavor: {tau_flavor}. Must be 'second-moment', 'first-moment', 'second-moment-massive', or 'first-moment-massive'")


    ## After removing gradient clipping in commit 70fba51, the gpt shows consistent training instabilities, suggesting that some amount of gradient clipping is needed.  The following command is applied post-v-tau updates, but before the m-update.  Clipping to a fixed multiple (4x) of the standard deviation is optimal in contexts where the standard deviation exists.  The 4x in principle should be tuned.
    #gradient_clipper = lambda u,v,tau,t : u/jnp.maximum(1.0,0.125*jnp.abs(u)*jnp.sqrt(tau_reg(tau, t)/(v+epsilon**2)))
    #gradient_clipper = lambda u,v,tau,t : u/jnp.maximum(1.0,0.125*jnp.sqrt(jnp.sum(u*u*tau_reg(tau, t))/jnp.sum(v+epsilon**2)))

    ##This is the default g2_momentum_term.
    g2_momentum_term = lambda u, md, v, tau, t: root_tau_reg(tau, t)/((jnp.sqrt(v)+epsilon))
    #This one was working close to init, but is not actually the theoretically informed one
    #g2_momentum_term = lambda u, md, v, tau, t: (root_tau_reg(tau, t)/((jnp.sqrt(v)+epsilon)))*jnp.minimum(1.0,clipsnr*jnp.abs(md)/(jnp.sqrt(v)*jnp.abs(u)+epsilon))
    #This is the correct schedule.
    g2_momentum_term = lambda u, md, v, tau, t: (root_tau_reg(tau, t)/((jnp.sqrt(v)+epsilon)))*jnp.minimum(1.0,clipsnr * jnp.sqrt(v)/(root_tau_reg(tau, t)*jnp.abs(u)+epsilon))

    ## The g3_momentum_term will be used to multiply the first moment estimator $m$ and the schedule.  The standard Adam scaling would simply output 1/(sqrt(v)+epsilon), times a learning rate, which is here g3(effective_time(tau, t)).  
    ## Now, in the sparse-in-time settig, where updates occur with some probability $p$, we ideally have something like $m = p*E(g)$, where $E(g)$ is some partial expectation of the gradient achieved by time averaging.  This $E(g)$ is a 'DANA-type' momentum estimate.  The $v = p*E(g^2)$ with the same sense of partial expectation.  The $tau$ is an approximation of $p$, and $\tau_reg$ stabilizes the estimate.  
    ## Now we want the momentum term to only be updated at the same speed as when the gradient terms are added.  
    ## To accomplish this, we consider the following instantaneous parameter-update-speed rule:
    ## abs(u * sqrt(tau_reg))/( abs(u * sqrt(tau_reg)) + sqrt(v) + epsilon) 
    ## This is similar to what is used to define tau itself.
    ## We then want to normalize the parameter updates, and so we also introduce
    ## sqrt(v/tau_reg) + epsilon/sqrt(tau_reg)
    ## 1. The "theory" version now takes the product of these factors.
    ## 2. The "effective-clip" version uses a more conservative estimate.  In the theory version we can represent the denominator as (a+b)*a, where a = sqrt(v)+epsilon and b = abs(u)*sqrt(tau_reg).  The 'effective-clip' version replaces this by (a+b)**2, which is always larger and moreover, is substantially larger if $b^2 \gg a^2$.  This can occur in settings where individual gradients have relatively heavy tails, in which case we expect the 'effective-clip' version to be more stable.
    ## 3. The "always-on" version allows momentum updates to always occur.  Since $m$ is effectiely scaled by the time-scale $p$, we expect to update (1/p) times between g2 updates.  Hence in mean this should behave the same way as the 'theory' version, but we expect it to be less stable.  This is the same as what is used for the 'g2' pure gradient term.
    ## 4. The "strong-clip" version is similar to the 'effective-clip'  This is actually a misnomer.  Effective-clip penalizes large 'u' more strongly than strong-clip.
    ## 5. The "mk2" version scales down the momentum term by a factor of sqrt(tau_reg), which accounts for higher noise in the low-probability directions, but is akin to the 'strong-clip' version.
    ## 6. The "mk3" version includes the scaled-down momentum from mk2, but also includes the implied clipping behavior of mk2.  The "mk2" version had uncontrolled loss spikes on nanogpt, which do not appear in the 'effective-clip' version.
    g3_momentum_term = lambda u, v, tau, t, m: abs(u)/((u**2) * tau_reg(tau, t)+v+epsilon**2)
    # Create lambda function for g3 momentum term based on flavor
    if momentum_flavor == "effective-clip":
        g3_momentum_term = lambda u, v, tau, t, m: abs(u)/((u**2) * tau_reg(tau, t)+v+epsilon**2)
    elif momentum_flavor == "theory":
        g3_momentum_term = lambda u, v, tau, t, m: abs(u)/((jnp.abs(u)*root_tau_reg(tau, t)+jnp.sqrt(v)+epsilon) * (jnp.sqrt(v)+epsilon) )
    elif momentum_flavor == "adam":
        g3_momentum_term = lambda u, v, tau, t, m: 1.0/((jnp.sqrt(v)+epsilon))
    elif momentum_flavor == "always-on":
        g3_momentum_term = lambda u, v, tau, t, m: root_tau_reg(tau, t)/((jnp.sqrt(v)+epsilon))
    elif momentum_flavor == "always-on-mk2":
        g3_momentum_term = lambda u, v, tau, t, m: tau_reg(tau, t)/((jnp.sqrt(v)+epsilon))
        #g3_momentum_term = lambda u, v, tau, t, m: jnp.minimum(abs(u),(jnp.sqrt(v/tau_reg(tau, t))))*quarter_root_tau_reg(tau, t)/(v+epsilon**2)
    elif momentum_flavor == "strong-clip":
        g3_momentum_term = lambda u, v, tau, t, m: ( tau_reg(tau, t) )**(1.5) /((jnp.sqrt(v)+epsilon))
    elif momentum_flavor == "mk2":
        g3_momentum_term = lambda u, v, tau, t, m: jnp.minimum(abs(u)*root_tau_reg(tau, t),(jnp.sqrt(v)))/(v+epsilon**2)
    elif momentum_flavor == "mk3":
        g3_momentum_term = lambda u, v, tau, t, m: (abs(u)*root_tau_reg(tau, t))/((u**2) * tau_reg(tau, t)+v+epsilon**2)
    elif momentum_flavor == "mk4":
        g3_momentum_term = lambda u, v, tau, t, m: jnp.abs(m)/(v+epsilon)
        #g3_momentum_term = lambda u, v, tau, t, m: 1.0/(jnp.sqrt(jnp.mean(v)*v)+epsilon)

    else:
        raise ValueError(f"Unknown momentum_flavor: {momentum_flavor}. Must be 'effective-clip', 'theory', 'adam', 'always-on', 'always-on-mk2', 'strong-clip', 'mk2', 'mk3', or 'mk4'")  

    def init_fn(params):

        m = otu.tree_zeros_like(params, dtype=y_dtype)  #First-Momentum
        v = otu.tree_zeros_like(params, dtype=y_dtype)  #Second-Momentum
        tau = otu.tree_zeros_like(params, dtype=y_dtype)  #Tau
        return TaneaOptimizerState(count=jnp.zeros([], jnp.int32), m=m, v=v, tau=tau)

    def update_fn(updates, state, params):

        new_wd = wd(state.count)
        newDelta = Delta(state.count)
        newg1 = g1(state.count)
        new_beta_m = beta_m(state.count)
        new_beta_v = beta_v(state.count)

        new_v = jax.tree.map(
            lambda v,u : None if v is None else v*(1-new_beta_v) + new_beta_v*(u**2),
            state.v,
            updates,
            is_leaf=lambda x: x is None,
        )

        new_tau = jax.tree.map(
            lambda tau,u,v : None if tau is None else tau*(1-newDelta) + newDelta*tau_updater(tau, u, v, state.count),
            state.tau,
            updates,
            new_v,
            is_leaf=lambda x: x is None,
        )
        
        # updates = jax.tree.map(
        #     lambda u,v,tau : u if v is None else gradient_clipper(u,v,tau,state.count),
        #     updates,
        #     new_v,
        #     new_tau,
        #     is_leaf=lambda x: x is None,
        # )

        new_m = jax.tree.map(
            lambda m,u : None if m is None else m*(1-new_beta_m) + newg1*u,
            state.m,
            updates,
            is_leaf=lambda x: x is None,
        )

        # This was used in an attempt to stabilize training without clipping.
        #else -1.0*(g2(effective_time(tau, state.count))*u*root_tau_reg(tau, state.count))/(jnp.sqrt(u**2 * tau_reg(tau, state.count)+v)+epsilon)-(g3(effective_time(tau, state.count))*m*g3_momentum_term(u, v, tau, state.count)),    
        updates = jax.tree.map(
            lambda m,u,v,tau : -1.0*g2(effective_time(tau, state.count))*u 
            if m is None 
            else -1.0*(g2(effective_time(tau, state.count))*u*g2_momentum_term(u, m*new_beta_m, v, tau, state.count))-(g3(effective_time(tau, state.count))*m*g3_momentum_term(u, v, tau, state.count, m)),
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

        return updates, TaneaOptimizerState(count=count_inc, m=new_m, v=new_v, tau=new_tau)

    return base.GradientTransformation(init_fn, update_fn)


class AdamWOptimizerState_withtau(NamedTuple):
  """State for the Tanea algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  m: base.Updates
  v: base.Updates
  tau: base.Updates
  vtau: base.Updates

def adamw_optimizer_withtau(
    Delta: base.ScalarOrSchedule = None,
    epsilon: float = 1e-8,
    lr : Optional[base.ScalarOrSchedule] = None,
    beta_1 : Optional[base.ScalarOrSchedule] = None,
    beta_2 : Optional[base.ScalarOrSchedule] = None,
    wd : Optional[base.ScalarOrSchedule] = None,
    *,
    y_dtype: Optional[chex.ArrayDType] = None,
  ) -> base.GradientTransformation:
    """AdamW optimizer with tau.  Implements AdamW (NO BIAS CORRECTION) and stores tau.

    Args:
        Delta: A scalar or schedule determining the tau decay rate.
        epsilon: Small constant for numerical stability.
        lr: A scalar or schedule determining the learning rate.
        beta_1: A scalar or schedule determining the first momentum decay rate.
        beta_2: A scalar or schedule determining the second momentum decay rate.
        wd: A scalar or schedule determining the weight decay.
        y_dtype: Optional `dtype` to be used for the momentum accumulator; if

    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    y_dtype = utils.canonicalize_dtype(y_dtype)

    if Delta is None:
        Delta = powerlaw_schedule(1.0, 0.0, -1.0, 8.0)
    elif not callable(Delta):
        Delta = lambda _: Delta
    if beta_1 is None:
        beta_1 = lambda _: 0.9
    if beta_2 is None:
        beta_2 = lambda _: 0.95
    if wd is None:
        wd = lambda _: 1e-3
    if lr is None:
        lr = lambda _: 1e-4
    elif not callable(lr):
        lr = lambda _: lr

    #In an idealized environment, this will lead to tau storing p/(1+p)
    tau_updater = lambda tau,u,v,t : (u**2)/ ( (u**2) + v + epsilon**2)

    def init_fn(params):

        m = otu.tree_zeros_like(params, dtype=y_dtype)  #First-Momentum
        v = otu.tree_zeros_like(params, dtype=y_dtype)  #Second-Momentum
        tau = otu.tree_zeros_like(params, dtype=y_dtype)  #Tau
        vtau = otu.tree_zeros_like(params, dtype=y_dtype)  #vtau
        return AdamWOptimizerState_withtau(count=jnp.zeros([], jnp.int32), m=m, v=v, tau=tau, vtau=vtau)

    def update_fn(updates, state, params):

        new_wd = wd(state.count)
        newDelta = Delta(state.count)
        newlr = lr(state.count)
        new_beta_1 = beta_1(state.count)
        new_beta_2 = beta_2(state.count)

        new_v = jax.tree.map(
            lambda v,u : None if v is None else v*(1-new_beta_2) + new_beta_2*(u**2),
            state.v,
            updates,
            is_leaf=lambda x: x is None,
        )
        
        new_vtau = jax.tree.map(
            lambda vtau,u : None if vtau is None else vtau*(1-newDelta) + newDelta*(u**2),
            state.vtau,
            updates,
            is_leaf=lambda x: x is None,
        )

        new_tau = jax.tree.map(
            lambda tau,u,v : None if tau is None else tau*(1-newDelta) + newDelta*tau_updater(tau, u, v, state.count),
            state.tau,
            updates,
            new_vtau,
            is_leaf=lambda x: x is None,
        )

        new_m = jax.tree.map(
            lambda m,u : None if m is None else m*(1-new_beta_1) + new_beta_1*u,
            state.m,
            updates,
            is_leaf=lambda x: x is None,
        )

        updates = jax.tree.map(
            lambda m,v,p : None if m is None else -newlr*(m/(jnp.sqrt(v)+epsilon) + new_wd*p),
            new_m,
            new_v,
            params,
            is_leaf=lambda x: x is None,
        )

        new_m = otu.tree_cast(new_m, y_dtype)
        new_v = otu.tree_cast(new_v, y_dtype)
        new_tau = otu.tree_cast(new_tau, y_dtype)
        new_vtau = otu.tree_cast(new_vtau, y_dtype)
        count_inc = numerics.safe_increment(state.count)

        return updates, AdamWOptimizerState_withtau(count=count_inc, m=new_m, v=new_v, tau=new_tau, vtau=new_vtau)

    return base.GradientTransformation(init_fn, update_fn)


class SparsifierState(NamedTuple):
    """Maintains count for weight decay scheduling."""
    count: chex.Array  # shape=(), dtype=jnp.int32


def subtract_decayed_weight_sparsifier(
    weight_decay: Union[float, jax.Array, base.ScalarOrSchedule] = 0.0,
    #mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """Add parameter scaled by `weight_decay`.

  Args:
    weight_decay: A scalar weight decay rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree, or a
      Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, `True` for leaves/subtrees you want to apply the
      transformation to, and `False` for those you want to skip.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """


  def init_fn(params):
    del params
    if callable(weight_decay):
      return SparsifierState(count=jnp.zeros([], jnp.int32))
    else:
      return base.EmptyState()


  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    s = weight_decay(state.count) if callable(weight_decay) else weight_decay
    updates = jax.tree.map(
        lambda g, p: None if g is None else g - s * jnp.sign(p),
        updates,
        params,
        is_leaf=lambda x: x is None,
    )
    return updates, state

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