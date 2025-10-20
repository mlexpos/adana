#!/usr/bin/env python3
"""
DANA optimizer on Power-Law Random Features (PLRF) regression.

This script benchmarks the DANA optimizer on synthetic PLRF problems,
comparing it against SGD, AdamW, and other baselines using ODE simulations.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from matplotlib import style
import optax

import optimizers
from power_law_rf import PowerLawRF
import deterministic_equivalent as theory


from typing import NamedTuple, Callable, Union, Optional, Literal
import math


def linear_warmup_scheduler(step, alpha_end, alpha_start=0, warmup=1):
    """Linear warmup for alpha parameter as in AdEMAMix (JAX-compatible)."""
    a = jnp.minimum(step / float(warmup), 1.0)
    return (1.0 - a) * alpha_start + a * alpha_end


def linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):
    """Linear half-life warmup for beta3 parameter as in AdEMAMix (JAX-compatible)."""
    def f(beta, eps=1e-8):
        return jnp.log(0.5) / jnp.log(beta + eps) - 1.0

    def f_inv(t):
        return jnp.power(0.5, 1.0 / (t + 1.0))

    a = jnp.minimum(step / float(warmup), 1.0)
    return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))


class DanaHparams(NamedTuple):
    g1: Callable[[Union[float, jnp.ndarray]], float]  # learning rate function
    g2: Callable[[Union[float, jnp.ndarray]], float]  # learning rate function
    g3: Callable[[Union[float, jnp.ndarray]], float]  # learning rate function
    delta: Callable[[Union[float, jnp.ndarray]], float]  # momentum function


class ODEInputs(NamedTuple):
    eigs_K: jnp.ndarray      # eigenvalues of covariance matrix (W^TDW)
    rho_init: jnp.ndarray    # initial rho_j's (rho_j^2)
    chi_init: jnp.ndarray    # initialization of chi's
    sigma_init: jnp.ndarray  # initialization of sigma's (xi^2_j)
    risk_infinity: float     # risk value at time infinity


def ode_resolvent_log_implicit(
    inputs: ODEInputs,
    opt_hparams: DanaHparams,
    batch: int,
    D: int,
    t_max: float,
    dt: float,
    approximate: bool = False,
    adaptive: Optional[Literal['adam', 'rmsprop_dana']] = None,
):
    """Generate the theoretical solution to momentum.
    Outputs TWICE the risk. Full ODE does NOT use coin-flip momentum.
    Approximate ODE for non-coin-flip and coin-flip are the same after dropping higher-order terms.
    Assumes the loss is AVERAGED over the batch, not summed.

    Parameters
    ----------
    inputs : ODEInputs
        eigs_K : array d
            eigenvalues of covariance matrix (W^TDW)
        rho_init : array d
            initial rho_j's (rho_j^2)
        chi_init : array (d)
            initialization of chi's
        sigma_init : array (d)
            initialization of sigma's (xi^2_j)
        risk_infinity : scalar
            represents the risk value at time infinity (note:
            this is NOT twice the risk)

    opt_hparams : optimizer hyperparameters for Dana
        g1, g2, g3 : function(time)
            learning rate functions
        delta : function(time)
            momentum function

    batch : int
        batch size
    D : int
        number of eigenvalues (i.e. shape of eigs_K)
    t_max : float
        The number of epochs
    dt : float
        time step used in Euler
    approximate : bool
        Whether to use the approximate ODE (drops higher order terms)
    adaptive : Optional[Literal['adam', 'rmsprop_dana']]
        Type of adaptive optimizer normalization:
        - None: no normalization
        - 'adam': normalize g3 terms (momentum entering parameters)
        - 'rmsprop_dana': normalize g1 terms (gradients entering momentum)

    Returns
    -------
    t_grid: numpy.array(float)
        the time steps used, which will discretize (0,t_max) into n_grid points
    twice_risks: numpy.array(float)
        twice the values of the risk, as used in the paper.
    """
    g1_fn, g2_fn, g3_fn, delta_fn = opt_hparams.g1, opt_hparams.g2, opt_hparams.g3, opt_hparams.delta
    eigs_K = inputs.eigs_K
    rho_init, chi_init, sigma_init = inputs.rho_init, inputs.chi_init, inputs.sigma_init
    twice_risk_infinity = 2.0*inputs.risk_infinity
    times = jnp.arange(0, jnp.log(t_max), step=dt, dtype=jnp.float32)
    risk_init = twice_risk_infinity + jnp.sum(inputs.eigs_K * inputs.rho_init)

    def get_normalization_factors(grad_norm):
        """Compute normalization factors based on adaptive optimizer type."""
        if adaptive == 'adam':
            return 1.0, grad_norm
        elif adaptive == 'rmsprop_dana':
            return grad_norm, 1.0
        else:
            return 1.0, 1.0

    def inverse_3x3(omega):
        # Extract matrix elements
        a11, a12, a13 = omega[0][0], omega[0][1], omega[0][2]
        a21, a22, a23 = omega[1][0], omega[1][1], omega[1][2]
        a31, a32, a33 = omega[2][0], omega[2][1], omega[2][2]

        # Calculate determinant
        det = (a11*a22*a33 + a12*a23*a31 + a13*a21*a32
               - a13*a22*a31 - a11*a23*a32 - a12*a21*a33)

        # Calculate each element of inverse matrix
        inv = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        inv[0][0] = (a22*a33 - a23*a32) / det
        inv[0][1] = (a13*a32 - a12*a33) / det
        inv[0][2] = (a12*a23 - a13*a22) / det

        inv[1][0] = (a23*a31 - a21*a33) / det
        inv[1][1] = (a11*a33 - a13*a31) / det
        inv[1][2] = (a13*a21 - a11*a23) / det

        inv[2][0] = (a21*a32 - a22*a31) / det
        inv[2][1] = (a12*a31 - a11*a32) / det
        inv[2][2] = (a11*a22 - a12*a21) / det

        return jnp.array(inv)

    def omega_full(time_plus, grad_norm):
        g1_norm, g3_norm = get_normalization_factors(grad_norm)
        
        # Normalize hyperparameters to apply adaptivity
        g1 = g1_fn(time_plus) / g1_norm
        g2 = g2_fn(time_plus)
        g3 = g3_fn(time_plus) / g3_norm
        delta = delta_fn(time_plus)
        
        # Row 1: Evolution of rho
        omega_11 = -2.0 * (g2 + g1 * g3) * eigs_K + \
                   ((batch + 1.0) / batch) * (g2**2 + 2.0 * g1 * g3 * g2 + g1**2 * g3**2) * eigs_K**2
        omega_12 = g3**2 * (1.0 - delta)**2 * jnp.ones_like(eigs_K)
        omega_13 = -2.0 * g3 * (1.0 - delta) + \
                   2.0 * (g2 * g3 + g3**2 * g1) * (1.0 - delta) * eigs_K
        omega_1 = jnp.array([omega_11, omega_12, omega_13])

        # Row 2: Evolution of sigma
        omega_21 = ((batch + 1.0) / batch) * g1**2 * eigs_K**2
        omega_22 = (-2.0 * delta + delta**2) * jnp.ones_like(eigs_K)
        omega_23 = 2.0 * g1 * eigs_K * (1.0 - delta)
        omega_2 = jnp.array([omega_21, omega_22, omega_23])

        # Row 3: Evolution of chi
        omega_31 = g1 * eigs_K - ((batch + 1.0) / batch) * eigs_K**2 * (g1 * g2 + g1**2 * g3)
        omega_32 = (-g3 + g3 * delta * (2.0 - delta)) * jnp.ones_like(eigs_K)
        omega_33 = -delta - (g2 - g2 * delta + 2.0 * (1.0 - delta) * g1 * g3) * eigs_K
        omega_3 = jnp.array([omega_31, omega_32, omega_33])

        omega = jnp.array([omega_1, omega_2, omega_3])  # 3 x 3 x d
        return omega

    def omega_approximate(time_plus, grad_norm):
        g1_norm, g3_norm = get_normalization_factors(grad_norm)
        
        # Normalize hyperparameters to apply adaptivity
        g1 = g1_fn(time_plus) / g1_norm
        g2 = g2_fn(time_plus)
        g3 = g3_fn(time_plus) / g3_norm
        delta = delta_fn(time_plus)
        
        omega11 = -2.0 * g2 * eigs_K
        omega12 = jnp.zeros_like(eigs_K)
        omega13 = -2.0 * g3 * jnp.ones_like(eigs_K)
        omega1 = jnp.array([omega11, omega12, omega13])

        omega21 = jnp.zeros_like(eigs_K)
        omega22 = -2.0 * delta * jnp.ones_like(eigs_K)
        omega23 = 2.0 * g1 * eigs_K
        omega2 = jnp.array([omega21, omega22, omega23])

        omega31 = g1 * eigs_K
        omega32 = -g3 * jnp.ones_like(eigs_K)
        omega33 = -delta - g2 * eigs_K
        omega3 = jnp.array([omega31, omega32, omega33])

        omega = jnp.array([omega1, omega2, omega3])  # 3 x 3 x d
        return omega

    def forcing_term(time_plus, grad_norm):
        g1_norm, g3_norm = get_normalization_factors(grad_norm)
        
        # Normalize hyperparameters to apply adaptivity
        g1 = g1_fn(time_plus) / g1_norm
        g2 = g2_fn(time_plus)
        g3 = g3_fn(time_plus) / g3_norm
        
        Gamma = jnp.array([
            (g2**2 + 2.0 * g1 * g2 * g3 + g1**2 * g3**2) / batch,
            g1**2 / batch,
            (-g1 * g2 - g1**2 * g3) / batch
        ])
        return jnp.einsum('i,j->ij', Gamma, inputs.eigs_K)  # 3 x d
    
    def forcing_term_approximate(time_plus, grad_norm):
        g1_norm, g3_norm = get_normalization_factors(grad_norm)
        
        # Normalize hyperparameters to apply adaptivity
        g1 = g1_fn(time_plus) / g1_norm
        g2 = g2_fn(time_plus)
        
        Gamma = jnp.array([
            g2**2 / batch,
            g1**2 / batch,
            0.0
        ])
        return jnp.einsum('i,j->ij', Gamma, inputs.eigs_K)  # 3 x d

    def ode_update(carry, time):
        v, twice_risk = carry
        time_plus = jnp.exp(time + dt)
        time_plus_minus_one = time_plus - 1.0
        
        # Use sqrt(twice_risk) as proxy for gradient norm when adaptive is not None
        grad_norm = jnp.sqrt(twice_risk) if adaptive is not None else 1.0
        
        omega = omega_approximate(time_plus_minus_one, grad_norm) if approximate else omega_full(time_plus_minus_one, grad_norm)
        identity = jnp.tensordot(jnp.eye(3), jnp.ones(D), 0)

        A = inverse_3x3(identity - (dt * time_plus) * omega)  # 3 x 3 x d

        z = jnp.einsum('i, j -> ij', jnp.array([1.0, 0.0, 0.0]), eigs_K)

        G_lambda = forcing_term_approximate(time_plus_minus_one, grad_norm) if approximate else forcing_term(time_plus_minus_one, grad_norm)
        x_temp = v + dt * time_plus * twice_risk_infinity * G_lambda

        x = jnp.einsum('ijk, jk -> ik', A, x_temp)

        y = jnp.einsum('ijk, jk -> ik', A, G_lambda)

        v_new = x + (dt * time_plus * y * jnp.sum(x * z) /
                    (1.0 - dt * time_plus * jnp.sum(y * z)))

        twice_risk_new = twice_risk_infinity + jnp.sum(eigs_K * v_new[0])
        return (v_new, twice_risk_new), twice_risk

    init_carry = (jnp.array([rho_init, sigma_init, chi_init]), risk_init)
    _, twice_risks = jax.lax.scan(ode_update, init_carry, times)
    return jnp.exp(times)-1.0, twice_risks

def main():
    """Run DANA on PLRF benchmark using ODE simulations."""
    
    # Set up plotting
    style.use('default')
    plt.rcParams['font.weight'] = 'light'
    plt.rcParams.update({'font.size': 14})
    
    # Experiment parameters
    ALPHA = 1.0                # Power law exponent for eigenvalues
    BETA = 0.5                    # Power law exponent for targets
    V = 4000                         # Hidden dimension
    D = 1000                         # Embedded dimension
    h = 0.0
    BATCH_SIZE = jnp.int32(D**h)     # Batch size
    STEPS = jnp.int32(D**(2*ALPHA))                    # Number of training steps
    DT = 10**(-3)                    # ODE time step
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Initialize random key
    key = random.PRNGKey(42)
    
    alpha = ALPHA
    print(f"\n{'='*60}")
    print(f"Running experiment with alpha={alpha}, beta={BETA}")
    print(f"{'='*60}")
    
    # Initialize problem
    key, subkey = random.split(key)
    problem = PowerLawRF.initialize_random(alpha=alpha, beta=BETA, v=V, d=D, key=subkey)
    
    # Get theory limit
    theory_limit = problem.get_theory_limit_loss()
    print(f"Theory limit loss: {theory_limit:.6e}")
    
    # Learning rate scalar
    #LRscalar = 0.1 / jnp.sqrt(jnp.float32(D))
    LRscalar = 1.0
    
    # Get theory parameters for ODE
    Keigs_theory, rho_weights = theory.theory_rhos(alpha, BETA, D)
    riskInftyTheory = theory_limit
    
    # Initialize ODE inputs
    num_grid_points = jnp.shape(rho_weights)[-1]
    sigma_init = jnp.zeros_like(rho_weights)
    chi_init = jnp.zeros_like(rho_weights)
    
    ode_inputs = ODEInputs(
        Keigs_theory.astype(jnp.float32),
        rho_weights,
        chi_init,
        sigma_init,
        riskInftyTheory
    )
    
    # Set up fixed g1, g2, and delta schedules
    g1 = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
    g2_constant = LRscalar * 0.5 * jnp.minimum(1.0, jnp.float32(BATCH_SIZE) / problem.population_trace)
    g2 = optimizers.powerlaw_schedule(g2_constant, 0.0, 0.0, 1)
    delta_constant = 4.0 + 2*(alpha + BETA)/(2*alpha)
    Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, delta_constant)
    
    print(f"\nDana parameters: g2={g2_constant:.4f}, delta={delta_constant:.4f}")
    
    # Run DANA with constant g3
    print("\nRunning DANA optimizer (constant g3)...")
    g3_constant = 0.5 * g2_constant / D
    g3 = optimizers.powerlaw_schedule(g3_constant, 0.0, 0.0, 1)
    dana_hparams = DanaHparams(g1, g2, g3, Delta)
    
    dana_times, dana_twice_risks = ode_resolvent_log_implicit(
        ode_inputs, dana_hparams, BATCH_SIZE, num_grid_points, STEPS, DT
    )
    dana_losses = 0.5 * dana_twice_risks
    print(f"g2_constant: {g2_constant:.4f}, g3_constant: {g3_constant:.4f}")
    
    # Run SGD (g3=0)
    print("\nRunning SGD optimizer (g3=0)...")
    g3_zero = optimizers.powerlaw_schedule(0.0, 0.0, 0.0, 1)
    sgd_hparams = DanaHparams(g1, g2, g3_zero, Delta)
    
    sgd_times, sgd_twice_risks = ode_resolvent_log_implicit(
        ode_inputs, sgd_hparams, BATCH_SIZE, num_grid_points, STEPS, DT
    )
    sgd_losses = 0.5 * sgd_twice_risks
    
    # Run DANA with decaying g3
    print("\nRunning DANA with decaying g3...")
    g3_constant_decay = 0.5 * g2_constant
    g3_decay_exponent = -1.0 / (2 * alpha)  # Adaptive decay based on alpha
    g3_decay = optimizers.powerlaw_schedule(g3_constant_decay, 0.0, g3_decay_exponent, 1.0)
    dana_decay_hparams = DanaHparams(g1, g2, g3_decay, Delta)
    
    dana_decay_times, dana_decay_twice_risks = ode_resolvent_log_implicit(
        ode_inputs, dana_decay_hparams, BATCH_SIZE, num_grid_points, STEPS, DT
    )
    dana_decay_losses = 0.5 * dana_decay_twice_risks
    
    # Run AdEMAMix variant
    print("\nRunning AdEMAMix-style optimizer with warmup...")
    
    # Parameters for AdEMAMix
    beta1 = 0.9        # Fast EMA coefficient
    T_ademamix = D**(ALPHA)
    print(f"T_ademamix: {T_ademamix}")
    beta3_final = 1.0 - delta_constant/T_ademamix    # Slow EMA coefficient (standard AdEMAMix value)
    print(f"beta3_final: {1-beta3_final}")
    alpha_mix_final = 0.5 * T_ademamix**(1-1/(2*ALPHA))    # Mixing coefficient (like in AdEMAMix paper)
    
    # Warmup parameters
    alpha_warmup = int(T_ademamix)  # 10% of total steps for alpha warmup
    beta3_warmup = int(T_ademamix)  # 20% of total steps for beta3 warmup
    
    # Create schedule functions with warmup for alpha and beta3
    def alpha_schedule(t):
        """Alpha with linear warmup."""
        return linear_warmup_scheduler(t, alpha_mix_final, alpha_start=0, warmup=alpha_warmup)
    
    def beta3_schedule(t):
        """Beta3 with linear half-life warmup."""
        return linear_hl_warmup_scheduler(t, beta3_final, beta_start=beta1, warmup=beta3_warmup)
    # g1: decays like 1/t to match Adam-like behavior for gradient->momentum
    g1_ademamix_constant = 1.0
    g1_ademamix = optimizers.powerlaw_schedule(g1_ademamix_constant, 0.0, -1.0, 1.0)
    
    # g2: standard learning rate (constant or with slight decay)
    g2_ademamix = g2
    
    # g3: grows with t, scaled by time-varying alpha
    # This matches alpha(t) * exp_avg_slow scaling in AdEMAMix
    def g3_ademamix(t):
        """g3 that grows with t and incorporates alpha warmup."""
        return g2_constant * alpha_schedule(t)
    
    # Delta: time-varying momentum decay using beta3 warmup
    # In AdEMAMix: m_t = beta3 * m_{t-1} + (1-beta3) * grad
    # In DANA ODE: delta represents the decay rate
    # So delta should correspond to (1 - beta3)
    def Delta_ademamix(t):
        """Delta with beta3 warmup schedule - returns 1-beta3."""
        return 1.0 - beta3_schedule(t)
    
    ademamix_hparams = DanaHparams(g1_ademamix, g2_ademamix, g3_ademamix, Delta_ademamix)
    
    print(f"  beta1={beta1}, beta3_final={beta3_final:.6f}, alpha_final={alpha_mix_final:.6f}")
    print(f"  alpha_warmup={alpha_warmup} steps, beta3_warmup={beta3_warmup} steps")
    
    # Debug: Test delta values at different times
    test_times = [1.0, 100.0, T_ademamix/2, T_ademamix, STEPS]
    print(f"\nDebug - Delta_ademamix values at different times:")
    for t in test_times:
        beta3_t = beta3_schedule(t)
        delta_t = Delta_ademamix(t)
        print(f"  t={t:.0f}: beta3={beta3_t:.10f}, delta={delta_t:.10f}")
    
    ademamix_times, ademamix_twice_risks = ode_resolvent_log_implicit(
        ode_inputs, ademamix_hparams, BATCH_SIZE, num_grid_points, STEPS, DT
    )
    ademamix_losses = 0.5 * ademamix_twice_risks
    
    # Plot 1: Population Risk
    ax1.loglog(sgd_times + 1, sgd_losses, label='SGD', color='tab:red', linewidth=2.5, alpha=0.8)
    ax1.loglog(dana_times + 1, dana_losses, label='DANA (constant g3)', color='tab:blue', linewidth=2.5, alpha=0.8)
    ax1.loglog(dana_decay_times + 1, dana_decay_losses, label='DANA (decaying g3)', 
              color='tab:green', linewidth=2.5, alpha=0.8, linestyle='--')
    ax1.loglog(ademamix_times + 1, ademamix_losses, label='AdEMAMix-style', 
              color='tab:purple', linewidth=2.5, alpha=0.8, linestyle='-.')
    
    # Add theory limit line
    ax1.axhline(y=theory_limit, color='black', linestyle=':', linewidth=2, 
               label=f'Theory limit ({theory_limit:.2e})')
    
    ax1.set_xlabel('Iterations', fontsize=16)
    ax1.set_ylabel('Population Risk', fontsize=16)
    ax1.set_title(f'Population Risk - α={alpha}, β={BETA}', fontsize=18)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, STEPS)
    
    # Add beta3 and warmup info as text annotation
    beta3_text = f'β₃={beta3_final:.6f}\nβ₃_warmup={beta3_warmup}'
    ax1.text(0.02, 0.02, beta3_text, transform=ax1.transAxes,
            fontsize=11, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Delta schedules (momentum decay rate in DANA ODE)
    # Create time grid for visualization
    time_grid = jnp.logspace(0, jnp.log10(STEPS), 1000)
    
    # Compute delta values for each optimizer
    # In DANA ODE, delta is the momentum decay rate
    sgd_delta_values = jnp.array([Delta(t) for t in time_grid])
    dana_delta_values = jnp.array([Delta(t) for t in time_grid])
    dana_decay_delta_values = jnp.array([Delta(t) for t in time_grid])
    ademamix_delta_values = jnp.array([Delta_ademamix(t) for t in time_grid])
    
    # Plot delta directly (which is the momentum decay rate in DANA)
    ax2.loglog(time_grid, sgd_delta_values, label='SGD', color='tab:red', linewidth=2.5, alpha=0.8)
    ax2.loglog(time_grid, dana_delta_values, label='DANA (constant g3)', color='tab:blue', linewidth=2.5, alpha=0.8)
    ax2.loglog(time_grid, dana_decay_delta_values, label='DANA (decaying g3)', 
              color='tab:green', linewidth=2.5, alpha=0.8, linestyle='--')
    ax2.loglog(time_grid, ademamix_delta_values, label='AdEMAMix-style', 
              color='tab:purple', linewidth=5.5, alpha=0.8, linestyle='-.')
    
    ax2.set_xlabel('Iterations', fontsize=16)
    ax2.set_ylabel('Delta (Momentum Decay Rate)', fontsize=16)
    ax2.set_title(f'Delta Schedules - α={alpha}, β={BETA}', fontsize=18)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, STEPS)
    
    # Add beta3 and warmup info as text annotation
    beta3_text = f'β₃={beta3_final:.6f}\nβ₃_warmup={beta3_warmup}'
    ax2.text(0.02, 0.02, beta3_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    print(f"\nFinal losses:")
    print(f"  SGD:              {sgd_losses[-1]:.6e}")
    print(f"  DANA (constant):  {dana_losses[-1]:.6e}")
    print(f"  DANA (decaying):  {dana_decay_losses[-1]:.6e}")
    print(f"  AdEMAMix-style:   {ademamix_losses[-1]:.6e}")
    print(f"  Theory limit:     {theory_limit:.6e}")
    
    # Add main title
    fig.suptitle(f'DANA & AdEMAMix Optimizers on PLRF (ODE) - D={D}, V={V}, batch={BATCH_SIZE}', 
                 fontsize=20, y=0.98)
    
    # Finalize plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_file = 'dana_plrf_benchmark.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"\n{'='*60}")
    print(f"Figure saved to {output_file}")
    print(f"{'='*60}\n")
    
    # Also save as PDF
    plt.savefig('dana_plrf_benchmark.pdf', bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    main()

