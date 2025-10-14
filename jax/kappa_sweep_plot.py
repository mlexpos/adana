#!/usr/bin/env python3
"""
Sweep over kappa values (g3 power law exponent) and plot risk and momentum-squared.
For a fixed alpha, we vary kappa in the range (-1, 0) and record the final risk
and momentum-squared from the ODE solutions.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from matplotlib import style
import sys
import os
import optimizers
from power_law_rf import PowerLawRF
import deterministic_equivalent as theory
from typing import Callable


class DanaHparams:
    """Hyperparameters for DANA optimizer schedules."""
    def __init__(self, g1: Callable, g2: Callable, g3: Callable, delta: Callable):
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.delta = delta


class ODEInputs:
    """Input parameters for ODE solver."""
    def __init__(self, eigs_K: jnp.ndarray, rho_init: jnp.ndarray, chi_init: jnp.ndarray,
                 sigma_init: jnp.ndarray, risk_infinity: float):
        self.eigs_K = eigs_K
        self.rho_init = rho_init
        self.chi_init = chi_init
        self.sigma_init = sigma_init
        self.risk_infinity = risk_infinity


# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================
ALPHA = 0.5                    # Fixed alpha value
BETA = 0.3                     # Fixed beta value
V = 2000                       # Number of features
D = 500                        # Number of parameters
h = 0.0
SGDBATCH = jnp.int32(D**h)
STEPS = 10**6                 # Number of steps
DT = 10**(-3)                 # ODE time step
NUM_KAPPA = 30                # Number of kappa values to test
#KAPPA_MIN = -1.5             # Minimum kappa value
#KAPPA_MAX = -0.2             # Maximum kappa value
KAPPA_MIN = 0.0               # Minimum kappa value
KAPPA_MAX = 1.0               # Maximum kappa value
OUTPUT_FILE = 'kappa_sweep.pdf'
OUTPUT_FILE_RATIO_GRID = 'kappa_sweep_ratio_grid.pdf'

BASE_KAPPA = 1.0             #Schedule looks like t^BASE_KAPPA *gradient-norm-squared^(A) *momentum-norm-squared^(B), clipped to [t^0,t^1]
A=0.0                        #Heuristic, power on twice-risk
B=0.75                        #Heuristic, power on momentum-squared

RISK_RENORMALIZED = True

# Plotting parameters
FIGURE_SIZE = (25, 6.5)
FONT_SIZE = 14
LINE_WIDTH = 2.5


# ============================================================================
# DANASTAR ODE SOLVER
# ============================================================================

def ode_resolvent_log_implicit_and_momentum_dynamic(
    inputs: ODEInputs,
    opt_hparams: DanaHparams,
    batch: int,
    D: int,
    t_max: float,
    dt: float,
    risk_renormalized: bool = RISK_RENORMALIZED,
    g3_mode: str = 'fixed',  # 'fixed' or 'dynamic'
    power_A: float = A,  # Power on twice-risk for dynamic mode
    power_B: float = B,  # Power on momentum-squared for dynamic mode
    base_kappa: float = BASE_KAPPA, # Base kappa value for dynamic mode
):
    """
    DanaStar ODE solver with fixed g3 schedule.
    ODE updates scaled by 1/sqrt(twice_risk).
    """
    g1_fn, g2_fn, g3_fn, delta_fn = opt_hparams.g1, opt_hparams.g2, opt_hparams.g3, opt_hparams.delta
    eigs_K = inputs.eigs_K
    rho_init, chi_init, sigma_init = inputs.rho_init, inputs.chi_init, inputs.sigma_init
    twice_risk_infinity = 2.0 * inputs.risk_infinity
    times = jnp.arange(0, jnp.log(t_max), step=dt, dtype=jnp.float32)
    risk_init = twice_risk_infinity + jnp.sum(inputs.eigs_K * inputs.rho_init)

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

    def omega_full(time_plus, g1, g2, g3, delta):
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

    def forcing_term(time_plus, g1, g2, g3, delta):
        Gamma = jnp.array([
            (g2**2 + 2.0 * g1 * g2 * g3 + g1**2 * g3**2) / batch,
            g1**2 / batch,
            (-g1 * g2 - g1**2 * g3) / batch
        ])
        return jnp.einsum('i,j->ij', Gamma, inputs.eigs_K)  # 3 x d

    def ode_update(carry, time):
        v, twice_risk, momentum_norm = carry
        time_plus = jnp.exp(time + dt)
        time_plus_minus_one = time_plus - 1.0

        # DanaStar scaling: 1/sqrt(twice_risk)
        if risk_renormalized:
            risk_scaling = 1.0 / jnp.sqrt(twice_risk + 1e-10)
        else:
            risk_scaling = 1.0

        # Compute g3 based on mode
        if g3_mode == 'dynamic':
            momentum_norm = jnp.where(momentum_norm == 0.0, 1.0, momentum_norm)
            factor = jnp.power(twice_risk, power_A) * jnp.power(momentum_norm, power_B)
            factor = jnp.minimum(jnp.maximum(factor*jnp.exp(base_kappa*time), 1.0),jnp.exp(time))
            g3 = g3_fn(time_plus_minus_one) * factor
        else:
            g3 = g3_fn(time_plus_minus_one)

        # DanaStar: scale g2,g3 by risk_scaling
        g1 = g1_fn(time_plus)
        g2 = g2_fn(time_plus) * risk_scaling
        delta = delta_fn(time_plus)
        g3 = g3 * risk_scaling

        omega = omega_full(time_plus_minus_one, g1, g2, g3, delta)
        identity = jnp.tensordot(jnp.eye(3), jnp.ones_like(eigs_K), 0)

        scaled_dt = dt * time_plus

        A = inverse_3x3(identity - scaled_dt * omega)  # 3 x 3 x d

        z = jnp.einsum('i, j -> ij', jnp.array([1.0, 0.0, 0.0]), eigs_K)

        G_lambda = forcing_term(time_plus_minus_one, g1, g2, g3, delta)
        # DanaStar: scale update by risk_scaling
        x_temp = v + scaled_dt * twice_risk_infinity * G_lambda

        x = jnp.einsum('ijk, jk -> ik', A, x_temp)

        y = jnp.einsum('ijk, jk -> ik', A, G_lambda)

        # DanaStar: scale update by risk_scaling
        v_new = x + (scaled_dt * y * jnp.sum(x * z) /
                    (1.0 - scaled_dt * jnp.sum(y * z)))

        twice_risk_new = twice_risk_infinity + jnp.sum(eigs_K * v_new[0])
        momentum_norm_new = jnp.sum(v_new[1])

        # Output current (old) state values, compute next state for carry
        return (v_new, twice_risk_new, momentum_norm_new), (twice_risk, jnp.sum(v[1]))

    # Initial state
    v_init = jnp.array([rho_init, sigma_init, chi_init])
    momentum_norm_init = jnp.sum(sigma_init)
    initial_carry = (v_init, risk_init, momentum_norm_init)

    # Run ODE
    _, outputs = jax.lax.scan(ode_update, initial_carry, times)

    twice_risks, momentum_norms = outputs

    return jnp.exp(times) - 1.0, twice_risks, momentum_norms


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    # Access global constants
    global A, B, BASE_KAPPA

    # Set up plotting style
    style.use('default')
    plt.rcParams['font.weight'] = 'light'
    plt.rcParams.update({'font.size': FONT_SIZE})

    # Initialize random key
    key = random.PRNGKey(0)

    print(f"Running kappa sweep with alpha = {ALPHA}...")
    print(f"Testing {NUM_KAPPA} kappa values in range ({KAPPA_MIN}, {KAPPA_MAX})")

    # Initialize problem
    key, subkey = random.split(key)
    problem = PowerLawRF.initialize_random(alpha=ALPHA, beta=BETA, v=V, d=D, key=subkey)

    # Learning rate scalar
    LRscalar = 0.1 / jnp.sqrt(jnp.float32(problem.d))

    # Get theory parameters
    Keigs_theory, rho_weights = theory.theory_rhos(ALPHA, BETA, D)
    riskInftyTheory = problem.get_theory_limit_loss()

    # Initialize ODE inputs
    num_grid_points = jnp.shape(rho_weights)[-1]
    sigma_init = jnp.zeros_like(rho_weights)
    chi_init = jnp.zeros_like(rho_weights)

    ode_inputs_deterministic = ODEInputs(
        Keigs_theory.astype(jnp.float32),
        rho_weights,
        chi_init,
        sigma_init,
        riskInftyTheory
    )

    # Set up fixed g1, g2, and delta schedules
    #g1 = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
    g2 = optimizers.powerlaw_schedule(LRscalar * 0.5 * jnp.minimum(1.0, jnp.float32(SGDBATCH) / problem.population_trace), 0.0, 0.0, 1)
    delta_constant = 4.0 + 2*(ALPHA + BETA)/(2*ALPHA)
    Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, delta_constant)
    g1=Delta

    # Generate kappa values (excluding endpoints)
    kappa_values = jnp.linspace(KAPPA_MIN, KAPPA_MAX, NUM_KAPPA, endpoint=False)

    # Storage for results
    all_times = []
    all_risks = []
    all_momentum_squared = []

    # Run ODE for each kappa value
    for i, kappa in enumerate(kappa_values):
        print(f"\n  [{i+1}/{NUM_KAPPA}] Running kappa = {kappa:.4f}...")


        #g3constant = 1.0
        g3constant = LRscalar * 0.5 * jnp.minimum(1.0, jnp.float32(SGDBATCH) / problem.population_trace)
        # Create g3 schedule with current kappa
        g3 = optimizers.powerlaw_schedule(
            g3constant,
            0.0,
            kappa,  # This is the exponent we're varying
            1
        )

        # Set up optimizer
        dana_hparams = DanaHparams(g1, g2, g3, Delta)

        # Run ODE
        times, twice_risks, momentum_norms = ode_resolvent_log_implicit_and_momentum_dynamic(
            ode_inputs_deterministic,
            dana_hparams,
            SGDBATCH,
            num_grid_points,
            STEPS,
            DT,
            g3_mode='fixed'
        )

        # Convert to risk (from twice_risk)
        risks = 0.5 * twice_risks

        # Store full curves
        all_times.append(times)
        all_risks.append(risks)
        all_momentum_squared.append(momentum_norms)

        print(f"      Final risk: {risks[-1]:.6e}, Final ||y||²: {momentum_norms[-1]:.6e}")

    # Run dynamic g3 schedule
    print(f"\n  Running dynamic g3 schedule...")
    #g3_dynamic = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
    g3_dynamic = optimizers.powerlaw_schedule(g3constant, 0.0, 0.0, 1)
    dana_hparams_dynamic = DanaHparams(g1, g2, g3_dynamic, Delta)
    times_dynamic, twice_risks_dynamic, momentum_norms_dynamic = ode_resolvent_log_implicit_and_momentum_dynamic(
        ode_inputs_deterministic,
        dana_hparams_dynamic,
        SGDBATCH,
        num_grid_points,
        STEPS,
        DT,
        g3_mode='dynamic'
    )
    risks_dynamic = 0.5 * twice_risks_dynamic
    print(f"      Final risk: {risks_dynamic[-1]:.6e}, Final ||y||²: {momentum_norms_dynamic[-1]:.6e}")

    # Extract values at specific iteration counts
    # Generate checkpoints: 10^k for k=1,2,3,...,N where 10^N <= STEPS
    import numpy as np
    N = int(np.floor(np.log10(STEPS)))
    iteration_checkpoints = [10**k for k in range(1, N+1)]
    num_checkpoints = len(iteration_checkpoints)

    print(f"\nGenerating {num_checkpoints} checkpoint plots at iterations: {iteration_checkpoints}")

    # For each checkpoint, find the closest time index and extract risk/momentum values
    checkpoint_risks = {T: [] for T in iteration_checkpoints}
    checkpoint_momentum = {T: [] for T in iteration_checkpoints}
    # Also extract from dynamic run for heuristic calculation
    checkpoint_risks_dynamic = {}
    checkpoint_momentum_dynamic = {}

    for times, risks, momentum in zip(all_times, all_risks, all_momentum_squared):
        for T in iteration_checkpoints:
            # Find closest time point
            idx = jnp.argmin(jnp.abs(times - T))
            checkpoint_risks[T].append(risks[idx])
            checkpoint_momentum[T].append(momentum[idx])

    # Extract from dynamic run
    for T in iteration_checkpoints:
        idx = jnp.argmin(jnp.abs(times_dynamic - T))
        checkpoint_risks_dynamic[T] = risks_dynamic[idx]
        checkpoint_momentum_dynamic[T] = momentum_norms_dynamic[idx]

    # Convert to arrays
    for T in iteration_checkpoints:
        checkpoint_risks[T] = jnp.array(checkpoint_risks[T])
        checkpoint_momentum[T] = jnp.array(checkpoint_momentum[T])

    # Create plots - 2 curve plots + num_checkpoints checkpoint plots
    total_plots = 2 + num_checkpoints
    fig_width = 5 * total_plots  # Scale width based on number of plots
    fig = plt.figure(figsize=(fig_width, 6.5))

    # Create subplots with custom positioning to make room for colorbar
    # [left, bottom, width, height] for each subplot
    plot_width = 0.8 / total_plots  # Distribute 80% of width among plots
    spacing = 0.02
    left_margin = 0.05

    # First two plots (curves)
    ax1 = fig.add_axes([left_margin, 0.25, plot_width, 0.65])
    ax2 = fig.add_axes([left_margin + plot_width + spacing, 0.25, plot_width, 0.65])

    # Checkpoint plots
    checkpoint_axes = []
    for i in range(num_checkpoints):
        left_pos = left_margin + 2*(plot_width + spacing) + i*(plot_width + spacing)
        ax = fig.add_axes([left_pos, 0.25, plot_width, 0.65])
        checkpoint_axes.append(ax)

    # Use a colormap for different kappa values
    cmap = plt.cm.viridis
    colors = [cmap(i / (NUM_KAPPA - 1)) for i in range(NUM_KAPPA)]

    # Plot 1: Risk curves (no legend, will add colorbar)
    for i, (times, risks, kappa) in enumerate(zip(all_times, all_risks, kappa_values)):
        ax1.loglog(times, risks, linewidth=LINE_WIDTH, color=colors[i], alpha=0.8)
    # Add dynamic curve in red
    ax1.loglog(times_dynamic, risks_dynamic, linewidth=LINE_WIDTH + 0.5, color='red',
               alpha=0.9, linestyle='--', label='Dynamic g3')
    ax1.set_xlabel('Iterations', fontsize=16)
    ax1.set_ylabel('Risk', fontsize=16)
    ax1.set_title('Risk Curves', fontsize=18)
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, None)

    # Plot 2: Momentum-squared curves (no legend, will add colorbar)
    for i, (times, momentum_sq, kappa) in enumerate(zip(all_times, all_momentum_squared, kappa_values)):
        ax2.loglog(times, momentum_sq, linewidth=LINE_WIDTH, color=colors[i], alpha=0.8)
    # Add dynamic curve in red
    ax2.loglog(times_dynamic, momentum_norms_dynamic, linewidth=LINE_WIDTH + 0.5, color='red',
               alpha=0.9, linestyle='--', label='Dynamic g3')
    ax2.set_xlabel('Iterations', fontsize=16)
    ax2.set_ylabel('||y||²', fontsize=16)
    ax2.set_title('Momentum² Curves', fontsize=18)
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, None)

    # Add colorbar for the curve plots below them
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=KAPPA_MIN, vmax=KAPPA_MAX)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Create an axis for the colorbar below the first two plots
    cbar_ax = fig.add_axes([0.08, 0.10, 0.32, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('κ (g3 exponent)', fontsize=14)

    # Plot checkpoints at T = 10^k for k=1,2,...,N
    for idx, (ax, T) in enumerate(zip(checkpoint_axes, iteration_checkpoints)):
        ax_twin = ax.twinx()

        # Compute log(risk) / log(T) and log(momentum) / log(T)
        risk_ratio = jnp.log(checkpoint_risks[T]) / jnp.log(T)
        momentum_ratio = jnp.log(checkpoint_momentum[T]) / jnp.log(T)

        # Compute difference ratio: (log(risk) - log(||y||²)) / log(T)
        log_risk = jnp.log(checkpoint_risks[T])
        log_momentum = jnp.log(checkpoint_momentum[T])
        diff_ratio = (A*log_risk + B*log_momentum) / jnp.log(T) + BASE_KAPPA

        # Find kappa that minimizes risk
        min_risk_idx = jnp.argmin(checkpoint_risks[T])
        optimal_kappa = kappa_values[min_risk_idx]

        # Add y=x line in dotted black
        ax.plot(kappa_values, kappa_values, 'k:', linewidth=1.5, label='y=x', zorder=0)

        # Add vertical line at optimal kappa (blue like risk curve)
        ax.axvline(x=optimal_kappa, color='blue', linestyle='-', linewidth=2,
                   label=f'Min Risk κ={optimal_kappa:.3f}', zorder=5)

        # Compute effective kappa from dynamic algorithm at this checkpoint
        # effective_kappa = (A*log(twice_risk) + B*log(momentum_norm²))/log(T)
        twice_risk_dynamic = checkpoint_risks_dynamic[T] * 2.0
        momentum_norm_sq_dynamic = checkpoint_momentum_dynamic[T]
        effective_kappa = (A * jnp.log(twice_risk_dynamic) + B * jnp.log(momentum_norm_sq_dynamic)) / jnp.log(T)
        effective_kappa = jnp.minimum(jnp.maximum(effective_kappa+BASE_KAPPA, 0), 1.0)

        # Add effective kappa vertical line (red, dashed)
        ax.axvline(x=effective_kappa, color='red', linestyle='--', linewidth=2,
                   label=f'Effective κ={effective_kappa:.3f}', zorder=5)

        line1 = ax.plot(kappa_values, risk_ratio, 'o-', linewidth=LINE_WIDTH,
                        markersize=8, color='tab:blue', label='log(Risk)/log(T)')
        ax.set_xlabel('κ', fontsize=16)
        ax.set_ylabel('log(Risk)/log(T)', fontsize=14, color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.grid(True, alpha=0.3)

        line2 = ax_twin.plot(kappa_values, momentum_ratio, 's-', linewidth=LINE_WIDTH,
                            markersize=8, color='tab:orange', label='log(||y||²)/log(T)')
        ax_twin.set_ylabel('log(||y||²)/log(T)', fontsize=14, color='tab:orange')
        ax_twin.tick_params(axis='y', labelcolor='tab:orange')

        # Add the difference ratio on the main axis
        line3 = ax.plot(kappa_values, diff_ratio, 'D-', linewidth=LINE_WIDTH,
                       markersize=6, color='tab:green', label=f'{A:.2f}·log(R)+{B:.2f}·log(||y||²))/log(T)', alpha=0.7)

        ax.set_title(f'T = {T:.0e}', fontsize=18)

        # Add legend only to first checkpoint
        if idx == 0:
            # Include both vertical lines in legend (min risk and effective kappa)
            vline_min_risk = ax.get_lines()[1]  # The min risk vertical line (blue)
            vline_effective = ax.get_lines()[2]  # The effective kappa vertical line (red, dashed)
            lines = line1 + line3 + line2 + [vline_min_risk, vline_effective]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, fontsize=10, loc='upper left')

    # Finalize and save
    fig.suptitle(f'DanaStar: Kappa Sweep (α={ALPHA}, β={BETA}, D={D}, batch={int(SGDBATCH)}, {NUM_KAPPA} values)',
                 fontsize=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE)
    print(f"\n{'='*60}")
    print(f"Figure saved to {OUTPUT_FILE}")
    print(f"{'='*60}")

    # ========================================================================
    # Create third figure with grid of (A*log(risk) + B*log(||y||²))/log(T)
    # ========================================================================

    import numpy as np
    A_values = np.arange(0.0, 1.25, 0.25)
    B_values = np.arange(0.0, 1.25, 0.25)
    num_A = len(A_values)
    num_B = len(B_values)

    # Create grid of subplots for each checkpoint
    for T in iteration_checkpoints:
        fig3 = plt.figure(figsize=(num_B * 4, num_A * 3.5))

        for i, A_val in enumerate(A_values):
            for j, B_val in enumerate(B_values):
                ax = plt.subplot(num_A, num_B, i * num_B + j + 1)

                # Compute (A*log(risk) + B*log(||y||²)) / log(T)
                log_risk = jnp.log(checkpoint_risks[T])
                log_momentum = jnp.log(checkpoint_momentum[T])
                ratio = (A_val * log_risk + B_val * log_momentum) / jnp.log(T) + BASE_KAPPA

                # Find kappa that minimizes risk
                min_risk_idx = jnp.argmin(checkpoint_risks[T])
                optimal_kappa = kappa_values[min_risk_idx]

                # Add y=x line in dotted black
                ax.plot(kappa_values, kappa_values, 'k:', linewidth=1.5, zorder=0)

                # Add vertical line at optimal kappa (blue like risk curve)
                ax.axvline(x=optimal_kappa, color='blue', linestyle='-', linewidth=1.5, zorder=5)

                # Plot the ratio
                ax.plot(kappa_values, ratio, 'o-', linewidth=2,
                        markersize=6, color='tab:purple')

                ax.set_xlabel('κ', fontsize=12)
                ax.set_ylabel(f'({A_val:.2f}·log(R) - {B_val:.2f}·log(||y||²))/log(T)', fontsize=10)
                ax.set_title(f'A={A_val:.2f}, B={B_val:.2f}', fontsize=12)
                ax.grid(True, alpha=0.3)

        fig3.suptitle(f'DanaStar: Weighted Ratio Grid at T={T:.0e} (α={ALPHA}, β={BETA}, D={D}, batch={int(SGDBATCH)})',
                      fontsize=18)
        fig3.tight_layout()

        # Save with checkpoint-specific filename
        output_file_grid = OUTPUT_FILE_RATIO_GRID.replace('.pdf', f'_T{T:.0e}.pdf')
        fig3.savefig(output_file_grid)
        print(f"\nRatio grid figure for T={T:.0e} saved to {output_file_grid}")

    print(f"{'='*60}")

    # ========================================================================
    # Create fourth figure with dynamic schedule diagnostics
    # ========================================================================

    print(f"\nCreating dynamic schedule diagnostic plots...")

    fig4 = plt.figure(figsize=(25, 5))

    # Create 5 subplots
    ax1 = plt.subplot(1, 5, 1)
    ax2 = plt.subplot(1, 5, 2)
    ax3 = plt.subplot(1, 5, 3)
    ax4 = plt.subplot(1, 5, 4)
    ax5 = plt.subplot(1, 5, 5)

    # Compute diagnostics for dynamic schedule
    # Normalized update factor: sqrt(momentum_norm²) / sqrt(twice_risk)
    normalized_update_factor = jnp.sqrt(momentum_norms_dynamic) / jnp.sqrt(twice_risks_dynamic + 1e-10)

    # Pre-sign magnitude: T^{BASE_KAPPA} * (twice_risk)^{A} * (momentum_norm²)^{B} * normalized_update_factor
    # This is the factor before clipping to [1, T]
    T_array = times_dynamic + 1.0  # times are (T-1), so add 1
    pre_sign_magnitude = (
        jnp.power(T_array, BASE_KAPPA) *
        jnp.power(twice_risks_dynamic + 1e-10, A) *
        jnp.power(momentum_norms_dynamic + 1e-10, B) *
        normalized_update_factor
    )

    # New diagnostic: T * (twice_risk)^{A} * (momentum_norm²)^{B}
    combined_diagnostic = (
        T_array *
        jnp.power(twice_risks_dynamic + 1e-10, A) *
        jnp.power(momentum_norms_dynamic + 1e-10, B)
    )

    # Define fitting range: 10^2 to 10^4
    fit_start = 100.0
    fit_end = 10000.0
    fit_mask = (times_dynamic >= fit_start) & (times_dynamic <= fit_end)
    times_fit = times_dynamic[fit_mask]

    # Helper function to perform log-log linear fit and return fitted line
    def loglog_fit(times, values, times_fit_range):
        """Fit log(y) = a*log(x) + b and return fitted values."""
        mask = (times >= fit_start) & (times <= fit_end)
        log_times = jnp.log(times[mask])
        log_values = jnp.log(values[mask] + 1e-10)

        # Linear regression in log-log space
        coeffs = jnp.polyfit(log_times, log_values, 1)
        slope, intercept = coeffs[0], coeffs[1]

        # Compute fitted values for the full time range
        fitted_log = slope * jnp.log(times_fit_range) + intercept
        fitted_values = jnp.exp(fitted_log)

        return fitted_values, slope

    # Plot 1: Twice-risk (log-log)
    ax1.loglog(times_dynamic, twice_risks_dynamic, linewidth=LINE_WIDTH, color='tab:blue',
               label='2×Risk')
    # Add linear fit
    fit_twice_risk, slope_risk = loglog_fit(times_dynamic, twice_risks_dynamic, times_fit)
    ax1.loglog(times_fit, fit_twice_risk, '--', linewidth=LINE_WIDTH-0.5, color='red',
               label=f'Fit: T^{{{slope_risk:.3f}}} ({fit_start:.0f}-{fit_end:.0f})', alpha=0.8)
    ax1.set_xlabel('Iterations', fontsize=14)
    ax1.set_ylabel('2×Risk', fontsize=14)
    ax1.set_title('Dynamic: Twice-Risk', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')
    ax1.set_xlim(1, None)

    # Plot 2: Momentum norm squared (log-log)
    ax2.loglog(times_dynamic, momentum_norms_dynamic, linewidth=LINE_WIDTH, color='tab:orange',
               label='||y||²')
    # Add linear fit
    fit_momentum, slope_momentum = loglog_fit(times_dynamic, momentum_norms_dynamic, times_fit)
    ax2.loglog(times_fit, fit_momentum, '--', linewidth=LINE_WIDTH-0.5, color='red',
               label=f'Fit: T^{{{slope_momentum:.3f}}} ({fit_start:.0f}-{fit_end:.0f})', alpha=0.8)
    ax2.set_xlabel('Iterations', fontsize=14)
    ax2.set_ylabel('||y||²', fontsize=14)
    ax2.set_title('Dynamic: Momentum² Norm', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')
    ax2.set_xlim(1, None)

    # Plot 3: Normalized update factor (log-log)
    ax3.loglog(times_dynamic, normalized_update_factor, linewidth=LINE_WIDTH, color='tab:green',
               label='√||y||² / √(2R)')
    # Add linear fit
    fit_normalized, slope_normalized = loglog_fit(times_dynamic, normalized_update_factor, times_fit)
    ax3.loglog(times_fit, fit_normalized, '--', linewidth=LINE_WIDTH-0.5, color='red',
               label=f'Fit: T^{{{slope_normalized:.3f}}} ({fit_start:.0f}-{fit_end:.0f})', alpha=0.8)
    ax3.set_xlabel('Iterations', fontsize=14)
    ax3.set_ylabel('√||y||² / √(2R)', fontsize=14)
    ax3.set_title('Dynamic: Normalized Update Factor', fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, loc='best')
    ax3.set_xlim(1, None)

    # Plot 4: Pre-sign magnitude (log-log)
    ax4.loglog(times_dynamic, pre_sign_magnitude, linewidth=LINE_WIDTH, color='tab:red',
               label='Pre-clip factor')
    # Add linear fit
    fit_pre_sign, slope_pre_sign = loglog_fit(times_dynamic, pre_sign_magnitude, times_fit)
    ax4.loglog(times_fit, fit_pre_sign, '--', linewidth=LINE_WIDTH-0.5, color='purple',
               label=f'Fit: T^{{{slope_pre_sign:.3f}}} ({fit_start:.0f}-{fit_end:.0f})', alpha=0.8)
    # Add reference lines: upper clip (1) and lower clip (normalized update factor)
    ax4.loglog(times_dynamic, jnp.ones_like(times_dynamic), 'k--', linewidth=1.5,
               label='Upper clip (1)', alpha=0.7)
    ax4.loglog(times_dynamic, normalized_update_factor, 'k:', linewidth=1.5,
               label='Lower clip (√||y||²/√2R)', alpha=0.7)
    ax4.set_xlabel('Iterations', fontsize=14)
    ax4.set_ylabel('Factor Value', fontsize=14)
    ax4.set_title(f'Dynamic: T^{{{BASE_KAPPA:.2f}}}·(2R)^{{{A:.2f}}}·||y||²^{{{B:.2f}}}·(√||y||²/√2R)', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9, loc='best')
    ax4.set_xlim(1, None)

    # Plot 5: Combined diagnostic T * (twice_risk)^A * (momentum_norm²)^B (log-log)
    ax5.loglog(times_dynamic, combined_diagnostic, linewidth=LINE_WIDTH, color='tab:purple',
               label='T·(2R)^A·||y||²^B')
    # Add linear fit
    fit_combined, slope_combined = loglog_fit(times_dynamic, combined_diagnostic, times_fit)
    ax5.loglog(times_fit, fit_combined, '--', linewidth=LINE_WIDTH-0.5, color='red',
               label=f'Fit: T^{{{slope_combined:.3f}}} ({fit_start:.0f}-{fit_end:.0f})', alpha=0.8)
    ax5.set_xlabel('Iterations', fontsize=14)
    ax5.set_ylabel('T·(2R)^A·||y||²^B', fontsize=14)
    ax5.set_title(f'Dynamic: T·(2R)^{{{A:.2f}}}·||y||²^{{{B:.2f}}}', fontsize=16)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10, loc='best')
    ax5.set_xlim(1, None)

    fig4.suptitle(f'DanaStar Dynamic Schedule Diagnostics (α={ALPHA}, β={BETA}, D={D}, A={A}, B={B}, κ_base={BASE_KAPPA})',
                  fontsize=18)
    fig4.tight_layout()

    output_file_dynamic = 'kappa_sweep_dynamic_diagnostics.pdf'
    fig4.savefig(output_file_dynamic)
    print(f"Dynamic diagnostics figure saved to {output_file_dynamic}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
