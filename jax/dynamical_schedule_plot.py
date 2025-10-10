#!/usr/bin/env python3
"""
Compare fixed g3 schedules with dynamical g3 schedule.
Dynamical g3 is computed as: g3 = g2 * min(risk/||y||^2, 1)
"""

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.stats import linregress
from typing import NamedTuple, Callable, Union, Optional, Literal

import sys
import os
from power_law_rf import optimizers
from power_law_rf.power_law_rf import PowerLawRF
from power_law_rf.ode import DanaHparams, ODEInputs
import power_law_rf.deterministic_equivalent as theory
from power_law_rf.least_squares import lsq_streaming_optax_simple

# ============================================================================
# GLOBAL PARAMETERS (modify these for different experiments)
# ============================================================================
ALPHA_VALUES = [0.3, 0.7, 1.0, 1.3]  # List of alpha values to test
BETA = 0.7                            # Fixed beta value
V = 4000                             # Number of features
D = 1000                              # Number of parameters
           # Batch size
h = 0.0 #jnp.float32(SGDBATCH / D)
SGDBATCH = jnp.int32(D**h)               
STEPS = 10**7                    # Number of steps
DT = 10**(-3)                         # ODE time step
OUTPUT_FILE_RISK = 'dynamical_schedule_risk.pdf'           # Output for risk plots
OUTPUT_FILE_RATIO = 'dynamical_schedule_risk_ratio.pdf'    # Output for risk/||y||^2 plots


# Plotting parameters
FIGURE_SIZE = (20, 12)
FONT_SIZE = 16
LINE_WIDTH = 2.5

# ============================================================================
# DYNAMICAL ODE SOLVER
# ============================================================================

def ode_resolvent_log_implicit_and_momentum_dynamic(
    inputs: ODEInputs,
    opt_hparams: DanaHparams,
    batch: int,
    D: int,
    t_max: float,
    dt: float,
    g3_mode: str = 'fixed',  # 'fixed' or 'dynamic'
):
    """
    ODE solver with optional dynamic g3 schedule.
    When g3_mode='dynamic', g3 is computed as: g3 = g2 * min(risk/||y||^2, 1)
    """
    g1_fn, g2_fn, g3_fn_fixed, delta_fn = opt_hparams.g1, opt_hparams.g2, opt_hparams.g3, opt_hparams.delta
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

    def omega_full(time_plus, g3):
        g1 = g1_fn(time_plus)
        g2 = g2_fn(time_plus)
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

    def forcing_term(time_plus, g3):
        g1 = g1_fn(time_plus)
        g2 = g2_fn(time_plus)

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

        # Compute g3 based on mode
        if g3_mode == 'dynamic':
            # g3 = g2 * min(risk/||y||^2, 1)
            risk_ratio = twice_risk / ( 2.0 * momentum_norm + 1e-10)  # Add small epsilon for numerical stability
            g3 =  g3_fn_fixed(time_plus_minus_one) * jnp.minimum( risk_ratio, 1.0) 
        else:
            g3 = g3_fn_fixed(time_plus_minus_one)

        omega = omega_full(time_plus_minus_one, g3)
        identity = jnp.tensordot(jnp.eye(3), jnp.ones_like(eigs_K), 0)

        A = inverse_3x3(identity - (dt * time_plus) * omega)  # 3 x 3 x d

        z = jnp.einsum('i, j -> ij', jnp.array([1.0, 0.0, 0.0]), eigs_K)

        G_lambda = forcing_term(time_plus_minus_one, g3)
        x_temp = v + dt * time_plus * twice_risk_infinity * G_lambda

        x = jnp.einsum('ijk, jk -> ik', A, x_temp)

        y = jnp.einsum('ijk, jk -> ik', A, G_lambda)

        v_new = x + (dt * time_plus * y * jnp.sum(x * z) /
                    (1.0 - dt * time_plus * jnp.sum(y * z)))

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
    # Set up plotting style
    style.use('default')
    plt.rcParams['font.weight'] = 'light'
    plt.rcParams.update({'font.size': FONT_SIZE})

    # Initialize random key
    key = random.PRNGKey(0)

    # Create figures with subplots for each alpha
    fig_risk, axes_risk = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    fig_ratio, axes_ratio = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    axes_risk = axes_risk.flatten()
    axes_ratio = axes_ratio.flatten()

    # Process each alpha value
    for idx, alpha in enumerate(ALPHA_VALUES):
        print(f"\nProcessing alpha = {alpha}...")

        # Initialize problem
        key, subkey = random.split(key)
        problem = PowerLawRF.initialize_random(alpha=alpha, beta=BETA, v=V, d=D, key=subkey)

        # Set up DANA optimizer schedules
        g1 = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
        g2 = optimizers.powerlaw_schedule(0.5 * jnp.minimum(1.0, jnp.float32(SGDBATCH) / problem.population_trace ), 0.0, 0.0, 1)

        # Two fixed g3 schedules
        g3_fixed1 = optimizers.powerlaw_schedule(0.1 * jnp.float32(SGDBATCH) / problem.d * jnp.minimum(1.0, jnp.float32(SGDBATCH) / problem.population_trace ), 0.0, 0.0, 1)
        g3_fixed2 = optimizers.powerlaw_schedule(0.1 * jnp.minimum(1.0, jnp.float32(SGDBATCH) / problem.population_trace ), 0.0, -(1.0-h)/(2*alpha), 1)
        g3_fixed3 = optimizers.powerlaw_schedule(0.1 * jnp.minimum(1.0, jnp.float32(SGDBATCH) / problem.population_trace ), 0.0, 0.0, 1)
        g3_sgd = optimizers.powerlaw_schedule(0.0, 0.0, 0.0, 1)

        delta_constant = 4.0 + 2*(alpha + BETA)/(2*alpha)
        Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, delta_constant)

        # Get theory parameters
        Keigs_theory, rho_weights = theory.theory_rhos(alpha, BETA, D)
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

        # Run ODE with fixed g3 (version 1: exponent -1.0)
        print("  Running fixed g3 (exp=-1.0)...")
        dana_hparams_fixed1 = DanaHparams(g1, g2, g3_fixed1, Delta)
        times_fixed1, risks_fixed1, momentum_fixed1 = ode_resolvent_log_implicit_and_momentum_dynamic(
            ode_inputs_deterministic,
            dana_hparams_fixed1,
            SGDBATCH,
            num_grid_points,
            STEPS,
            DT,
            g3_mode='fixed'
        )
        risks_fixed1 = 0.5 * risks_fixed1

        # #Runs the stochastic version of dana-constant
        # danaconstantopt = optimizers.dana_optimizer(g1,g2,g3_fixed1,Delta)
        # key, newkey = random.split(key)
        # danaconstanttimes,danaconstantlosses = lsq_streaming_optax_simple(newkey, 
        #                         problem.get_data, 
        #                         SGDBATCH, 
        #                         STEPS, 
        #                         danaconstantopt, 
        #                         jnp.zeros((problem.d,1)), 
        #                         problem.get_population_risk)


        # Run ODE with fixed g3 (version 2: exponent -1/(2*alpha))
        print("  Running fixed g3 (exp=-1/(2α))...")
        dana_hparams_fixed2 = DanaHparams(g1, g2, g3_fixed2, Delta)
        times_fixed2, risks_fixed2, momentum_fixed2 = ode_resolvent_log_implicit_and_momentum_dynamic(
            ode_inputs_deterministic,
            dana_hparams_fixed2,
            SGDBATCH,
            num_grid_points,
            STEPS,
            DT,
            g3_mode='fixed'
        )
        risks_fixed2 = 0.5 * risks_fixed2

        #Runs the stochastic version of dana-decaying
        # danadecayingopt = optimizers.dana_optimizer(g1,g2,g3_fixed2,Delta)
        # key, newkey = random.split(key)
        # danadecayingtimes,danadecayinglosses = lsq_streaming_optax_simple(newkey, 
        #                         problem.get_data, 
        #                         SGDBATCH, 
        #                         STEPS, 
        #                         danadecayingopt, 
        #                         jnp.zeros((problem.d,1)), 
        #                         problem.get_population_risk)

        # Run ODE with dynamic g3
        print("  Running dynamic g3...")
        # Use fixed1 as base for dynamic (the g3 function won't be used in dynamic mode)
        dana_hparams_dynamic = DanaHparams(g1, g2, g3_fixed3, Delta)
        times_dynamic, risks_dynamic, momentum_dynamic = ode_resolvent_log_implicit_and_momentum_dynamic(
            ode_inputs_deterministic,
            dana_hparams_dynamic,
            SGDBATCH,
            num_grid_points,
            STEPS,
            DT,
            g3_mode='dynamic'
        )
        risks_dynamic = 0.5 * risks_dynamic

        #Runs the stochastic version of Auto-dana
        # danaautoopt = optimizers.auto_dana_optimizer(g1, g2, g3_fixed3, Delta)
        # key, newkey = random.split(key)
        # danaautotimes,danaautolosses = lsq_streaming_optax_simple(newkey, 
        #                         problem.get_data, 
        #                         SGDBATCH, 
        #                         STEPS, 
        #                         danaautoopt, 
        #                         jnp.zeros((problem.d,1)), 
        #                         problem.get_population_risk)

        print("  Running fixed SGD...")
        dana_hparams_sgd = DanaHparams(g1, g2, g3_sgd, Delta)
        times_sgd, risks_sgd, momentum_sgd = ode_resolvent_log_implicit_and_momentum_dynamic(
            ode_inputs_deterministic,
            dana_hparams_sgd,
            SGDBATCH,
            num_grid_points,
            STEPS,
            DT,
            g3_mode='fixed'
        )
        risks_sgd = 0.5 * risks_sgd

        #Runs the stochastic version of SGD
        # sgdopt = optimizers.dana_optimizer(g1,g2,g3_sgd,Delta)
        # key, newkey = random.split(key)
        # sgdtimes, sgdlosses = lsq_streaming_optax_simple(newkey, 
        #                         problem.get_data, 
        #                         SGDBATCH, 
        #                         STEPS, 
        #                         sgdopt, 
        #                         jnp.zeros((problem.d,1)), 
        #                         problem.get_population_risk)

        # Debug: Print some statistics
        print(f"    Fixed1: risk range [{jnp.min(risks_fixed1):.2e}, {jnp.max(risks_fixed1):.2e}], momentum range [{jnp.min(momentum_fixed1):.2e}, {jnp.max(momentum_fixed1):.2e}]")
        print(f"    Fixed2: risk range [{jnp.min(risks_fixed2):.2e}, {jnp.max(risks_fixed2):.2e}], momentum range [{jnp.min(momentum_fixed2):.2e}, {jnp.max(momentum_fixed2):.2e}]")
        print(f"    Dynamic: risk range [{jnp.min(risks_dynamic):.2e}, {jnp.max(risks_dynamic):.2e}], momentum range [{jnp.min(momentum_dynamic):.2e}, {jnp.max(momentum_dynamic):.2e}]")
        print(f"    SGD: risk range [{jnp.min(risks_sgd):.2e}, {jnp.max(risks_sgd):.2e}], momentum range [{jnp.min(momentum_sgd):.2e}, {jnp.max(momentum_sgd):.2e}]")

        # Plot risks
        ax_risk = axes_risk[idx]
        ax_risk.loglog(times_fixed1, risks_fixed1, label='Fixed g3 (exp=-1.0)',
                       color='tab:blue', linewidth=LINE_WIDTH, alpha=0.8)
        ax_risk.loglog(times_fixed2, risks_fixed2, label='Fixed g3 (exp=-1/(2α))',
                       color='tab:orange', linewidth=LINE_WIDTH, alpha=0.8)
        ax_risk.loglog(times_dynamic, risks_dynamic, label='Dynamic g3',
                       color='tab:green', linewidth=LINE_WIDTH, alpha=0.8)
        ax_risk.loglog(times_sgd, risks_sgd, label='sgd',
                       color='tab:red', linewidth=LINE_WIDTH, alpha=0.8)

        # ax_risk.loglog(danaconstanttimes, danaconstantlosses, label='Fixed g3 (exp=-1.0)',
        #                color='tab:blue', linewidth=LINE_WIDTH, linestyle = '--', alpha=0.8)
        # ax_risk.loglog(danadecayingtimes, danadecayinglosses, label='Fixed g3 (exp=-1/(2α))',
        #                color='tab:orange', linewidth=LINE_WIDTH, linestyle = '--', alpha=0.8)
        # ax_risk.loglog(danaautotimes, danaautolosses, label='Dynamic g3',
        #                color='tab:green', linewidth=LINE_WIDTH, linestyle = '--', alpha=0.8)
        # ax_risk.loglog(sgdtimes, sgdlosses, label='sgd',
        #                color='tab:red', linewidth=LINE_WIDTH, linestyle = '--', alpha=0.8)

        ax_risk.set_xlabel('Iterations', fontsize=18)
        ax_risk.set_ylabel('Risk', fontsize=18)
        ax_risk.set_title(f'α={alpha}', fontsize=20)
        ax_risk.legend(fontsize=14)
        ax_risk.grid(True, alpha=0.3)
        ax_risk.set_xlim(1, None)

        # Plot risk/||y||^2 ratios
        ax_ratio = axes_ratio[idx]
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        ratio_fixed1 = risks_fixed1 / (momentum_fixed1 + eps)
        ratio_fixed2 = risks_fixed2 / (momentum_fixed2 + eps)
        ratio_dynamic = risks_dynamic / (momentum_dynamic + eps)
        ratio_sgd = risks_sgd / (momentum_sgd + eps)

        # Filter out invalid values (inf, nan)
        mask_fixed1 = jnp.isfinite(ratio_fixed1) & (ratio_fixed1 > 0)
        mask_fixed2 = jnp.isfinite(ratio_fixed2) & (ratio_fixed2 > 0)
        mask_dynamic = jnp.isfinite(ratio_dynamic) & (ratio_dynamic > 0)
        mask_sgd = jnp.isfinite(ratio_sgd) & (ratio_sgd > 0)

        ax_ratio.loglog(times_fixed1[mask_fixed1], ratio_fixed1[mask_fixed1], label='Fixed g3 (exp=-1.0)',
                        color='tab:blue', linewidth=LINE_WIDTH)
        ax_ratio.loglog(times_fixed2[mask_fixed2], ratio_fixed2[mask_fixed2], label='Fixed g3 (exp=-1/(2α))',
                        color='tab:orange', linewidth=LINE_WIDTH)
        ax_ratio.loglog(times_dynamic[mask_dynamic], ratio_dynamic[mask_dynamic], label='Dynamic g3',
                        color='tab:green', linewidth=LINE_WIDTH, linestyle='--')
        ax_ratio.loglog(times_sgd[mask_sgd], ratio_sgd[mask_sgd], label='sgd',
                        color='tab:red', linewidth=LINE_WIDTH)

        ax_ratio.set_xlabel('Iterations', fontsize=18)
        ax_ratio.set_ylabel('Risk / ||y||²', fontsize=18)
        ax_ratio.set_title(f'α={alpha}', fontsize=20)
        ax_ratio.legend(fontsize=14)
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.set_xlim(1, None)
        ax_ratio.set_ylim(None, 1)

    # Finalize and save plots
    fig_risk.suptitle(f'Risk Comparison: Fixed vs Dynamic g3 (β={BETA})', fontsize=24)
    fig_risk.tight_layout()
    fig_risk.savefig(OUTPUT_FILE_RISK)
    print(f"\nRisk figure saved to {OUTPUT_FILE_RISK}")

    fig_ratio.suptitle(f'Risk/||y||² Comparison: Fixed vs Dynamic g3 (β={BETA})', fontsize=24)
    fig_ratio.tight_layout()
    fig_ratio.savefig(OUTPUT_FILE_RATIO)
    print(f"Ratio figure saved to {OUTPUT_FILE_RATIO}")


if __name__ == "__main__":
    main()
