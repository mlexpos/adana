#!/usr/bin/env python3
"""
Sweep over alpha values, run kappa_sweep_plot for each, and extract the crossing point
where y=x intersects the green curve (log(Risk) - log(||y||²))/log(T) at T=10^6.
Then plot the crossing kappa value as a function of 1/(2*alpha).
"""

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import optimizers
from power_law_rf import PowerLawRF
import deterministic_equivalent as theory


class DanaHparams:
    """Hyperparameters for DANA optimizer schedules."""
    def __init__(self, g1, g2, g3, delta):
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.delta = delta


class ODEInputs:
    """Input parameters for ODE solver."""
    def __init__(self, eigs_K, rho_init, chi_init, sigma_init, risk_infinity):
        self.eigs_K = eigs_K
        self.rho_init = rho_init
        self.chi_init = chi_init
        self.sigma_init = sigma_init
        self.risk_infinity = risk_infinity


# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================
BETA = 0.5                    # Fixed beta value
V = 4000                      # Number of features
D = 1000                      # Number of parameters
h = 0.0
SGDBATCH = jnp.int32(D**h)
STEPS = 10**7                 # Number of steps
DT = 10**(-3)                 # ODE time step
NUM_KAPPA = 30                # Number of kappa values to test
KAPPA_MIN = -1.5              # Minimum kappa value
KAPPA_MAX = -0.2             # Maximum kappa value
TARGET_T = 10**6              # Target iteration for extracting crossing

# Alpha sweep parameters
NUM_ALPHA = 5               # Number of alpha values
ALPHA_MIN = 0.7               # Minimum alpha
ALPHA_MAX = 1.2               # Maximum alpha

OUTPUT_FILE = 'alpha_sweep_crossing.pdf'


# ============================================================================
# DANASTAR ODE SOLVER (copied from kappa_sweep_plot.py)
# ============================================================================

def ode_resolvent_log_implicit_and_momentum_dynamic(
    inputs: ODEInputs,
    opt_hparams: DanaHparams,
    batch: int,
    D: int,
    t_max: float,
    dt: float,
):
    """DanaStar ODE solver with fixed g3 schedule."""
    g1_fn, g2_fn, g3_fn, delta_fn = opt_hparams.g1, opt_hparams.g2, opt_hparams.g3, opt_hparams.delta
    eigs_K = inputs.eigs_K
    rho_init, chi_init, sigma_init = inputs.rho_init, inputs.chi_init, inputs.sigma_init
    twice_risk_infinity = 2.0 * inputs.risk_infinity
    times = jnp.arange(0, jnp.log(t_max), step=dt, dtype=jnp.float32)
    risk_init = twice_risk_infinity + jnp.sum(inputs.eigs_K * inputs.rho_init)

    def inverse_3x3(omega):
        a11, a12, a13 = omega[0][0], omega[0][1], omega[0][2]
        a21, a22, a23 = omega[1][0], omega[1][1], omega[1][2]
        a31, a32, a33 = omega[2][0], omega[2][1], omega[2][2]

        det = (a11*a22*a33 + a12*a23*a31 + a13*a21*a32
               - a13*a22*a31 - a11*a23*a32 - a12*a21*a33)

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
        omega_11 = -2.0 * (g2 + g1 * g3) * eigs_K + \
                   ((batch + 1.0) / batch) * (g2**2 + 2.0 * g1 * g3 * g2 + g1**2 * g3**2) * eigs_K**2
        omega_12 = g3**2 * (1.0 - delta)**2 * jnp.ones_like(eigs_K)
        omega_13 = -2.0 * g3 * (1.0 - delta) + \
                   2.0 * (g2 * g3 + g3**2 * g1) * (1.0 - delta) * eigs_K
        omega_1 = jnp.array([omega_11, omega_12, omega_13])

        omega_21 = ((batch + 1.0) / batch) * g1**2 * eigs_K**2
        omega_22 = (-2.0 * delta + delta**2) * jnp.ones_like(eigs_K)
        omega_23 = 2.0 * g1 * eigs_K * (1.0 - delta)
        omega_2 = jnp.array([omega_21, omega_22, omega_23])

        omega_31 = g1 * eigs_K - ((batch + 1.0) / batch) * eigs_K**2 * (g1 * g2 + g1**2 * g3)
        omega_32 = (-g3 + g3 * delta * (2.0 - delta)) * jnp.ones_like(eigs_K)
        omega_33 = -delta - (g2 - g2 * delta + 2.0 * (1.0 - delta) * g1 * g3) * eigs_K
        omega_3 = jnp.array([omega_31, omega_32, omega_33])

        omega = jnp.array([omega_1, omega_2, omega_3])
        return omega

    def forcing_term(time_plus, g1, g2, g3, delta):
        Gamma = jnp.array([
            (g2**2 + 2.0 * g1 * g2 * g3 + g1**2 * g3**2) / batch,
            g1**2 / batch,
            (-g1 * g2 - g1**2 * g3) / batch
        ])
        return jnp.einsum('i,j->ij', Gamma, inputs.eigs_K)

    def ode_update(carry, time):
        v, twice_risk, momentum_norm = carry
        time_plus = jnp.exp(time + dt)
        time_plus_minus_one = time_plus - 1.0

        risk_scaling = 1.0 / jnp.sqrt(twice_risk + 1e-10)
        g3 = g3_fn(time_plus_minus_one)
        g1 = g1_fn(time_plus)
        g2 = g2_fn(time_plus) * risk_scaling
        delta = delta_fn(time_plus)
        g3 = g3 * risk_scaling

        omega = omega_full(time_plus_minus_one, g1, g2, g3, delta)
        identity = jnp.tensordot(jnp.eye(3), jnp.ones_like(eigs_K), 0)
        scaled_dt = dt * time_plus

        A = inverse_3x3(identity - scaled_dt * omega)
        z = jnp.einsum('i, j -> ij', jnp.array([1.0, 0.0, 0.0]), eigs_K)
        G_lambda = forcing_term(time_plus_minus_one, g1, g2, g3, delta)
        x_temp = v + scaled_dt * twice_risk_infinity * G_lambda
        x = jnp.einsum('ijk, jk -> ik', A, x_temp)
        y = jnp.einsum('ijk, jk -> ik', A, G_lambda)
        v_new = x + (scaled_dt * y * jnp.sum(x * z) /
                    (1.0 - scaled_dt * jnp.sum(y * z)))

        twice_risk_new = twice_risk_infinity + jnp.sum(eigs_K * v_new[0])
        momentum_norm_new = jnp.sum(v_new[1])

        return (v_new, twice_risk_new, momentum_norm_new), (twice_risk, jnp.sum(v[1]))

    v_init = jnp.array([rho_init, sigma_init, chi_init])
    momentum_norm_init = jnp.sum(sigma_init)
    initial_carry = (v_init, risk_init, momentum_norm_init)

    _, outputs = jax.lax.scan(ode_update, initial_carry, times)
    twice_risks, momentum_norms = outputs

    return jnp.exp(times) - 1.0, twice_risks, momentum_norms


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def find_crossing_kappa(alpha):
    """
    Run kappa sweep for given alpha and find where y=x crosses
    the (log(Risk) - log(||y||²))/log(T) curve at T=TARGET_T.

    Returns the kappa value at the crossing point.
    """
    key = random.PRNGKey(0)

    # Initialize problem
    key, subkey = random.split(key)
    problem = PowerLawRF.initialize_random(alpha=alpha, beta=BETA, v=V, d=D, key=subkey)

    # Learning rate scalar
    LRscalar = 0.1 / jnp.sqrt(jnp.float32(problem.d))

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

    # Set up fixed g1, g2, and delta schedules
    g1 = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
    g2 = optimizers.powerlaw_schedule(LRscalar * 0.5 * jnp.minimum(1.0, jnp.float32(SGDBATCH) / problem.population_trace), 0.0, 0.0, 1)
    delta_constant = 4.0 + 2*(alpha + BETA)/(2*alpha)
    Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, delta_constant)

    # Generate kappa values
    kappa_values = jnp.linspace(KAPPA_MIN, KAPPA_MAX, NUM_KAPPA, endpoint=False)

    # Storage for ratio at T=TARGET_T
    ratio_values = []

    # Run ODE for each kappa value
    for i, kappa in enumerate(kappa_values):
        g3constant = 1.0
        g3 = optimizers.powerlaw_schedule(g3constant, 0.0, kappa, 1)
        dana_hparams = DanaHparams(g1, g2, g3, Delta)

        # Run ODE
        times, twice_risks, momentum_norms = ode_resolvent_log_implicit_and_momentum_dynamic(
            ode_inputs_deterministic,
            dana_hparams,
            SGDBATCH,
            num_grid_points,
            STEPS,
            DT
        )

        # Convert to risk
        risks = 0.5 * twice_risks

        # Find closest time to TARGET_T
        idx = jnp.argmin(jnp.abs(times - TARGET_T))

        # Compute ratio: (log(Risk) - log(||y||²)) / log(T)
        log_risk = jnp.log(risks[idx])
        log_momentum = jnp.log(momentum_norms[idx])
        ratio = (log_risk - log_momentum) / jnp.log(TARGET_T)

        ratio_values.append(float(ratio))

    ratio_values = np.array(ratio_values)
    kappa_array = np.array(kappa_values)

    # Find crossing point where ratio = kappa (y = x)
    # Linear interpolation to find where ratio - kappa = 0
    diff = ratio_values - kappa_array

    # Find sign change
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        print(f"  Warning: No crossing found for alpha={alpha:.3f}")
        return None

    # Take first crossing
    idx = sign_changes[0]

    # Linear interpolation between kappa_array[idx] and kappa_array[idx+1]
    x1, x2 = kappa_array[idx], kappa_array[idx+1]
    y1, y2 = diff[idx], diff[idx+1]

    # Find x where y = 0
    kappa_crossing = x1 - y1 * (x2 - x1) / (y2 - y1)

    return float(kappa_crossing)


def main():
    style.use('default')
    plt.rcParams['font.weight'] = 'light'
    plt.rcParams.update({'font.size': 14})

    print("="*70)
    print(f"Alpha Sweep: Finding crossing points at T={TARGET_T:.0e}")
    print(f"Testing {NUM_ALPHA} alpha values in range ({ALPHA_MIN}, {ALPHA_MAX})")
    print("="*70)

    # Generate alpha values
    alpha_values = np.linspace(ALPHA_MIN, ALPHA_MAX, NUM_ALPHA)

    # Storage for crossing points
    crossing_kappas = []
    valid_alphas = []

    # Run for each alpha
    for i, alpha in enumerate(alpha_values):
        print(f"\n[{i+1}/{NUM_ALPHA}] Processing alpha = {alpha:.3f}...")

        kappa_cross = find_crossing_kappa(alpha)

        if kappa_cross is not None:
            crossing_kappas.append(kappa_cross)
            valid_alphas.append(alpha)
            print(f"  -> Crossing at κ = {kappa_cross:.4f}")

    crossing_kappas = np.array(crossing_kappas)
    valid_alphas = np.array(valid_alphas)

    # Compute 1/(2*alpha)
    x_values = 1.0 / (2.0 * valid_alphas)

    # Compute linear fit: κ = m * (1/(2α)) + b
    coeffs = np.polyfit(x_values, crossing_kappas, 1)
    slope, intercept = coeffs[0], coeffs[1]
    fit_line = slope * x_values + intercept

    # Compute R² for the fit
    residuals = crossing_kappas - fit_line
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((crossing_kappas - np.mean(crossing_kappas))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(x_values, crossing_kappas, 'o', markersize=10,
            color='tab:green', label='Crossing κ', zorder=3)
    ax.plot(x_values, fit_line, '--', linewidth=2.5, color='tab:red',
            label=f'Linear fit: κ = {slope:.4f} · (1/2α) + {intercept:.4f}', zorder=2)

    ax.set_xlabel('1/(2α)', fontsize=18)
    ax.set_ylabel('Crossing κ', fontsize=18)
    ax.set_title(f'Crossing Point vs 1/(2α) at T={TARGET_T:.0e}\n(β={BETA}, D={D})',
                 fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')

    # Add text box with info
    textstr = f'T = {TARGET_T:.0e}\nβ = {BETA}\nD = {D}\nR² = {r_squared:.5f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)

    print("\n" + "="*70)
    print(f"Figure saved to {OUTPUT_FILE}")
    print("="*70)

    # Print summary table
    print("\nLinear Fit Results:")
    print(f"  κ = {slope:.6f} · (1/2α) + {intercept:.6f}")
    print(f"  R² = {r_squared:.6f}")

    print("\nSummary:")
    print(f"{'α':>8}  {'1/(2α)':>10}  {'κ_crossing':>12}  {'κ_fit':>12}")
    print("-" * 50)
    for alpha, x, kappa, fit in zip(valid_alphas, x_values, crossing_kappas, fit_line):
        print(f"{alpha:>8.3f}  {x:>10.4f}  {kappa:>12.4f}  {fit:>12.4f}")


if __name__ == "__main__":
    main()
