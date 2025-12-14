#!/usr/bin/env python
"""
PLRF Optimizer Comparison using ODE simulation.

Compares: SGD, Dana-constant, Dana-decay, AdEMAMix
USAGE: python plrf_optimizer_comparison.py
"""
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import NamedTuple, Callable, Union

from optimizers import powerlaw_schedule
from power_law_rf import PowerLawRF
import deterministic_equivalent as theory

# ============================================================================
# Configuration
# ============================================================================

ALPHA = 1.5
D = 500
T = D ** (1.5*ALPHA)

CONFIG = {
    'alpha': ALPHA,
    'beta': 0.3,
    'v': 5 * D,
    'd': D,
    'steps': int(D ** (2 * ALPHA)),
    'batch_size': 1,
    'lr_scale': 0.5,
    'g2': 0.5,
    'g3': 0.5,
    'delta': 8.0,
    'output_dir': 'jax',
    'random_seed': 42,
    'optimizers': ['sgd', 'dana-constant', 'dana-decay', 'ademamix'],
}

# ============================================================================
# ODE Framework
# ============================================================================

class DanaHparams(NamedTuple):
    g1: Callable
    g2: Callable
    g3: Callable
    delta: Callable


class ODEInputs(NamedTuple):
    eigs_K: jnp.ndarray
    rho_init: jnp.ndarray
    chi_init: jnp.ndarray
    sigma_init: jnp.ndarray
    risk_infinity: float


def ode_solve(inputs, hparams, batch, D, t_max, dt=1e-3):
    """Solve the DANA ODE using implicit log-time stepping."""
    g1_fn, g2_fn, g3_fn, delta_fn = hparams
    eigs_K = inputs.eigs_K
    rho_init, chi_init, sigma_init = inputs.rho_init, inputs.chi_init, inputs.sigma_init
    twice_risk_inf = 2.0 * inputs.risk_infinity
    times = jnp.arange(0, jnp.log(t_max), step=dt, dtype=jnp.float32)
    risk_init = twice_risk_inf + jnp.sum(eigs_K * rho_init)

    def inv3x3(omega):
        a11, a12, a13 = omega[0][0], omega[0][1], omega[0][2]
        a21, a22, a23 = omega[1][0], omega[1][1], omega[1][2]
        a31, a32, a33 = omega[2][0], omega[2][1], omega[2][2]
        det = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31 - a11*a23*a32 - a12*a21*a33
        return jnp.array([
            [(a22*a33 - a23*a32)/det, (a13*a32 - a12*a33)/det, (a12*a23 - a13*a22)/det],
            [(a23*a31 - a21*a33)/det, (a11*a33 - a13*a31)/det, (a13*a21 - a11*a23)/det],
            [(a21*a32 - a22*a31)/det, (a12*a31 - a11*a32)/det, (a11*a22 - a12*a21)/det]
        ])

    def omega(t):
        g1, g2, g3, delta = g1_fn(t), g2_fn(t), g3_fn(t), delta_fn(t)
        c = (batch + 1.0) / batch
        return jnp.array([
            [-2*(g2 + g1*g3)*eigs_K + c*(g2 + g1*g3)**2*eigs_K**2,
             g3**2*(1-delta)**2*jnp.ones_like(eigs_K),
             -2*g3*(1-delta) + 2*(g2*g3 + g3**2*g1)*(1-delta)*eigs_K],
            [c*g1**2*eigs_K**2,
             (-2*delta + delta**2)*jnp.ones_like(eigs_K),
             2*g1*eigs_K*(1-delta)],
            [g1*eigs_K - c*eigs_K**2*(g1*g2 + g1**2*g3),
             (-g3 + g3*delta*(2-delta))*jnp.ones_like(eigs_K),
             -delta - (g2 - g2*delta + 2*(1-delta)*g1*g3)*eigs_K]
        ])

    def forcing(t):
        g1, g2, g3 = g1_fn(t), g2_fn(t), g3_fn(t)
        Gamma = jnp.array([(g2 + g1*g3)**2/batch, g1**2/batch, -(g1*g2 + g1**2*g3)/batch])
        return jnp.einsum('i,j->ij', Gamma, eigs_K)

    def step(carry, time):
        v, risk = carry
        tp = jnp.exp(time + dt)
        Om = omega(tp - 1)
        I = jnp.tensordot(jnp.eye(3), jnp.ones(D), 0)
        A = inv3x3(I - dt*tp*Om)
        z = jnp.einsum('i,j->ij', jnp.array([1., 0., 0.]), eigs_K)
        G = forcing(tp - 1)
        x = jnp.einsum('ijk,jk->ik', A, v + dt*tp*twice_risk_inf*G)
        y = jnp.einsum('ijk,jk->ik', A, G)
        v_new = x + dt*tp*y*jnp.sum(x*z)/(1 - dt*tp*jnp.sum(y*z))
        return (v_new, twice_risk_inf + jnp.sum(eigs_K*v_new[0])), risk

    _, risks = jax.lax.scan(step, (jnp.array([rho_init, sigma_init, chi_init]), risk_init), times)
    return jnp.exp(times) - 1, risks


def run_ode(cfg):
    """Run ODE for all optimizers."""
    alpha, beta, d = cfg['alpha'], cfg['beta'], cfg['d']
    batch, steps, dt = cfg['batch_size'], cfg['steps'], 1e-3
    
    # Theory inputs
    Keigs, rho_w = theory.theory_rhos(alpha, beta, d)
    key = random.PRNGKey(cfg['random_seed'])
    problem = PowerLawRF.initialize_random(alpha=alpha, beta=beta, v=cfg['v'], d=d, key=key)
    risk_inf = problem.get_theory_limit_loss()
    
    inputs = ODEInputs(Keigs.astype(jnp.float32), rho_w, jnp.zeros_like(rho_w), jnp.zeros_like(rho_w), risk_inf)
    n = jnp.shape(rho_w)[-1]
    
    # Learning rate
    lr = cfg['lr_scale'] * jnp.minimum(1.0, jnp.float32(batch) / problem.population_trace)
    g2_base = lr * cfg['g2']
    delta_const = cfg['delta']
    
    # Common schedules
    g1 = powerlaw_schedule(1.0, 0.0, 0.0, 1)
    g2 = powerlaw_schedule(g2_base, 0.0, 0.0, 1)
    Delta = powerlaw_schedule(1.0, 0.0, -1.0, delta_const)
    
    results = {}
    
    # SGD
    if 'sgd' in cfg['optimizers']:
        print("  SGD...")
        h = DanaHparams(g1, g2, powerlaw_schedule(0.0, 0.0, 0.0, 1), Delta)
        t, r = ode_solve(inputs, h, batch, n, steps, dt)
        results['sgd'] = {'t': np.array(t+1), 'loss': np.array(0.5*r)}
    
    # Dana-constant
    if 'dana-constant' in cfg['optimizers']:
        print("  Dana-constant...")
        g3 = powerlaw_schedule(cfg['g3']*g2_base/d, 0.0, 0.0, 1)
        h = DanaHparams(g1, g2, g3, Delta)
        t, r = ode_solve(inputs, h, batch, n, steps, dt)
        results['dana-constant'] = {'t': np.array(t+1), 'loss': np.array(0.5*r)}
    
    # Dana-decay
    if 'dana-decay' in cfg['optimizers']:
        print("  Dana-decay...")
        g3 = powerlaw_schedule(cfg['g3']*g2_base, 0.0, -1.0/(2*alpha), 1.0)
        h = DanaHparams(g1, g2, g3, Delta)
        t, r = ode_solve(inputs, h, batch, n, steps, dt)
        results['dana-decay'] = {'t': np.array(t+1), 'loss': np.array(0.5*r)}
    
    # AdEMAMix
    if 'ademamix' in cfg['optimizers']:
        print("  AdEMAMix...")
        T_mix = d ** alpha
        beta3_final = 1.0 - delta_const / T_mix
        alpha_final = 0.5 * T_mix ** (1 - 1/(2*alpha))
        
        def g1_mix(t): return 1.0 / (1.0 + t)
        def g3_mix(t): return g2_base * jnp.minimum(t/T_mix, 1.0) * alpha_final
        def delta_mix(t):
            a = jnp.minimum(t/T_mix, 1.0)
            hl_s, hl_e = jnp.log(0.5)/jnp.log(0.9+1e-8)-1, jnp.log(0.5)/jnp.log(beta3_final+1e-8)-1
            return 1.0 - jnp.power(0.5, 1.0/((1-a)*hl_s + a*hl_e + 1))
        
        h = DanaHparams(g1_mix, g2, g3_mix, delta_mix)
        t, r = ode_solve(inputs, h, batch, n, steps, dt)
        results['ademamix'] = {'t': np.array(t+1), 'loss': np.array(0.5*r)}
    
    return results, risk_inf


def plot(results, cfg, risk_inf):
    # Professional plot styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.5,
        'figure.dpi': 150,
    })
    
    colors = {'sgd': '#1f77b4', 'dana-constant': '#2ca02c', 'dana-decay': '#ff7f0e', 'ademamix': '#d62728'}
    labels = {'sgd': 'SGD', 'dana-constant': 'DANA (constant)', 'dana-decay': 'DANA (decay)', 'ademamix': 'AdEMAMix'}
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    for name, res in results.items():
        ax.loglog(res['t'], res['loss'], color=colors.get(name, 'black'), 
                  lw=2.5, label=labels.get(name, name), alpha=0.9)
    
    ax.axhline(y=risk_inf, color='gray', ls='--', lw=1.5, alpha=0.8, label='Limit risk')
    
    # Vertical line at T_ademamix (warmup end for beta3)
    T_mix = cfg['d'] ** cfg['alpha']
    ax.axvline(x=T_mix, color='gray', ls='--', lw=2, alpha=0.8, label=f'$T_{{\\beta_3}}$')
    
    ax.set_xlabel('Iteration $t$')
    ax.set_ylabel('Population Risk $\\mathcal{R}(\\theta_t)$')
    ax.set_xlim(1, cfg['steps'])
    
    # Clean legend
    ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=0.95, loc='upper right')
    
    # Subtle grid
    ax.grid(True, which='major', ls='-', alpha=0.2, color='gray')
    ax.grid(True, which='minor', ls=':', alpha=0.1, color='gray')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    os.makedirs(cfg['output_dir'], exist_ok=True)
    plt.savefig(os.path.join(cfg['output_dir'], 'plrf_comparison_ode.pdf'), bbox_inches='tight')
    print(f"Saved to {cfg['output_dir']}/plrf_comparison_ode.pdf")
    plt.close()


if __name__ == "__main__":
    print(f"Config: α={CONFIG['alpha']}, β={CONFIG['beta']}, d={CONFIG['d']}, T={CONFIG['steps']}")
    results, risk_inf = run_ode(CONFIG)
    print("\nFinal risks:")
    for name, res in results.items():
        print(f"  {name}: {res['loss'][-1]:.6e}")
    print(f"  Limit: {risk_inf:.6e}")
    plot(results, CONFIG, risk_inf)
