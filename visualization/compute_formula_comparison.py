#!/usr/bin/env python3
"""
Generate residual plots comparing different compute formulas and fitting approaches for AdamW.

Creates a 3-row figure:
- Top: Single power law L = a + b*C^{-c} with fitted saturation
- Middle: Single power law with forced a=0
- Bottom: Broken power law L = a + b*C^{-c} + e*C^{-f}

This demonstrates the superiority of the 6D Chinchilla formula for scaling law extrapolation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style, rc, rcParams
import argparse
import warnings
import os
import pickle
import jax
import jax.numpy as jnp
from jax import grad, jit
import optax

warnings.filterwarnings('ignore')

# Matplotlib formatting
style.use('seaborn-v0_8-darkgrid')
rc('font', family='sans-serif')
rcParams['font.weight'] = 'normal'
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.0
rcParams['axes.edgecolor'] = '#333333'
rcParams['grid.alpha'] = 0.3
rcParams['grid.linestyle'] = '--'

# Compute formulas to compare
COMPUTE_FORMULAS = [
    ('6N1', 'Kaplan (6P)', '#FF6347'),        # Red - older formula
    ('6N2', 'Chinchilla (6D)', '#00CED1'),    # Cyan - recommended
    ('M', 'DeepSeek (M)', '#9370DB'),          # Purple - attention-aware
]

# Fitting approaches
FIT_APPROACHES = [
    ('single_fitted_a', 'Single power law (fitted $a$)'),
    ('single_forced_a0', 'Single power law ($a=0$)'),
    ('broken', 'Broken power law'),
]


def load_data_for_formula(cache_dir='wandb_cache', compute_formula='6N2', optimizer='adamw'):
    """Load optimizer data for Enoki from formula-specific cache."""
    suffix = compute_formula

    possible_paths = [
        f'cache_Enoki_Scaled_{optimizer}_danastar_ep-rmt-ml-opt_{suffix}.pkl',
        f'cache_Enoki_{optimizer}_danastar_ep-rmt-ml-opt_{suffix}.pkl',
    ]

    for filename in possible_paths:
        cache_path = os.path.join(cache_dir, filename)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data['df']

    return None


def fit_single_power_law(compute_arr, loss_arr, force_a_zero=False, n_steps=30000, learning_rate=0.1):
    """
    Fit single power law L = a + b*C^{-c}.
    If force_a_zero=True, fits L = b*C^{-c}.
    """
    compute_arr = jnp.array(compute_arr, dtype=jnp.float32)
    loss_arr = jnp.array(loss_arr, dtype=jnp.float32)

    if force_a_zero:
        # Fit L = b*C^{-c} (2 parameters)
        init_b = 0.5
        init_c = 0.1
        fit_params = jnp.array([jnp.log(init_b), init_c], dtype=jnp.float32)

        @jit
        def loss_fn(params):
            log_b, c = params
            b = jnp.exp(log_b)
            predictions = b * jnp.power(compute_arr, -c)
            return jnp.mean((loss_arr - predictions) ** 2)
    else:
        # Fit L = a + b*C^{-c} (3 parameters)
        init_a = 0.1
        init_b = 0.5
        init_c = 0.1
        fit_params = jnp.array([init_a, jnp.log(init_b), init_c], dtype=jnp.float32)

        @jit
        def loss_fn(params):
            a, log_b, c = params
            b = jnp.exp(log_b)
            predictions = a + b * jnp.power(compute_arr, -c)
            return jnp.mean((loss_arr - predictions) ** 2)

    # Optimize
    optimizer = optax.adagrad(learning_rate)
    opt_state = optimizer.init(fit_params)
    grad_fn = jit(grad(loss_fn))

    best_loss = float('inf')
    best_params = fit_params

    for step in range(n_steps):
        grads = grad_fn(fit_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        fit_params = optax.apply_updates(fit_params, updates)

        current_loss = float(loss_fn(fit_params))
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = fit_params

    # Extract parameters
    if force_a_zero:
        a = 0.0
        b = float(jnp.exp(best_params[0]))
        c = float(best_params[1])
    else:
        a = float(best_params[0])
        b = float(jnp.exp(best_params[1]))
        c = float(best_params[2])

    # Compute predictions and residuals
    compute_np = np.array(compute_arr)
    loss_np = np.array(loss_arr)
    predictions = a + b * np.power(compute_np, -c)
    residuals = loss_np - predictions
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((loss_np - np.mean(loss_np))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'a': a, 'b': b, 'c': c,
        'r_squared': r_squared,
        'predictions': predictions,
        'residuals': residuals,
    }


def fit_broken_power_law(compute_arr, loss_arr, n_steps=50000, learning_rate=0.1):
    """
    Fit broken power law L = a + b*C^{-c} + e*C^{-f}.
    """
    compute_arr = jnp.array(compute_arr, dtype=jnp.float32)
    loss_arr = jnp.array(loss_arr, dtype=jnp.float32)

    # Initialize parameters: [a, log(b), c, log(e), f]
    init_a = 0.1
    init_b = 0.5
    init_c = 0.05
    init_e = 0.1
    init_f = 0.2

    fit_params = jnp.array([init_a, jnp.log(init_b), init_c, jnp.log(init_e), init_f], dtype=jnp.float32)

    @jit
    def loss_fn(params):
        a = params[0]
        log_b = params[1]
        c = params[2]
        log_e = params[3]
        f = params[4]

        b = jnp.exp(log_b)
        e = jnp.exp(log_e)

        predictions = a + b * jnp.power(compute_arr, -c) + e * jnp.power(compute_arr, -f)
        return jnp.mean((loss_arr - predictions) ** 2)

    # Optimize
    optimizer = optax.adagrad(learning_rate)
    opt_state = optimizer.init(fit_params)
    grad_fn = jit(grad(loss_fn))

    best_loss = float('inf')
    best_params = fit_params

    for step in range(n_steps):
        grads = grad_fn(fit_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        fit_params = optax.apply_updates(fit_params, updates)

        current_loss = float(loss_fn(fit_params))
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = fit_params

    # Extract parameters
    a = float(best_params[0])
    b = float(jnp.exp(best_params[1]))
    c = float(best_params[2])
    e = float(jnp.exp(best_params[3]))
    f = float(best_params[4])

    # Compute predictions and residuals
    compute_np = np.array(compute_arr)
    loss_np = np.array(loss_arr)
    predictions = a + b * np.power(compute_np, -c) + e * np.power(compute_np, -f)
    residuals = loss_np - predictions
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((loss_np - np.mean(loss_np))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'a': a, 'b': b, 'c': c, 'e': e, 'f': f,
        'r_squared': r_squared,
        'predictions': predictions,
        'residuals': residuals,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare compute formulas and fitting approaches')
    parser.add_argument('--cache-dir', type=str, default='wandb_cache',
                        help='Directory with cached WandB data')
    parser.add_argument('--output', type=str, default='compute_formula_adamw_residuals.pdf',
                        help='Output filename')
    parser.add_argument('--ylim', type=float, default=0.02,
                        help='Y-axis limit for residuals')
    args = parser.parse_args()

    # Load data for each compute formula
    print("Loading AdamW data for different compute formulas...")
    all_data = {}

    for formula_key, formula_name, formula_color in COMPUTE_FORMULAS:
        df = load_data_for_formula(args.cache_dir, compute_formula=formula_key, optimizer='adamw')
        if df is not None:
            df = df.sort_values('size')
            all_data[formula_key] = {
                'compute': df['compute'].values,
                'loss': df['val_loss'].values,
                'sizes': df['size'].values,
                'name': formula_name,
                'color': formula_color,
            }
            print(f"  {formula_name}: {len(df)} points, compute {df['compute'].min():.4e} to {df['compute'].max():.4e}")
        else:
            print(f"  {formula_name}: not found")

    if len(all_data) == 0:
        raise RuntimeError("No data found! Ensure cache files exist for different compute formulas.")

    # Fit all combinations
    print("\nFitting power laws for each formula and approach...")
    all_fits = {}

    for formula_key in all_data:
        compute = all_data[formula_key]['compute']
        loss = all_data[formula_key]['loss']

        all_fits[formula_key] = {
            'single_fitted_a': fit_single_power_law(compute, loss, force_a_zero=False),
            'single_forced_a0': fit_single_power_law(compute, loss, force_a_zero=True),
            'broken': fit_broken_power_law(compute, loss),
        }

        for approach_key, approach_name in FIT_APPROACHES:
            fit = all_fits[formula_key][approach_key]
            print(f"  {all_data[formula_key]['name']} + {approach_name}: R²={fit['r_squared']:.6f}")

    # Create the figure: 3 rows (fitting approaches) x len(formulas) columns
    n_approaches = len(FIT_APPROACHES)
    n_formulas = len(all_data)

    fig, axes = plt.subplots(n_approaches, n_formulas, figsize=(4 * n_formulas, 2.5 * n_approaches),
                              sharex='col', sharey='row')
    fig.subplots_adjust(hspace=0.25, wspace=0.15, left=0.08, right=0.98, top=0.90, bottom=0.10)

    # Ensure axes is 2D
    if n_formulas == 1:
        axes = axes.reshape(-1, 1)

    # Plot
    formula_keys = list(all_data.keys())
    for col_idx, formula_key in enumerate(formula_keys):
        formula_data = all_data[formula_key]
        compute = formula_data['compute']

        for row_idx, (approach_key, approach_name) in enumerate(FIT_APPROACHES):
            ax = axes[row_idx, col_idx]
            fit = all_fits[formula_key][approach_key]

            # Plot residuals
            ax.scatter(compute, fit['residuals'], c=formula_data['color'], s=30, alpha=0.8,
                       edgecolors='black', linewidths=0.4)

            # Horizontal line at 0
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

            # Formatting
            ax.set_xscale('log')
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_ylim(-args.ylim, args.ylim)

            # R² label
            r2_label = f"$R^2$={fit['r_squared']:.4f}"
            ax.text(0.97, 0.92, r2_label, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            # Column headers (formula names)
            if row_idx == 0:
                ax.set_title(formula_data['name'], fontsize=11, fontweight='bold')

            # Row labels (fitting approaches)
            if col_idx == 0:
                ax.set_ylabel(approach_name + '\nResidual', fontsize=9)

            # X-axis label on bottom row
            if row_idx == n_approaches - 1:
                ax.set_xlabel('Compute (PFH)', fontsize=10)

    # Overall title
    fig.suptitle('Fit residuals for different compute formulas and fitting approaches (AdamW)',
                 fontsize=12, fontweight='bold', y=0.96)

    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', '68a78fcfedd649b2fea47ddd', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to {output_path}")

    # Also save as PNG
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {png_path}")

    # Print summary table
    print("\n" + "="*70)
    print("R² Summary:")
    print("="*70)
    print(f"{'Formula':<20} {'Fitted a':<12} {'Forced a=0':<12} {'Broken':<12}")
    print("-"*70)
    for formula_key in formula_keys:
        name = all_data[formula_key]['name']
        r2_fitted = all_fits[formula_key]['single_fitted_a']['r_squared']
        r2_forced = all_fits[formula_key]['single_forced_a0']['r_squared']
        r2_broken = all_fits[formula_key]['broken']['r_squared']
        print(f"{name:<20} {r2_fitted:<12.6f} {r2_forced:<12.6f} {r2_broken:<12.6f}")
    print("="*70)

    plt.close()


if __name__ == '__main__':
    main()
