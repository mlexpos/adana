#!/usr/bin/env python3
"""
Generate residual plots for broken power law fits across multiple optimizers.

Creates a figure with horizontal subplots showing residuals for each optimizer
using the 6D (Chinchilla) compute formula.

Supports both Enoki and Qwen3 architectures via --model argument.
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

# Optimizer display names and colors (matching compare_scaling_rules_ep.py)
# These are the default for Enoki
OPTIMIZERS_ENOKI = [
    ('adamw', 'AdamW', '#00CED1'),
    ('dana-star-mk4-kappa-0-85', 'Dana*-MK4-κ0.85', '#FF6347'),
    ('dana-star-no-tau-kappa-0-85', 'ADANA-κ0.85', '#32CD32'),
    ('d-muon', 'Muon', '#FF69B4'),
    ('ademamix-decaying-wd', 'Ademamix-DecayWD', '#9370DB'),
]

# Optimizers available for Qwen3
OPTIMIZERS_QWEN3 = [
    ('adamw', 'AdamW', '#00CED1'),
    ('adana-kappa-0-85', 'ADANA-κ0.85', '#32CD32'),
    ('dana-star-mk4-kappa-0-75', 'Dana*-MK4-κ0.75', '#FF6347'),
    ('dana-star-mk4-kappa-0-85', 'Dana*-MK4-κ0.85', '#DC143C'),
    ('dana-mk4-kappa-0-75', 'Dana-MK4-κ0.75', '#FFA500'),
    ('dana-mk4-kappa-0-85', 'Dana-MK4-κ0.85', '#FF8C00'),
    ('ademamix', 'Ademamix', '#9370DB'),
]


def load_data_for_formula(cache_dir='wandb_cache', compute_formula='6N2', optimizer='adamw', model='Enoki'):
    """Load optimizer data from formula-specific cache."""
    suffix = compute_formula

    possible_paths = [
        f'cache_{model}_Scaled_{optimizer}_danastar_ep-rmt-ml-opt_{suffix}.pkl',
        f'cache_{model}_{optimizer}_danastar_ep-rmt-ml-opt_{suffix}.pkl',
    ]

    for filename in possible_paths:
        cache_path = os.path.join(cache_dir, filename)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data['df']

    return None


def fit_broken_power_law_shared_a(all_data, shared_a, n_steps=50000, learning_rate=0.1):
    """
    Fit broken power law L = a + b*C^{-c} + e*C^{-f} with shared a across all optimizers.
    Returns fit results for each optimizer.

    Uses JOINT fitting (all curves simultaneously) matching compare_scaling_rules_ep.py:
    - Log-space fitting for numerical stability
    - Initialization: b=0.50, c=0.050, e=2.50, f=0.045
    - Equal weights (not compute-weighted)
    - Fixed shared 'a' parameter
    """
    # Prepare data for all optimizers
    opt_keys = list(all_data.keys())
    n_curves = len(opt_keys)

    jax_data = []
    for opt_key in opt_keys:
        opt_data = all_data[opt_key]
        compute_arr = jnp.array(opt_data['compute'], dtype=jnp.float32)
        loss_arr = jnp.array(opt_data['loss'], dtype=jnp.float32)
        weights_arr = jnp.ones_like(compute_arr)  # Equal weights
        jax_data.append({
            'compute': compute_arr,
            'loss': loss_arr,
            'weights': weights_arr,
            'name': opt_key
        })

    # Initialize parameters for all curves jointly
    # Params: [log(b_0), c_0, log(e_0), f_0, log(b_1), c_1, log(e_1), f_1, ...]
    # Initial values: b=0.40, c=0.200, e=2.50, f=0.030
    init_params = []
    for i in range(n_curves):
        init_params.extend([
            jnp.log(0.40),  # log(b)
            0.200,          # c
            jnp.log(2.50),  # log(e)
            0.030           # f
        ])

    fit_params = jnp.array(init_params, dtype=jnp.float32)

    # Use fixed shared_a (converted to JAX scalar for JIT)
    a_fixed = jnp.array(shared_a, dtype=jnp.float32)

    @jit
    def loss_fn(params):
        """Joint loss function for all curves with fixed shared saturation."""
        total_loss = 0.0
        total_weight = 0.0

        for i in range(n_curves):
            # Extract per-curve parameters: [log(b), c, log(e), f]
            log_b = params[4*i]
            c = params[4*i + 1]
            log_e = params[4*i + 2]
            f = params[4*i + 3]

            b = jnp.exp(log_b)
            e = jnp.exp(log_e)

            compute_i = jax_data[i]['compute']
            loss_i = jax_data[i]['loss']
            weights_i = jax_data[i]['weights']

            # Broken power law: loss = a + b*C^{-c} + e*C^{-f}
            pred_loss = a_fixed + b * jnp.power(compute_i, -c) + e * jnp.power(compute_i, -f)

            # Log-space fitting for numerical stability
            log_loss_true = jnp.log(loss_i)
            log_loss_pred = jnp.log(jnp.maximum(pred_loss, 1e-10))

            # Weighted MSE in log space
            residuals = (log_loss_pred - log_loss_true) ** 2
            curve_loss = jnp.sum(weights_i * residuals)
            curve_weight = jnp.sum(weights_i)

            total_loss += curve_loss
            total_weight += curve_weight

        return total_loss / total_weight

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

    # Extract final parameters and compute residuals for each optimizer
    results = {}
    for i, opt_key in enumerate(opt_keys):
        log_b = float(best_params[4*i])
        c = float(best_params[4*i + 1])
        log_e = float(best_params[4*i + 2])
        f = float(best_params[4*i + 3])

        b = float(jnp.exp(log_b))
        e = float(jnp.exp(log_e))

        # Compute R-squared and residuals (in original space)
        compute_np = np.array(all_data[opt_key]['compute'])
        loss_np = np.array(all_data[opt_key]['loss'])
        predictions = shared_a + b * np.power(compute_np, -c) + e * np.power(compute_np, -f)
        residuals = loss_np - predictions
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((loss_np - np.mean(loss_np))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        results[opt_key] = {
            'a': shared_a,
            'b': b,
            'c': c,
            'e': e,
            'f': f,
            'r_squared': r_squared,
            'predictions': predictions,
            'residuals': residuals,
            'compute': compute_np,
            'loss': loss_np,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate compute formula residual plots')
    parser.add_argument('--cache-dir', type=str, default='wandb_cache',
                        help='Directory with cached WandB data')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: compute_formula_residuals_{model}.pdf)')
    parser.add_argument('--shared-a', type=float, default=None,
                        help='Shared saturation parameter a (default: 0.1056 for Enoki, auto-fit for Qwen3)')
    parser.add_argument('--ylim', type=float, default=0.015,
                        help='Y-axis limit for residuals')
    parser.add_argument('--model', type=str, default='Enoki', choices=['Enoki', 'Qwen3'],
                        help='Model architecture to use (default: Enoki)')
    parser.add_argument('--head-min', type=int, default=None,
                        help='Minimum head count for fitting (default: None, use all data)')
    args = parser.parse_args()

    # Select optimizer list based on model
    if args.model == 'Qwen3':
        OPTIMIZERS = OPTIMIZERS_QWEN3
        default_shared_a = 0.12  # Will be refined from data
    else:
        OPTIMIZERS = OPTIMIZERS_ENOKI
        default_shared_a = 0.1056

    shared_a = args.shared_a if args.shared_a is not None else default_shared_a

    # Set default output filename based on model
    if args.output is None:
        args.output = f'compute_formula_residuals_{args.model.lower()}.pdf'

    # Load data for all optimizers
    print(f"Loading data for {args.model} with 6D (Chinchilla) compute formula...")
    all_data = {}

    for opt_key, opt_name, opt_color in OPTIMIZERS:
        df = load_data_for_formula(args.cache_dir, compute_formula='6N2', optimizer=opt_key, model=args.model)
        if df is not None:
            df = df.sort_values('size')
            # Filter by head-min if specified
            if args.head_min is not None:
                df_filtered = df[df['size'] >= args.head_min]
                print(f"  {opt_name}: {len(df_filtered)}/{len(df)} points (head >= {args.head_min}), compute {df_filtered['compute'].min():.4e} to {df_filtered['compute'].max():.4e} PFH")
                df = df_filtered
            else:
                print(f"  {opt_name}: {len(df)} points, compute {df['compute'].min():.4e} to {df['compute'].max():.4e} PFH")

            if len(df) > 0:
                all_data[opt_key] = {
                    'compute': df['compute'].values,
                    'loss': df['val_loss'].values,
                    'sizes': df['size'].values,
                    'name': opt_name,
                    'color': opt_color,
                }
        else:
            print(f"  {opt_name}: not found")

    if len(all_data) == 0:
        raise RuntimeError("No optimizer data found!")

    # Fit broken power law with shared a
    print(f"\nFitting broken power laws with shared a = {shared_a}...")
    fits = fit_broken_power_law_shared_a(all_data, shared_a)

    for opt_key, fit in fits.items():
        print(f"  {all_data[opt_key]['name']}: b={fit['b']:.3f}, c={fit['c']:.3f}, e={fit['e']:.3f}, f={fit['f']:.3f}, R²={fit['r_squared']:.6f}")

    # Create the figure
    n_opts = len(fits)
    fig, axes = plt.subplots(n_opts, 1, figsize=(10, 2 * n_opts), sharex=True)
    fig.subplots_adjust(hspace=0.15, left=0.10, right=0.95, top=0.92, bottom=0.08)

    if n_opts == 1:
        axes = [axes]

    # Plot residuals for each optimizer
    for idx, (opt_key, opt_name, opt_color) in enumerate(OPTIMIZERS):
        if opt_key not in fits:
            continue

        ax = axes[idx]
        fit = fits[opt_key]

        # Plot residuals vs compute
        ax.scatter(fit['compute'], fit['residuals'], c=opt_color, s=40, alpha=0.8,
                   edgecolors='black', linewidths=0.5)

        # Add horizontal line at 0
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

        # Formatting
        ax.set_xscale('log')
        ax.set_ylabel('Residual', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.set_ylim(-args.ylim, args.ylim)

        # Add label with fit info
        label = f"{opt_name}: $c$={fit['c']:.3f}, $f$={fit['f']:.3f}, $R^2$={fit['r_squared']:.6f}"
        ax.text(0.02, 0.92, label, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # X-axis label on bottom plot
    axes[-1].set_xlabel('Compute (PFH)', fontsize=11)

    # Overall title
    fig.suptitle(f'{args.model}: Broken Power Law Residuals ($6D$ Chinchilla, shared $a$={shared_a:.4f})',
                 fontsize=13, fontweight='bold', y=0.97)

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to {output_path}")

    # Also save as PNG
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {png_path}")

    plt.close()


if __name__ == '__main__':
    main()
