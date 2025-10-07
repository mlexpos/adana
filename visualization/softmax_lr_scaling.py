#!/usr/bin/env python3
"""
Learning Rate Scaling Law Analysis - Direct Power Law Fitting

This script estimates a power law fit for the optimal learning rate across model sizes
using a direct weighted fitting approach. For a fixed omega value, it finds the K best
LRs for each model size, weights them (smallest gets weight K, second gets K-1, etc.),
and fits: LR = A * (n_layer)^B using weighted MSE loss with Adagrad optimization.

Usage:
    python softmax_lr_scaling.py --optimizer DS --target-omega 4.0 --top-k 5
    python softmax_lr_scaling.py --optimizer AW --target-omega 4.0 --dataset fineweb_100 --top-k 3
"""

import wandb
import jax
import jax.numpy as jnp
from jax import grad, jit
import optax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import warnings
from scipy.optimize import curve_fit
from model_library import MODEL_CONFIGS, get_models_by_optimizer, get_model_keys_by_optimizer

warnings.filterwarnings('ignore')

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description='Fit power law for optimal LR across model sizes using direct weighted fitting')
parser.add_argument('--optimizer', type=str, required=True, choices=['DS', 'AW'],
                    help='Optimizer type: DS (DanaStar) or AW (AdamW)')
parser.add_argument('--target-omega', type=float, default=4.0,
                    help='Target omega value to find optimal LR (default: 4.0)')
parser.add_argument('--top-k', type=int, default=5,
                    help='Number of best LRs to use for each model size (default: 5)')
parser.add_argument('--dataset', type=str, default='fineweb_100',
                    help='Dataset to filter runs (default: fineweb_100)')
parser.add_argument('--project', type=str, default='danastar',
                    help='WandB project name (default: danastar)')
parser.add_argument('--output', type=str, default=None,
                    help='Output filename for plot (default: auto-generated)')
parser.add_argument('--n-steps', type=int, default=5000,
                    help='Number of optimization steps for power law fitting (default: 5000)')
parser.add_argument('--learning-rate', type=float, default=0.1,
                    help='Optimizer learning rate for power law fitting (default: 0.1)')
parser.add_argument('--exclude-small', action='store_true',
                    help='Exclude small model size (4 layers) from the fit')
args = parser.parse_args()

# Map optimizer abbreviations
optimizer_map = {'DS': 'danastar', 'AW': 'adamw'}
optimizer_type = optimizer_map[args.optimizer]

# =============================================================================
# DIRECT POWER LAW FITTING FUNCTIONS
# =============================================================================

@jit
def power_law_function(n_layer, A, B):
    """Power law function: LR = A * (n_layer)^B"""
    return A * (n_layer ** B)

def get_top_k_lrs_for_omega(data_df, target_omega, top_k=5, omega_tolerance=0.1):
    """
    Get top K LRs with smallest validation losses for a given omega value.
    Returns list of (LR, val_loss, weight) tuples where weight = K for best, K-1 for second best, etc.
    """
    # Filter data to target omega
    omega_data = data_df[np.abs(data_df['omega'] - target_omega) < omega_tolerance].copy()

    if len(omega_data) == 0:
        print(f"    No data found for omega={target_omega:.3f}")
        return []

    # Sort by validation loss and take top K
    omega_data = omega_data.nsmallest(top_k, 'val_loss')

    # Assign weights: smallest gets weight K, second smallest gets K-1, etc.
    results = []
    for rank, (idx, row) in enumerate(omega_data.iterrows()):
        weight = top_k - rank  # K, K-1, K-2, ..., 1
        results.append({
            'lr': row['lr'],
            'val_loss': row['val_loss'],
            'weight': weight
        })

    print(f"    Found {len(results)} LRs at omega≈{target_omega:.3f}")
    print(f"    LR range: {min(r['lr'] for r in results):.6f} to {max(r['lr'] for r in results):.6f}")
    print(f"    Weights: {[r['weight'] for r in results]}")

    return results

def load_wandb_data_simple(project_name, group_name, optimizer_type, dataset_filter=None):
    """Load data from WandB"""
    api = wandb.Api()

    print(f"Loading data from {group_name}...")

    runs = api.runs(f"ep-rmt-ml-opt/{project_name}", filters={"group": group_name})

    data = []
    total_runs = 0
    skipped_incomplete = 0
    skipped_dataset = 0
    skipped_missing_data = 0

    for run in runs:
        total_runs += 1
        config = run.config
        summary = run.summary

        # Filter by dataset if specified
        if dataset_filter and config.get('dataset') != dataset_filter:
            skipped_dataset += 1
            continue

        # Check if run completed (only if both fields exist)
        actual_iter = summary.get('iter', 0)
        iterations_config = config.get('iterations', 0)
        if actual_iter > 0 and iterations_config > 0 and actual_iter < iterations_config:
            skipped_incomplete += 1
            continue

        lr = config.get('lr')
        val_loss = summary.get('val/loss')

        if lr is None or val_loss is None:
            skipped_missing_data += 1
            continue

        # Calculate omega based on optimizer type
        if optimizer_type == 'danastar':
            wd_ts = config.get('wd_ts', 1.0)
            weight_decay = config.get('weight_decay', 1.0)
            omega = wd_ts * lr * weight_decay
        else:  # adamw
            weight_decay = config.get('weight_decay', 0.1)
            iterations = config.get('iterations', 1)
            omega = weight_decay * lr * iterations

        data.append({
            'lr': lr,
            'val_loss': val_loss,
            'omega': omega,
            'run_name': run.name
        })

    print(f"  Total runs: {total_runs}, Loaded: {len(data)}")

    df = pd.DataFrame(data)
    return df

# =============================================================================
# DIRECT WEIGHTED POWER LAW FITTING
# =============================================================================

def fit_power_law_weighted(n_layers_list, lrs_list, weights_list, n_steps=5000, learning_rate=0.1):
    """
    Fit power law LR = A * (n_layer)^B using weighted MSE loss with Adagrad optimization.

    Args:
        n_layers_list: List of n_layer values
        lrs_list: List of corresponding LRs
        weights_list: List of weights for each data point
        n_steps: Number of optimization steps
        learning_rate: Learning rate for Adagrad

    Returns:
        (A, B): Fitted parameters
    """
    # Convert to JAX arrays
    n_layers = jnp.array(n_layers_list, dtype=jnp.float32)
    lrs = jnp.array(lrs_list, dtype=jnp.float32)
    weights = jnp.array(weights_list, dtype=jnp.float32)

    # Take log of LRs for log-space fitting
    log_lrs = jnp.log(lrs)
    log_n_layers = jnp.log(n_layers)

    # Initialize parameters
    # log(LR) = log(A) + B * log(n_layer)
    # Initial guess: log(A) ≈ log(1e-3), B ≈ -0.5
    params = jnp.array([jnp.log(1e-3), -0.5])

    @jit
    def loss_fn(params):
        log_A, B = params

        # Predictions in log space
        # log(LR) = log(A) + B * log(n_layer)
        pred_log_lrs = log_A + B * log_n_layers

        # Weighted MSE loss in log space (multiply weights by n_layer)
        residuals = (log_lrs - pred_log_lrs) ** 2
        combined_weights = weights * n_layers
        weighted_loss = jnp.sum(combined_weights * residuals) / jnp.sum(combined_weights)

        return weighted_loss

    # Set up optimizer
    optimizer = optax.adagrad(learning_rate)
    opt_state = optimizer.init(params)

    # JIT compile gradient
    grad_fn = jit(grad(loss_fn))

    # Optimization loop
    best_loss = float('inf')
    best_params = params

    for step in range(n_steps):
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        current_loss = float(loss_fn(params))
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = params

        if step % 1000 == 0 or step == n_steps - 1:
            log_A, B = params
            A = float(jnp.exp(log_A))
            print(f"  Step {step:5d}: loss={current_loss:.6e}, A={A:.6e}, B={B:.4f}")

    # Extract final parameters
    log_A, B = best_params
    A = float(jnp.exp(log_A))
    B = float(B)

    return A, B

def collect_weighted_data_for_models(optimizer_type, target_omega, top_k, dataset, project, exclude_small=False):
    """
    Collect top-K LRs for each model size at target omega, with weights.
    Returns lists of n_layers, lrs, and weights for power law fitting.

    Args:
        exclude_small: If True, exclude small model (n_layer=4) from fitting data
    """
    model_configs = get_models_by_optimizer(optimizer_type)

    all_n_layers = []
    all_lrs = []
    all_weights = []
    all_n_layers_excluded = []  # For plotting excluded points
    all_lrs_excluded = []
    all_weights_excluded = []
    results = {}

    for model_key, model_info in model_configs.items():
        print(f"\n{'='*70}")
        print(f"Processing {model_key} (n_layer={model_info['n_layer']})")
        print(f"{'='*70}")

        # Load data for this model
        group_name = model_info['group']
        data_df = load_wandb_data_simple(project, group_name, optimizer_type, dataset)

        if len(data_df) == 0:
            print(f"No data found for {model_key}")
            continue

        # Find closest omega to target
        available_omegas = sorted(data_df['omega'].unique())
        closest_omega = min(available_omegas, key=lambda x: abs(x - target_omega))

        print(f"  Target omega: {target_omega:.2f}, Closest: {closest_omega:.2f}")

        # Get top K LRs at this omega
        top_k_data = get_top_k_lrs_for_omega(data_df, closest_omega, top_k=top_k)

        if len(top_k_data) == 0:
            continue

        # Add to global lists (or excluded lists if small model and exclude_small=True)
        n_layer = model_info['n_layer']
        is_small = (n_layer == 4)

        if exclude_small and is_small:
            print(f"  Excluding small model (n_layer=4) from fit")
            for item in top_k_data:
                all_n_layers_excluded.append(n_layer)
                all_lrs_excluded.append(item['lr'])
                all_weights_excluded.append(item['weight'])
        else:
            for item in top_k_data:
                all_n_layers.append(n_layer)
                all_lrs.append(item['lr'])
                all_weights.append(item['weight'])

        # Store for plotting
        results[model_key] = {
            'n_layer': n_layer,
            'closest_omega': closest_omega,
            'top_k_data': top_k_data,
            'data_df': data_df,
            'excluded': exclude_small and is_small
        }

    return all_n_layers, all_lrs, all_weights, all_n_layers_excluded, all_lrs_excluded, all_weights_excluded, results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print(f"Learning Rate Scaling Law Analysis - Direct Weighted Fitting")
    print(f"Optimizer: {args.optimizer} ({optimizer_type})")
    print(f"Target Omega: {args.target_omega}")
    print(f"Top K: {args.top_k}")
    print(f"Dataset: {args.dataset}")
    print(f"Exclude Small: {args.exclude_small}")
    print("="*70)

    # Collect weighted data from all model sizes
    n_layers, lrs, weights, n_layers_exc, lrs_exc, weights_exc, model_results = collect_weighted_data_for_models(
        optimizer_type=optimizer_type,
        target_omega=args.target_omega,
        top_k=args.top_k,
        dataset=args.dataset,
        project=args.project,
        exclude_small=args.exclude_small
    )

    if len(n_layers) == 0:
        print("No data collected. Exiting.")
        exit(1)

    print(f"\n{'='*70}")
    print("Collected Data Summary")
    print(f"{'='*70}")
    print(f"Total data points: {len(n_layers)}")
    print(f"Model sizes (n_layer): {sorted(set(n_layers))}")
    print(f"LR range: {min(lrs):.6e} to {max(lrs):.6e}")
    print(f"Weight range: {min(weights)} to {max(weights)}")

    # Fit power law using weighted optimization
    print(f"\n{'='*70}")
    print("Fitting Power Law: LR = A * (n_layer)^B")
    print(f"{'='*70}")
    A_fit, B_fit = fit_power_law_weighted(n_layers, lrs, weights,
                                          n_steps=args.n_steps,
                                          learning_rate=args.learning_rate)

    print(f"\nFitted parameters:")
    print(f"  A = {A_fit:.6e}")
    print(f"  B = {B_fit:.4f}")
    print(f"Power law: LR = {A_fit:.6e} * (n_layer)^{B_fit:.4f}")

    # =============================================================================
    # VISUALIZATION
    # =============================================================================

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all data points with size proportional to weight
    all_unique_n_layers = sorted(set(n_layers) | set(n_layers_exc))
    colors = cm.viridis(np.linspace(0, 1, len(all_unique_n_layers)))
    color_map = {n: colors[i] for i, n in enumerate(all_unique_n_layers)}

    # Plot included points (used in fit)
    for n, lr, w in zip(n_layers, lrs, weights):
        ax.scatter(n, lr, s=w*50, c=[color_map[n]], alpha=0.6, edgecolors='black', linewidths=0.5)

    # Plot excluded points (grayed out with 'x' marker)
    if len(n_layers_exc) > 0:
        for n, lr, w in zip(n_layers_exc, lrs_exc, weights_exc):
            ax.scatter(n, lr, s=w*50, c='gray', alpha=0.3, marker='x', linewidths=1.5,
                      label='Excluded (not in fit)' if n == n_layers_exc[0] and lr == lrs_exc[0] else '')

    # Plot power law fit line and extrapolation
    unique_n_layers = sorted(set(n_layers))
    n_layer_range = np.linspace(min(unique_n_layers), 30, 100)
    lr_fit = power_law_function(n_layer_range, A_fit, B_fit)
    ax.plot(n_layer_range, lr_fit, 'r--', linewidth=2.5,
            label=f'Power Law Fit: {A_fit:.2e} × $n_{{layer}}^{{{B_fit:.3f}}}$', zorder=10)

    # Mark and annotate predictions at specific n_layer values
    prediction_points = [15, 18, 24, 30]
    for n_pred in prediction_points:
        lr_pred = float(power_law_function(n_pred, A_fit, B_fit))
        ax.scatter([n_pred], [lr_pred], s=150, marker='D', c='red',
                  edgecolors='black', linewidths=1.5, zorder=11)
        ax.text(n_pred, lr_pred * 1.15, f'n={n_pred}\nLR={lr_pred:.2e}',
               ha='center', va='bottom', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Formatting
    ax.set_xlabel('Number of Layers (n_layer)', fontsize=12)
    ax.set_ylabel('Learning Rate (LR)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(f'{args.optimizer} Optimal LR Scaling Law (Direct Weighted Fit)\n(Target ω = {args.target_omega}, Top-K = {args.top_k}, Dataset = {args.dataset})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add text annotation showing point size = weight
    ax.text(0.02, 0.98, 'Point size ∝ weight\n(best LR has largest weight)',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    if args.output:
        output_file = args.output
    else:
        # Map optimizer type to algorithm name
        alg_name_map = {'danastar': 'DanaStar', 'adamw': 'AdamW'}
        alg_name = alg_name_map.get(optimizer_type, optimizer_type)
        output_file = f'{alg_name}-lr-extrapolation.pdf'

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Print extrapolations
    print(f"\nExtrapolated optimal LRs:")
    for n_pred in [15, 18, 24, 30]:
        lr_pred = float(power_law_function(n_pred, A_fit, B_fit))
        print(f"  n_layer={n_pred}: LR={lr_pred:.6e}")

    plt.show()
