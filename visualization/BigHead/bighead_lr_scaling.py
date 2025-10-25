#!/usr/bin/env python3
"""
BigHead Learning Rate Scaling Law Analysis - Direct Power Law Fitting

This script estimates a power law fit for the optimal learning rate across BigHead model sizes
using a direct weighted fitting approach. For a fixed omega value, it finds the K best
LRs for each model depth, weights them (smallest gets weight K, second gets K-1, etc.),
and fits: LR = A * (parameter_metric)^B using weighted MSE loss with Adagrad optimization.

BigHead Architecture:
- head_dim = 16 * depth
- n_embd = 16 * depth^2
- mlp_hidden = 32 * depth^2
- n_head = depth
- n_layer = depth

Usage:
    python bighead_lr_scaling.py --optimizer adamw --target-omega 4.0 --top-k 5
    python bighead_lr_scaling.py --optimizer mk4 --target-omega 4.0 --top-k 5 --target-clipsnr 1.0
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
import matplotlib.font_manager as fm
import matplotlib as mpl
from matplotlib import style
from matplotlib import rc, rcParams
import argparse
import warnings
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

# Matplotlib formatting
style.use('default')
rc('font', family='sans-serif')
rcParams['font.weight'] = 'light'
rcParams['font.size'] = 20
rcParams['figure.figsize'] = (1 * 10.0, 1 * 8.0)

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description='Fit power law for optimal LR across BigHead model depths')
parser.add_argument('--optimizer', type=str, required=True, choices=['adamw', 'mk4', 'dana', 'ademamix'],
                    help='Optimizer type: adamw, mk4 (dana-star-mk4), dana, or ademamix')
parser.add_argument('--target-omega', type=float, default=4.0,
                    help='Target omega value to find optimal LR (default: 4.0)')
parser.add_argument('--top-k', type=int, default=5,
                    help='Number of best LRs to use for each model size (default: 5)')
parser.add_argument('--project', type=str, default='danastar',
                    help='WandB project name (default: danastar)')
parser.add_argument('--group', type=str, default='DanaStar_MK4_BigHead_Sweep',
                    help='WandB group name (default: DanaStar_MK4_BigHead_Sweep)')
parser.add_argument('--entity', type=str, default='ep-rmt-ml-opt',
                    help='WandB entity name (default: ep-rmt-ml-opt)')
parser.add_argument('--output', type=str, default=None,
                    help='Output filename for plot (default: auto-generated)')
parser.add_argument('--n-steps', type=int, default=100000,
                    help='Number of optimization steps for power law fitting (default: 100000)')
parser.add_argument('--learning-rate', type=float, default=100.0,
                    help='Optimizer learning rate for power law fitting (default: 100.0)')
parser.add_argument('--exclude-small', action='store_true',
                    help='Exclude small model size (4 layers) from the fit')
parser.add_argument('--fit-total-params', action='store_true',
                    help='Include fit using total parameters (default: off)')
parser.add_argument('--fit-compute', action='store_true',
                    help='Include fit using compute metric (default: off)')
parser.add_argument('--target-clipsnr', type=float, default=None,
                    help='Target clipsnr value for MK4 optimizer (filters runs within tolerance, default: None)')
parser.add_argument('--clipsnr-tolerance', type=float, default=0.1,
                    help='Tolerance for clipsnr matching (default: 0.1)')
args = parser.parse_args()

# Map optimizer abbreviations
optimizer_map = {'adamw': 'adamw', 'mk4': 'dana-star-mk4', 'dana': 'dana', 'ademamix': 'ademamix'}
optimizer_type = optimizer_map[args.optimizer]

# =============================================================================
# PARAMETER CALCULATION FUNCTIONS (Based on BigHead.sh)
# =============================================================================

def compute_non_embedding_params(depth):
    """
    Compute non-embedding parameters based on BigHead architecture.

    From BigHead.sh:
    head_dim = 16 * depth
    n_embd = 16 * depth^2
    mlp_hidden = 32 * depth^2
    n_head = depth
    n_layer = depth

    Non-emb = depth * (3 * head_dim * n_embd * n_head + n_embd^2 + 2 * n_embd * mlp + 8 * n_embd) + 2 * n_embd
    """
    head_dim = 16 * depth
    n_embd = 16 * depth * depth
    mlp_hidden = 32 * depth * depth
    n_head = depth
    n_layer = depth

    non_emb = depth * (3 * head_dim * n_embd * n_head + n_embd * n_embd + 2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd
    return int(non_emb)

def compute_total_params(depth):
    """
    Compute total parameters including embeddings.
    Total params = non_emb + 2 * n_embd * 50304
    """
    non_emb = compute_non_embedding_params(depth)
    n_embd = 16 * depth * depth
    vocab_size = 50304
    total_params = non_emb + 2 * n_embd * vocab_size
    return int(total_params)

def compute_compute(depth):
    """
    Compute compute metric: non_emb * total_params * 20 
    (Based on the iterations calculation: ITERATIONS = 20 * TOTAL_PARAMS )
    """
    non_emb = compute_non_embedding_params(depth)
    total_params = compute_total_params(depth)
    compute = non_emb * total_params * 20 
    return compute

@jit
def power_law_function(params, A, B):
    """Power law function: LR = A * (params)^B"""
    params_float = jnp.asarray(params, dtype=jnp.float32)
    return A * (params_float ** B)

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

def load_wandb_data_simple(project_name, group_name, entity, optimizer_type,
                           target_clipsnr=None, clipsnr_tolerance=0.1):
    """Load data from WandB"""
    api = wandb.Api()

    print(f"Loading data from {group_name}...")

    runs = api.runs(f"{entity}/{project_name}", filters={"group": group_name})

    data = []
    total_runs = 0
    skipped_optimizer = 0
    skipped_incomplete = 0
    skipped_missing_data = 0
    skipped_clipsnr = 0

    for run in runs:
        total_runs += 1
        config = run.config
        summary = run.summary

        # Filter by optimizer (stored as 'opt' in config)
        opt = config.get('opt', '')
        if opt != optimizer_type:
            skipped_optimizer += 1
            continue

        # Filter by clipsnr if specified (for MK4 optimizer)
        if target_clipsnr is not None:
            clipsnr = config.get('clipsnr')
            if clipsnr is None or abs(clipsnr - target_clipsnr) > clipsnr_tolerance:
                skipped_clipsnr += 1
                continue

        # Check if run completed
        actual_iter = summary.get('iter', 0)
        iterations_config = config.get('iterations', 0)
        if actual_iter > 0 and iterations_config > 0 and actual_iter < iterations_config:
            skipped_incomplete += 1
            continue

        lr = config.get('lr')
        val_loss = summary.get('final-val/loss')

        if lr is None or val_loss is None:
            skipped_missing_data += 1
            continue

        # Extract depth (stored as n_layer)
        depth = config.get('n_layer')
        if depth is None:
            skipped_missing_data += 1
            continue

        # Calculate omega based on optimizer type
        if optimizer_type in ['dana-star-mk4', 'dana']:
            wd_ts = config.get('wd_ts', 1.0)
            weight_decay = config.get('weight_decay', 1.0)
            omega = wd_ts * lr * weight_decay
        else:  # adamw, ademamix
            weight_decay = config.get('weight_decay', 0.1)
            iterations = config.get('iterations', 1)
            omega = weight_decay * lr * iterations

        data.append({
            'depth': depth,
            'lr': lr,
            'val_loss': val_loss,
            'omega': omega,
            'run_name': run.name
        })

    print(f"  Total runs: {total_runs}, Loaded: {len(data)}")
    if skipped_optimizer > 0:
        print(f"  Skipped {skipped_optimizer} runs due to optimizer filter")
    if skipped_clipsnr > 0:
        print(f"  Skipped {skipped_clipsnr} runs due to clipsnr filter")
    if skipped_incomplete > 0:
        print(f"  Skipped {skipped_incomplete} incomplete runs")

    df = pd.DataFrame(data)
    return df

# =============================================================================
# DIRECT WEIGHTED POWER LAW FITTING
# =============================================================================

def fit_power_law_weighted(params_list, lrs_list, weights_list, n_steps=5000, learning_rate=0.1):
    """
    Fit power law LR = A * (params)^B using weighted MSE loss with Adagrad optimization.
    """
    # Convert to JAX arrays
    params_arr = jnp.array(params_list, dtype=jnp.float32)
    lrs = jnp.array(lrs_list, dtype=jnp.float32)
    weights = jnp.array(weights_list, dtype=jnp.float32)

    # Take log of LRs for log-space fitting
    log_lrs = jnp.log(lrs)
    log_params = jnp.log(params_arr)

    # Initialize parameters
    # log(LR) = log(A) + B * log(params)
    # Initial guess: log(A) ≈ log(1e-3), B ≈ -0.5
    fit_params = jnp.array([jnp.log(1e-3), -0.5])

    @jit
    def loss_fn(fit_params):
        log_A, B = fit_params

        # Predictions in log space
        pred_log_lrs = log_A + B * log_params

        # Weighted MSE loss in log space (multiply weights by params)
        residuals = (log_lrs - pred_log_lrs) ** 2
        combined_weights = weights * params_arr
        weighted_loss = jnp.sum(combined_weights * residuals) / jnp.sum(combined_weights)

        return weighted_loss

    # Set up optimizer
    optimizer = optax.adagrad(learning_rate)
    opt_state = optimizer.init(fit_params)

    # JIT compile gradient
    grad_fn = jit(grad(loss_fn))

    # Optimization loop
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

        if step % 10000 == 0 or step == n_steps - 1:
            log_A, B = fit_params
            A = float(jnp.exp(log_A))
            print(f"  Step {step:5d}: loss={current_loss:.6e}, A={A:.6e}, B={B:.4f}")

    # Extract final parameters
    log_A, B = best_params
    A = float(jnp.exp(log_A))
    B = float(B)

    return A, B

def collect_weighted_data_for_depths(optimizer_type, target_omega, top_k, project, group, entity,
                                      exclude_small=False, target_clipsnr=None, clipsnr_tolerance=0.1):
    """
    Collect top-K LRs for each depth at target omega, with weights.
    """
    # Load all data for the optimizer
    data_df = load_wandb_data_simple(project, group, entity, optimizer_type,
                                     target_clipsnr, clipsnr_tolerance)

    if len(data_df) == 0:
        print("No data found. Exiting.")
        return None

    # Get unique depths
    available_depths = sorted(data_df['depth'].unique())
    print(f"\nAvailable depths: {available_depths}")

    all_non_emb_params = []
    all_total_params = []
    all_compute = []
    all_lrs = []
    all_weights = []
    all_non_emb_params_excluded = []
    all_total_params_excluded = []
    all_compute_excluded = []
    all_lrs_excluded = []
    all_weights_excluded = []
    results = {}

    for depth in available_depths:
        print(f"\n{'='*70}")
        print(f"Processing depth={depth}")
        print(f"{'='*70}")

        # Filter data for this depth
        depth_data = data_df[data_df['depth'] == depth].copy()

        if len(depth_data) == 0:
            print(f"No data found for depth={depth}")
            continue

        # Find closest omega to target
        available_omegas = sorted(depth_data['omega'].unique())
        closest_omega = min(available_omegas, key=lambda x: abs(x - target_omega))

        print(f"  Target omega: {target_omega:.2f}, Closest: {closest_omega:.2f}")

        # Get top K LRs at this omega
        top_k_data = get_top_k_lrs_for_omega(depth_data, closest_omega, top_k=top_k)

        if len(top_k_data) == 0:
            continue

        # Compute parameter counts
        non_emb_params = compute_non_embedding_params(depth)
        total_params = compute_total_params(depth)
        compute_metric = compute_compute(depth)

        print(f"  Non-embedding params: {non_emb_params:,}")
        print(f"  Total params: {total_params:,}")
        print(f"  Compute: {compute_metric:.2e}")

        # Add to global lists (or excluded lists if small model and exclude_small=True)
        is_small = (depth == 4)

        if exclude_small and is_small:
            print(f"  Excluding depth=4 from fit")
            for item in top_k_data:
                all_non_emb_params_excluded.append(non_emb_params)
                all_total_params_excluded.append(total_params)
                all_compute_excluded.append(compute_metric)
                all_lrs_excluded.append(item['lr'])
                all_weights_excluded.append(item['weight'])
        else:
            for item in top_k_data:
                all_non_emb_params.append(non_emb_params)
                all_total_params.append(total_params)
                all_compute.append(compute_metric)
                all_lrs.append(item['lr'])
                all_weights.append(item['weight'])

        # Store for plotting
        results[depth] = {
            'non_emb_params': non_emb_params,
            'total_params': total_params,
            'compute': compute_metric,
            'closest_omega': closest_omega,
            'top_k_data': top_k_data,
            'data_df': depth_data,
            'excluded': exclude_small and is_small
        }

    return (all_non_emb_params, all_total_params, all_compute, all_lrs, all_weights,
            all_non_emb_params_excluded, all_total_params_excluded, all_compute_excluded,
            all_lrs_excluded, all_weights_excluded, results)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print(f"BigHead Learning Rate Scaling Law Analysis")
    print(f"Optimizer: {args.optimizer} ({optimizer_type})")
    print(f"Target Omega: {args.target_omega}")
    print(f"Top K: {args.top_k}")
    print(f"Group: {args.group}")
    print(f"Exclude Small: {args.exclude_small}")
    if args.target_clipsnr is not None:
        print(f"Target ClipSNR: {args.target_clipsnr} (tolerance: {args.clipsnr_tolerance})")
    print("="*70)

    # Collect weighted data from all depths
    result = collect_weighted_data_for_depths(
        optimizer_type=optimizer_type,
        target_omega=args.target_omega,
        top_k=args.top_k,
        project=args.project,
        group=args.group,
        entity=args.entity,
        exclude_small=args.exclude_small,
        target_clipsnr=args.target_clipsnr,
        clipsnr_tolerance=args.clipsnr_tolerance
    )

    if result is None:
        exit(1)

    (non_emb_params, total_params, compute, lrs, weights,
     non_emb_params_exc, total_params_exc, compute_exc, lrs_exc, weights_exc, model_results) = result

    if len(non_emb_params) == 0:
        print("No data collected. Exiting.")
        exit(1)

    print(f"\n{'='*70}")
    print("Collected Data Summary")
    print(f"{'='*70}")
    print(f"Total data points: {len(non_emb_params)}")
    print(f"Non-embedding params range: {min(non_emb_params):,} to {max(non_emb_params):,}")
    print(f"Total params range: {min(total_params):,} to {max(total_params):,}")
    print(f"Compute range: {min(compute):.2e} to {max(compute):.2e}")
    print(f"LR range: {min(lrs):.6e} to {max(lrs):.6e}")
    print(f"Weight range: {min(weights)} to {max(weights)}")

    # Fit power law using non-embedding parameters (always on)
    print(f"\n{'='*70}")
    print("Fitting Power Law: LR = A * (non_emb_params)^B")
    print(f"{'='*70}")
    A_fit, B_fit = fit_power_law_weighted(non_emb_params, lrs, weights,
                                           n_steps=args.n_steps,
                                           learning_rate=args.learning_rate)

    print(f"\nFitted parameters (non-embedding):")
    print(f"  A = {A_fit:.6e}")
    print(f"  B = {B_fit:.4f}")
    print(f"Power law: LR = {A_fit:.6e} * (non_emb_params)^{B_fit:.4f}")

    # =============================================================================
    # VISUALIZATION
    # =============================================================================

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot all data points with size proportional to weight
    all_unique_params = sorted(set(non_emb_params) | set(non_emb_params_exc))
    colors = cm.viridis(np.linspace(0, 1, len(all_unique_params)))
    color_map = {p: colors[i] for i, p in enumerate(all_unique_params)}

    # Plot included points (used in fit)
    for p, lr, w in zip(non_emb_params, lrs, weights):
        ax.scatter(p, lr, s=w*50, c=[color_map[p]], alpha=0.6, edgecolors='black', linewidths=0.5)

    # Plot excluded points (grayed out with 'x' marker)
    if len(non_emb_params_exc) > 0:
        for p, lr, w in zip(non_emb_params_exc, lrs_exc, weights_exc):
            ax.scatter(p, lr, s=w*50, c='gray', alpha=0.3, marker='x', linewidths=1.5,
                      label='Excluded (not in fit)' if p == non_emb_params_exc[0] and lr == lrs_exc[0] else '')

    # Plot power law fit line and extrapolation
    unique_params = sorted(set(non_emb_params))
    # Extrapolate to depth 12
    max_depth = 12
    max_non_emb = compute_non_embedding_params(max_depth)
    params_range = np.linspace(min(unique_params) * 0.5, max_non_emb, 200)

    lr_fit = power_law_function(params_range, A_fit, B_fit)
    ax.plot(params_range, lr_fit, '--', color='tab:orange', linewidth=3,
            label=f'Power law: {A_fit:.2e} × $P^{{{B_fit:.3f}}}$', zorder=10)

    # Mark predictions at specific depths
    prediction_depths = [8, 9, 10, 11, 12]

    for i, depth_pred in enumerate(prediction_depths):
        non_emb_pred = float(compute_non_embedding_params(depth_pred))
        lr_pred = float(power_law_function(non_emb_pred, A_fit, B_fit))

        ax.scatter([non_emb_pred], [lr_pred], s=150, marker='D', c='tab:orange',
                  edgecolors='black', linewidths=1.5, zorder=11)

        vertical_offset = 1.5 if i != 1 else 1.3
        ax.text(non_emb_pred, lr_pred * vertical_offset, f'Depth={depth_pred}\nLR={lr_pred:.2e}',
               ha='left', va='bottom', fontsize=15,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='tab:orange', alpha=0.2))

    # Formatting
    ax.set_xlabel('Non-embedding Parameters', fontsize=20)
    ax.set_ylabel('Learning Rate (LR)', fontsize=20)
    ax.set_xscale('log')
    ax.set_yscale('log')

    optimizer_title_map = {'adamw': 'AdamW', 'mk4': 'Dana-Star-MK4', 'dana': 'Dana-Star', 'ademamix': 'AdemaMix'}
    optimizer_title = optimizer_title_map[args.optimizer]

    title_parts = [f'ω = {args.target_omega}', f'Top-K = {args.top_k}']
    if args.target_clipsnr is not None:
        title_parts.append(f'ClipSNR = {args.target_clipsnr}')
    title_params = ', '.join(title_parts)

    ax.set_title(f'BigHead {optimizer_title} Optimal Learning Rate Scaling Law\n({title_params})',
                 fontsize=20, fontweight='bold')
    ax.legend(fontsize=15, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.text(0.02, 0.02, 'Point size ∝ weight\n(best LR has largest weight)',
            transform=ax.transAxes, fontsize=15, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    if args.output:
        output_file = args.output
    else:
        optimizer_filename_map = {'adamw': 'AdamW', 'mk4': 'DanaStar-MK4', 'dana': 'DanaStar', 'ademamix': 'AdemaMix'}
        optimizer_name = optimizer_filename_map[args.optimizer]
        output_file = f'BigHead-{optimizer_name}-lr-extrapolation.pdf'

    import os
    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    print(f"\nPlot saved to: {os.path.abspath(output_file)}")

    # Print extrapolations
    print(f"\nExtrapolated optimal LRs for BigHead architecture:")
    for depth_pred in prediction_depths:
        non_emb_pred = float(compute_non_embedding_params(depth_pred))
        lr_pred = float(power_law_function(non_emb_pred, A_fit, B_fit))
        print(f"  Depth={depth_pred}: LR={lr_pred:.6e}")
