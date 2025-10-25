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
    python softmax_lr_scaling.py --optimizer AS --target-omega 4.0 --top-k 5
    python softmax_lr_scaling.py --optimizer MK4 --target-omega 4.0 --top-k 5 --target-clipsnr 5.0
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
from model_library import MODEL_CONFIGS, get_models_by_optimizer, get_model_keys_by_optimizer

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

parser = argparse.ArgumentParser(description='Fit power law for optimal LR across model sizes using direct weighted fitting')
parser.add_argument('--optimizer', type=str, required=True, choices=['DS', 'AW', 'AS', 'MK4'],
                    help='Optimizer type: DS (DanaStar), AW (AdamW), AS (AdamStar), or MK4 (DanaStar-MK4)')
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
parser.add_argument('--n-steps', type=int, default=100000,
                    help='Number of optimization steps for power law fitting (default: 5000)')
parser.add_argument('--learning-rate', type=float, default=100.0,
                    help='Optimizer learning rate for power law fitting (default: 0.1)')
parser.add_argument('--exclude-small', action='store_true',
                    help='Exclude small model size (4 layers) from the fit')
parser.add_argument('--fit-total-params', action='store_true',
                    help='Include fit using total non-normalization parameters (default: off)')
parser.add_argument('--fit-compute', action='store_true',
                    help='Include fit using compute: (non-emb)*(total_params)*20/(32*2048) (default: off)')
parser.add_argument('--target-clipsnr', type=float, default=None,
                    help='Target clipsnr value for MK4 optimizer (filters runs within tolerance, default: None - no filtering)')
parser.add_argument('--clipsnr-tolerance', type=float, default=0.1,
                    help='Tolerance for clipsnr matching (default: 0.1)')
args = parser.parse_args()

# Map optimizer abbreviations
optimizer_map = {'DS': 'danastar', 'AW': 'adamw', 'AS': 'adamstar', 'MK4': 'mk4'}
optimizer_type = optimizer_map[args.optimizer]

# =============================================================================
# PARAMETER CALCULATION FUNCTIONS (Based on DiLoCoAttention)
# =============================================================================

def compute_non_embedding_params(n_head, qkv_dim, n_layer):
    """
    Compute non-embedding parameters based on DiLoCoAttention and DiLoCoMLP.

    (These decisions depend on how the MLP hidden dimension being 4x embed and head*qkv = embed.)

    qkv -- 3
    proj -- 1
    mlp -- 2 x 4
    Formula: ( (3+1+8) * (n_head × qkv_dim)² +  ) × n_layer
    """
    n_embd = n_head * qkv_dim
    return 12 * n_embd * n_embd * n_layer

def compute_total_non_norm_params(n_head, qkv_dim, n_layer):
    """
    Compute total non-normalization parameters including embeddings.

    Adds:
    - Token embeddings: vocab_size × n_embd (50340 × n_head × qkv_dim)
    - LM head: n_embd × vocab_size (n_head × qkv_dim × 50340)
    - Total embeddings: 2 × (n_head × qkv_dim × 50340)
    """
    non_emb = compute_non_embedding_params(n_head, qkv_dim, n_layer)
    n_embd = n_head * qkv_dim
    vocab_size = 50340
    embeddings = n_embd * 2 * vocab_size
    return non_emb + embeddings

def compute_compute(n_head, qkv_dim, n_layer):
    """Compute compute metric: (non-emb) * (total_params) * 20 / (32 * 2048)"""
    non_emb = compute_non_embedding_params(n_head, qkv_dim, n_layer)
    total_params = compute_total_non_norm_params(n_head, qkv_dim, n_layer)
    return 6*non_emb * total_params * 20 / (32 * 2048)

@jit
def power_law_function(params, A, B):
    """Power law function: LR = A * (params)^B"""
    # Ensure params is float to avoid int32 overflow
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

def load_wandb_data_simple(project_name, group_name, optimizer_type, dataset_filter=None,
                           target_clipsnr=None, clipsnr_tolerance=0.1):
    """Load data from WandB"""
    api = wandb.Api()

    print(f"Loading data from {group_name}...")

    runs = api.runs(f"ep-rmt-ml-opt/{project_name}", filters={"group": group_name})

    data = []
    total_runs = 0
    skipped_incomplete = 0
    skipped_dataset = 0
    skipped_missing_data = 0
    skipped_clipsnr = 0

    for run in runs:
        total_runs += 1
        config = run.config
        summary = run.summary

        # Filter by dataset if specified
        if dataset_filter and config.get('dataset') != dataset_filter:
            skipped_dataset += 1
            continue

        # Filter by clipsnr if specified (for MK4 optimizer)
        if target_clipsnr is not None:
            clipsnr = config.get('clipsnr')
            if clipsnr is None or abs(clipsnr - target_clipsnr) > clipsnr_tolerance:
                skipped_clipsnr += 1
                continue

        # Check if run completed (only if both fields exist)
        actual_iter = summary.get('iter', 0)
        iterations_config = config.get('iterations', 0)
        if actual_iter > 0 and iterations_config > 0 and actual_iter < iterations_config:
            skipped_incomplete += 1
            continue

        lr = config.get('lr')
        # Use final-val/loss instead of val/loss
        val_loss = summary.get('final-val/loss')

        if lr is None or val_loss is None:
            skipped_missing_data += 1
            continue

        # Calculate omega based on optimizer type
        if optimizer_type in ['danastar', 'adamstar', 'mk4']:
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
    if skipped_clipsnr > 0:
        print(f"  Skipped {skipped_clipsnr} runs due to clipsnr filter")

    df = pd.DataFrame(data)
    return df

# =============================================================================
# DIRECT WEIGHTED POWER LAW FITTING
# =============================================================================

def fit_power_law_weighted(params_list, lrs_list, weights_list, n_steps=5000, learning_rate=0.1):
    """
    Fit power law LR = A * (params)^B using weighted MSE loss with Adagrad optimization.

    Args:
        params_list: List of parameter count values (non-embedding or total non-norm)
        lrs_list: List of corresponding LRs
        weights_list: List of weights for each data point
        n_steps: Number of optimization steps
        learning_rate: Learning rate for Adagrad

    Returns:
        (A, B): Fitted parameters
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
        # log(LR) = log(A) + B * log(params)
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

        if step % 1000 == 0 or step == n_steps - 1:
            log_A, B = fit_params
            A = float(jnp.exp(log_A))
            print(f"  Step {step:5d}: loss={current_loss:.6e}, A={A:.6e}, B={B:.4f}")

    # Extract final parameters
    log_A, B = best_params
    A = float(jnp.exp(log_A))
    B = float(B)

    return A, B

def collect_weighted_data_for_models(optimizer_type, target_omega, top_k, dataset, project, exclude_small=False,
                                     target_clipsnr=None, clipsnr_tolerance=0.1):
    """
    Collect top-K LRs for each model size at target omega, with weights.
    Returns lists of non-embedding params, total non-norm params, compute, lrs, and weights for power law fitting.

    Args:
        exclude_small: If True, exclude small model (n_layer=4) from fitting data
        target_clipsnr: If specified, filter runs by clipsnr value (for MK4 optimizer)
        clipsnr_tolerance: Tolerance for clipsnr matching
    """
    model_configs = get_models_by_optimizer(optimizer_type)

    all_non_emb_params = []
    all_total_params = []
    all_compute = []
    all_lrs = []
    all_weights = []
    all_non_emb_params_excluded = []  # For plotting excluded points
    all_total_params_excluded = []
    all_compute_excluded = []
    all_lrs_excluded = []
    all_weights_excluded = []
    results = {}

    for model_key, model_info in model_configs.items():
        print(f"\n{'='*70}")
        print(f"Processing {model_key} (n_layer={model_info['n_layer']})")
        print(f"{'='*70}")

        # Load data for this model
        group_name = model_info['group']
        data_df = load_wandb_data_simple(project, group_name, optimizer_type, dataset,
                                         target_clipsnr, clipsnr_tolerance)

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

        # Compute parameter counts
        n_layer = model_info['n_layer']
        n_head = model_info['n_head']
        qkv_dim = model_info['qkv_dim']
        non_emb_params = compute_non_embedding_params(n_head, qkv_dim, n_layer)
        total_params = compute_total_non_norm_params(n_head, qkv_dim, n_layer)
        compute_metric = compute_compute(n_head, qkv_dim, n_layer)

        print(f"  Non-embedding params: {non_emb_params:,}")
        print(f"  Total non-norm params: {total_params:,}")
        print(f"  Compute: {compute_metric:.2e}")

        # Add to global lists (or excluded lists if small model and exclude_small=True)
        is_small = (n_layer == 4)

        if exclude_small and is_small:
            print(f"  Excluding small model (n_layer=4) from fit")
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
        results[model_key] = {
            'n_layer': n_layer,
            'n_head': n_head,
            'qkv_dim': qkv_dim,
            'non_emb_params': non_emb_params,
            'total_params': total_params,
            'compute': compute_metric,
            'closest_omega': closest_omega,
            'top_k_data': top_k_data,
            'data_df': data_df,
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
    print(f"Learning Rate Scaling Law Analysis - Direct Weighted Fitting")
    print(f"Optimizer: {args.optimizer} ({optimizer_type})")
    print(f"Target Omega: {args.target_omega}")
    print(f"Top K: {args.top_k}")
    print(f"Dataset: {args.dataset}")
    print(f"Exclude Small: {args.exclude_small}")
    if args.target_clipsnr is not None:
        print(f"Target ClipSNR: {args.target_clipsnr} (tolerance: {args.clipsnr_tolerance})")
    print("="*70)

    # Collect weighted data from all model sizes
    (non_emb_params, total_params, compute, lrs, weights,
     non_emb_params_exc, total_params_exc, compute_exc, lrs_exc, weights_exc, model_results) = collect_weighted_data_for_models(
        optimizer_type=optimizer_type,
        target_omega=args.target_omega,
        top_k=args.top_k,
        dataset=args.dataset,
        project=args.project,
        exclude_small=args.exclude_small,
        target_clipsnr=args.target_clipsnr,
        clipsnr_tolerance=args.clipsnr_tolerance
    )

    if len(non_emb_params) == 0:
        print("No data collected. Exiting.")
        exit(1)

    print(f"\n{'='*70}")
    print("Collected Data Summary")
    print(f"{'='*70}")
    print(f"Total data points: {len(non_emb_params)}")
    print(f"Non-embedding params range: {min(non_emb_params):,} to {max(non_emb_params):,}")
    print(f"Total non-norm params range: {min(total_params):,} to {max(total_params):,}")
    print(f"Compute range: {min(compute):.2e} to {max(compute):.2e}")
    print(f"LR range: {min(lrs):.6e} to {max(lrs):.6e}")
    print(f"Weight range: {min(weights)} to {max(weights)}")

    # Fit power law using non-embedding parameters (always on)
    print(f"\n{'='*70}")
    print("Fitting Power Law 1: LR = A * (non_emb_params)^B")
    print(f"{'='*70}")
    A_fit1, B_fit1 = fit_power_law_weighted(non_emb_params, lrs, weights,
                                             n_steps=args.n_steps,
                                             learning_rate=args.learning_rate)

    print(f"\nFitted parameters (non-embedding):")
    print(f"  A = {A_fit1:.6e}")
    print(f"  B = {B_fit1:.4f}")
    print(f"Power law: LR = {A_fit1:.6e} * (non_emb_params)^{B_fit1:.4f}")

    # Optionally fit power law using total non-norm parameters
    A_fit2, B_fit2 = None, None
    if args.fit_total_params:
        print(f"\n{'='*70}")
        print("Fitting Power Law 2: LR = A * (total_non_norm_params)^B")
        print(f"{'='*70}")
        A_fit2, B_fit2 = fit_power_law_weighted(total_params, lrs, weights,
                                                 n_steps=args.n_steps,
                                                 learning_rate=args.learning_rate)

        print(f"\nFitted parameters (total non-norm):")
        print(f"  A = {A_fit2:.6e}")
        print(f"  B = {B_fit2:.4f}")
        print(f"Power law: LR = {A_fit2:.6e} * (total_non_norm_params)^{B_fit2:.4f}")

    # Optionally fit power law using compute
    A_fit3, B_fit3 = None, None
    if args.fit_compute:
        print(f"\n{'='*70}")
        print("Fitting Power Law 3: LR = A * (compute)^B")
        print(f"{'='*70}")
        A_fit3, B_fit3 = fit_power_law_weighted(compute, lrs, weights,
                                                 n_steps=args.n_steps,
                                                 learning_rate=args.learning_rate)

        print(f"\nFitted parameters (compute):")
        print(f"  A = {A_fit3:.6e}")
        print(f"  B = {B_fit3:.4f}")
        print(f"Power law: LR = {A_fit3:.6e} * (compute)^{B_fit3:.4f}")

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

    # Plot power law fit lines and extrapolation
    unique_params = sorted(set(non_emb_params))
    # Stop at L=15, H=20
    max_n_layer = 15
    max_n_head = 20
    max_qkv_dim = 64
    max_non_emb = compute_non_embedding_params(max_n_head, max_qkv_dim, max_n_layer)
    params_range = np.linspace(min(unique_params) * 0.5, max_non_emb, 200)

    # Fit 1: Non-embedding parameters (always shown)
    lr_fit1 = power_law_function(params_range, A_fit1, B_fit1)
    ax.plot(params_range, lr_fit1, '--', color='tab:orange', linewidth=3,
            label=f'Fit 1 (Non-emb): {A_fit1:.2e} × $P^{{{B_fit1:.3f}}}$', zorder=10)

    # Fit 2: Total non-norm parameters (optional)
    # Note: For total params fit, we need to plot it correctly by mapping total params to their
    # corresponding non-embedding params for the x-axis
    if args.fit_total_params and A_fit2 is not None:
        # Create arrays of (non_emb, total_params) pairs from the actual data
        total_to_nonemb = []
        for model_key, model_info in model_results.items():
            if not model_info.get('excluded', False):
                total_to_nonemb.append((model_info['total_params'], model_info['non_emb_params']))

        # Sort by total params value
        total_to_nonemb.sort()

        # For plotting, use total params values to predict LR, then plot at corresponding non_emb x values
        nonemb_vals = [x[1] for x in total_to_nonemb]
        total_vals = [x[0] for x in total_to_nonemb]
        lr_vals = [float(power_law_function(t, A_fit2, B_fit2)) for t in total_vals]
        ax.plot(nonemb_vals, lr_vals, 'b:', linewidth=2.5,
                label=f'Fit 2 (Total non-norm): {A_fit2:.2e} × $P^{{{B_fit2:.3f}}}$', zorder=10)

    # Fit 3: Compute (optional)
    # Note: For compute fit, we need to plot it correctly by mapping compute values to their
    # corresponding non-embedding params for the x-axis
    if args.fit_compute and A_fit3 is not None:
        # Create arrays of (non_emb, compute) pairs from the actual data
        # For each unique model size, compute the relationship
        compute_to_nonemb = []
        for model_key, model_info in model_results.items():
            if not model_info.get('excluded', False):
                compute_to_nonemb.append((model_info['compute'], model_info['non_emb_params']))

        # Sort by compute value
        compute_to_nonemb.sort()

        # For plotting, we'll use the compute values to predict LR, then plot at corresponding non_emb x values
        # This is the correct way since compute and non_emb are related through the model architecture
        # We'll plot points along the curve
        for comp, nonemb in compute_to_nonemb:
            lr_pred = float(power_law_function(comp, A_fit3, B_fit3))
            ax.plot(nonemb, lr_pred, 'o', color='tab:green', markersize=8, zorder=9)

        # Draw a line stopping at L=15, H=20
        # Compute the max compute value for L=15, H=20
        max_compute = compute_compute(max_n_head, max_qkv_dim, max_n_layer)
        compute_range = np.linspace(min(compute) * 0.5, max_compute, 200)
        lr_fit3 = power_law_function(compute_range, A_fit3, B_fit3)

        # For x-axis, we need to map compute back to non_emb_params
        # Use the relationship from model configs to extrapolate
        # Compute = non_emb * total_params * 20 / (32 * 2048)
        # For consistent extrapolation, plot directly using compute_range mapped to equivalent non_emb
        # Get the scaling relationship from existing data
        if len(compute_to_nonemb) >= 2:
            # Linear relationship in log space for extrapolation
            log_comp = np.log([x[0] for x in compute_to_nonemb])
            log_nonemb = np.log([x[1] for x in compute_to_nonemb])
            # Fit line: log(nonemb) = a * log(comp) + b
            from numpy.polynomial import Polynomial
            p = Polynomial.fit(log_comp, log_nonemb, 1)
            log_nonemb_extended = p(np.log(compute_range))
            nonemb_extended = np.exp(log_nonemb_extended)

            ax.plot(nonemb_extended, lr_fit3, '-.', color='tab:green', linewidth=3,
                    label=f'Fit 3 (Compute): {A_fit3:.2e} × $C^{{{B_fit3:.3f}}}$', zorder=10)
        else:
            # Fallback to original method if not enough data points
            nonemb_vals = [x[1] for x in compute_to_nonemb]
            comp_vals = [x[0] for x in compute_to_nonemb]
            lr_vals = [float(power_law_function(c, A_fit3, B_fit3)) for c in comp_vals]
            ax.plot(nonemb_vals, lr_vals, '-.', color='tab:green', linewidth=3,
                    label=f'Fit 3 (Compute): {A_fit3:.2e} × $C^{{{B_fit3:.3f}}}$', zorder=10)

    # Mark and annotate predictions at specific n_layer values
    prediction_layers = [15, 18, 24, 30]
    # Use H = (4/3) * L and Q = 64
    pred_qkv_dim = 64

    for i, n_pred in enumerate(prediction_layers):
        # Compute n_head using H = (4/3) * L
        pred_n_head = int(round((4/3) * n_pred))

        # Compute params for this n_layer (convert to float to avoid int32 overflow)
        non_emb_pred = float(compute_non_embedding_params(pred_n_head, pred_qkv_dim, n_pred))
        total_pred = float(compute_total_non_norm_params(pred_n_head, pred_qkv_dim, n_pred))
        compute_pred = float(compute_compute(pred_n_head, pred_qkv_dim, n_pred))

        # LR prediction from fit 1 (always shown)
        lr_pred1 = float(power_law_function(non_emb_pred, A_fit1, B_fit1))
        ax.scatter([non_emb_pred], [lr_pred1], s=150, marker='D', c='tab:orange',
                  edgecolors='black', linewidths=1.5, zorder=11)
        # Move orange labels up a bit more (1.3 -> 1.5), and second from top down (i==1)
        vertical_offset = 1.5 if i != 1 else 1.3
        ax.text(non_emb_pred, lr_pred1 * vertical_offset, f'L={n_pred},H={pred_n_head}\nLR={lr_pred1:.2e}',
               ha='left', va='bottom', fontsize=15,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='tab:orange', alpha=0.2))

        # LR prediction from fit 2 (optional)
        # Note: Use non_emb_pred for x-coordinate since that's our x-axis
        if args.fit_total_params and A_fit2 is not None:
            lr_pred2 = float(power_law_function(total_pred, A_fit2, B_fit2))
            ax.scatter([non_emb_pred], [lr_pred2], s=150, marker='s', c='blue',
                      edgecolors='black', linewidths=1.5, zorder=11)
            ax.text(non_emb_pred, lr_pred2 * 0.7, f'L={n_pred},H={pred_n_head}\nLR={lr_pred2:.2e}',
                   ha='center', va='top', fontsize=15,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.2))

        # LR prediction from fit 3 (optional)
        # Note: Use non_emb_pred for x-coordinate since that's our x-axis
        if args.fit_compute and A_fit3 is not None:
            lr_pred3 = float(power_law_function(compute_pred, A_fit3, B_fit3))
            ax.scatter([non_emb_pred], [lr_pred3], s=150, marker='^', c='tab:green',
                      edgecolors='black', linewidths=1.5, zorder=11)
            # Move second from top down (i==1), and last (bottom) label to the right
            vertical_offset = 0.7 if i != 1 else 0.6
            horiz_align = 'left' if i == 3 else 'right'
            ax.text(non_emb_pred, lr_pred3 * vertical_offset, f'L={n_pred},H={pred_n_head}\nLR={lr_pred3:.2e}',
                   ha=horiz_align, va='top', fontsize=15,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='tab:green', alpha=0.2))

    # Formatting
    ax.set_xlabel('Non-embedding Parameters', fontsize=20)
    ax.set_ylabel('Learning Rate (LR)', fontsize=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Map optimizer abbreviations for title
    optimizer_title_map = {'DS': 'Dana-Star', 'AW': 'AdamW', 'AS': 'Adam-Star', 'MK4': 'Dana-Star-MK4'}
    optimizer_title = optimizer_title_map[args.optimizer]

    # Build title with optional clipsnr info
    title_parts = [f'ω = {args.target_omega}', f'Top-K = {args.top_k}', f'Dataset = {args.dataset}']
    if args.target_clipsnr is not None:
        title_parts.append(f'ClipSNR = {args.target_clipsnr}')
    title_params = ', '.join(title_parts)

    ax.set_title(f'{optimizer_title} Optimal Learning Rate Scaling Law\n({title_params})',
                 fontsize=20, fontweight='bold')
    ax.legend(fontsize=15, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add text annotation showing point size = weight
    ax.text(0.02, 0.02, 'Point size ∝ weight\n(best LR has largest weight)\n\nPredictions use H=(4/3)L, Q=64',
            transform=ax.transAxes, fontsize=15, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    if args.output:
        output_file = args.output
    else:
        # Map optimizer abbreviations for filename
        optimizer_filename_map = {'DS': 'DanaStar', 'AW': 'AdamW', 'AS': 'AdamStar', 'MK4': 'DanaStar-MK4'}
        optimizer_name = optimizer_filename_map[args.optimizer]
        output_file = f'{optimizer_name}-lr-extrapolation.pdf'

    import os
    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    print(f"\nPlot saved to: {os.path.abspath(output_file)}")

    # Print extrapolations
    print(f"\nExtrapolated optimal LRs (using H=(4/3)L, Q=64):")
    for n_pred in prediction_layers:
        pred_n_head = int(round((4/3) * n_pred))
        non_emb_pred = float(compute_non_embedding_params(pred_n_head, pred_qkv_dim, n_pred))
        total_pred = float(compute_total_non_norm_params(pred_n_head, pred_qkv_dim, n_pred))
        compute_pred = float(compute_compute(pred_n_head, pred_qkv_dim, n_pred))

        lr_pred1 = float(power_law_function(non_emb_pred, A_fit1, B_fit1))

        output = f"  n_layer={n_pred}, n_head={pred_n_head}: LR={lr_pred1:.6e} (Fit 1)"

        if args.fit_total_params and A_fit2 is not None:
            lr_pred2 = float(power_law_function(total_pred, A_fit2, B_fit2))
            output += f", LR={lr_pred2:.6e} (Fit 2)"

        if args.fit_compute and A_fit3 is not None:
            lr_pred3 = float(power_law_function(compute_pred, A_fit3, B_fit3))
            output += f", LR={lr_pred3:.6e} (Fit 3)"

        print(output)
