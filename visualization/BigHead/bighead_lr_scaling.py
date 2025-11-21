#!/usr/bin/env python3
"""
Learning Rate Scaling Law Analysis - Alternative Power Law Fitting

This script estimates an alternative power law fit for the optimal learning rate across different
model scaling rules using a direct weighted fitting approach. For a fixed omega value, it finds
the K best LRs for each model size, weights them (smallest gets weight K, second gets K-1, etc.),
and fits: LR = a * (b + parameter_metric)^d using weighted MSE loss with Adagrad optimization.
Constraints: a, b > 0 (enforced via exponential parameterization).

Supported Scaling Rules:
1. BigHead: depth-based scaling
   - n_layer = depth, n_head = depth
   - n_embd = 16 * depth^2, mlp_hidden = 32 * depth^2

2. EggHead: heads-based quadratic depth scaling
   - n_layer = heads * (heads - 1) / 2, n_head = heads
   - n_embd = 16 * heads^2, mlp_hidden = 32 * heads^2

3. Enoki: heads-based DiLoco scaling
   - n_layer = 3 * heads / 4, n_head = heads
   - n_embd = heads * 64, mlp_hidden = 4 * n_embd

4. Eryngii: heads-based scaling with increased head dimension and depth
   - n_layer = heads^2 / 8, n_head = heads
   - head_dim = 32 * heads / 3 (rounded to multiple of 8)
   - n_embd = n_head * head_dim, mlp_hidden = 4 * n_embd

Usage:
    python bighead_lr_scaling.py --scaling-rule BigHead --optimizer adamw --target-omega 4.0 --top-k 5
    python bighead_lr_scaling.py --scaling-rule Enoki --optimizer mk4 --target-omega 4.0 --top-k 5
    python bighead_lr_scaling.py --scaling-rule EggHead --optimizer d-muon --target-omega 4.0 --top-k 5
    python bighead_lr_scaling.py --scaling-rule Eryngii --optimizer adamw --target-omega 4.0 --top-k 5
    python bighead_lr_scaling.py --scaling-rule BigHead --optimizer manau --target-omega 4.0 --top-k 5 --wd-decaying
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
import json
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

# =============================================================================
# SCALING RULE CONFIGURATION
# =============================================================================

SCALING_RULE_CONFIG = {
    'BigHead': {
        'group': 'DanaStar_MK4_BigHead_Sweep',
        'extrapolation_sizes': [8, 9, 10, 11, 12, 13, 14, 15],  # Show these if no data
        'size_step': 1,  # Show all consecutive integers
    },
    'EggHead': {
        'group': 'DanaStar_MK4_EggHead_Sweep',
        'extrapolation_sizes': [8, 9, 10, 11, 12],  # Show these if no data
        'size_step': 1,  # Show all consecutive integers
    },
    'Enoki': {
        'group': 'DanaStar_MK4_Enoki_Sweep',
        'extrapolation_sizes': [8, 12, 16, 20, 24, 28, 32, 36, 40],  # Only multiples of 4
        'size_step': 4,  # Only show multiples of 4
    },
    'Enoki_std': {
        'group': 'Enoki_Sweep_std',
        'extrapolation_sizes': [8, 12, 16, 20, 24, 28, 32, 36, 40],  # Only multiples of 4
        'size_step': 4,  # Only show multiples of 4
    },
    'Enoki_Scaled': {
        'group': 'Enoki_ScaledGPT',
        'extrapolation_sizes': [8, 12, 16, 20, 24, 28, 32, 36, 40],  # Only multiples of 4
        'size_step': 4,  # Only show multiples of 4
    },
    'Eryngii': {
        'group': 'eryngii_sweeps',
        'extrapolation_sizes': [8, 9, 10, 11, 12, 13, 14, 15],
        'size_step': 1,  # Show all consecutive integers
    },
    'Eryngii_Scaled': {
        'group': 'Eryngii_ScaledGPT',
        'extrapolation_sizes': [8, 9, 10, 11, 12, 13, 14, 15],
        'size_step': 1,  # Show all consecutive integers
    }
}

# Matplotlib formatting
style.use('default')
rc('font', family='sans-serif')
rcParams['font.weight'] = 'light'
rcParams['font.size'] = 20
rcParams['figure.figsize'] = (1 * 10.0, 1 * 8.0)

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description='Fit power law for optimal LR across different model scaling rules')
parser.add_argument('--scaling-rule', type=str, required=True, choices=['BigHead', 'EggHead', 'Enoki', 'Enoki_std', 'Eryngii', 'Enoki_Scaled', 'Eryngii_Scaled'],
                    help='Model scaling rule: BigHead (depth-based), EggHead (quadratic depth), Enoki (DiLoco), Enoki_std (standard init), Enoki_Scaled (ScaledGPT init), Eryngii (increased head dim and depth), or Eryngii_Scaled (ScaledGPT init)')
parser.add_argument('--optimizer', type=str, required=True, choices=['adamw', 'mk4', 'dana', 'ademamix', 'd-muon', 'manau', 'adamw-decaying-wd', 'dana-mk4'],
                    help='Optimizer type: adamw, mk4 (dana-star-mk4), dana, ademamix, d-muon, manau, adamw-decaying-wd, or dana-mk4')
parser.add_argument('--target-omega', type=float, default=4.0,
                    help='Target omega value to find optimal LR (default: 4.0)')
parser.add_argument('--target-residual-exponent', type=float, default=None,
                    help='Target residual exponent for Enoki_std: filters runs where residual_stream_scalar ≈ n_layer^exponent (default: None, no filtering)')
parser.add_argument('--top-k', type=int, default=5,
                    help='Number of best LRs to use for each model size (default: 5)')
parser.add_argument('--project', type=str, default='danastar',
                    help='WandB project name (default: danastar)')
parser.add_argument('--group', type=str, default=None,
                    help='WandB group name (default: auto-determined from scaling-rule)')
parser.add_argument('--entity', type=str, default='ep-rmt-ml-opt',
                    help='WandB entity name (default: ep-rmt-ml-opt)')
parser.add_argument('--output', type=str, default=None,
                    help='Output filename for plot (default: auto-generated)')
parser.add_argument('--n-steps', type=int, default=200000,
                    help='Number of optimization steps for power law fitting (default: 100000)')
parser.add_argument('--learning-rate', type=float, default=1.0,
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
parser.add_argument('--wd-decaying', action='store_true',
                    help='For Manau optimizer: filter for runs with wd_decaying=True (default: False, meaning no filter)')
parser.add_argument('--show-predictions', action='store_true',
                    help='Show text boxes with LR predictions on plot (default: off)')
args = parser.parse_args()

# Map optimizer abbreviations
optimizer_map = {'adamw': 'adamw', 'mk4': 'dana-star-mk4', 'dana': 'dana', 'ademamix': 'ademamix', 'd-muon': 'd-muon', 'manau': 'manau', 'adamw-decaying-wd': 'adamw-decaying-wd', 'dana-mk4': 'dana-mk4'}
optimizer_type = optimizer_map[args.optimizer]

# Get scaling rule configuration
scaling_config = SCALING_RULE_CONFIG[args.scaling_rule]

# Determine WandB group based on scaling rule if not specified
if args.group is None:
    wandb_group = scaling_config['group']
else:
    wandb_group = args.group

# =============================================================================
# PARAMETER CALCULATION FUNCTIONS (Support for BigHead, EggHead, and Enoki)
# =============================================================================

def compute_non_embedding_params(size, scaling_rule):
    """
    Compute non-embedding parameters based on scaling rule.

    Args:
        size: For BigHead, this is depth. For EggHead/Enoki/Eryngii, this is heads.
        scaling_rule: One of 'BigHead', 'EggHead', 'Enoki', 'Enoki_std', 'Enoki_Scaled', 'Eryngii', or 'Eryngii_Scaled'

    Returns:
        int: Number of non-embedding parameters
    """
    if scaling_rule == 'BigHead':
        # BigHead: depth-based scaling
        depth = size
        head_dim = 16 * depth
        n_embd = 16 * depth * depth
        mlp_hidden = 32 * depth * depth
        n_head = depth
        n_layer = depth
        # Non-emb = n_layer * (3 * head_dim * n_embd * n_head + n_embd^2 + 2 * n_embd * mlp + 8 * n_embd) + 2 * n_embd
        non_emb = n_layer * (3 * head_dim * n_embd * n_head + n_embd * n_embd + 2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd

    elif scaling_rule == 'EggHead':
        # EggHead: heads-based quadratic depth scaling
        heads = size
        head_dim = 16 * heads
        n_embd = 16 * heads * heads
        mlp_hidden = 32 * heads * heads
        n_head = heads
        n_layer = int(heads * (heads - 1) / 2)
        # Non-emb = n_layer * (3 * head_dim * n_embd * n_head + n_embd^2 + 2 * n_embd * mlp + 8 * n_embd) + 2 * n_embd
        non_emb = n_layer * (3 * head_dim * n_embd * n_head + n_embd * n_embd + 2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd

    elif scaling_rule == 'Enoki' or scaling_rule == 'Enoki_std' or scaling_rule == 'Enoki_Scaled':
        # Enoki and Enoki_std: heads-based DiLoco scaling
        heads = size
        head_dim = 64  # Fixed for Enoki
        n_embd = heads * 64
        mlp_hidden = 4 * n_embd
        n_head = heads
        n_layer = int(3 * heads // 4)
        # Non-emb = 12 * n_embd^2 * n_layer (standard DiLoco formula)
        non_emb = 12 * n_embd * n_embd * n_layer

    elif scaling_rule == 'Eryngii' or scaling_rule == 'Eryngii_Scaled':
        # Eryngii and Eryngii_Scaled: heads-based scaling with increased head dimension and depth
        heads = size
        head_dim = int(round(32 * heads / 3 / 8) * 8)  # Rounded to multiple of 8
        n_head = heads
        n_layer = int(heads * heads // 8)
        n_embd = n_head * head_dim
        mlp_hidden = 4 * n_embd
        # Non-emb = 12 * n_embd^2 * n_layer (standard DiLoco formula)
        non_emb = 12 * n_embd * n_embd * n_layer

    else:
        raise ValueError(f"Unknown scaling rule: {scaling_rule}")

    return int(non_emb)

def compute_total_params(size, scaling_rule):
    """
    Compute total parameters including embeddings.
    Total params = non_emb + 2 * n_embd * vocab_size
    """
    non_emb = compute_non_embedding_params(size, scaling_rule)

    if scaling_rule == 'BigHead':
        n_embd = 16 * size * size
        vocab_size = 50304
    elif scaling_rule == 'EggHead':
        n_embd = 16 * size * size
        vocab_size = 50304
    elif scaling_rule == 'Enoki' or scaling_rule == 'Enoki_std' or scaling_rule == 'Enoki_Scaled':
        n_embd = size * 64
        vocab_size = 50304
    elif scaling_rule == 'Eryngii' or scaling_rule == 'Eryngii_Scaled':
        heads = size
        head_dim = int(round(32 * heads / 3 / 8) * 8)  # Rounded to multiple of 8
        n_embd = heads * head_dim
        vocab_size = 50304
    else:
        raise ValueError(f"Unknown scaling rule: {scaling_rule}")

    total_params = non_emb + 2 * n_embd * vocab_size
    return int(total_params)

def compute_compute(size, scaling_rule):
    """
    Compute compute metric: non_emb * total_params * 20
    (Based on the iterations calculation: ITERATIONS = 20 * TOTAL_PARAMS )
    """
    non_emb = compute_non_embedding_params(size, scaling_rule)
    total_params = compute_total_params(size, scaling_rule)
    compute = non_emb * total_params * 20
    return compute

@jit
def saturated_power_law_function(params, a, b, d):
    """Alternative power law function: LR = a * (b + params)^d"""
    params_float = jnp.asarray(params, dtype=jnp.float32)
    return a * ((b + params_float) ** d)

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

def load_wandb_data_simple(project_name, group_name, entity, optimizer_type, scaling_rule,
                           target_clipsnr=None, clipsnr_tolerance=0.1, wd_decaying_filter=False,
                           target_residual_exponent=None):
    """Load data from WandB

    Args:
        scaling_rule: One of 'BigHead', 'EggHead', 'Enoki', 'Enoki_std', 'Enoki_Scaled', 'Eryngii', or 'Eryngii_Scaled' to determine size parameter
        wd_decaying_filter: If True, only include runs with wd_decaying=True (for Manau)
        target_residual_exponent: If provided, filter runs where residual_stream_scalar ≈ n_layer^exponent (for Enoki_std)
    """
    api = wandb.Api()

    print(f"Loading data from {group_name}...")
    print(f"Scaling rule: {scaling_rule}")
    if wd_decaying_filter:
        print(f"Filtering for wd_decaying=True")
    if target_residual_exponent is not None:
        print(f"Filtering for residual_stream_scalar ≈ n_layer^{target_residual_exponent}")

    runs = api.runs(f"{entity}/{project_name}", filters={"group": group_name})

    data = []
    total_runs = 0
    skipped_optimizer = 0
    skipped_incomplete = 0
    skipped_missing_data = 0
    skipped_clipsnr = 0
    skipped_wd_decaying = 0
    skipped_residual_exponent = 0

    for run in runs:
        total_runs += 1

        # Handle different wandb API versions where config might be a string or dict
        config = run.config
        if isinstance(config, str):
            # In wandb version 0.22.2, config is returned as a JSON string
            # Parse it to get a dictionary
            try:
                config = json.loads(config)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  Warning: Could not parse config JSON for run {run.name}, skipping...")
                continue

        # Convert to dict if it's a wandb Config object
        if hasattr(config, 'as_dict'):
            config = config.as_dict()
        elif not isinstance(config, dict):
            # If it's not a dict and doesn't have as_dict, try to convert
            try:
                config = dict(config)
            except (TypeError, ValueError):
                print(f"  Warning: Could not convert config for run {run.name}, skipping...")
                continue

        # Extract value from nested dict structure if needed
        # In some versions, config has structure {"key": {"value": actual_value}}
        def extract_value(config_dict):
            """Extract values from nested config structure."""
            result = {}
            for key, val in config_dict.items():
                if isinstance(val, dict) and 'value' in val:
                    result[key] = val['value']
                else:
                    result[key] = val
            return result

        config = extract_value(config)

        # Handle summary (also may be a JSON string in some wandb versions)
        summary = run.summary
        if hasattr(summary, '_json_dict') and isinstance(summary._json_dict, str):
            try:
                summary = json.loads(summary._json_dict)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  Warning: Could not parse summary JSON for run {run.name}, skipping...")
                continue
        elif not isinstance(summary, dict) and hasattr(summary, '__getitem__'):
            # If it's a wandb Summary object with dict-like access, use it as-is
            pass

        # Filter by optimizer (stored as 'opt' in config)
        opt = config.get('opt', '')
        if opt != optimizer_type:
            skipped_optimizer += 1
            continue

        # Filter by clipsnr if specified (for MK4 and Manau optimizers)
        if target_clipsnr is not None:
            clipsnr = config.get('clipsnr')
            if clipsnr is None or abs(clipsnr - target_clipsnr) > clipsnr_tolerance:
                skipped_clipsnr += 1
                continue

        # Filter by wd_decaying if requested (for Manau optimizer)
        if wd_decaying_filter:
            wd_decaying = config.get('wd_decaying', False)
            if not wd_decaying:
                skipped_wd_decaying += 1
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

        # Extract size parameter based on scaling rule
        if scaling_rule == 'BigHead':
            # BigHead uses n_layer directly as depth (n_layer == depth)
            size = config.get('n_layer')
            size_name = 'depth'
        elif scaling_rule == 'EggHead':
            # EggHead uses n_head as heads
            size = config.get('n_head')
            size_name = 'heads'
        elif scaling_rule == 'Enoki' or scaling_rule == 'Enoki_std' or scaling_rule == 'Enoki_Scaled':
            # Enoki and Enoki_std use n_head as heads
            size = config.get('n_head')
            size_name = 'heads'
        elif scaling_rule == 'Eryngii' or scaling_rule == 'Eryngii_Scaled':
            # Eryngii and Eryngii_Scaled use n_head as heads
            size = config.get('n_head')
            size_name = 'heads'
        else:
            raise ValueError(f"Unknown scaling rule: {scaling_rule}")

        if size is None:
            skipped_missing_data += 1
            continue

        # Filter by residual exponent if specified (for Enoki_std)
        if target_residual_exponent is not None:
            residual_stream_scalar = config.get('residual_stream_scalar')
            n_layer = config.get('n_layer')

            if residual_stream_scalar is None or n_layer is None:
                skipped_missing_data += 1
                continue

            # Calculate target value: n_layer^exponent
            target_residual_value = n_layer ** target_residual_exponent

            # Check if residual_stream_scalar is within 10% of target
            relative_diff = abs(residual_stream_scalar - target_residual_value) / target_residual_value
            if relative_diff > 0.10:  # 10% tolerance
                skipped_residual_exponent += 1
                continue

        # Calculate omega based on optimizer type
        if optimizer_type in ['dana-star-mk4', 'dana', 'adamw-decaying-wd']:
            wd_ts = config.get('wd_ts', 1.0)
            weight_decay = config.get('weight_decay', 1.0)
            omega = wd_ts * lr * weight_decay
        elif optimizer_type == 'manau':
            # Manau can use either wd_decaying mode or AdamW-style weight decay
            wd_decaying = config.get('wd_decaying', False)
            if wd_decaying:
                # When wd_decaying=True, use wd_ts * lr * weight_decay (like DANA)
                wd_ts = config.get('wd_ts', 1.0)
                weight_decay = config.get('weight_decay', 1.0)
                omega = wd_ts * lr * weight_decay
            else:
                # When wd_decaying=False, use weight_decay * lr * iterations (like AdamW)
                weight_decay = config.get('weight_decay', 0.1)
                iterations = config.get('iterations', 1)
                omega = weight_decay * lr * iterations
        else:  # adamw, ademamix, d-muon
            weight_decay = config.get('weight_decay', 0.1)
            iterations = config.get('iterations', 1)
            omega = weight_decay * lr * iterations

        data.append({
            'size': size,
            'size_name': size_name,
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
    if skipped_wd_decaying > 0:
        print(f"  Skipped {skipped_wd_decaying} runs due to wd_decaying filter")
    if skipped_residual_exponent > 0:
        print(f"  Skipped {skipped_residual_exponent} runs due to residual exponent filter")
    if skipped_incomplete > 0:
        print(f"  Skipped {skipped_incomplete} incomplete runs")

    df = pd.DataFrame(data)
    return df

# =============================================================================
# DIRECT WEIGHTED POWER LAW FITTING
# =============================================================================

def fit_saturated_power_law_weighted(params_list, lrs_list, weights_list, n_steps=5000, learning_rate=0.1):
    """
    Fit alternative power law LR = a * (b + params)^d using weighted MSE loss with Adagrad optimization.
    Constraints: a, b > 0

    Uses parameterization:
    - a_raw -> a = exp(a_raw) to ensure a > 0
    - b_raw -> b = exp(b_raw) to ensure b > 0
    - d is unconstrained (typically negative for learning rate scaling)
    """
    # Convert to JAX arrays
    params_arr = jnp.array(params_list, dtype=jnp.float32)
    lrs = jnp.array(lrs_list, dtype=jnp.float32)
    weights = jnp.array(weights_list, dtype=jnp.float32)
    log_lrs = jnp.log(lrs)

    # Initialize parameters: [a_raw, b_raw, d]
    # Initial guess: a ≈ 1e3, b ≈ min(params), d ≈ -1.0
    max_lr = float(jnp.max(lrs))
    min_params = float(jnp.min(params_arr))
    fit_params = jnp.array([
        jnp.log(1e3),  # a_raw -> a = exp(a_raw)
        jnp.log(min_params),    # b_raw -> b = exp(b_raw)
        -1.0                     # d (unconstrained)
    ], dtype=jnp.float32)

    @jit
    def loss_fn(fit_params):
        a_raw, b_raw, d = fit_params

        # Apply constraints: a, b > 0
        a = jnp.exp(a_raw)
        b = jnp.exp(b_raw)

        # Predictions: LR = a * (b + params)^d
        pred_lrs = a * ((b + params_arr) ** d)
        log_pred_lrs = jnp.log(pred_lrs)

        # Weighted MSE loss (multiply weights by params for larger model emphasis)
        residuals = (log_lrs - log_pred_lrs) ** 2
        combined_weights = weights**2 * params_arr
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
            a_raw, b_raw, d = fit_params
            a = float(jnp.exp(a_raw))
            b = float(jnp.exp(b_raw))
            print(f"  Step {step:5d}: loss={current_loss:.6e}, a={a:.6e}, b={b:.6e}, d={d:.4f}")

    # Extract final parameters
    a_raw, b_raw, d = best_params
    a = float(jnp.exp(a_raw))
    b = float(jnp.exp(b_raw))
    d = float(d)

    return a, b, d

def collect_weighted_data_for_depths(optimizer_type, scaling_rule, target_omega, top_k, project, group, entity,
                                      exclude_small=False, target_clipsnr=None, clipsnr_tolerance=0.1,
                                      wd_decaying_filter=False, target_residual_exponent=None):
    """
    Collect top-K LRs for each size at target omega, with weights.

    Args:
        scaling_rule: One of 'BigHead', 'EggHead', 'Enoki', 'Enoki_std', 'Enoki_Scaled', 'Eryngii', or 'Eryngii_Scaled'
        wd_decaying_filter: If True, only include runs with wd_decaying=True (for Manau)
        target_residual_exponent: If provided, filter runs where residual_stream_scalar ≈ n_layer^exponent
    """
    # Load all data for the optimizer
    data_df = load_wandb_data_simple(project, group, entity, optimizer_type, scaling_rule,
                                     target_clipsnr, clipsnr_tolerance, wd_decaying_filter,
                                     target_residual_exponent)

    if len(data_df) == 0:
        print("No data found. Exiting.")
        return None

    # Get unique sizes
    available_sizes = sorted(data_df['size'].unique())
    size_name = data_df['size_name'].iloc[0] if len(data_df) > 0 else 'size'
    print(f"\nAvailable {size_name}s: {available_sizes}")

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

    for size in available_sizes:
        print(f"\n{'='*70}")
        print(f"Processing {size_name}={size}")
        print(f"{'='*70}")

        # Filter data for this size
        size_data = data_df[data_df['size'] == size].copy()

        if len(size_data) == 0:
            print(f"No data found for {size_name}={size}")
            continue

        # Find closest omega to target
        available_omegas = sorted(size_data['omega'].unique())
        closest_omega = min(available_omegas, key=lambda x: abs(x - target_omega))

        print(f"  Target omega: {target_omega:.2f}, Closest: {closest_omega:.2f}")

        # Get top K LRs at this omega
        top_k_data = get_top_k_lrs_for_omega(size_data, closest_omega, top_k=top_k)

        if len(top_k_data) == 0:
            continue

        # Compute parameter counts
        non_emb_params = compute_non_embedding_params(size, scaling_rule)
        total_params = compute_total_params(size, scaling_rule)
        compute_metric = compute_compute(size, scaling_rule)

        print(f"  Non-embedding params: {non_emb_params:,}")
        print(f"  Total params: {total_params:,}")
        print(f"  Compute: {compute_metric:.2e}")

        # Add to global lists (or excluded lists if small model and exclude_small=True)
        is_small = (size == 4)

        if exclude_small and is_small:
            print(f"  Excluding {size_name}=4 from fit")
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
        results[size] = {
            'size': size,
            'size_name': size_name,
            'non_emb_params': non_emb_params,
            'total_params': total_params,
            'compute': compute_metric,
            'closest_omega': closest_omega,
            'top_k_data': top_k_data,
            'data_df': size_data,
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
    print(f"Learning Rate Scaling Law Analysis - {args.scaling_rule}")
    print(f"Scaling Rule: {args.scaling_rule}")
    print(f"Optimizer: {args.optimizer} ({optimizer_type})")
    print(f"Target Omega: {args.target_omega}")
    print(f"Top K: {args.top_k}")
    print(f"Group: {wandb_group}")
    print(f"Exclude Small: {args.exclude_small}")
    if args.target_clipsnr is not None:
        print(f"Target ClipSNR: {args.target_clipsnr} (tolerance: {args.clipsnr_tolerance})")
    if args.wd_decaying:
        print(f"Filtering for wd_decaying=True")
    if args.target_residual_exponent is not None:
        print(f"Target Residual Exponent: {args.target_residual_exponent} (residual_stream_scalar ≈ n_layer^{args.target_residual_exponent}, 10% tolerance)")
    print("="*70)

    # Collect weighted data from all sizes
    result = collect_weighted_data_for_depths(
        optimizer_type=optimizer_type,
        scaling_rule=args.scaling_rule,
        target_omega=args.target_omega,
        top_k=args.top_k,
        project=args.project,
        group=wandb_group,
        entity=args.entity,
        exclude_small=args.exclude_small,
        target_clipsnr=args.target_clipsnr,
        clipsnr_tolerance=args.clipsnr_tolerance,
        wd_decaying_filter=args.wd_decaying,
        target_residual_exponent=args.target_residual_exponent
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

    # Fit saturated power law using non-embedding parameters (always on)
    print(f"\n{'='*70}")
    print("Fitting Saturated Power Law: LR = a * (b + non_emb_params)^d")
    print(f"{'='*70}")
    a_fit, b_fit, d_fit = fit_saturated_power_law_weighted(non_emb_params, lrs, weights,
                                                             n_steps=args.n_steps,
                                                             learning_rate=args.learning_rate)

    print(f"\nFitted parameters (non-embedding):")
    print(f"  a = {a_fit:.6e}")
    print(f"  b = {b_fit:.6e}")
    print(f"  d = {d_fit:.4f}")
    print(f"Alternative power law: LR = {a_fit:.6e} * ({b_fit:.6e} + non_emb_params)^{d_fit:.4f}")

    # Fit using total parameters if requested
    a_fit_total, b_fit_total, d_fit_total = None, None, None
    if args.fit_total_params:
        print(f"\n{'='*70}")
        print("Fitting Alternative Power Law: LR = a * (b + total_params)^d")
        print(f"{'='*70}")
        a_fit_total, b_fit_total, d_fit_total = fit_saturated_power_law_weighted(
            total_params, lrs, weights,
            n_steps=args.n_steps,
            learning_rate=args.learning_rate)

        print(f"\nFitted parameters (total params):")
        print(f"  a = {a_fit_total:.6e}")
        print(f"  b = {b_fit_total:.6e}")
        print(f"  d = {d_fit_total:.4f}")
        print(f"Alternative power law: LR = {a_fit_total:.6e} * ({b_fit_total:.6e} + total_params)^{d_fit_total:.4f}")

    # Fit using compute if requested
    a_fit_compute, b_fit_compute, d_fit_compute = None, None, None
    if args.fit_compute:
        print(f"\n{'='*70}")
        print("Fitting Alternative Power Law: LR = a * (b + compute)^d")
        print(f"{'='*70}")
        a_fit_compute, b_fit_compute, d_fit_compute = fit_saturated_power_law_weighted(
            compute, lrs, weights,
            n_steps=args.n_steps,
            learning_rate=args.learning_rate)

        print(f"\nFitted parameters (compute):")
        print(f"  a = {a_fit_compute:.6e}")
        print(f"  b = {b_fit_compute:.6e}")
        print(f"  d = {d_fit_compute:.4f}")
        print(f"Alternative power law: LR = {a_fit_compute:.6e} * ({b_fit_compute:.6e} + compute)^{d_fit_compute:.4f}")

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

    # Plot saturated power law fit line and extrapolation
    unique_params = sorted(set(non_emb_params))

    # Get sizes with actual data
    sizes_with_data = set(model_results.keys())

    # Determine prediction sizes: extrapolation_sizes that don't have data
    all_prediction_sizes = scaling_config['extrapolation_sizes']
    prediction_sizes_no_data = [s for s in all_prediction_sizes if s not in sizes_with_data]

    # For plotting the fit line, use max of extrapolation sizes
    max_size = max(all_prediction_sizes)
    max_non_emb = compute_non_embedding_params(max_size, args.scaling_rule)
    params_range = np.linspace(min(unique_params) * 0.5, max_non_emb, 200)

    lr_fit = saturated_power_law_function(params_range, a_fit, b_fit, d_fit)
    ax.plot(params_range, lr_fit, '--', color='tab:orange', linewidth=3,
            label=f'Fit: {a_fit:.2e} × $({b_fit:.2e} + P)^{{{d_fit:.3f}}}$', zorder=10)

    # Plot compute fit line if --fit-compute is specified
    if args.fit_compute and a_fit_compute is not None:
        # Create compute range corresponding to the params_range
        # Use the relationship: compute = non_emb * total * 20
        # We can derive compute from non_emb params for each point
        compute_range = []
        for p in params_range:
            # Find the approximate size for this parameter count
            # Use inverse relationship to estimate size from parameters
            best_size = None
            best_diff = float('inf')
            for size_val in all_prediction_sizes:
                size_params = compute_non_embedding_params(size_val, args.scaling_rule)
                diff = abs(size_params - p)
                if diff < best_diff:
                    best_diff = diff
                    best_size = size_val

            # Use the best matching size to compute the compute metric
            if best_size is not None:
                # Scale the compute metric proportionally to parameter difference
                size_params = compute_non_embedding_params(best_size, args.scaling_rule)
                size_compute = compute_compute(best_size, args.scaling_rule)
                # Compute scales as non_emb * total * 20, approximately as p^2 for these architectures
                ratio = p / size_params
                compute_range.append(size_compute * (ratio ** 2))

        compute_range = np.array(compute_range)
        lr_fit_compute = saturated_power_law_function(compute_range, a_fit_compute, b_fit_compute, d_fit_compute)
        ax.plot(params_range, lr_fit_compute, '-.', color='tab:green', linewidth=3,
                label=f'Compute fit: {a_fit_compute:.2e} + {b_fit_compute:.2e} × $C^{{{d_fit_compute:.3f}}}$', zorder=10)

    # Get size name from results
    size_name = list(model_results.values())[0]['size_name'] if model_results else 'size'

    # Mark predictions at sizes WITHOUT data
    for i, size_pred in enumerate(prediction_sizes_no_data):
        non_emb_pred = float(compute_non_embedding_params(size_pred, args.scaling_rule))
        lr_pred = float(saturated_power_law_function(non_emb_pred, a_fit, b_fit, d_fit))

        ax.scatter([non_emb_pred], [lr_pred], s=150, marker='D', c='tab:orange',
                  edgecolors='black', linewidths=1.5, zorder=11)

        # Only show text boxes if --show-predictions flag is set
        if args.show_predictions:
            vertical_offset = 1.5 if i != 1 else 1.3
            ax.text(non_emb_pred, lr_pred * vertical_offset, f'{size_name.capitalize()}={size_pred}\nLR={lr_pred:.2e}',
                   ha='left', va='bottom', fontsize=15,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='tab:orange', alpha=0.2))

    # Formatting
    ax.set_xlabel('Non-embedding Parameters', fontsize=20)
    ax.set_ylabel('Learning Rate (LR)', fontsize=20)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add second x-axis showing 'size' variable
    ax2 = ax.twiny()

    # Get all sizes (both with data and extrapolated)
    all_sizes_for_axis = sorted(set(list(sizes_with_data) + all_prediction_sizes))

    # Compute non-embedding params for these sizes
    size_to_params = {size: compute_non_embedding_params(size, args.scaling_rule) for size in all_sizes_for_axis}

    # Set up the second axis with size labels at corresponding parameter positions
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([size_to_params[size] for size in all_sizes_for_axis])
    ax2.set_xticklabels([str(size) for size in all_sizes_for_axis])
    ax2.set_xlabel(f'{size_name.capitalize()}', fontsize=20)

    optimizer_title_map = {'adamw': 'AdamW', 'mk4': 'Dana-Star-MK4', 'dana': 'Dana-Star', 'ademamix': 'AdemaMix', 'd-muon': 'D-Muon', 'manau': 'Manau', 'adamw-decaying-wd': 'AdamW-Decaying-WD', 'dana-mk4': 'Dana-MK4'}
    optimizer_title = optimizer_title_map[args.optimizer]

    title_parts = [f'ω = {args.target_omega}', f'Top-K = {args.top_k}']
    if args.target_clipsnr is not None:
        title_parts.append(f'ClipSNR = {args.target_clipsnr}')
    if args.wd_decaying:
        title_parts.append('wd_decaying=True')
    title_params = ', '.join(title_parts)

    ax.set_title(f'{args.scaling_rule} {optimizer_title} Optimal Learning Rate Scaling Law\n({title_params})',
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
        optimizer_filename_map = {'adamw': 'AdamW', 'mk4': 'DanaStar-MK4', 'dana': 'DanaStar', 'ademamix': 'AdemaMix', 'd-muon': 'D-Muon', 'manau': 'Manau', 'adamw-decaying-wd': 'AdamW-Decaying-WD', 'dana-mk4': 'Dana-MK4'}
        optimizer_name = optimizer_filename_map[args.optimizer]
        output_file = f'{args.scaling_rule}-{optimizer_name}-lr-extrapolation.pdf'

    import os
    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    print(f"\nPlot saved to: {os.path.abspath(output_file)}")

    # Print extrapolations for ALL sizes (both with and without data)
    print(f"\n{'='*70}")
    print(f"Optimal LRs for {args.scaling_rule} Architecture")
    print(f"{'='*70}")

    # Combine sizes with data and extrapolation sizes, then sort
    all_sizes_to_show = sorted(set(list(sizes_with_data) + all_prediction_sizes))

    for size_pred in all_sizes_to_show:
        non_emb_pred = float(compute_non_embedding_params(size_pred, args.scaling_rule))
        total_pred = float(compute_total_params(size_pred, args.scaling_rule))
        compute_pred = float(compute_compute(size_pred, args.scaling_rule))

        lr_pred_non_emb = float(saturated_power_law_function(non_emb_pred, a_fit, b_fit, d_fit))

        # Mark whether this size has data or is extrapolated
        has_data_marker = " [HAS DATA]" if size_pred in sizes_with_data else " [EXTRAPOLATED]"

        print(f"\n{size_name.capitalize()}={size_pred}{has_data_marker}:")
        print(f"  Non-emb params: {non_emb_pred:,}")
        print(f"  Total params: {total_pred:,}")
        print(f"  Compute: {compute_pred:.2e}")
        print(f"  LR (non-emb fit): {lr_pred_non_emb:.6e}")

        if args.fit_total_params and a_fit_total is not None:
            lr_pred_total = float(saturated_power_law_function(total_pred, a_fit_total, b_fit_total, d_fit_total))
            print(f"  LR (total params fit): {lr_pred_total:.6e}")

        if args.fit_compute and a_fit_compute is not None:
            lr_pred_compute = float(saturated_power_law_function(compute_pred, a_fit_compute, b_fit_compute, d_fit_compute))
            print(f"  LR (compute fit): {lr_pred_compute:.6e}")
