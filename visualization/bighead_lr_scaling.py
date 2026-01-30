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
    python bighead_lr_scaling.py --scaling-rule Enoki --optimizer mk4 --target-omega 4.0 --top-k 5 --target-kappa 0.85
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
from typing import Dict, List, Tuple, Optional, Any

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
    'Enoki_Scaled_noqk': {
        'group': 'Enoki_ScaledGPT_noqk',
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
    },
    'Qwen3_Scaled': {
        'group': 'Qwen3_ScaledGPT',
        'extrapolation_sizes': [4, 6, 8, 10, 12, 14, 16],
        'size_step': 2,  # Show multiples of 2
    },
    'Qwen3_Hoyer': {
        'group': 'Qwen3_Hoyer',
        'extrapolation_sizes': [4, 6, 8, 10, 12, 14, 16],
        'size_step': 2,  # Show multiples of 2
    },
    'Enoki_512': {
        'group': 'enoki_512',
        'extrapolation_sizes': [8, 12, 16, 20, 24, 28, 32, 36, 40],  # Only multiples of 4
        'size_step': 4,  # Only show multiples of 4
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
parser.add_argument('--scaling-rule', type=str, required=True, choices=['BigHead', 'EggHead', 'Enoki', 'Enoki_std', 'Eryngii', 'Enoki_Scaled', 'Enoki_Scaled_noqk', 'Eryngii_Scaled', 'Qwen3_Scaled', 'Qwen3_Hoyer', 'Enoki_512'],
                    help='Model scaling rule: BigHead (depth-based), EggHead (quadratic depth), Enoki (DiLoco), Enoki_std (standard init), Enoki_Scaled (ScaledGPT init), Enoki_Scaled_noqk (ScaledGPT init without QK norm), Eryngii (increased head dim and depth), Eryngii_Scaled (ScaledGPT init), Qwen3_Scaled (ScaledGPT init), or Qwen3_Hoyer (ScaledGPT init with Hoyer loss)')
parser.add_argument('--optimizer', type=str, required=True, choices=['adamw', 'mk4', 'dana', 'ademamix', 'd-muon', 'manau', 'adamw-decaying-wd', 'dana-mk4', 'ademamix-decaying-wd', 'dana-star-no-tau', 'dana-star', 'dana-star-no-tau-kappa-0-8', 'dana-star-no-tau-kappa-0-85', 'dana-star-no-tau-kappa-0-9', 'dana-mk4-kappa-0-85', 'dana-star-no-tau-beta1', 'dana-star-no-tau-dana-constant', 'dana-star-no-tau-beta2-constant', 'dana-star-no-tau-dana-constant-beta2-constant', 'dana-star-no-tau-dana-constant-beta1', 'dana-star-no-tau-dana-constant-beta2-constant-beta1'],
                    help='Optimizer type: adamw, mk4 (dana-star-mk4), dana, ademamix, d-muon, manau, adamw-decaying-wd, dana-mk4, ademamix-decaying-wd, dana-star-no-tau, dana-star, dana-star-no-tau-kappa-0-8/85/9, dana-mk4-kappa-0-85, dana-star-no-tau-beta1, dana-star-no-tau-dana-constant, dana-star-no-tau-beta2-constant, dana-star-no-tau-dana-constant-beta2-constant, dana-star-no-tau-dana-constant-beta1 or dana-star-no-tau-dana-constant-beta2-constant-beta1')
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
parser.add_argument('--target-kappa', type=float, default=None,
                    help='Target kappa value for DANA optimizers (filters runs within tolerance, default: None)')
parser.add_argument('--kappa-tolerance', type=float, default=0.05,
                    help='Tolerance for kappa matching (default: 0.05)')
parser.add_argument('--wd-decaying', action='store_true',
                    help='For Manau optimizer: filter for runs with wd_decaying=True (default: False, meaning no filter)')
parser.add_argument('--show-predictions', action='store_true',
                    help='Show text boxes with LR predictions on plot (default: off)')
parser.add_argument('--smallest-head', type=int, default=None,
                    help='Ignore data with head sizes smaller than this value (default: None, no filtering)')
parser.add_argument('--simple-title', action='store_true',
                    help='Use simplified title: "<Model> <Optimizer> LR Scaling Law Fit"')
parser.add_argument('--title-fontsize', type=int, default=20,
                    help='Title font size (default: 20)')
parser.add_argument('--legend-fontsize', type=int, default=15,
                    help='Legend font size (default: 15)')
parser.add_argument('--tick-fontsize', type=int, default=None,
                    help='Tick label font size (default: auto)')
parser.add_argument('--skip-ticks-after', type=int, default=None,
                    help='Show every other tick label past this head count (default: None, show all)')
parser.add_argument('--hide-ticks', type=int, nargs='+', default=None,
                    help='List of specific tick labels to hide (default: None)')
parser.add_argument('--no-diamonds', action='store_true',
                    help='Do not show orange diamond markers for extrapolated predictions')
parser.add_argument('--figsize', type=float, nargs=2, default=[12, 7],
                    help='Figure size (width height) in inches (default: 12 7)')
parser.add_argument('--output-suffix', type=str, default=None,
                    help='Suffix for output filename (e.g., "lr-scaling" produces "<Model>-<Optimizer>-lr-scaling.pdf")')
parser.add_argument('--bootstrap', action='store_true',
                    help='Enable bootstrap analysis to compute confidence intervals on LR fit parameters')
parser.add_argument('--n-bootstrap', type=int, default=1000,
                    help='Number of bootstrap samples (default: 1000)')
parser.add_argument('--output-bootstrap-json', type=str, default=None,
                    help='Output file for bootstrap results in JSON format')
parser.add_argument('--plot-confidence-band', action='store_true',
                    help='Add shaded confidence band to the LR scaling plot')
parser.add_argument('--confidence-level', type=float, default=0.95,
                    help='Confidence level for bootstrap intervals (default: 0.95)')
args = parser.parse_args()

# Map optimizer abbreviations
optimizer_map = {'adamw': 'adamw', 'mk4': 'dana-star-mk4', 'dana': 'dana', 'ademamix': 'ademamix', 'd-muon': 'd-muon', 'manau': 'manau', 'adamw-decaying-wd': 'adamw-decaying-wd', 'dana-mk4': 'dana-mk4', 'ademamix-decaying-wd': 'ademamix-decaying-wd', 'dana-star-no-tau': 'dana-star-no-tau', 'dana-star': 'dana-star', 'dana-star-no-tau-kappa-0-8': 'dana-star-no-tau-kappa-0-8', 'dana-star-no-tau-kappa-0-85': 'dana-star-no-tau-kappa-0-85', 'dana-star-no-tau-kappa-0-9': 'dana-star-no-tau-kappa-0-9', 'dana-mk4-kappa-0-85': 'dana-mk4-kappa-0-85', 'dana-star-no-tau-beta1': 'dana-star-no-tau-beta1', 'dana-star-no-tau-dana-constant': 'dana-star-no-tau-dana-constant', 'dana-star-no-tau-beta2-constant': 'dana-star-no-tau-beta2-constant', 'dana-star-no-tau-dana-constant-beta2-constant': 'dana-star-no-tau-dana-constant-beta2-constant', 'dana-star-no-tau-dana-constant-beta1': 'dana-star-no-tau-dana-constant-beta1', 'dana-star-no-tau-dana-constant-beta2-constant-beta1': 'dana-star-no-tau-dana-constant-beta2-constant-beta1'}
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
        size: For BigHead, this is depth. For EggHead/Enoki/Eryngii/Qwen3_Scaled/Qwen3_Hoyer, this is heads.
        scaling_rule: One of 'BigHead', 'EggHead', 'Enoki', 'Enoki_std', 'Enoki_Scaled', 'Enoki_Scaled_noqk', 'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', or 'Qwen3_Hoyer'

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

    elif scaling_rule in ('Enoki', 'Enoki_std', 'Enoki_Scaled', 'Enoki_Scaled_noqk', 'Enoki_512'):
        # Enoki variants: heads-based DiLoco scaling
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

    elif scaling_rule == 'Qwen3_Scaled' or scaling_rule == 'Qwen3_Hoyer':
        # Qwen3_Scaled / Qwen3_Hoyer: heads-based Qwen3 scaling with elementwise gating
        # head_dim=128, n_layer=2*heads, n_embd=128*heads, mlp_hidden=3*n_embd
        heads = size
        head_dim = 128
        n_head = heads
        n_layer = 2 * heads
        n_embd = 128 * heads
        total_qkv_dim = n_head * head_dim

        # Qwen3 with gating: non_emb = n_layer * (5 * n_embd * total_qkv_dim + 2 * head_dim + 9 * n_embd^2 + 2 * n_embd) + n_embd
        # Per layer:
        # - attn = 5 * n_embd * total_qkv_dim  # q_proj (2x with gating) + k_proj + v_proj + o_proj
        # - qk_norm = 2 * head_dim
        # - mlp = 9 * n_embd^2  # SwiGLU: gate_proj + up_proj + down_proj (mlp_hidden = 3 * n_embd)
        # - layer_norms = 2 * n_embd
        per_layer = 5 * n_embd * total_qkv_dim + 2 * head_dim + 9 * n_embd * n_embd + 2 * n_embd
        non_emb = n_layer * per_layer + n_embd  # +n_embd for final norm

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
    elif scaling_rule in ('Enoki', 'Enoki_std', 'Enoki_Scaled', 'Enoki_Scaled_noqk', 'Enoki_512'):
        n_embd = size * 64
        vocab_size = 50304
    elif scaling_rule == 'Eryngii' or scaling_rule == 'Eryngii_Scaled':
        heads = size
        head_dim = int(round(32 * heads / 3 / 8) * 8)  # Rounded to multiple of 8
        n_embd = heads * head_dim
        vocab_size = 50304
    elif scaling_rule == 'Qwen3_Scaled' or scaling_rule == 'Qwen3_Hoyer':
        n_embd = size * 128
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

def deduplicate_lrs(data_df, lr_tolerance=0.05):
    """
    Remove duplicate learning rates that differ by less than lr_tolerance (relative error).
    For each group of similar LRs, keep the one with the lowest validation loss.

    Args:
        data_df: DataFrame with 'lr' and 'val_loss' columns
        lr_tolerance: Relative tolerance for considering LRs as duplicates (default: 5%)

    Returns:
        DataFrame with duplicates removed
    """
    if len(data_df) == 0:
        return data_df

    # Sort by validation loss (best first) so we keep the best when deduplicating
    df_sorted = data_df.sort_values('val_loss').reset_index(drop=True)

    # Track which rows to keep
    keep_mask = [True] * len(df_sorted)

    for i in range(len(df_sorted)):
        if not keep_mask[i]:
            continue
        lr_i = df_sorted.loc[i, 'lr']

        # Check all subsequent rows for duplicates
        for j in range(i + 1, len(df_sorted)):
            if not keep_mask[j]:
                continue
            lr_j = df_sorted.loc[j, 'lr']

            # Calculate relative difference
            rel_diff = abs(lr_i - lr_j) / max(lr_i, lr_j)

            if rel_diff < lr_tolerance:
                # Mark the later one (higher loss) for removal
                keep_mask[j] = False

    result = df_sorted[keep_mask].copy()
    n_removed = len(df_sorted) - len(result)
    if n_removed > 0:
        print(f"    Deduplication: removed {n_removed} duplicate LRs (within {lr_tolerance*100:.0f}% tolerance)")

    return result


def get_top_k_lrs_for_omega(data_df, target_omega, top_k=5, omega_tolerance=0.1, lr_tolerance=0.05):
    """
    Get top K LRs with smallest validation losses for a given omega value.
    Returns tuple of (top_k_results, other_results) where:
    - top_k_results: list of dicts with 'lr', 'val_loss', 'weight' for top K points
    - other_results: list of dicts with 'lr', 'val_loss', 'distance_from_optimal' for other points

    Deduplicates LRs that differ by less than lr_tolerance before selecting top K.
    """
    # Filter data to target omega
    omega_data = data_df[np.abs(data_df['omega'] - target_omega) < omega_tolerance].copy()

    if len(omega_data) == 0:
        print(f"    No data found for omega={target_omega:.3f}")
        return [], []

    # Deduplicate similar LRs, keeping the one with lowest loss
    omega_data = deduplicate_lrs(omega_data, lr_tolerance=lr_tolerance)

    # Sort by validation loss
    omega_data_sorted = omega_data.sort_values('val_loss')

    # Get the best validation loss for distance calculation
    best_val_loss = omega_data_sorted['val_loss'].iloc[0]

    # Top K points with weights
    top_k_results = []
    for rank, (idx, row) in enumerate(omega_data_sorted.head(top_k).iterrows()):
        weight = top_k - rank  # K, K-1, K-2, ..., 1
        top_k_results.append({
            'lr': row['lr'],
            'val_loss': row['val_loss'],
            'weight': weight
        })

    # Other points (not in top K) with distance from optimal
    other_results = []
    for idx, row in omega_data_sorted.iloc[top_k:].iterrows():
        distance = row['val_loss'] - best_val_loss
        other_results.append({
            'lr': row['lr'],
            'val_loss': row['val_loss'],
            'distance_from_optimal': distance
        })

    print(f"    Found {len(top_k_results)} top-K LRs and {len(other_results)} other LRs at omega≈{target_omega:.3f}")
    if len(top_k_results) > 0:
        print(f"    Top-K LR range: {min(r['lr'] for r in top_k_results):.6f} to {max(r['lr'] for r in top_k_results):.6f}")
        print(f"    Top-K weights: {[r['weight'] for r in top_k_results]}")
    if len(other_results) > 0:
        print(f"    Other LR range: {min(r['lr'] for r in other_results):.6f} to {max(r['lr'] for r in other_results):.6f}")
        print(f"    Distance range: {min(r['distance_from_optimal'] for r in other_results):.6f} to {max(r['distance_from_optimal'] for r in other_results):.6f}")

    return top_k_results, other_results

def load_wandb_data_simple(project_name, group_name, entity, optimizer_type, scaling_rule,
                           target_clipsnr=None, clipsnr_tolerance=0.1, wd_decaying_filter=False,
                           target_residual_exponent=None, smallest_head=None,
                           target_kappa=None, kappa_tolerance=0.05):
    """Load data from WandB

    Args:
        scaling_rule: One of 'BigHead', 'EggHead', 'Enoki', 'Enoki_std', 'Enoki_Scaled', 'Enoki_Scaled_noqk', 'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', or 'Qwen3_Hoyer' to determine size parameter
        wd_decaying_filter: If True, only include runs with wd_decaying=True (for Manau)
        target_residual_exponent: If provided, filter runs where residual_stream_scalar ≈ n_layer^exponent (for Enoki_std)
        smallest_head: If provided, only include runs with head sizes >= this value
        target_kappa: If provided, filter runs where kappa ≈ target_kappa (for DANA optimizers)
        kappa_tolerance: Tolerance for kappa matching
    """
    api = wandb.Api()

    print(f"Loading data from {group_name}...")
    print(f"Scaling rule: {scaling_rule}")
    if wd_decaying_filter:
        print(f"Filtering for wd_decaying=True")
    if target_residual_exponent is not None:
        print(f"Filtering for residual_stream_scalar ≈ n_layer^{target_residual_exponent}")
    if smallest_head is not None:
        print(f"Filtering for head sizes >= {smallest_head}")
    if target_kappa is not None:
        print(f"Filtering for kappa ≈ {target_kappa} (tolerance: {kappa_tolerance})")

    runs = api.runs(f"{entity}/{project_name}", filters={"group": group_name})

    data = []
    total_runs = 0
    skipped_optimizer = 0
    skipped_incomplete = 0
    skipped_missing_data = 0
    skipped_clipsnr = 0
    skipped_kappa = 0
    skipped_wd_decaying = 0
    skipped_residual_exponent = 0
    skipped_smallest_head = 0

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

        # Filter by kappa if specified (for DANA optimizers)
        if target_kappa is not None:
            kappa = config.get('kappa')
            if kappa is None or abs(kappa - target_kappa) > kappa_tolerance:
                skipped_kappa += 1
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
        elif scaling_rule in ('Enoki', 'Enoki_std', 'Enoki_Scaled', 'Enoki_Scaled_noqk', 'Enoki_512'):
            # Enoki variants use n_head as heads
            size = config.get('n_head')
            size_name = 'heads'
        elif scaling_rule == 'Eryngii' or scaling_rule == 'Eryngii_Scaled':
            # Eryngii and Eryngii_Scaled use n_head as heads
            size = config.get('n_head')
            size_name = 'heads'
        elif scaling_rule == 'Qwen3_Scaled' or scaling_rule == 'Qwen3_Hoyer':
            # Qwen3_Scaled / Qwen3_Hoyer use n_head as heads
            size = config.get('n_head')
            size_name = 'heads'
        else:
            raise ValueError(f"Unknown scaling rule: {scaling_rule}")

        if size is None:
            skipped_missing_data += 1
            continue

        # Filter by smallest_head if specified
        if smallest_head is not None:
            # For head-based scaling rules, check n_head directly
            if scaling_rule in ['EggHead', 'Enoki', 'Enoki_std', 'Enoki_Scaled', 'Enoki_Scaled_noqk', 'Enoki_512', 'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', 'Qwen3_Hoyer']:
                if size < smallest_head:
                    skipped_smallest_head += 1
                    continue
            # For BigHead, n_head == n_layer == depth, so also filter
            elif scaling_rule == 'BigHead':
                if size < smallest_head:
                    skipped_smallest_head += 1
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
        if optimizer_type in ['dana-star-mk4', 'dana', 'adamw-decaying-wd', 'ademamix-decaying-wd', 'dana-mk4', 'dana-star-no-tau', 'dana-star', 'dana-star-no-tau-kappa-0-8', 'dana-star-no-tau-kappa-0-85', 'dana-star-no-tau-kappa-0-9', 'dana-mk4-kappa-0-85', 'dana-star-no-tau-beta1', 'dana-star-no-tau-dana-constant', 'dana-star-no-tau-beta2-constant', 'dana-star-no-tau-dana-constant-beta2-constant', 'dana-star-no-tau-dana-constant-beta1', 'dana-star-no-tau-dana-constant-beta2-constant-beta1']:
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
    if skipped_kappa > 0:
        print(f"  Skipped {skipped_kappa} runs due to kappa filter")
    if skipped_wd_decaying > 0:
        print(f"  Skipped {skipped_wd_decaying} runs due to wd_decaying filter")
    if skipped_smallest_head > 0:
        print(f"  Skipped {skipped_smallest_head} runs due to smallest_head filter")
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
    # Initial guess: a = 10, b = 30000, d = -0.5
    init_a_raw = jnp.log(10.0)
    init_b_raw = jnp.log(30000.0)
    init_d = -0.5

    # =========================================================================
    # WARMUP PHASE: Optimize only 'a' for 1000 steps (keep b and d fixed)
    # =========================================================================
    @jit
    def warmup_loss_fn(a_raw, b_raw, d):
        a = jnp.exp(a_raw)
        b = jnp.exp(b_raw)
        pred_lrs = a * ((b + params_arr) ** d)
        log_pred_lrs = jnp.log(pred_lrs)
        residuals = (log_lrs - log_pred_lrs) ** 2
        combined_weights = weights**2 * params_arr
        return jnp.sum(combined_weights * residuals) / jnp.sum(combined_weights)

    warmup_grad_fn = jit(grad(warmup_loss_fn, argnums=0))  # gradient w.r.t. a_raw only
    warmup_optimizer = optax.adagrad(learning_rate)
    warmup_opt_state = warmup_optimizer.init(init_a_raw)

    a_raw = init_a_raw
    b_raw = init_b_raw
    d = init_d

    for step in range(1000):
        grad_a = warmup_grad_fn(a_raw, b_raw, d)
        updates, warmup_opt_state = warmup_optimizer.update(grad_a, warmup_opt_state)
        a_raw = optax.apply_updates(a_raw, updates)

    # Print warmup result
    warmup_loss = float(warmup_loss_fn(a_raw, b_raw, d))
    print(f"  Warmup complete: loss={warmup_loss:.6e}, a={float(jnp.exp(a_raw)):.6e}, b={float(jnp.exp(b_raw)):.6e}, d={float(d):.4f}")

    # =========================================================================
    # MAIN OPTIMIZATION: Optimize all parameters
    # =========================================================================
    fit_params = jnp.array([a_raw, b_raw, d], dtype=jnp.float32)

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


# =============================================================================
# BOOTSTRAP ANALYSIS FUNCTIONS
# =============================================================================

def bootstrap_power_law_fit(
    model_results: Dict,
    scaling_rule: str,
    n_bootstrap: int = 1000,
    n_steps: int = 50000,
    learning_rate: float = 1.0,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap analysis at the model-size level.

    For each bootstrap sample:
    - Resample model sizes with replacement
    - For each sampled size, use all its top-K LRs and weights
    - Fit the power law: η(P) = a × (b + P)^d

    Args:
        model_results: Dictionary mapping size -> {'non_emb_params', 'top_k_data', ...}
        scaling_rule: Scaling rule name for parameter computation
        n_bootstrap: Number of bootstrap samples
        n_steps: Optimization steps for each fit
        learning_rate: Learning rate for optimizer
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (a_samples, b_samples, d_samples) arrays of shape (n_bootstrap,)
    """
    np.random.seed(random_seed)

    # Get list of sizes that have data (and are not excluded)
    sizes_with_data = [
        size for size, data in model_results.items()
        if not data.get('excluded', False) and len(data.get('top_k_data', [])) > 0
    ]

    if len(sizes_with_data) < 2:
        raise ValueError(f"Need at least 2 sizes with data for bootstrap, got {len(sizes_with_data)}")

    n_sizes = len(sizes_with_data)
    print(f"\n{'='*70}")
    print(f"Bootstrap Analysis: {n_bootstrap} samples, {n_sizes} model sizes")
    print(f"Sizes with data: {sizes_with_data}")
    print(f"{'='*70}")

    a_samples = np.zeros(n_bootstrap)
    b_samples = np.zeros(n_bootstrap)
    d_samples = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        if i % 100 == 0:
            print(f"  Bootstrap sample {i}/{n_bootstrap}...")

        # Resample model sizes with replacement
        resampled_sizes = np.random.choice(sizes_with_data, size=n_sizes, replace=True)

        # Collect data from resampled sizes
        params_list = []
        lrs_list = []
        weights_list = []

        for size in resampled_sizes:
            data = model_results[size]
            non_emb_params = data['non_emb_params']

            for item in data['top_k_data']:
                params_list.append(non_emb_params)
                lrs_list.append(item['lr'])
                weights_list.append(item['weight'])

        # Fit power law to this bootstrap sample (with reduced verbosity)
        try:
            a, b, d = fit_saturated_power_law_weighted_quiet(
                params_list, lrs_list, weights_list,
                n_steps=n_steps, learning_rate=learning_rate
            )
            a_samples[i] = a
            b_samples[i] = b
            d_samples[i] = d
        except Exception as e:
            print(f"    Warning: Bootstrap sample {i} failed: {e}")
            # Use NaN for failed fits
            a_samples[i] = np.nan
            b_samples[i] = np.nan
            d_samples[i] = np.nan

    # Report how many samples succeeded
    valid_mask = ~np.isnan(a_samples)
    n_valid = np.sum(valid_mask)
    print(f"\n  Bootstrap complete: {n_valid}/{n_bootstrap} samples succeeded")

    return a_samples, b_samples, d_samples


def fit_saturated_power_law_weighted_quiet(params_list, lrs_list, weights_list, n_steps=50000, learning_rate=1.0):
    """
    Same as fit_saturated_power_law_weighted but without print statements.
    Used for bootstrap resampling.
    """
    # Convert to JAX arrays
    params_arr = jnp.array(params_list, dtype=jnp.float32)
    lrs = jnp.array(lrs_list, dtype=jnp.float32)
    weights = jnp.array(weights_list, dtype=jnp.float32)
    log_lrs = jnp.log(lrs)

    # Initialize parameters
    init_a_raw = jnp.log(10.0)
    init_b_raw = jnp.log(30000.0)
    init_d = -0.5

    # Warmup phase
    @jit
    def warmup_loss_fn(a_raw, b_raw, d):
        a = jnp.exp(a_raw)
        b = jnp.exp(b_raw)
        pred_lrs = a * ((b + params_arr) ** d)
        log_pred_lrs = jnp.log(pred_lrs)
        residuals = (log_lrs - log_pred_lrs) ** 2
        combined_weights = weights**2 * params_arr
        return jnp.sum(combined_weights * residuals) / jnp.sum(combined_weights)

    warmup_grad_fn = jit(grad(warmup_loss_fn, argnums=0))
    warmup_optimizer = optax.adagrad(learning_rate)
    warmup_opt_state = warmup_optimizer.init(init_a_raw)

    a_raw = init_a_raw
    b_raw = init_b_raw
    d = init_d

    for step in range(1000):
        grad_a = warmup_grad_fn(a_raw, b_raw, d)
        updates, warmup_opt_state = warmup_optimizer.update(grad_a, warmup_opt_state)
        a_raw = optax.apply_updates(a_raw, updates)

    # Main optimization
    fit_params = jnp.array([a_raw, b_raw, d], dtype=jnp.float32)

    @jit
    def loss_fn(fit_params):
        a_raw, b_raw, d = fit_params
        a = jnp.exp(a_raw)
        b = jnp.exp(b_raw)
        pred_lrs = a * ((b + params_arr) ** d)
        log_pred_lrs = jnp.log(pred_lrs)
        residuals = (log_lrs - log_pred_lrs) ** 2
        combined_weights = weights**2 * params_arr
        weighted_loss = jnp.sum(combined_weights * residuals) / jnp.sum(combined_weights)
        return weighted_loss

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

    a_raw, b_raw, d = best_params
    a = float(jnp.exp(a_raw))
    b = float(jnp.exp(b_raw))
    d = float(d)

    return a, b, d


def compute_confidence_intervals(
    a_samples: np.ndarray,
    b_samples: np.ndarray,
    d_samples: np.ndarray,
    confidence_level: float = 0.95,
    extrapolation_params: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Compute confidence intervals from bootstrap samples.

    Args:
        a_samples, b_samples, d_samples: Bootstrap samples of parameters
        confidence_level: Confidence level (default 0.95 for 95% CI)
        extrapolation_params: List of parameter values at which to compute LR predictions with CIs

    Returns:
        Dictionary with confidence intervals for parameters and (optionally) LR predictions
    """
    # Remove NaN samples
    valid_mask = ~(np.isnan(a_samples) | np.isnan(b_samples) | np.isnan(d_samples))
    a_valid = a_samples[valid_mask]
    b_valid = b_samples[valid_mask]
    d_valid = d_samples[valid_mask]

    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    results = {
        'confidence_level': confidence_level,
        'n_valid_samples': int(np.sum(valid_mask)),
        'parameters': {
            'a': {
                'mean': float(np.mean(a_valid)),
                'std': float(np.std(a_valid)),
                'median': float(np.median(a_valid)),
                'ci_lower': float(np.percentile(a_valid, lower_percentile)),
                'ci_upper': float(np.percentile(a_valid, upper_percentile)),
            },
            'b': {
                'mean': float(np.mean(b_valid)),
                'std': float(np.std(b_valid)),
                'median': float(np.median(b_valid)),
                'ci_lower': float(np.percentile(b_valid, lower_percentile)),
                'ci_upper': float(np.percentile(b_valid, upper_percentile)),
            },
            'd': {
                'mean': float(np.mean(d_valid)),
                'std': float(np.std(d_valid)),
                'median': float(np.median(d_valid)),
                'ci_lower': float(np.percentile(d_valid, lower_percentile)),
                'ci_upper': float(np.percentile(d_valid, upper_percentile)),
            },
        }
    }

    # Compute LR predictions with confidence intervals at extrapolation points
    if extrapolation_params is not None:
        lr_predictions = {}
        for p in extrapolation_params:
            # Compute LR for each bootstrap sample
            lr_samples = a_valid * ((b_valid + p) ** d_valid)

            lr_predictions[str(int(p))] = {
                'params': float(p),
                'lr_mean': float(np.mean(lr_samples)),
                'lr_std': float(np.std(lr_samples)),
                'lr_median': float(np.median(lr_samples)),
                'lr_ci_lower': float(np.percentile(lr_samples, lower_percentile)),
                'lr_ci_upper': float(np.percentile(lr_samples, upper_percentile)),
            }

        results['lr_predictions'] = lr_predictions

    return results


def compute_lr_confidence_band(
    params_range: np.ndarray,
    a_samples: np.ndarray,
    b_samples: np.ndarray,
    d_samples: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute confidence band for LR predictions across a range of parameters.

    Args:
        params_range: Array of parameter values
        a_samples, b_samples, d_samples: Bootstrap samples
        confidence_level: Confidence level

    Returns:
        Tuple of (lr_median, lr_lower, lr_upper) arrays
    """
    # Remove NaN samples
    valid_mask = ~(np.isnan(a_samples) | np.isnan(b_samples) | np.isnan(d_samples))
    a_valid = a_samples[valid_mask]
    b_valid = b_samples[valid_mask]
    d_valid = d_samples[valid_mask]

    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    n_params = len(params_range)
    lr_median = np.zeros(n_params)
    lr_lower = np.zeros(n_params)
    lr_upper = np.zeros(n_params)

    for i, p in enumerate(params_range):
        # Compute LR for each valid bootstrap sample
        lr_samples = a_valid * ((b_valid + p) ** d_valid)

        lr_median[i] = np.median(lr_samples)
        lr_lower[i] = np.percentile(lr_samples, lower_percentile)
        lr_upper[i] = np.percentile(lr_samples, upper_percentile)

    return lr_median, lr_lower, lr_upper


def collect_weighted_data_for_depths(optimizer_type, scaling_rule, target_omega, top_k, project, group, entity,
                                      exclude_small=False, target_clipsnr=None, clipsnr_tolerance=0.1,
                                      wd_decaying_filter=False, target_residual_exponent=None, smallest_head=None,
                                      target_kappa=None, kappa_tolerance=0.05):
    """
    Collect top-K LRs for each size at target omega, with weights.

    Args:
        scaling_rule: One of 'BigHead', 'EggHead', 'Enoki', 'Enoki_std', 'Enoki_Scaled', 'Enoki_Scaled_noqk', 'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', or 'Qwen3_Hoyer'
        wd_decaying_filter: If True, only include runs with wd_decaying=True (for Manau)
        target_residual_exponent: If provided, filter runs where residual_stream_scalar ≈ n_layer^exponent
        smallest_head: If provided, only include runs with head sizes >= this value
        target_kappa: If provided, filter runs where kappa ≈ target_kappa
        kappa_tolerance: Tolerance for kappa matching
    """
    # Load all data for the optimizer
    data_df = load_wandb_data_simple(project, group, entity, optimizer_type, scaling_rule,
                                     target_clipsnr, clipsnr_tolerance, wd_decaying_filter,
                                     target_residual_exponent, smallest_head,
                                     target_kappa, kappa_tolerance)

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
    # New lists for non-top-K points (to be shown as grey dots)
    all_non_emb_params_other = []
    all_lrs_other = []
    all_distances_other = []
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

        # Get top K LRs at this omega (now returns tuple of top_k and other)
        top_k_data, other_data = get_top_k_lrs_for_omega(size_data, closest_omega, top_k=top_k)

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

        # Collect non-top-K points (always collect, regardless of exclusion)
        for item in other_data:
            all_non_emb_params_other.append(non_emb_params)
            all_lrs_other.append(item['lr'])
            all_distances_other.append(item['distance_from_optimal'])

        # Store for plotting
        results[size] = {
            'size': size,
            'size_name': size_name,
            'non_emb_params': non_emb_params,
            'total_params': total_params,
            'compute': compute_metric,
            'closest_omega': closest_omega,
            'top_k_data': top_k_data,
            'other_data': other_data,  # Add other data for plotting
            'data_df': size_data,
            'excluded': exclude_small and is_small
        }

    return (all_non_emb_params, all_total_params, all_compute, all_lrs, all_weights,
            all_non_emb_params_excluded, all_total_params_excluded, all_compute_excluded,
            all_lrs_excluded, all_weights_excluded,
            all_non_emb_params_other, all_lrs_other, all_distances_other, results)

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
    if args.target_kappa is not None:
        print(f"Target Kappa: {args.target_kappa} (tolerance: {args.kappa_tolerance})")
    if args.wd_decaying:
        print(f"Filtering for wd_decaying=True")
    if args.target_residual_exponent is not None:
        print(f"Target Residual Exponent: {args.target_residual_exponent} (residual_stream_scalar ≈ n_layer^{args.target_residual_exponent}, 10% tolerance)")
    if args.smallest_head is not None:
        print(f"Smallest Head: {args.smallest_head} (ignoring head sizes < {args.smallest_head})")
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
        target_residual_exponent=args.target_residual_exponent,
        smallest_head=args.smallest_head,
        target_kappa=args.target_kappa,
        kappa_tolerance=args.kappa_tolerance
    )

    if result is None:
        exit(1)

    (non_emb_params, total_params, compute, lrs, weights,
     non_emb_params_exc, total_params_exc, compute_exc, lrs_exc, weights_exc,
     non_emb_params_other, lrs_other, distances_other, model_results) = result

    if len(non_emb_params) == 0:
        print("No data collected. Exiting.")
        exit(1)

    print(f"\n{'='*70}")
    print("Collected Data Summary")
    print(f"{'='*70}")
    print(f"Top-K data points (used in fit): {len(non_emb_params)}")
    print(f"Other data points (shown as grey): {len(non_emb_params_other)}")
    print(f"Non-embedding params range: {min(non_emb_params):,} to {max(non_emb_params):,}")
    print(f"Total params range: {min(total_params):,} to {max(total_params):,}")
    print(f"Compute range: {min(compute):.2e} to {max(compute):.2e}")
    print(f"LR range: {min(lrs):.6e} to {max(lrs):.6e}")
    print(f"Weight range: {min(weights)} to {max(weights)}")
    if len(distances_other) > 0:
        print(f"Distance from optimal range (grey points): {min(distances_other):.6f} to {max(distances_other):.6f}")

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
    # BOOTSTRAP ANALYSIS
    # =============================================================================

    bootstrap_results = None
    a_samples, b_samples, d_samples = None, None, None

    if args.bootstrap:
        # Run bootstrap analysis
        a_samples, b_samples, d_samples = bootstrap_power_law_fit(
            model_results=model_results,
            scaling_rule=args.scaling_rule,
            n_bootstrap=args.n_bootstrap,
            n_steps=50000,  # Reduced steps for bootstrap (faster)
            learning_rate=args.learning_rate,
            random_seed=42
        )

        # Compute confidence intervals
        # Get extrapolation parameter values for LR predictions
        all_prediction_sizes = scaling_config['extrapolation_sizes']
        extrapolation_params = [
            float(compute_non_embedding_params(s, args.scaling_rule))
            for s in all_prediction_sizes
        ]

        bootstrap_results = compute_confidence_intervals(
            a_samples, b_samples, d_samples,
            confidence_level=args.confidence_level,
            extrapolation_params=extrapolation_params
        )

        # Print bootstrap results
        print(f"\n{'='*70}")
        print(f"Bootstrap Confidence Intervals ({args.confidence_level*100:.0f}%)")
        print(f"{'='*70}")
        print(f"Valid bootstrap samples: {bootstrap_results['n_valid_samples']}/{args.n_bootstrap}")

        print("\nParameter estimates with confidence intervals:")
        for param_name in ['a', 'b', 'd']:
            p = bootstrap_results['parameters'][param_name]
            if param_name == 'd':
                print(f"  {param_name}: {p['mean']:.4f} [{p['ci_lower']:.4f}, {p['ci_upper']:.4f}] (std: {p['std']:.4f})")
            else:
                print(f"  {param_name}: {p['mean']:.2e} [{p['ci_lower']:.2e}, {p['ci_upper']:.2e}] (std: {p['std']:.2e})")

        if 'lr_predictions' in bootstrap_results:
            print("\nExtrapolated LR predictions with confidence intervals:")
            sizes_with_data = set(model_results.keys())
            for size_pred in all_prediction_sizes:
                non_emb_pred = float(compute_non_embedding_params(size_pred, args.scaling_rule))
                key = str(int(non_emb_pred))
                if key in bootstrap_results['lr_predictions']:
                    pred = bootstrap_results['lr_predictions'][key]
                    has_data = size_pred in sizes_with_data
                    marker = "[HAS DATA]" if has_data else "[EXTRAPOLATED]"
                    print(f"  Size={size_pred} {marker}: LR = {pred['lr_mean']:.2e} [{pred['lr_ci_lower']:.2e}, {pred['lr_ci_upper']:.2e}]")

        # Save bootstrap results to JSON if requested
        if args.output_bootstrap_json:
            # Add metadata to results
            bootstrap_results['metadata'] = {
                'scaling_rule': args.scaling_rule,
                'optimizer': args.optimizer,
                'target_omega': args.target_omega,
                'top_k': args.top_k,
                'n_bootstrap': args.n_bootstrap,
                'point_estimate': {
                    'a': a_fit,
                    'b': b_fit,
                    'd': d_fit
                }
            }

            with open(args.output_bootstrap_json, 'w') as f:
                json.dump(bootstrap_results, f, indent=2)
            print(f"\nBootstrap results saved to: {args.output_bootstrap_json}")

    # =============================================================================
    # VISUALIZATION
    # =============================================================================

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    # Plot all data points with size proportional to weight
    all_unique_params = sorted(set(non_emb_params) | set(non_emb_params_exc) | set(non_emb_params_other))
    colors = cm.viridis(np.linspace(0, 1, len(all_unique_params)))
    color_map = {p: colors[i] for i, p in enumerate(all_unique_params)}

    # Plot non-top-K points first (grey dots with size inversely proportional to distance from optimal)
    # These are plotted first so they appear behind the top-K points
    if len(non_emb_params_other) > 0:
        distances_arr = np.array(distances_other)
        # Calculate sizes: larger distance = smaller dot
        # Use inverse relationship: size = base_size / (1 + distance * scale_factor)
        # Normalize distances to [0, 1] range for consistent sizing
        max_dist = distances_arr.max() if distances_arr.max() > 0 else 1.0
        normalized_dist = distances_arr / max_dist
        # Size ranges from ~10 (worst) to ~100 (best among non-top-K)
        # Smallest size for largest distance, larger size for smaller distance
        other_sizes = 100 * (1 - 0.9 * normalized_dist)  # ranges from 10 to 100

        for p, lr, sz in zip(non_emb_params_other, lrs_other, other_sizes):
            ax.scatter(p, lr, s=sz, c='lightgray', alpha=0.4, edgecolors='gray', linewidths=0.3, zorder=1)

    # Plot included points (used in fit) - top-K
    for p, lr, w in zip(non_emb_params, lrs, weights):
        ax.scatter(p, lr, s=w*50, c=[color_map[p]], alpha=0.6, edgecolors='black', linewidths=0.5, zorder=2)

    # Plot excluded points (grayed out with 'x' marker) - top-K that were excluded
    if len(non_emb_params_exc) > 0:
        for p, lr, w in zip(non_emb_params_exc, lrs_exc, weights_exc):
            ax.scatter(p, lr, s=w*50, c='gray', alpha=0.3, marker='x', linewidths=1.5, zorder=2,
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

    # Plot confidence band if bootstrap was run and --plot-confidence-band is set
    if args.bootstrap and args.plot_confidence_band and a_samples is not None:
        lr_median, lr_lower, lr_upper = compute_lr_confidence_band(
            params_range, a_samples, b_samples, d_samples,
            confidence_level=args.confidence_level
        )
        ax.fill_between(params_range, lr_lower, lr_upper, alpha=0.2, color='tab:orange',
                       label=f'{args.confidence_level*100:.0f}% CI', zorder=5)

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

    # Mark predictions at sizes WITHOUT data (unless --no-diamonds is set)
    if not args.no_diamonds:
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
    axis_fontsize = args.title_fontsize if args.title_fontsize else 20
    ax.set_xlabel('Non-embedding Parameters', fontsize=axis_fontsize)
    ax.set_ylabel('Learning Rate (LR)', fontsize=axis_fontsize)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set y-limits based on top-k LR range with 10% margin
    if len(lrs) > 0:
        min_lr = min(lrs)
        max_lr = max(lrs)
        ax.set_ylim(min_lr * 0.9, max_lr * 1.1)

    # Add second x-axis showing 'size' variable
    ax2 = ax.twiny()

    # Get all sizes (both with data and extrapolated)
    all_sizes_for_axis = sorted(set(list(sizes_with_data) + all_prediction_sizes))

    # Compute non-embedding params for these sizes
    size_to_params = {size: compute_non_embedding_params(size, args.scaling_rule) for size in all_sizes_for_axis}

    # Set up the second axis with size labels at corresponding parameter positions
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())

    # Apply skip-ticks-after logic: show every other tick past a certain size
    # Also apply hide-ticks to suppress specific labels
    hide_set = set(args.hide_ticks) if args.hide_ticks else set()

    if args.skip_ticks_after is not None:
        # Separate sizes into "always show" and "conditionally show"
        tick_positions = []
        tick_labels = []
        skip_next = False
        for size in all_sizes_for_axis:
            tick_positions.append(size_to_params[size])
            if size in hide_set:
                tick_labels.append('')  # Hidden
            elif size <= args.skip_ticks_after:
                tick_labels.append(str(size))
                skip_next = False
            else:
                # Past the threshold, show every other
                if skip_next:
                    tick_labels.append('')  # Empty label
                    skip_next = False
                else:
                    tick_labels.append(str(size))
                    skip_next = True
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels)
    else:
        tick_positions = [size_to_params[size] for size in all_sizes_for_axis]
        tick_labels = ['' if size in hide_set else str(size) for size in all_sizes_for_axis]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels)

    ax2.set_xlabel(f'{size_name.capitalize()}', fontsize=axis_fontsize)

    # Apply tick fontsize if specified
    if args.tick_fontsize is not None:
        ax.tick_params(axis='both', which='major', labelsize=args.tick_fontsize)
        ax2.tick_params(axis='both', which='major', labelsize=args.tick_fontsize)

    optimizer_title_map = {'adamw': 'AdamW', 'mk4': 'Dana-Star-MK4', 'dana': 'Dana-Star', 'ademamix': 'AdemaMix', 'd-muon': 'D-Muon', 'manau': 'Manau', 'adamw-decaying-wd': 'AdamW-Decaying-WD', 'dana-mk4': 'DANA-MK4', 'ademamix-decaying-wd': 'Ademamix-Decaying-WD', 'dana-star-no-tau': 'ADana', 'dana-star': 'Dana-Star', 'dana-star-no-tau-kappa-0-8': r'ADana-$\kappa$=0.8', 'dana-star-no-tau-kappa-0-85': r'ADana-$\kappa$=0.85', 'dana-star-no-tau-kappa-0-9': r'ADana-$\kappa$=0.9', 'dana-mk4-kappa-0-85': r'Dana-MK4-$\kappa$=0.85', 'dana-star-no-tau-beta1': 'Dana-Star-No-Tau-beta1', 'dana-star-no-tau-dana-constant': 'Dana-Star-No-Tau-Dana-Constant', 'dana-star-no-tau-beta2-constant': 'Dana-Star-No-Tau-beta2-Constant', 'dana-star-no-tau-dana-constant-beta2-constant': 'Dana-Star-No-Tau-Dana-Constant-beta2-Constant', 'dana-star-no-tau-dana-constant-beta1': 'Dana-Star-No-Tau-Dana-Constant-beta1', 'dana-star-no-tau-dana-constant-beta2-constant-beta1': 'Dana-Star-No-Tau-Dana-Constant-beta2-Constant-beta1'}
    optimizer_title = optimizer_title_map[args.optimizer]

    # Build title based on --simple-title flag
    if args.simple_title:
        # Map scaling rule to model name
        model_name_map = {
            'Enoki': 'Enoki',
            'Enoki_Scaled': 'Enoki',
            'Enoki_std': 'Enoki',
            'Enoki_Scaled_noqk': 'Enoki',
            'Enoki_512': 'Enoki_512',
            'Qwen3_Scaled': 'Qwen3',
            'Qwen3_Hoyer': 'Qwen3',
            'BigHead': 'BigHead',
            'EggHead': 'EggHead',
            'Eryngii': 'Eryngii',
            'Eryngii_Scaled': 'Eryngii',
        }
        model_name = model_name_map.get(args.scaling_rule, args.scaling_rule)
        title = f'{model_name} {optimizer_title} LR Scaling Law Fit'
    else:
        title_parts = [f'ω = {args.target_omega}', f'Top-K = {args.top_k}']
        if args.target_clipsnr is not None:
            title_parts.append(f'ClipSNR = {args.target_clipsnr}')
        if args.wd_decaying:
            title_parts.append('wd_decaying=True')
        title_params = ', '.join(title_parts)
        title = f'{args.scaling_rule} {optimizer_title} Optimal Learning Rate Scaling Law\n({title_params})'

    ax.set_title(title, fontsize=args.title_fontsize)
    ax.legend(fontsize=args.legend_fontsize, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.text(0.02, 0.02, 'Colored: top-K LRs (size ∝ weight)\nGrey: other LRs (size ∝ closeness to optimal)',
            transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    if args.output:
        output_file = args.output
    else:
        optimizer_filename_map = {'adamw': 'AdamW', 'mk4': 'DanaStar-MK4', 'dana': 'DanaStar', 'ademamix': 'Ademamix', 'd-muon': 'DMuon', 'manau': 'Manau', 'adamw-decaying-wd': 'AdamW-Decaying-WD', 'dana-mk4': 'DanaMK4', 'ademamix-decaying-wd': 'Ademamix-Decaying-WD', 'dana-star-no-tau': 'Dana-Star-No-Tau', 'dana-star': 'Dana-Star', 'dana-star-no-tau-kappa-0-8': 'Dana-Star-No-Tau-kappa-0-8', 'dana-star-no-tau-kappa-0-85': 'Dana-Star-No-Tau-kappa-0-85', 'dana-star-no-tau-kappa-0-9': 'Dana-Star-No-Tau-kappa-0-9', 'dana-mk4-kappa-0-85': 'Dana-MK4-kappa-0-85', 'dana-star-no-tau-beta1': 'Dana-Star-No-Tau-beta1', 'dana-star-no-tau-dana-constant': 'Dana-Star-No-Tau-dana-constant', 'dana-star-no-tau-beta2-constant': 'Dana-Star-No-Tau-beta2-constant', 'dana-star-no-tau-dana-constant-beta2-constant': 'Dana-Star-No-Tau-dana-constant-beta2-constant', 'dana-star-no-tau-dana-constant-beta1': 'Dana-Star-No-Tau-dana-constant-beta1', 'dana-star-no-tau-dana-constant-beta2-constant-beta1': 'Dana-Star-No-Tau-dana-constant-beta2-constant-beta1'}
        optimizer_name = optimizer_filename_map[args.optimizer]
        # Map scaling rule to model name for filename
        model_name_map = {
            'Enoki': 'Enoki',
            'Enoki_Scaled': 'Enoki',
            'Enoki_std': 'Enoki',
            'Enoki_Scaled_noqk': 'Enoki',
            'Enoki_512': 'Enoki_512',
            'Qwen3_Scaled': 'Qwen3',
            'Qwen3_Hoyer': 'Qwen3',
            'BigHead': 'BigHead',
            'EggHead': 'EggHead',
            'Eryngii': 'Eryngii',
            'Eryngii_Scaled': 'Eryngii',
        }
        model_name = model_name_map.get(args.scaling_rule, args.scaling_rule)
        if args.output_suffix:
            output_file = f'{model_name}-{optimizer_name}-{args.output_suffix}.pdf'
        else:
            output_file = f'{model_name}-{optimizer_name}-lr-extrapolation.pdf'

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
