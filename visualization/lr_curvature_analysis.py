#!/usr/bin/env python3
"""
Learning Rate Curvature Analysis

This script analyzes how loss curvature with respect to learning rate varies across model scales.
Key insight: The loss landscape curvature κ around the optimal LR varies with model size,
affecting how LR errors translate to loss increases at different scales.

Analysis Steps:
1. For each model size, load all LRs and losses from sweeps
2. Fit parabola in log-LR space: loss(log η) ≈ L*(P) + κ(P) × (log η - log η*(P))²
3. Fit curvature scaling law: κ(P) = κ₀ × P^α
4. Quantify LR error impact: Δloss(P, ε) = κ(P) × (log(1 + ε))²

Usage:
    python lr_curvature_analysis.py --scaling-rule Enoki_Scaled --optimizer adamw --target-omega 4.0
    python lr_curvature_analysis.py --scaling-rule Enoki_Scaled --optimizer dana-mk4 --target-omega 4.0 --output-json curvature_results.json
"""

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style, rc, rcParams
import argparse
import warnings
import json
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# =============================================================================
# SCALING RULE CONFIGURATION (same as bighead_lr_scaling.py)
# =============================================================================

SCALING_RULE_CONFIG = {
    'BigHead': {
        'group': 'DanaStar_MK4_BigHead_Sweep',
        'extrapolation_sizes': [8, 9, 10, 11, 12, 13, 14, 15],
        'size_step': 1,
    },
    'EggHead': {
        'group': 'DanaStar_MK4_EggHead_Sweep',
        'extrapolation_sizes': [8, 9, 10, 11, 12],
        'size_step': 1,
    },
    'Enoki': {
        'group': 'DanaStar_MK4_Enoki_Sweep',
        'extrapolation_sizes': [8, 12, 16, 20, 24, 28, 32, 36, 40],
        'size_step': 4,
    },
    'Enoki_Scaled': {
        'group': 'Enoki_ScaledGPT',
        'extrapolation_sizes': [8, 12, 16, 20, 24, 28, 32, 36, 40],
        'size_step': 4,
    },
    'Qwen3_Scaled': {
        'group': 'Qwen3_ScaledGPT',
        'extrapolation_sizes': [4, 6, 8, 10, 12, 14, 16],
        'size_step': 2,
    },
    'Qwen3_Hoyer': {
        'group': 'Qwen3_Hoyer',
        'extrapolation_sizes': [4, 6, 8, 10, 12, 14, 16],
        'size_step': 2,
    }
}

# Matplotlib formatting
style.use('default')
rc('font', family='sans-serif')
rcParams['font.weight'] = 'light'
rcParams['font.size'] = 14

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description='Analyze LR curvature scaling across model sizes')
parser.add_argument('--scaling-rule', type=str, required=True,
                    choices=['BigHead', 'EggHead', 'Enoki', 'Enoki_Scaled', 'Qwen3_Scaled', 'Qwen3_Hoyer'],
                    help='Model scaling rule')
parser.add_argument('--optimizer', type=str, required=True,
                    choices=['adamw', 'mk4', 'dana', 'ademamix', 'd-muon', 'dana-mk4'],
                    help='Optimizer type')
parser.add_argument('--target-omega', type=float, default=4.0,
                    help='Target omega value (default: 4.0)')
parser.add_argument('--omega-tolerance', type=float, default=0.1,
                    help='Tolerance for omega matching (default: 0.1)')
parser.add_argument('--project', type=str, default='danastar',
                    help='WandB project name (default: danastar)')
parser.add_argument('--entity', type=str, default='ep-rmt-ml-opt',
                    help='WandB entity name (default: ep-rmt-ml-opt)')
parser.add_argument('--output-json', type=str, default=None,
                    help='Output file for curvature analysis results')
parser.add_argument('--output-plot', type=str, default=None,
                    help='Output file for curvature scaling plot')
parser.add_argument('--min-points-per-size', type=int, default=5,
                    help='Minimum number of LR data points required per size (default: 5)')
parser.add_argument('--target-kappa', type=float, default=None,
                    help='Target kappa for DANA optimizers (default: None)')
parser.add_argument('--kappa-tolerance', type=float, default=0.05,
                    help='Tolerance for kappa matching (default: 0.05)')
args = parser.parse_args()

# Map optimizer abbreviations
optimizer_map = {
    'adamw': 'adamw', 'mk4': 'dana-star-mk4', 'dana': 'dana',
    'ademamix': 'ademamix', 'd-muon': 'd-muon', 'dana-mk4': 'dana-mk4'
}
optimizer_type = optimizer_map[args.optimizer]

# Get scaling rule configuration
scaling_config = SCALING_RULE_CONFIG[args.scaling_rule]
wandb_group = scaling_config['group']

# =============================================================================
# PARAMETER CALCULATION FUNCTIONS
# =============================================================================

def compute_non_embedding_params(size, scaling_rule):
    """Compute non-embedding parameters based on scaling rule."""
    if scaling_rule == 'BigHead':
        depth = size
        n_embd = 16 * depth * depth
        mlp_hidden = 32 * depth * depth
        head_dim = 16 * depth
        n_head = depth
        n_layer = depth
        non_emb = n_layer * (3 * head_dim * n_embd * n_head + n_embd * n_embd + 2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd
    elif scaling_rule == 'EggHead':
        heads = size
        n_embd = 16 * heads * heads
        mlp_hidden = 32 * heads * heads
        head_dim = 16 * heads
        n_head = heads
        n_layer = int(heads * (heads - 1) / 2)
        non_emb = n_layer * (3 * head_dim * n_embd * n_head + n_embd * n_embd + 2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd
    elif scaling_rule in ['Enoki', 'Enoki_Scaled']:
        heads = size
        n_embd = heads * 64
        n_layer = int(3 * heads // 4)
        non_emb = 12 * n_embd * n_embd * n_layer
    elif scaling_rule in ['Qwen3_Scaled', 'Qwen3_Hoyer']:
        heads = size
        head_dim = 128
        n_head = heads
        n_layer = 2 * heads
        n_embd = 128 * heads
        total_qkv_dim = n_head * head_dim
        per_layer = 5 * n_embd * total_qkv_dim + 2 * head_dim + 9 * n_embd * n_embd + 2 * n_embd
        non_emb = n_layer * per_layer + n_embd
    else:
        raise ValueError(f"Unknown scaling rule: {scaling_rule}")
    return int(non_emb)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_sweep_data(project_name, group_name, entity, optimizer_type, scaling_rule,
                    target_omega=4.0, omega_tolerance=0.1,
                    target_kappa=None, kappa_tolerance=0.05):
    """
    Load ALL LRs and losses for each model size at target omega.
    Returns a dictionary mapping size -> DataFrame with 'lr', 'val_loss' columns.
    """
    api = wandb.Api()

    print(f"Loading data from {group_name}...")
    runs = api.runs(f"{entity}/{project_name}", filters={"group": group_name})

    data_by_size = {}

    for run in runs:
        config = run.config
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except:
                continue
        if hasattr(config, 'as_dict'):
            config = config.as_dict()

        # Extract value from nested dict structure
        def extract_value(config_dict):
            result = {}
            for key, val in config_dict.items():
                if isinstance(val, dict) and 'value' in val:
                    result[key] = val['value']
                else:
                    result[key] = val
            return result
        config = extract_value(config)

        summary = run.summary

        # Filter by optimizer
        opt = config.get('opt', '')
        if opt != optimizer_type:
            continue

        # Filter by kappa if specified
        if target_kappa is not None:
            kappa = config.get('kappa')
            if kappa is None or abs(kappa - target_kappa) > kappa_tolerance:
                continue

        # Check completion
        actual_iter = summary.get('iter', 0)
        iterations_config = config.get('iterations', 0)
        if actual_iter > 0 and iterations_config > 0 and actual_iter < iterations_config:
            continue

        lr = config.get('lr')
        val_loss = summary.get('final-val/loss')
        if lr is None or val_loss is None:
            continue

        # Get size
        if scaling_rule == 'BigHead':
            size = config.get('n_layer')
        else:
            size = config.get('n_head')
        if size is None:
            continue

        # Calculate omega
        if optimizer_type in ['dana-star-mk4', 'dana', 'dana-mk4']:
            wd_ts = config.get('wd_ts', 1.0)
            weight_decay = config.get('weight_decay', 1.0)
            omega = wd_ts * lr * weight_decay
        else:
            weight_decay = config.get('weight_decay', 0.1)
            iterations = config.get('iterations', 1)
            omega = weight_decay * lr * iterations

        # Filter by omega
        if abs(omega - target_omega) > omega_tolerance:
            continue

        # Add to data
        if size not in data_by_size:
            data_by_size[size] = []
        data_by_size[size].append({
            'lr': lr,
            'val_loss': val_loss,
            'omega': omega
        })

    # Convert to DataFrames
    result = {}
    for size, data_list in data_by_size.items():
        df = pd.DataFrame(data_list)
        # Deduplicate similar LRs, keeping the best
        df = df.sort_values('val_loss')
        df_dedup = []
        for _, row in df.iterrows():
            is_duplicate = False
            for kept in df_dedup:
                if abs(row['lr'] - kept['lr']) / max(row['lr'], kept['lr']) < 0.05:
                    is_duplicate = True
                    break
            if not is_duplicate:
                df_dedup.append(row.to_dict())
        result[size] = pd.DataFrame(df_dedup)

    return result

# =============================================================================
# CURVATURE FITTING
# =============================================================================

def fit_local_curvature(lrs: np.ndarray, losses: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit parabola in log-LR space: loss = L* + κ(log η - log η*)²

    Args:
        lrs: Array of learning rates
        losses: Array of corresponding losses

    Returns:
        Tuple of (L_star, lr_star, kappa) where:
        - L_star: Optimal loss
        - lr_star: Optimal learning rate
        - kappa: Curvature parameter
    """
    log_lrs = np.log(lrs)
    losses = np.array(losses)

    # Fit quadratic: loss = a*log_lr^2 + b*log_lr + c
    # Then: kappa = a, log_lr* = -b/(2a), L* = c - b^2/(4a)
    coeffs = np.polyfit(log_lrs, losses, 2)
    a, b, c = coeffs

    if a <= 0:
        # Convex fit failed - curvature should be positive
        # Try robust approach: use only points near the minimum
        min_idx = np.argmin(losses)
        n = len(losses)
        # Use points within 1 index of minimum, or at least 3 points
        start = max(0, min_idx - 1)
        end = min(n, min_idx + 2)
        if end - start < 3:
            start = max(0, min_idx - 2)
            end = min(n, min_idx + 3)

        if end - start >= 3:
            local_log_lrs = log_lrs[start:end]
            local_losses = losses[start:end]
            coeffs = np.polyfit(local_log_lrs, local_losses, 2)
            a, b, c = coeffs

    if a <= 0:
        # Still failed - return NaN
        return np.nan, np.nan, np.nan

    kappa = a
    log_lr_star = -b / (2 * a)
    lr_star = np.exp(log_lr_star)
    L_star = c - b**2 / (4 * a)

    return L_star, lr_star, kappa


def compute_average_curvature(curvatures: np.ndarray) -> Tuple[float, float]:
    """
    Compute average curvature (avoiding power law fit due to systematic errors).

    Args:
        curvatures: Array of curvatures for each size

    Returns:
        Tuple of (mean_kappa, std_kappa)
    """
    # Remove NaN values
    valid_mask = ~np.isnan(curvatures)
    curvatures_valid = curvatures[valid_mask]

    if len(curvatures_valid) < 1:
        return np.nan, np.nan

    mean_kappa = np.mean(curvatures_valid)
    std_kappa = np.std(curvatures_valid)

    return mean_kappa, std_kappa


def fit_curvature_scaling_law(sizes: np.ndarray, curvatures: np.ndarray) -> Tuple[float, float]:
    """
    Fit curvature scaling law: κ(P) = κ₀ × P^α

    NOTE: This function is deprecated due to systematic errors in the fit.
    Use compute_average_curvature() instead.

    Args:
        sizes: Array of model sizes (number of non-embedding params)
        curvatures: Array of curvatures for each size

    Returns:
        Tuple of (kappa_0, alpha)
    """
    # Remove NaN values
    valid_mask = ~np.isnan(curvatures)
    sizes_valid = sizes[valid_mask]
    curvatures_valid = curvatures[valid_mask]

    if len(sizes_valid) < 2:
        return np.nan, np.nan

    # Log-linear fit: log(κ) = log(κ₀) + α*log(P)
    log_sizes = np.log(sizes_valid)
    log_curvatures = np.log(curvatures_valid)

    coeffs = np.polyfit(log_sizes, log_curvatures, 1)
    alpha = coeffs[0]
    kappa_0 = np.exp(coeffs[1])

    return kappa_0, alpha


def predict_loss_increase(params: float, lr_error_fraction: float,
                          kappa_0: float, alpha: float) -> float:
    """
    Predict loss increase for a given LR error at a given model size.

    Δloss(P, ε) = κ(P) × (log(1 + ε))²
                = κ₀ × P^α × (log(1 + ε))²

    Args:
        params: Number of parameters
        lr_error_fraction: Relative LR error (e.g., 0.2 for 20% error)
        kappa_0: Curvature scaling coefficient
        alpha: Curvature scaling exponent

    Returns:
        Expected loss increase
    """
    kappa = kappa_0 * (params ** alpha)
    delta_loss = kappa * (np.log(1 + lr_error_fraction))**2
    return delta_loss

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print(f"Learning Rate Curvature Analysis")
    print(f"Scaling Rule: {args.scaling_rule}")
    print(f"Optimizer: {args.optimizer} ({optimizer_type})")
    print(f"Target Omega: {args.target_omega}")
    print("="*70)

    # Load sweep data
    sweep_data = load_sweep_data(
        args.project, wandb_group, args.entity, optimizer_type, args.scaling_rule,
        target_omega=args.target_omega, omega_tolerance=args.omega_tolerance,
        target_kappa=args.target_kappa, kappa_tolerance=args.kappa_tolerance
    )

    if len(sweep_data) == 0:
        print("No data found. Exiting.")
        exit(1)

    print(f"\nFound data for {len(sweep_data)} model sizes")
    for size in sorted(sweep_data.keys()):
        df = sweep_data[size]
        print(f"  Size {size}: {len(df)} LR data points, LR range [{df['lr'].min():.2e}, {df['lr'].max():.2e}]")

    # Fit curvature at each size
    print(f"\n{'='*70}")
    print("Fitting local curvature at each model size")
    print(f"{'='*70}")

    curvature_results = []

    for size in sorted(sweep_data.keys()):
        df = sweep_data[size]

        if len(df) < args.min_points_per_size:
            print(f"  Size {size}: Skipping (only {len(df)} points, need {args.min_points_per_size})")
            continue

        lrs = df['lr'].values
        losses = df['val_loss'].values

        L_star, lr_star, kappa = fit_local_curvature(lrs, losses)

        if np.isnan(kappa):
            print(f"  Size {size}: Curvature fit failed (non-convex)")
            continue

        non_emb_params = compute_non_embedding_params(size, args.scaling_rule)

        curvature_results.append({
            'size': size,
            'non_emb_params': non_emb_params,
            'L_star': L_star,
            'lr_star': lr_star,
            'kappa': kappa,
            'n_points': len(df)
        })

        print(f"  Size {size} (P={non_emb_params:,.0f}):")
        print(f"    L* = {L_star:.4f}, η* = {lr_star:.2e}, κ = {kappa:.4f}")

    if len(curvature_results) < 2:
        print("\nInsufficient data for curvature scaling law fit. Exiting.")
        exit(1)

    # Compute average curvature (avoiding power law fit due to systematic errors)
    print(f"\n{'='*70}")
    print("Computing average curvature across model sizes")
    print(f"{'='*70}")

    sizes_arr = np.array([r['non_emb_params'] for r in curvature_results])
    kappas_arr = np.array([r['kappa'] for r in curvature_results])

    mean_kappa, std_kappa = compute_average_curvature(kappas_arr)

    print(f"\nAverage curvature:")
    print(f"  κ_mean = {mean_kappa:.4e}")
    print(f"  κ_std  = {std_kappa:.4e}")
    print(f"  CV (std/mean) = {std_kappa/mean_kappa:.2%}")

    # For compatibility with JSON output, set these values
    kappa_0 = mean_kappa
    alpha = 0.0

    # Compute loss increase for various LR errors using average curvature
    print(f"\n{'='*70}")
    print("Loss increase for LR errors (using average κ)")
    print(f"{'='*70}")

    lr_errors = [0.1, 0.2, 0.3, 0.5]
    extrapolation_sizes = scaling_config['extrapolation_sizes']

    print("\n  Size    |   Params   |", "  |  ".join([f"ε={e:.0%}" for e in lr_errors]))
    print("-" * (25 + len(lr_errors) * 12))

    for size in extrapolation_sizes:
        params = compute_non_embedding_params(size, args.scaling_rule)
        # Use constant average curvature (no scale dependence)
        delta_losses = [mean_kappa * (np.log(1 + e))**2 for e in lr_errors]

        params_str = f"{params:>10,.0f}" if params < 1e9 else f"{params/1e9:>8.2f}B"
        delta_str = "  |  ".join([f"{dl:.4f}" for dl in delta_losses])
        print(f"  {size:>4}   |{params_str}  | {delta_str}")

    # Save results to JSON
    if args.output_json:
        results = {
            'metadata': {
                'scaling_rule': args.scaling_rule,
                'optimizer': args.optimizer,
                'target_omega': args.target_omega,
            },
            'average_curvature': {
                'kappa_mean': mean_kappa,
                'kappa_std': std_kappa,
            },
            'curvature_by_size': curvature_results,
            'loss_sensitivity': {
                'lr_errors': lr_errors,
                'extrapolation_sizes': extrapolation_sizes,
                'delta_losses': {
                    size: {
                        f'error_{int(e*100)}pct': mean_kappa * (np.log(1 + e))**2
                        for e in lr_errors
                    }
                    for size in extrapolation_sizes
                }
            }
        }

        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    # Create curvature plot (data points only, no fit line due to systematic errors)
    if args.output_plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data points
        ax.scatter(sizes_arr, kappas_arr, s=100, c='tab:blue', edgecolors='black',
                  label=r'Measured $\zeta$', zorder=10)

        # Plot horizontal line for average curvature with ±1 std band
        params_range = np.logspace(np.log10(sizes_arr.min() * 0.5),
                                   np.log10(sizes_arr.max() * 2), 100)
        ax.axhline(y=mean_kappa, color='tab:orange', linestyle='--', linewidth=2,
                  label=fr'Mean $\zeta$ = {mean_kappa:.3e}', zorder=5)
        ax.fill_between(params_range, mean_kappa - std_kappa, mean_kappa + std_kappa,
                       color='tab:orange', alpha=0.2, label=fr'$\pm 1$ std = {std_kappa:.3e}')

        ax.set_xlabel('Non-embedding Parameters', fontsize=14)
        ax.set_ylabel(r'Curvature $\zeta$', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'{args.scaling_rule} {args.optimizer.upper()} LR Curvature', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output_plot}")

    print("\nDone.")
