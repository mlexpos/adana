#!/usr/bin/env python3
"""
Gamma3 Factor Scaling Analysis

This script analyzes the optimal gamma_3_factor value as a function of training iterations
for different model sizes. It fetches runs from wandb with dana optimizer and renormalized
weight decay (omega) = 4, then plots and fits the relationship.

Usage:
    python gamma3_factor_scaling.py --target-omega 4.0 --scaling-rule Enoki_Scaled --fit-function power --group gamma3_scaling_search_new --top-k 1 --show-kappa --fac-min 0.1 --fac-max 10 --exp-constant --fit-kappa-using-constant
    python gamma3_factor_scaling.py --target-omega 4.0 --scaling-rule Enoki_Scaled --fit-function power --group gamma3_scaling_search_new --top-k 3 --show-kappa --exp-constant
    python gamma3_factor_scaling.py --target-omega 4.0 --scaling-rule Enoki_Scaled --fit-function power --group gamma3_scaling_search_new --top-k 3 --exp-constant --fit-kappa-using-constant
    python gamma3_factor_scaling.py --target-omega 4.0 --scaling-rule Enoki_Scaled --fit-function power --group gamma3_scaling_search_new --top-k 3
""" 

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style, rc, rcParams
import argparse
import warnings
import json
import os
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

# Matplotlib formatting
style.use('default')
rc('font', family='sans-serif')
rcParams['font.weight'] = 'light'
rcParams['font.size'] = 18
rcParams['figure.figsize'] = (14, 8)

# =============================================================================
# PLOT STYLING CONSTANTS
# =============================================================================

# Tick label sizes
TICK_LABELSIZE_MAJOR = 42
TICK_LABELSIZE_MINOR = 42

# Axis label font sizes
AXIS_LABEL_FONTSIZE = 32

# Legend font size
LEGEND_FONTSIZE = 28

# Colorbar font sizes
COLORBAR_LABEL_FONTSIZE = 30
COLORBAR_TICK_LABELSIZE = 28

# Annotation font sizes
ANNOTATION_FONTSIZE = 8
ANNOTATION_FONTSIZE_SMALL = 7

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description='Analyze optimal gamma_3_factor vs iterations for different model sizes')
parser.add_argument('--group', type=str, default='gamma3factor_scaling_search',
                    help='WandB group name (default: gamma3factor_scaling_search)')
parser.add_argument('--project', type=str, default='danastar',
                    help='WandB project name (default: danastar)')
parser.add_argument('--entity', type=str, default='ep-rmt-ml-opt',
                    help='WandB entity name (default: ep-rmt-ml-opt)')
parser.add_argument('--target-omega', type=float, default=4.0,
                    help='Target renorm_weight_decay value (default: 4.0)')
parser.add_argument('--omega-tolerance', type=float, default=0.2,
                    help='Tolerance for renorm_weight_decay matching (default: 0.2)')
parser.add_argument('--no-omega-filter', action='store_true',
                    help='Disable renorm_weight_decay filtering (load all dana runs regardless of value)')
parser.add_argument('--scaling-rule', type=str, default='Enoki_Scaled',
                    choices=['BigHead', 'EggHead', 'Enoki', 'Enoki_std', 'Enoki_Scaled', 'Eryngii', 'Eryngii_Scaled'],
                    help='Scaling rule to determine model size (default: Enoki_Scaled)')
parser.add_argument('--output', type=str, default=None,
                    help='Output filename for plot (default: auto-generated)')
parser.add_argument('--fit-function', type=str, default='power',
                    choices=['power', 'linear', 'exponential'],
                    help='Function to fit: power (a*x^b), linear (a*x+b), or exponential (a*exp(b*x)) (default: power)')
parser.add_argument('--top-k', type=int, default=5,
                    help='Number of best gamma_3_factor values to use for each iteration count (default: 5)')
def parse_float_or_inf(value):
    """Parse a float value or 'inf' string."""
    if isinstance(value, str) and value.lower() in ['inf', 'infinity', '+inf', '+infinity']:
        return float('inf')
    return float(value)

parser.add_argument('--fac-min', type=float, default=0.0,
                    help='Minimum factor of Chinchilla iterations to include (default: 0.0, i.e., no lower bound)')
parser.add_argument('--fac-max', type=str, default='inf',
                    help='Maximum factor of Chinchilla iterations to include (default: inf, i.e., no upper bound; can also use numeric value)')
parser.add_argument('--show-kappa', action='store_true',
                    help='Display kappa value next to each point, where gamma_3_factor = iteration^(-kappa)')
parser.add_argument('--exp-constant', action='store_true',
                    help='Plot kappa (where gamma_3_factor = iterations^kappa) on y-axis instead of gamma_3_factor, and show average as horizontal lines')
parser.add_argument('--fit-kappa-using-constant', action='store_true',
                    help='When in exp-constant mode, fit delta * T^kappa to the points by minimizing MSE in log space instead of computing mean kappa')
parser.add_argument('--sizes', type=int, nargs='+', default=None,
                    help='Filter to only show specific model sizes (e.g., --sizes 6 8)')
parser.add_argument('--no-within-head-fit', action='store_true',
                    help='Do not show individual fits for each head size, only show the global fit')
parser.add_argument('--no-grey-points', action='store_true',
                    help='Do not show suboptimal grey points, only show top-K colored points')
args = parser.parse_args()

# Convert fac_max to float, handling 'inf' string
if isinstance(args.fac_max, str):
    args.fac_max = parse_float_or_inf(args.fac_max)
else:
    args.fac_max = float(args.fac_max)

# =============================================================================
# CHINCHILLA ITERATIONS CALCULATION
# =============================================================================

def calculate_chinchilla_iterations(size, scaling_rule):
    """
    Calculate Chinchilla iterations for a given model size.
    
    For Enoki scaling:
    - head_dim = 64 (fixed)
    - n_layer = 3 * heads / 4
    - n_embd = 64 * heads
    - mlp = 4 * n_embd
    - NON_EMB = 12 * n_embd^2 * n_layer
    - TOTAL_PARAMS = NON_EMB + 2 * n_embd * 50304
    - Chinchilla iterations = 20 * TOTAL_PARAMS / 65536
    
    Args:
        size: Model size (n_head for Enoki, n_layer for BigHead)
        scaling_rule: Scaling rule name
    
    Returns:
        Chinchilla iterations for this model size
    """
    if scaling_rule == 'BigHead':
        # For BigHead, size is n_layer
        n_layer = size
        # Need to infer other parameters - this might need adjustment based on actual BigHead scaling
        # For now, using a placeholder that matches the pattern
        n_embd = 64 * 16  # Assuming default head count
        n_head = 16
    else:
        # For Enoki, Enoki_Scaled, etc., size is n_head
        n_head = size
        n_layer = int(3 * n_head // 4)
        n_embd = 64 * n_head
    
    # Calculate non-embedding parameters
    # Non-emb = 12 * n_embd^2 * n_layer
    non_emb = 12 * n_embd * n_embd * n_layer
    
    # Calculate total parameters (including embeddings)
    # TOTAL_PARAMS = NON_EMB + 2 * n_embd * vocab_size
    vocab_size = 50304
    total_params = non_emb + 2 * n_embd * vocab_size
    
    # Chinchilla iterations formula: 20 * TOTAL_PARAMS / 65536
    chinchilla_iterations = int(20 * total_params / 65536)
    
    return chinchilla_iterations

# =============================================================================
# DATA LOADING
# =============================================================================

def load_gamma3_data(project, group, entity, target_omega, omega_tolerance, scaling_rule, use_omega_filter=True):
    """
    Load runs from wandb and extract gamma_3_factor, iterations, and model size.
    
    Filters for:
    - opt=dana
    - renorm_weight_decay ≈ target_omega (if use_omega_filter=True)
    - completed runs
    
    Returns:
        DataFrame with columns: size, gamma_3_factor, iterations, val_loss, renorm_weight_decay
    """
    api = wandb.Api()
    
    print(f"Loading data from {group}...")
    print(f"Target omega: {target_omega} ± {omega_tolerance}")
    print(f"Scaling rule: {scaling_rule}")
    
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    
    data = []
    total_runs = 0
    skipped_optimizer = 0
    skipped_omega = 0
    skipped_incomplete = 0
    skipped_missing_data = 0
    omega_values = []  # Track all omega values for debugging
    all_sizes_found = []  # Track all sizes regardless of omega filter
    skipped_omega_by_size = {}  # Track skipped sizes due to omega mismatch
    
    for run in runs:
        total_runs += 1
        
        # Handle config parsing
        config = run.config
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except (json.JSONDecodeError, TypeError):
                print(f"  Warning: Could not parse config for run {run.name}")
                continue
        
        if hasattr(config, 'as_dict'):
            config = config.as_dict()
        elif not isinstance(config, dict):
            try:
                config = dict(config)
            except (TypeError, ValueError):
                print(f"  Warning: Could not convert config for run {run.name}")
                continue
        
        # Extract values from nested structure
        def extract_value(config_dict):
            result = {}
            for key, val in config_dict.items():
                if isinstance(val, dict) and 'value' in val:
                    result[key] = val['value']
                else:
                    result[key] = val
            return result
        
        config = extract_value(config)
        
        # Handle summary
        summary = run.summary
        if hasattr(summary, '_json_dict') and isinstance(summary._json_dict, str):
            try:
                summary = json.loads(summary._json_dict)
            except (json.JSONDecodeError, TypeError):
                continue
        
        # Filter by optimizer (must be dana)
        opt = config.get('opt', '')
        if opt != 'dana':
            skipped_optimizer += 1
            continue
        
        # Check if run completed
        actual_iter = summary.get('iter', 0)
        iterations_config = config.get('iterations', 0)
        if actual_iter > 0 and iterations_config > 0 and actual_iter < iterations_config:
            skipped_incomplete += 1
            continue
        
        # Extract required fields
        gamma_3_factor = config.get('gamma_3_factor')
        val_loss = summary.get('final-val/loss')
        iterations = config.get('iterations')
        renorm_weight_decay = config.get('renorm_weight_decay')
        
        # Check for None values
        if None in [gamma_3_factor, val_loss, iterations, renorm_weight_decay]:
            skipped_missing_data += 1
            continue
        
        # Convert val_loss to numeric (handle string values)
        try:
            val_loss = float(val_loss)
        except (ValueError, TypeError):
            skipped_missing_data += 1
            continue
        
        # Ensure all numeric fields are actually numeric
        try:
            gamma_3_factor = float(gamma_3_factor)
            iterations = int(iterations)
            renorm_weight_decay = float(renorm_weight_decay)
        except (ValueError, TypeError):
            skipped_missing_data += 1
            continue
        
        # Extract size based on scaling rule
        if scaling_rule == 'BigHead':
            size = config.get('n_layer')  # depth
        else:  # EggHead, Enoki, Enoki_std, Enoki_Scaled, Eryngii, Eryngii_Scaled
            size = config.get('n_head')  # heads
        
        if size is None:
            skipped_missing_data += 1
            continue
        
        # Track all sizes found
        all_sizes_found.append(size)
        
        # Use renorm_weight_decay as omega
        omega = renorm_weight_decay
        omega_values.append(omega)
        
        # Filter by omega (only if use_omega_filter is True)
        if use_omega_filter and abs(omega - target_omega) > omega_tolerance:
            skipped_omega += 1
            if size not in skipped_omega_by_size:
                skipped_omega_by_size[size] = []
            skipped_omega_by_size[size].append(omega)
            continue
        
        data.append({
            'size': size,
            'gamma_3_factor': gamma_3_factor,
            'iterations': iterations,
            'val_loss': val_loss,
            'renorm_weight_decay': renorm_weight_decay,
            'run_name': run.name
        })
    
    print(f"\n  Total runs: {total_runs}")
    print(f"  Loaded: {len(data)}")
    
    # Show all sizes found in the data (before filtering)
    if len(all_sizes_found) > 0:
        unique_sizes = sorted(set(all_sizes_found))
        print(f"  All sizes found in group: {unique_sizes}")
    
    if skipped_optimizer > 0:
        print(f"  Skipped {skipped_optimizer} runs (not dana optimizer)")
    if skipped_omega > 0:
        print(f"  Skipped {skipped_omega} runs (renorm_weight_decay mismatch)")
        if len(omega_values) > 0:
            print(f"  renorm_weight_decay values found: min={min(omega_values):.4f}, max={max(omega_values):.4f}, mean={np.mean(omega_values):.4f}")
            print(f"  Target: {target_omega} ± {omega_tolerance}")
        if skipped_omega_by_size:
            print(f"  Sizes skipped due to omega mismatch:")
            for size in sorted(skipped_omega_by_size.keys()):
                omegas = skipped_omega_by_size[size]
                print(f"    Size {size}: {len(omegas)} runs, omega range: {min(omegas):.4f} to {max(omegas):.4f}")
    if skipped_incomplete > 0:
        print(f"  Skipped {skipped_incomplete} incomplete runs")
    if skipped_missing_data > 0:
        print(f"  Skipped {skipped_missing_data} runs (missing data)")
    
    df = pd.DataFrame(data)
    
    # Show loaded data breakdown by size and renorm_weight_decay
    if len(df) > 0:
        print(f"\n  Loaded data by size:")
        for size in sorted(df['size'].unique()):
            size_df = df[df['size'] == size]
            renorm_wds = size_df['renorm_weight_decay'].unique()
            iterations_unique = sorted(size_df['iterations'].unique())
            print(f"    Size {size}: {len(size_df)} runs, renorm_weight_decay values: {sorted(renorm_wds)}")
            print(f"      Iterations values: {iterations_unique}")
    
    return df

def filter_by_chinchilla_iterations(df, scaling_rule, fac_min, fac_max):
    """
    Filter dataframe to only include iterations within fac_min to fac_max times
    the Chinchilla iterations for each model size.
    
    Args:
        df: DataFrame with columns including 'size' and 'iterations'
        scaling_rule: Scaling rule name
        fac_min: Minimum factor (default 0.0 means no lower bound)
        fac_max: Maximum factor (default inf means no upper bound)
    
    Returns:
        Filtered DataFrame
    """
    if len(df) == 0:
        return df
    
    original_count = len(df)
    
    # Calculate Chinchilla iterations for each unique size
    size_to_chinchilla = {}
    for size in df['size'].unique():
        chinchilla_iters = calculate_chinchilla_iterations(size, scaling_rule)
        size_to_chinchilla[size] = chinchilla_iters
    
    # Filter based on factor range
    mask = pd.Series([False] * len(df), index=df.index)
    
    for size, chinchilla_iters in size_to_chinchilla.items():
        size_mask = df['size'] == size
        size_df = df[size_mask]
        
        # Calculate factor for each row
        factors = size_df['iterations'] / chinchilla_iters
        
        # Apply factor filter
        size_factor_mask = (factors >= fac_min) & (factors <= fac_max)
        mask[size_mask] = size_factor_mask
    
    df_filtered = df[mask].copy()
    
    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count
    
    if removed_count > 0:
        print(f"\n  Filtered by Chinchilla iterations (fac_min={fac_min}, fac_max={fac_max}):")
        print(f"    Removed {removed_count} runs ({original_count} -> {filtered_count})")
        print(f"    Chinchilla iterations by size:")
        for size in sorted(size_to_chinchilla.keys()):
            chinchilla_iters = size_to_chinchilla[size]
            size_df = df_filtered[df_filtered['size'] == size]
            if len(size_df) > 0:
                iter_range = (size_df['iterations'].min(), size_df['iterations'].max())
                factor_range = (iter_range[0] / chinchilla_iters, iter_range[1] / chinchilla_iters)
                print(f"      Size {size}: Chinchilla={chinchilla_iters}, "
                      f"iterations range={iter_range[0]:.0f}-{iter_range[1]:.0f} "
                      f"(factor={factor_range[0]:.3f}-{factor_range[1]:.3f})")
    
    return df_filtered

# =============================================================================
# FITTING FUNCTIONS
# =============================================================================

def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * np.power(x, b)

def linear_func(x, a, b):
    """Linear: y = a * x + b"""
    return a * x + b

def exponential_func(x, a, b):
    """Exponential: y = a * exp(b * x)"""
    return a * np.exp(b * x)

def fit_kappa_to_gamma_formula(iterations, gamma_values, weights=None):
    """
    Fit delta * T^kappa to gamma_3_factor points by minimizing weighted MSE in log space.
    This finds optimal (delta, kappa) such that gamma_3_factor = delta * T^kappa
    by minimizing: sum(weights * (log(gamma_3_factor) - log(delta * T^kappa))^2)
    
    Args:
        iterations: array of iteration values (T)
        gamma_values: array of gamma_3_factor values
        weights: optional array of weights for weighted fitting
    
    Returns:
        params: [delta, kappa]
        kappa_values: array of kappa values corresponding to each point (for plotting)
    """
    if weights is None:
        weights = np.ones_like(iterations)
    
    # Filter out invalid values
    valid_mask = (iterations > 0) & (gamma_values > 0) & (weights > 0) & np.isfinite(iterations) & np.isfinite(gamma_values)
    if np.sum(valid_mask) < 2:
        return None, None
    
    iterations_valid = iterations[valid_mask]
    gamma_valid = gamma_values[valid_mask]
    weights_valid = weights[valid_mask]
    
    # Compute kappa for each point: kappa = log(gamma_3_factor) / log(T) (for display only)
    kappa_values = np.log(gamma_valid) / np.log(iterations_valid)
    
    # Fit delta * T^kappa by minimizing weighted MSE in log space
    # Model: gamma_3_factor = delta * T^kappa
    # In log space: log(gamma_3_factor) = log(delta) + kappa * log(T)
    # Minimize: sum(w_i * (log(gamma_i) - log(delta) - kappa * log(T_i))^2)
    
    log_iterations = np.log(iterations_valid)
    log_gamma = np.log(gamma_valid)
    
    # Weighted linear regression in log-log space minimizes weighted MSE
    # This is equivalent to minimizing: sum(w * (log(gamma) - (log(delta) + kappa * log(T)))^2)
    coeffs = np.polyfit(log_iterations, log_gamma, 1, w=weights_valid)
    kappa_fit = coeffs[0]  # slope = kappa
    log_delta = coeffs[1]  # intercept = log(delta)
    delta_fit = np.exp(log_delta)
    
    return [delta_fit, kappa_fit], kappa_values

def fit_kappa_only_to_gamma_formula(iterations, gamma_values, weights=None, negative=False):
    """
    Fit T^kappa or T^(-kappa) to gamma_3_factor points by minimizing weighted MSE in log space.
    This finds optimal kappa such that gamma_3_factor = T^kappa (or T^(-kappa) if negative=True)
    by minimizing: sum(weights * (log(gamma_3_factor) - kappa * log(T))^2)
    (or sum(weights * (log(gamma_3_factor) + kappa * log(T))^2) for negative)
    
    Args:
        iterations: array of iteration values (T)
        gamma_values: array of gamma_3_factor values
        weights: optional array of weights for weighted fitting
        negative: If True, fit T^(-kappa), otherwise fit T^kappa
    
    Returns:
        kappa: fitted kappa value (always positive)
    """
    if weights is None:
        weights = np.ones_like(iterations)
    
    # Filter out invalid values
    valid_mask = (iterations > 0) & (gamma_values > 0) & (weights > 0) & np.isfinite(iterations) & np.isfinite(gamma_values)
    if np.sum(valid_mask) < 2:
        return None
    
    iterations_valid = iterations[valid_mask]
    gamma_valid = gamma_values[valid_mask]
    weights_valid = weights[valid_mask]
    
    log_iterations = np.log(iterations_valid)
    log_gamma = np.log(gamma_valid)
    
    if negative:
        # Fit T^(-kappa): log(gamma_3_factor) = -kappa * log(T)
        # This is: log(gamma_3_factor) = (-kappa) * log(T) + 0
        # Use weighted linear regression with zero intercept
        # We need to fit: y = a*x where y=log_gamma, x=log_iterations, a=-kappa
        # For weighted regression with zero intercept: a = sum(w*x*y) / sum(w*x^2)
        numerator = np.sum(weights_valid * log_iterations * log_gamma)
        denominator = np.sum(weights_valid * log_iterations * log_iterations)
        if denominator == 0:
            return None
        negative_kappa = numerator / denominator
        kappa_fit = -negative_kappa  # Make kappa positive
    else:
        # Fit T^kappa: log(gamma_3_factor) = kappa * log(T)
        # This is: log(gamma_3_factor) = kappa * log(T) + 0
        # Use weighted linear regression with zero intercept
        numerator = np.sum(weights_valid * log_iterations * log_gamma)
        denominator = np.sum(weights_valid * log_iterations * log_iterations)
        if denominator == 0:
            return None
        kappa_fit = numerator / denominator
    
    return kappa_fit

def fit_function(iterations, gamma_values, fit_type='power', weights=None):
    """
    Fit a function to the data with optional weights.
    
    For power law: fits log(y) = log(a) + b * log(x) using weighted linear regression in log space.
    
    Args:
        iterations: array of iteration values
        gamma_values: array of gamma_3_factor values
        fit_type: 'power', 'linear', or 'exponential'
        weights: optional array of weights for weighted fitting
    
    Returns:
        params: fitted parameters
        func: the function used for fitting
    """
    if weights is None:
        weights = np.ones_like(iterations)
    
    if fit_type == 'power':
        # Weighted linear regression in log-log space: log(y) = log(a) + b * log(x)
        # This is equivalent to y = a * x^b
        log_iterations = np.log(iterations)
        log_gamma = np.log(gamma_values)
        
        # Filter out any invalid values (NaN, inf, or non-positive)
        valid_mask = np.isfinite(log_iterations) & np.isfinite(log_gamma) & (iterations > 0) & (gamma_values > 0) & (weights > 0)
        if np.sum(valid_mask) < 2:
            return None, power_law
        
        log_iterations_valid = log_iterations[valid_mask]
        log_gamma_valid = log_gamma[valid_mask]
        weights_valid = weights[valid_mask]
        
        # Weighted linear regression: y = a*x + b, where y=log_gamma, x=log_iterations
        # Use np.polyfit with weights (weights are applied to squared residuals)
        # For weighted polyfit, we need to use np.polyfit with w parameter
        coeffs = np.polyfit(log_iterations_valid, log_gamma_valid, 1, w=weights_valid)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        log_a = intercept
        b = slope
        a = np.exp(log_a)
        
        params = [a, b]
        func = power_law
        return params, func
    elif fit_type == 'linear':
        func = linear_func
        # Weighted linear regression
        valid_mask = np.isfinite(iterations) & np.isfinite(gamma_values) & (weights > 0)
        if np.sum(valid_mask) < 2:
            return None, linear_func
        
        iterations_valid = iterations[valid_mask]
        gamma_valid = gamma_values[valid_mask]
        weights_valid = weights[valid_mask]
        
        coeffs = np.polyfit(iterations_valid, gamma_valid, 1, w=weights_valid)
        slope = coeffs[0]
        intercept = coeffs[1]
        params = [slope, intercept]
        return params, func
    elif fit_type == 'exponential':
        func = exponential_func
        # Weighted fit in log space: log(y) = log(a) + b * x
        valid_mask = (gamma_values > 0) & np.isfinite(gamma_values) & np.isfinite(iterations) & (weights > 0)
        if np.sum(valid_mask) < 2:
            return None, exponential_func
        
        log_gamma_valid = np.log(gamma_values[valid_mask])
        iterations_valid = iterations[valid_mask]
        weights_valid = weights[valid_mask]
        
        coeffs = np.polyfit(iterations_valid, log_gamma_valid, 1, w=weights_valid)
        slope = coeffs[0]
        intercept = coeffs[1]
        b = slope
        log_a = intercept
        a = np.exp(log_a)
        params = [a, b]
        return params, func
    else:
        raise ValueError(f"Unknown fit type: {fit_type}")

# =============================================================================
# PLOTTING
# =============================================================================

def plot_gamma3_vs_iterations(df, fit_type='power', scaling_rule='Enoki_Scaled', top_k=5, show_kappa=False, exp_constant=False, fit_kappa=False, no_within_head_fit=False, no_grey_points=False):
    """
    Plot optimal gamma_3_factor vs iterations for each model size.
    Shows all data points (gray) and highlights top-K ones (colored) with decreasing weights.
    Uses weighted fitting based on top-K procedure similar to bighead_lr_scaling.py.
    
    Args:
        df: DataFrame with columns including 'size', 'iterations', 'gamma_3_factor'
        fit_type: Type of fit function ('power', 'linear', 'exponential')
        scaling_rule: Scaling rule name
        top_k: Number of top points to highlight at each iteration count
        show_kappa: If True, annotate each point with kappa where gamma_3_factor = iteration^(-kappa)
        exp_constant: If True, plot kappa on y-axis and show average horizontal lines instead of fitting
        fit_kappa: If True (with exp_constant), fit delta * T^kappa by minimizing MSE instead of computing mean
    """
    # Create figure with subplots: if fit_kappa is True, add a second subplot for gamma_3_factor vs T
    if fit_kappa:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 7), facecolor='white')
    else:
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
        ax2 = None
    
    # Get unique sizes
    sizes = sorted(df['size'].unique())
    
    # Color map for sizes (using cm.viridis like bighead_lr_scaling.py)
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.colorbar import ColorbarBase
    
    # Normalize head counts for colormap
    norm = Normalize(vmin=min(sizes), vmax=max(sizes))
    cmap = cm.viridis
    colors = [cmap(norm(size)) for size in sizes]
    
    # Global collections for all top-k points across all sizes
    global_weighted_gamma = []
    global_weighted_iterations = []
    global_weighted_weights = []
    global_weighted_kappa = []
    
    # For each size, plot all points and find top-K gamma_3_factor at each iteration count
    for idx, size in enumerate(sizes):
        size_df = df[df['size'] == size].copy()
        
        if len(size_df) == 0:
            continue
        
        # Plot ALL data points with size proportional to rank (like bighead_lr_scaling.py)
        # Group by iterations to rank within each iteration count
        unique_iters = sorted(size_df['iterations'].unique())
        
        # Collect top-K points with weights for fitting
        weighted_gamma = []
        weighted_iterations = []
        weighted_weights = []
        weighted_kappa = []
        
        for iters in unique_iters:
            iter_df = size_df[size_df['iterations'] == iters].copy()
            
            # Skip if less than 3 runs for this size and iteration count
            n_runs = len(iter_df)
            if n_runs < 3:
                continue
            
            # Sort by val_loss to get ranks
            iter_df = iter_df.sort_values('val_loss')
            
            # Take top K
            top_k_df = iter_df.head(top_k).copy()
            
            # Plot each point with size based on rank (matching bighead_lr_scaling.py style)
            # For weighting, use n_runs - rank, but for point size use a fixed scale based on rank only
            for rank, (_, row) in enumerate(iter_df.iterrows()):
                weight = n_runs - rank  # Best gets n_runs, worst gets 1 (for fitting weights)
                # Point size based on rank only (not n_runs) so same rank has same size across iterations
                point_size = (top_k - rank) * 200 if rank < top_k else 80  # 4x: was 50, now 200
                
                # Calculate kappa: gamma_3_factor = iteration^kappa
                # => kappa = log(gamma_3_factor) / log(iterations)
                kappa_value = None
                if row['iterations'] > 0 and row['gamma_3_factor'] > 0:
                    kappa_value = np.log(row['gamma_3_factor']) / np.log(row['iterations'])
                
                # Determine y_value based on exp_constant mode
                if exp_constant:
                    if kappa_value is None:
                        continue  # Skip points without valid kappa
                    y_value = kappa_value
                else:
                    y_value = row['gamma_3_factor']
                
                if rank < top_k:
                    # Top-K points: colored, larger, black edge
                    ax.scatter([row['iterations']], [y_value], 
                              s=point_size, c=[colors[idx]], alpha=0.6, 
                              edgecolors='black', linewidths=0.5, zorder=10)
                    
                    # Annotate with kappa if requested (only in non-exp_constant mode)
                    if show_kappa and not exp_constant and kappa_value is not None:
                        ax.annotate(f'{kappa_value:.3f}', 
                                   xy=(row['iterations'], y_value),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=ANNOTATION_FONTSIZE, alpha=0.7, color=colors[idx],
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='none'))
                    
                    # Track top-K for weighted fitting (per-size)
                    weighted_gamma.append(row['gamma_3_factor'])
                    weighted_iterations.append(row['iterations'])
                    weighted_weights.append(weight)
                    if kappa_value is not None:
                        weighted_kappa.append(kappa_value)
                    
                    # Also track for global fit (across all sizes)
                    global_weighted_gamma.append(row['gamma_3_factor'])
                    global_weighted_iterations.append(row['iterations'])
                    global_weighted_weights.append(weight)
                    if kappa_value is not None:
                        global_weighted_kappa.append(kappa_value)
                else:
                    # Other points: gray, smaller (only if not no_grey_points)
                    if not no_grey_points:
                        ax.scatter([row['iterations']], [y_value], 
                                  s=point_size, c='gray', alpha=0.3, 
                                  edgecolors='none', zorder=5)
                        
                        # Annotate with kappa if requested (for non-top-K points too, only in non-exp_constant mode)
                        if show_kappa and not exp_constant and kappa_value is not None:
                            ax.annotate(f'{kappa_value:.3f}', 
                                       xy=(row['iterations'], y_value),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=ANNOTATION_FONTSIZE_SMALL, alpha=0.5, color='gray',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.4, edgecolor='none'))
        
        if len(weighted_iterations) < 2:
            print(f"  Warning: Head {size} has only {len(weighted_iterations)} data points, skipping fit")
            continue
        
        # Only plot individual head fits if requested
        if not no_within_head_fit:
            # Fit function with weights (or compute average kappa for exp_constant mode)
            weighted_iterations_arr = np.array(weighted_iterations)
            weighted_gamma_arr = np.array(weighted_gamma)
            weighted_weights_arr = np.array(weighted_weights)
            
            if exp_constant:
                # Compute weighted average kappa or fit kappa
                if len(weighted_kappa) > 0:
                    weighted_kappa_arr = np.array(weighted_kappa)
                    
                    if fit_kappa:
                        # Fit delta * T^kappa to minimize MSE
                        fit_params, _ = fit_kappa_to_gamma_formula(
                            weighted_iterations_arr, 
                            weighted_gamma_arr, 
                            weights=weighted_weights_arr
                        )
                        
                        if fit_params is not None:
                            delta_fit, kappa_fit = fit_params
                            
                            # Compute residuals for statistics
                            predicted_kappa = kappa_fit * np.ones_like(weighted_kappa_arr)
                            residuals = weighted_kappa_arr - predicted_kappa
                            std_kappa = np.sqrt(np.average(residuals**2, weights=weighted_weights_arr[:len(weighted_kappa_arr)]))
                            
                            # Plot horizontal line for fitted kappa
                            iter_range = np.linspace(min(weighted_iterations_arr), max(weighted_iterations_arr) * 1.2, 200)
                            ax.axhline(y=kappa_fit, color=colors[idx], linestyle='--', linewidth=3, 
                                      label=f'Head {size}: κ = {kappa_fit:.2f} (fit)', zorder=9)
                            
                            print(f"\nHead {size}:")
                            print(f"  Data points: {len(weighted_kappa)} (top-{top_k} at each iteration)")
                            print(f"  Iterations range: {min(weighted_iterations)} to {max(weighted_iterations)}")
                            print(f"  Fitted: gamma_3_factor = {delta_fit:.6e} × T^{kappa_fit:.4f}")
                            print(f"  Kappa (fitted): {kappa_fit:.4f} ± {std_kappa:.4f} (MSE residual)")
                            print(f"  Kappa range: {min(weighted_kappa):.4f} to {max(weighted_kappa):.4f}")
                    else:
                        # Use weights for weighted average
                        avg_kappa = np.average(weighted_kappa_arr, weights=weighted_weights_arr[:len(weighted_kappa_arr)])
                        std_kappa = np.sqrt(np.average((weighted_kappa_arr - avg_kappa)**2, weights=weighted_weights_arr[:len(weighted_kappa_arr)]))
                        
                        # Plot horizontal line for average kappa
                        iter_range = np.linspace(min(weighted_iterations_arr), max(weighted_iterations_arr) * 1.2, 200)
                        ax.axhline(y=avg_kappa, color=colors[idx], linestyle='--', linewidth=3, 
                                  label=f'Head {size}: κ = {avg_kappa:.2f}', zorder=9)
                        
                        print(f"\nHead {size}:")
                        print(f"  Data points: {len(weighted_kappa)} (top-{top_k} at each iteration)")
                        print(f"  Iterations range: {min(weighted_iterations)} to {max(weighted_iterations)}")
                        print(f"  Kappa (weighted avg): {avg_kappa:.4f} ± {std_kappa:.4f}")
                        print(f"  Kappa range: {min(weighted_kappa):.4f} to {max(weighted_kappa):.4f}")
            else:
                # When fit_kappa is True, use fit_kappa_to_gamma_formula instead of regular fit
                if fit_kappa:
                    fit_params, _ = fit_kappa_to_gamma_formula(
                        weighted_iterations_arr, 
                        weighted_gamma_arr, 
                        weights=weighted_weights_arr
                    )
                    
                    if fit_params is not None:
                        delta_fit, kappa_fit = fit_params
                        
                        # Plot fitted curve: gamma_3_factor = delta * T^kappa
                        iter_range = np.linspace(min(weighted_iterations_arr), max(weighted_iterations_arr) * 1.2, 200)
                        gamma_fit = delta_fit * np.power(iter_range, kappa_fit)
                        
                        label = f'Head {size} fit: {delta_fit:.2e} × $T^{{{kappa_fit:.3f}}}$'
                        
                        ax.plot(iter_range, gamma_fit, '--', color=colors[idx], linewidth=3, 
                               label=label, zorder=9)
                        
                        print(f"\nHead {size}:")
                        print(f"  Data points: {len(weighted_iterations)} (top-{top_k} at each iteration)")
                        print(f"  Iterations range: {min(weighted_iterations)} to {max(weighted_iterations)}")
                        print(f"  Gamma_3_factor range: {min(weighted_gamma):.4f} to {max(weighted_gamma):.4f}")
                        print(f"  Fit: gamma_3_factor = {delta_fit:.6e} × T^{kappa_fit:.4f}")
                else:
                    params, func = fit_function(weighted_iterations_arr, weighted_gamma_arr, fit_type, weights=weighted_weights_arr)
                    
                    if params is not None:
                        # Plot fitted curve
                        iter_range = np.linspace(min(weighted_iterations_arr), max(weighted_iterations_arr) * 1.2, 200)
                        gamma_fit = func(iter_range, *params)
                        
                        # Create label based on fit type
                        if fit_type == 'power':
                            label = f'Head {size} fit: {params[0]:.2e} × $T^{{{params[1]:.3f}}}$'
                        elif fit_type == 'linear':
                            label = f'Head {size} fit: {params[0]:.2e} × T + {params[1]:.3f}'
                        elif fit_type == 'exponential':
                            label = f'Head {size} fit: {params[0]:.2e} × exp({params[1]:.2e} × T)'
                        
                        ax.plot(iter_range, gamma_fit, '--', color=colors[idx], linewidth=3, 
                               label=label, zorder=9)
                        
                        print(f"\nHead {size}:")
                        print(f"  Data points: {len(weighted_iterations)} (top-{top_k} at each iteration)")
                        print(f"  Iterations range: {min(weighted_iterations)} to {max(weighted_iterations)}")
                        print(f"  Gamma_3_factor range: {min(weighted_gamma):.4f} to {max(weighted_gamma):.4f}")
                        if fit_type == 'power':
                            print(f"  Fit: gamma_3_factor = {params[0]:.6e} × iterations^{params[1]:.4f}")
                        elif fit_type == 'linear':
                            print(f"  Fit: gamma_3_factor = {params[0]:.6e} × iterations + {params[1]:.4f}")
                        elif fit_type == 'exponential':
                            print(f"  Fit: gamma_3_factor = {params[0]:.6e} × exp({params[1]:.6e} × iterations)")
        # End of individual head fitting block (if not no_within_head_fit)
    
    # Global fit across all sizes
    if len(global_weighted_iterations) >= 2:
        print(f"\n{'='*70}")
        if exp_constant:
            print("Global Average (all sizes combined):")
        else:
            print("Global Fit (all sizes combined):")
        print(f"{'='*70}")
        
        global_iterations_arr = np.array(global_weighted_iterations)
        global_gamma_arr = np.array(global_weighted_gamma)
        global_weights_arr = np.array(global_weighted_weights)
        
        if exp_constant:
            # Compute global weighted average kappa or fit kappa
            if len(global_weighted_kappa) > 0:
                global_kappa_arr = np.array(global_weighted_kappa)
                
                if fit_kappa:
                    # Fit delta * T^kappa to minimize MSE globally
                    global_fit_params, _ = fit_kappa_to_gamma_formula(
                        global_iterations_arr, 
                        global_gamma_arr, 
                        weights=global_weights_arr
                    )
                    
                    if global_fit_params is not None:
                        global_delta_fit, global_kappa_fit = global_fit_params
                        
                        # Compute residuals for statistics
                        predicted_kappa = global_kappa_fit * np.ones_like(global_kappa_arr)
                        residuals = global_kappa_arr - predicted_kappa
                        global_std_kappa = np.sqrt(np.average(residuals**2, weights=global_weights_arr[:len(global_kappa_arr)]))
                        
                        # Plot global horizontal line
                        ax.axhline(y=global_kappa_fit, color='red', linestyle='-', linewidth=4, 
                                  label=f'Fit: κ = {global_kappa_fit:.2f}', zorder=11, alpha=0.8)
                        
                        print(f"  Total data points: {len(global_weighted_kappa)} (top-{top_k} at each iteration for all sizes)")
                        print(f"  Iterations range: {min(global_weighted_iterations)} to {max(global_weighted_iterations)}")
                        print(f"  Global fitted: gamma_3_factor = {global_delta_fit:.6e} × T^{global_kappa_fit:.4f}")
                        print(f"  Global kappa (fitted): {global_kappa_fit:.4f} ± {global_std_kappa:.4f} (MSE residual)")
                        print(f"  Global kappa range: {min(global_weighted_kappa):.4f} to {max(global_weighted_kappa):.4f}")
                else:
                    global_avg_kappa = np.average(global_kappa_arr, weights=global_weights_arr[:len(global_kappa_arr)])
                    global_std_kappa = np.sqrt(np.average((global_kappa_arr - global_avg_kappa)**2, weights=global_weights_arr[:len(global_kappa_arr)]))
                    
                    # Plot global horizontal line
                    ax.axhline(y=global_avg_kappa, color='red', linestyle='-', linewidth=4, 
                              label=f'Global: κ = {global_avg_kappa:.2f}', zorder=11, alpha=0.8)
                    
                    print(f"  Total data points: {len(global_weighted_kappa)} (top-{top_k} at each iteration for all sizes)")
                    print(f"  Iterations range: {min(global_weighted_iterations)} to {max(global_weighted_iterations)}")
                    print(f"  Global kappa (weighted avg): {global_avg_kappa:.4f} ± {global_std_kappa:.4f}")
                    print(f"  Global kappa range: {min(global_weighted_kappa):.4f} to {max(global_weighted_kappa):.4f}")
        else:
            # When fit_kappa is True, use fit_kappa_to_gamma_formula instead of regular fit
            if fit_kappa:
                global_fit_params, _ = fit_kappa_to_gamma_formula(
                    global_iterations_arr, 
                    global_gamma_arr, 
                    weights=global_weights_arr
                )
                
                if global_fit_params is not None:
                    global_delta_fit, global_kappa_fit = global_fit_params
                    
                    # Plot global fitted curve: gamma_3_factor = delta * T^kappa
                    iter_range_global = np.linspace(
                        min(global_iterations_arr), 
                        max(global_iterations_arr) * 1.2, 
                        200
                    )
                    gamma_fit_global = global_delta_fit * np.power(iter_range_global, global_kappa_fit)
                    
                    global_label = f'Fit: {global_delta_fit:.2e} × $T^{{{global_kappa_fit:.3f}}}$'
                    
                    # Plot with thicker, solid line in red to stand out
                    ax.plot(iter_range_global, gamma_fit_global, '-', color='red', linewidth=4, 
                           label=global_label, zorder=11, alpha=0.8)
                    
                    print(f"  Total data points: {len(global_weighted_iterations)} (top-{top_k} at each iteration for all sizes)")
                    print(f"  Iterations range: {min(global_weighted_iterations)} to {max(global_weighted_iterations)}")
                    print(f"  Gamma_3_factor range: {min(global_weighted_gamma):.4f} to {max(global_weighted_gamma):.4f}")
                    print(f"  Global fit: gamma_3_factor = {global_delta_fit:.6e} × T^{global_kappa_fit:.4f}")
                else:
                    print("  Warning: Could not compute global fit (fit_kappa)")
            else:
                global_params, global_func = fit_function(
                    global_iterations_arr, 
                    global_gamma_arr, 
                    fit_type, 
                    weights=global_weights_arr
                )
                
                if global_params is not None:
                    # Plot global fitted curve with distinct style
                    iter_range_global = np.linspace(
                        min(global_iterations_arr), 
                        max(global_iterations_arr) * 1.2, 
                        200
                    )
                    gamma_fit_global = global_func(iter_range_global, *global_params)
                    
                    # Create label based on fit type
                    if fit_type == 'power':
                        global_label = f'Fit: {global_params[0]:.2e} × $T^{{{global_params[1]:.3f}}}$'
                    elif fit_type == 'linear':
                        global_label = f'Fit: {global_params[0]:.2e} × T + {global_params[1]:.3f}'
                    elif fit_type == 'exponential':
                        global_label = f'Fit: {global_params[0]:.2e} × exp({global_params[1]:.2e} × T)'
                    
                    # Plot with thicker, solid line in black or red to stand out
                    ax.plot(iter_range_global, gamma_fit_global, '-', color='red', linewidth=4, 
                           label=global_label, zorder=11, alpha=0.8)
                    
                    print(f"  Total data points: {len(global_weighted_iterations)} (top-{top_k} at each iteration for all sizes)")
                    print(f"  Iterations range: {min(global_weighted_iterations)} to {max(global_weighted_iterations)}")
                    print(f"  Gamma_3_factor range: {min(global_weighted_gamma):.4f} to {max(global_weighted_gamma):.4f}")
                    if fit_type == 'power':
                        print(f"  Global fit: gamma_3_factor = {global_params[0]:.6e} × iterations^{global_params[1]:.4f}")
                    elif fit_type == 'linear':
                        print(f"  Global fit: gamma_3_factor = {global_params[0]:.6e} × iterations + {global_params[1]:.4f}")
                    elif fit_type == 'exponential':
                        print(f"  Global fit: gamma_3_factor = {global_params[0]:.6e} × exp({global_params[1]:.6e} × iterations)")
                else:
                    print("  Warning: Could not compute global fit")
    else:
        print(f"\nWarning: Not enough data points for global fit ({len(global_weighted_iterations)} points)")
    
    # Set y-axis limits and formatting based on mode
    if exp_constant:
        # For kappa mode, use linear scale and fixed limits
        ax.set_xlabel('Training Iterations', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax.set_ylabel('Spectral Dimension κ', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax.set_xscale('log')
        ax.set_ylim(-1.1, -0.6)  # Zoom in on typical kappa range
        # Keep y-axis linear for kappa values, no title in exp_constant mode
    else:
        ax.set_ylim(1e-7, 1e-1)
        ax.set_xlabel('Training Iterations', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax.set_ylabel('Gamma 3 Factor', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_title(f'Optimal Gamma 3 Factor vs Training Iterations',
        #             fontsize=20, fontweight='bold')
    
    # Set tick label sizes for both axes
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE_MAJOR)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_LABELSIZE_MINOR)
    
    # Add colorbar to show head count mapping
    # If fit_kappa is True (both axes shown), colorbar will be added to ax2 later, otherwise add to ax now
    if not fit_kappa or ax2 is None:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label('Heads', fontsize=COLORBAR_LABEL_FONTSIZE, fontweight='bold')
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_LABELSIZE)
    
    # Add legend to show global fit (if it exists)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=LEGEND_FONTSIZE, loc='upper right', framealpha=0.9)
    
    # Remove grid lines and unused spines for clean professional look
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')
    
    # Add annotation about point sizes (matching bighead_lr_scaling.py style) - only in non-exp_constant mode
    # if not exp_constant:
    #     ax.text(0.02, 0.02, f'Point size ∝ weight\n(top-{top_k} at each iteration, weighted fit)',
    #             transform=ax.transAxes, fontsize=15, verticalalignment='bottom',
    #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add text box showing global fit parameters if in exp_constant + fit_kappa mode
    # Note: The actual fits are shown on ax2, but we can show a summary here
    # (This section is no longer needed as legend is added after ax2 plots are created)
    
    # Add secondary x-axis at the top showing ratio of iterations to Chinchilla iterations
    # Only works well if there's a single size (otherwise ratios would differ per size)
    # Skip this if we have a second subplot (fit_kappa mode)
    if len(sizes) == 1 and not fit_kappa:
        size = sizes[0]
        chinchilla_iters = calculate_chinchilla_iterations(size, scaling_rule)
        
        # Create secondary axis with a transform: x_top = x_bottom / chinchilla_iters
        ax_twin = ax.twiny()
        ax_twin.set_xscale('log')
        
        # Get the primary axis limits and convert to ratio
        xlim = ax.get_xlim()
        ax_twin.set_xlim(xlim[0] / chinchilla_iters, xlim[1] / chinchilla_iters)
        
        ax_twin.set_xlabel(f'Iterations / Chinchilla (head {size}, Chinchilla={chinchilla_iters})', fontsize=COLORBAR_LABEL_FONTSIZE)
        # Remove top spine for clean look
        ax_twin.spines['top'].set_visible(False)
        # Set tick label sizes for twin axis
        ax_twin.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE_MAJOR)
        ax_twin.tick_params(axis='both', which='minor', labelsize=TICK_LABELSIZE_MINOR)
    
    # If fit_kappa is True, add second subplot showing gamma_3_factor vs T in log-log with fitted curves
    if fit_kappa and ax2 is not None:
        # Re-plot all data points on the second subplot
        for idx, size in enumerate(sizes):
            size_df = df[df['size'] == size].copy()
            
            if len(size_df) == 0:
                continue
            
            unique_iters = sorted(size_df['iterations'].unique())
            
            for iters in unique_iters:
                iter_df = size_df[size_df['iterations'] == iters].copy()
                
                n_runs = len(iter_df)
                if n_runs < 3:
                    continue
                
                iter_df = iter_df.sort_values('val_loss')
                top_k_df = iter_df.head(top_k).copy()
                
                for rank, (_, row) in enumerate(iter_df.iterrows()):
                    weight = n_runs - rank  # For fitting weights
                    # Point size based on rank only (not n_runs) so same rank has same size across iterations
                    point_size = (top_k - rank) * 200 if rank < top_k else 80  # 4x: was 50, now 200
                    
                    if rank < top_k:
                        ax2.scatter([row['iterations']], [row['gamma_3_factor']], 
                                  s=point_size, c=[colors[idx]], alpha=0.6, 
                                  edgecolors='black', linewidths=0.5, zorder=10)
                    else:
                        # Other points: gray (only if not no_grey_points)
                        if not no_grey_points:
                            ax2.scatter([row['iterations']], [row['gamma_3_factor']], 
                                      s=point_size, c='gray', alpha=0.3, 
                                      edgecolors='none', zorder=5)
        
        # Plot fitted curves for each size (only if not no_within_head_fit)
        if not no_within_head_fit:
            for idx, size in enumerate(sizes):
                size_df = df[df['size'] == size].copy()
                if len(size_df) == 0:
                    continue
                
                unique_iters = sorted(size_df['iterations'].unique())
                weighted_gamma = []
                weighted_iterations = []
                weighted_weights = []
                
                for iters in unique_iters:
                    iter_df = size_df[size_df['iterations'] == iters].copy()
                    n_runs = len(iter_df)
                    if n_runs < 3:
                        continue
                    
                    iter_df = iter_df.sort_values('val_loss')
                    top_k_df = iter_df.head(top_k).copy()
                    
                    for rank, (_, row) in enumerate(top_k_df.iterrows()):
                        weight = n_runs - rank
                        weighted_gamma.append(row['gamma_3_factor'])
                        weighted_iterations.append(row['iterations'])
                        weighted_weights.append(weight)
                
                if len(weighted_iterations) >= 2:
                    # Fit T^kappa (delta = 1, no delta constant)
                    kappa_fit = fit_kappa_only_to_gamma_formula(
                        np.array(weighted_iterations),
                        np.array(weighted_gamma),
                        weights=np.array(weighted_weights),
                        negative=False
                    )
                    
                    if kappa_fit is not None:
                        iter_range = np.linspace(min(weighted_iterations), max(weighted_iterations) * 1.2, 200)
                        gamma_fit = np.power(iter_range, kappa_fit)
                        
                        label = f'Head {size}: $T^{{{kappa_fit:.3f}}}$'
                        ax2.plot(iter_range, gamma_fit, '--', color=colors[idx], linewidth=3, 
                               label=label, zorder=9)
                    
                    # Fit delta * T^kappa (with delta constant)
                    fit_params, _ = fit_kappa_to_gamma_formula(
                        np.array(weighted_iterations),
                        np.array(weighted_gamma),
                        weights=np.array(weighted_weights)
                    )
                    
                    if fit_params is not None:
                        delta_fit, kappa_fit_delta = fit_params
                        iter_range = np.linspace(min(weighted_iterations), max(weighted_iterations) * 1.2, 200)
                        gamma_fit_delta = delta_fit * np.power(iter_range, kappa_fit_delta)
                        
                        label_delta = f'Head {size}: {delta_fit:.2e} × $T^{{{kappa_fit_delta:.3f}}}$'
                        ax2.plot(iter_range, gamma_fit_delta, ':', color=colors[idx], linewidth=2.5, 
                               label=label_delta, zorder=8, alpha=0.7)
        
        # Plot global fitted curves (delta * T^kappa with both parameters fitted)
        if len(global_weighted_iterations) >= 2:
            # Fit delta * T^kappa (with delta constant)
            global_fit_params, _ = fit_kappa_to_gamma_formula(
                np.array(global_weighted_iterations),
                np.array(global_weighted_gamma),
                weights=np.array(global_weighted_weights)
            )
            
            if global_fit_params is not None:
                global_delta_fit, global_kappa_fit_delta = global_fit_params
                iter_range_global = np.linspace(
                    min(global_weighted_iterations),
                    max(global_weighted_iterations) * 1.2,
                    200
                )
                gamma_fit_global_delta = global_delta_fit * np.power(iter_range_global, global_kappa_fit_delta)
                
                global_label_delta = f'Fit: {global_delta_fit:.2e} × $T^{{{global_kappa_fit_delta:.3f}}}$'
                ax2.plot(iter_range_global, gamma_fit_global_delta, '-', color='red', linewidth=4, 
                       label=global_label_delta, zorder=11, alpha=0.8)
                print(f"  Global fit (with delta): gamma_3_factor = {global_delta_fit:.6e} × T^{global_kappa_fit_delta:.4f}")
        
        # Configure second subplot
        ax2.set_xlabel('Training Iterations', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax2.set_ylabel('Gamma 3 Factor', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_ylim(1e-7, 1e-1)
        
        # Set tick label sizes for ax2 (larger)
        ax2.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE_MAJOR)
        ax2.tick_params(axis='both', which='minor', labelsize=TICK_LABELSIZE_MINOR)
        
        # Add colorbar to ax2 (right plot) when both axes are shown
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, pad=0.02, fraction=0.046)
        cbar.set_label('Heads', fontsize=COLORBAR_LABEL_FONTSIZE, fontweight='bold')
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_LABELSIZE)
        
        # Add legend to show global fits (larger font, matching ax)
        handles2, labels2 = ax2.get_legend_handles_labels()
        if handles2:
            ax2.legend(handles2, labels2, fontsize=LEGEND_FONTSIZE, loc='best', framealpha=0.9)
        
        # Remove grid lines and unused spines for clean professional look
        ax2.grid(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_facecolor('white')
        # ax2.text(0.02, 0.02, f'Point size ∝ weight\n(top-{top_k} at each iteration)',
        #         transform=ax2.transAxes, fontsize=15, verticalalignment='bottom',
        #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("Gamma 3 Factor Scaling Analysis")
    print(f"Group: {args.group}")
    print(f"Target Omega: {args.target_omega}")
    print(f"Scaling Rule: {args.scaling_rule}")
    print(f"Fit Function: {args.fit_function}")
    print(f"Exp Constant Mode: {args.exp_constant}")
    if args.fac_min > 0.0 or (args.fac_max != float('inf') and not np.isinf(args.fac_max)):
        print(f"Chinchilla iteration filter: fac_min={args.fac_min}, fac_max={args.fac_max}")
    else:
        print(f"Chinchilla iteration filter: disabled (fac_min=0.0, fac_max=inf)")
    print("="*70)
    
    # Load data
    df = load_gamma3_data(
        project=args.project,
        group=args.group,
        entity=args.entity,
        target_omega=args.target_omega,
        omega_tolerance=args.omega_tolerance,
        scaling_rule=args.scaling_rule,
        use_omega_filter=not args.no_omega_filter
    )
    
    if len(df) == 0:
        print("\nNo data found matching criteria. Exiting.")
        exit(1)
    
    # Filter by Chinchilla iterations if requested
    # Only filter if fac_min > 0 or fac_max < inf (i.e., if filtering is actually requested)
    if args.fac_min > 0.0 or (args.fac_max != float('inf') and not np.isinf(args.fac_max)):
        df = filter_by_chinchilla_iterations(df, args.scaling_rule, args.fac_min, args.fac_max)
        if len(df) == 0:
            print("\nNo data remaining after Chinchilla iterations filtering. Exiting.")
            exit(1)
    
    # Filter by specific sizes if requested
    if args.sizes is not None:
        df = df[df['size'].isin(args.sizes)]
        if len(df) == 0:
            print(f"\nNo data remaining after filtering for sizes {args.sizes}. Exiting.")
            exit(1)
        print(f"\nFiltered to sizes: {args.sizes}")
    
    print(f"\n{'='*70}")
    print("Data Summary")
    print(f"{'='*70}")
    print(f"Total runs loaded: {len(df)}")
    print(f"Unique sizes: {sorted(df['size'].unique())}")
    
    # Ensure numeric columns are numeric before computing min/max
    df['iterations'] = pd.to_numeric(df['iterations'], errors='coerce')
    df['gamma_3_factor'] = pd.to_numeric(df['gamma_3_factor'], errors='coerce')
    df['val_loss'] = pd.to_numeric(df['val_loss'], errors='coerce')
    
    # Drop any rows with NaN values after conversion
    df = df.dropna(subset=['iterations', 'gamma_3_factor', 'val_loss'])
    
    if len(df) == 0:
        print("\nNo valid numeric data found after conversion. Exiting.")
        exit(1)
    
    print(f"Iterations range: {df['iterations'].min()} to {df['iterations'].max()}")
    print(f"Gamma_3_factor range: {df['gamma_3_factor'].min():.4f} to {df['gamma_3_factor'].max():.4f}")
    print(f"Val loss range: {df['val_loss'].min():.4f} to {df['val_loss'].max():.4f}")
    
    # Create plot
    print(f"\n{'='*70}")
    print("Creating plot...")
    print(f"{'='*70}")
    
    fig = plot_gamma3_vs_iterations(df, fit_type=args.fit_function, scaling_rule=args.scaling_rule, top_k=args.top_k, show_kappa=args.show_kappa, exp_constant=args.exp_constant, fit_kappa=args.fit_kappa_using_constant, no_within_head_fit=args.no_within_head_fit, no_grey_points=args.no_grey_points)
    
    # Save plot
    if args.output:
        output_file = args.output
    else:
        output_file = f'gamma3_factor_scaling_{args.scaling_rule}_omega{args.target_omega}_{args.fit_function}.pdf'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\n{'='*70}")
    print(f"Plot saved to: {os.path.abspath(output_file)}")
    print(f"{'='*70}")

