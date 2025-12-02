#!/usr/bin/env python3
"""
Gamma3 Factor Scaling Analysis

This script analyzes the optimal gamma_3_factor value as a function of training iterations
for different model sizes. It fetches runs from wandb with dana optimizer and renormalized
weight decay (omega) = 4, then plots and fits the relationship.

Usage:
    python gamma3_factor_scaling.py --target-omega 4.0 --scaling-rule Enoki_Scaled
    python gamma3_factor_scaling.py --target-omega 4.0 --scaling-rule Enoki_Scaled --fit-function power
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
args = parser.parse_args()

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
        
        if None in [gamma_3_factor, val_loss, iterations, renorm_weight_decay]:
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
            print(f"    Size {size}: {len(size_df)} runs, renorm_weight_decay values: {sorted(renorm_wds)}")
    
    return df

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

def fit_function(iterations, gamma_values, fit_type='power'):
    """
    Fit a function to the data.
    
    Returns:
        params: fitted parameters
        func: the function used for fitting
    """
    if fit_type == 'power':
        func = power_law
        # Initial guess for power law
        p0 = [gamma_values[0], -0.5]
    elif fit_type == 'linear':
        func = linear_func
        # Initial guess for linear
        p0 = [0, gamma_values[0]]
    elif fit_type == 'exponential':
        func = exponential_func
        # Initial guess for exponential
        p0 = [gamma_values[0], -1e-6]
    else:
        raise ValueError(f"Unknown fit type: {fit_type}")
    
    try:
        params, _ = curve_fit(func, iterations, gamma_values, p0=p0, maxfev=10000)
        return params, func
    except Exception as e:
        print(f"  Warning: Fitting failed with error: {e}")
        return None, func

# =============================================================================
# PLOTTING
# =============================================================================

def plot_gamma3_vs_iterations(df, fit_type='power', scaling_rule='Enoki_Scaled'):
    """
    Plot optimal gamma_3_factor vs iterations for each model size.
    Shows all data points (gray) and highlights optimal ones (colored).
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get unique sizes
    sizes = sorted(df['size'].unique())
    
    # Color map for sizes
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    
    # Track if we've added the "all points" legend entry
    all_points_legend_added = False
    
    # For each size, plot all points and find the best gamma_3_factor at each iteration count
    for idx, size in enumerate(sizes):
        size_df = df[df['size'] == size].copy()
        
        if len(size_df) == 0:
            continue
        
        # Plot ALL data points with size proportional to rank (like bighead_lr_scaling.py)
        # Group by iterations to rank within each iteration count
        unique_iters = sorted(size_df['iterations'].unique())
        
        best_gamma = []
        best_iterations = []
        
        for iters in unique_iters:
            iter_df = size_df[size_df['iterations'] == iters].copy()
            
            # Sort by val_loss to get ranks
            iter_df = iter_df.sort_values('val_loss')
            n_runs = len(iter_df)
            
            # Plot each point with size based on rank
            for rank, (_, row) in enumerate(iter_df.iterrows()):
                weight = n_runs - rank  # Best gets n_runs, worst gets 1
                point_size = weight * 15  # Scale factor for visibility
                
                if rank == 0:
                    # Best point: colored, larger, black edge
                    if not all_points_legend_added:
                        ax.scatter([row['iterations']], [row['gamma_3_factor']], 
                                  s=point_size, c=[colors[idx]], alpha=0.8, 
                                  edgecolors='black', linewidths=1.5, zorder=10,
                                  label=f'Size {size} (best at each iteration)')
                        all_points_legend_added = True
                    else:
                        ax.scatter([row['iterations']], [row['gamma_3_factor']], 
                                  s=point_size, c=[colors[idx]], alpha=0.8, 
                                  edgecolors='black', linewidths=1.5, zorder=10)
                    
                    # Track best for fitting
                    best_gamma.append(row['gamma_3_factor'])
                    best_iterations.append(row['iterations'])
                else:
                    # Other points: gray, smaller
                    ax.scatter([row['iterations']], [row['gamma_3_factor']], 
                              s=point_size, c='gray', alpha=0.3, 
                              edgecolors='none', zorder=5)
        
        if len(best_iterations) < 2:
            print(f"  Warning: Size {size} has only {len(best_iterations)} data points, skipping fit")
            continue
        
        # Fit function
        best_iterations_arr = np.array(best_iterations)
        best_gamma_arr = np.array(best_gamma)
        
        params, func = fit_function(best_iterations_arr, best_gamma_arr, fit_type)
        
        if params is not None:
            # Plot fitted curve
            iter_range = np.linspace(min(best_iterations_arr), max(best_iterations_arr) * 1.2, 200)
            gamma_fit = func(iter_range, *params)
            
            # Create label based on fit type
            if fit_type == 'power':
                label = f'Size {size} fit: {params[0]:.2e} × $T^{{{params[1]:.3f}}}$'
            elif fit_type == 'linear':
                label = f'Size {size} fit: {params[0]:.2e} × T + {params[1]:.3f}'
            elif fit_type == 'exponential':
                label = f'Size {size} fit: {params[0]:.2e} × exp({params[1]:.2e} × T)'
            
            ax.plot(iter_range, gamma_fit, '--', color=colors[idx], linewidth=2.5, 
                   label=label, zorder=9)
            
            print(f"\nSize {size}:")
            print(f"  Data points: {len(best_iterations)}")
            print(f"  Iterations range: {min(best_iterations)} to {max(best_iterations)}")
            print(f"  Gamma_3_factor range: {min(best_gamma):.4f} to {max(best_gamma):.4f}")
            if fit_type == 'power':
                print(f"  Fit: gamma_3_factor = {params[0]:.6e} × iterations^{params[1]:.4f}")
            elif fit_type == 'linear':
                print(f"  Fit: gamma_3_factor = {params[0]:.6e} × iterations + {params[1]:.4f}")
            elif fit_type == 'exponential':
                print(f"  Fit: gamma_3_factor = {params[0]:.6e} × exp({params[1]:.6e} × iterations)")
    
    # Formatting
    ax.set_xlabel('Training Iterations', fontsize=20)
    ax.set_ylabel('Gamma 3 Factor', fontsize=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Optimal Gamma 3 Factor vs Training Iterations\n(Dana, renorm_weight_decay = {args.target_omega}, {scaling_rule})',
                fontsize=20, fontweight='bold')
    ax.legend(fontsize=13, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotation about point sizes
    ax.text(0.02, 0.02, 'Point size ∝ rank\n(best at each iteration is largest)',
            transform=ax.transAxes, fontsize=14, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
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
    
    print(f"\n{'='*70}")
    print("Data Summary")
    print(f"{'='*70}")
    print(f"Total runs loaded: {len(df)}")
    print(f"Unique sizes: {sorted(df['size'].unique())}")
    print(f"Iterations range: {df['iterations'].min()} to {df['iterations'].max()}")
    print(f"Gamma_3_factor range: {df['gamma_3_factor'].min():.4f} to {df['gamma_3_factor'].max():.4f}")
    print(f"Val loss range: {df['val_loss'].min():.4f} to {df['val_loss'].max():.4f}")
    
    # Create plot
    print(f"\n{'='*70}")
    print("Creating plot...")
    print(f"{'='*70}")
    
    fig = plot_gamma3_vs_iterations(df, fit_type=args.fit_function, scaling_rule=args.scaling_rule)
    
    # Save plot
    if args.output:
        output_file = args.output
    else:
        output_file = f'gamma3_factor_scaling_{args.scaling_rule}_omega{args.target_omega}_{args.fit_function}.pdf'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    print(f"\n{'='*70}")
    print(f"Plot saved to: {os.path.abspath(output_file)}")
    print(f"{'='*70}")

