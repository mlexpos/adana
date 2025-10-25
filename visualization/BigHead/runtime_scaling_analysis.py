#!/usr/bin/env python3
"""
Runtime Scaling Analysis - Power Law Fit for Runtime vs Depth

This script analyzes the runtime of BigHead sweep runs from wandb,
extrapolates the total runtime based on iter_dt and planned iterations,
and performs a power law fit of runtime vs depth.

Usage:
    python runtime_scaling_analysis.py --optimizer adamw
    python runtime_scaling_analysis.py --optimizer mk4
"""

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style, rc, rcParams
from scipy.optimize import curve_fit
import argparse
import warnings

warnings.filterwarnings('ignore')

# Matplotlib formatting
style.use('default')
rc('font', family='sans-serif')
rcParams['font.weight'] = 'light'
rcParams['font.size'] = 18
rcParams['figure.figsize'] = (12, 8)

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description='Analyze BigHead runtime scaling for different optimizers')
parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'mk4', 'dana', 'ademamix'],
                    help='Optimizer type: adamw, mk4 (dana-star-mk4), dana, or ademamix (default: adamw)')
parser.add_argument('--project', type=str, default='danastar',
                    help='WandB project name (default: danastar)')
parser.add_argument('--group', type=str, default='DanaStar_MK4_BigHead_Sweep',
                    help='WandB group name (default: DanaStar_MK4_BigHead_Sweep)')
parser.add_argument('--entity', type=str, default='ep-rmt-ml-opt',
                    help='WandB entity name (default: ep-rmt-ml-opt)')
args = parser.parse_args()

# Map optimizer abbreviations
optimizer_map = {'adamw': 'adamw', 'mk4': 'dana-star-mk4', 'dana': 'dana', 'ademamix': 'ademamix'}
optimizer_type = optimizer_map[args.optimizer]

# Depths from the sweep script
DEPTHS = [4, 5, 6, 7]

# =============================================================================
# DATA LOADING
# =============================================================================

def load_wandb_runtime_data(project_name, group_name, entity, optimizer_type):
    """
    Load runtime data from WandB for specified optimizer runs.

    Returns a DataFrame with columns: depth, iter_dt, iterations, lr_mult, run_name
    """
    api = wandb.Api()

    print(f"Loading data from group: {group_name}")
    print(f"Project: {project_name}")
    print(f"Entity: {entity}")
    print(f"Optimizer: {optimizer_type}")
    print()

    runs = api.runs(f"{entity}/{project_name}", filters={"group": group_name})

    data = []
    total_runs = 0
    skipped_runs = 0

    for run in runs:
        total_runs += 1
        config = run.config
        summary = run.summary

        # Filter for specified optimizer (from 'opt' config field)
        optimizer = config.get('opt', '')
        if optimizer != optimizer_type:
            skipped_runs += 1
            continue

        # Extract depth from config (stored as 'n_layer')
        depth = config.get('n_layer')
        if depth is None:
            skipped_runs += 1
            continue

        # Extract planned iterations
        iterations = config.get('iterations')
        if iterations is None:
            skipped_runs += 1
            continue

        # Extract iter_dt from summary (average time per iteration)
        iter_dt = summary.get('iter_dt')
        if iter_dt is None or iter_dt == 0:
            skipped_runs += 1
            continue

        # Extract learning rate to determine multiplier
        lr = config.get('lr')

        # Try to extract actual iteration count
        actual_iter = summary.get('iter', 0)

        # Extract model architecture parameters
        n_head = config.get('n_head')
        qkv_dim = config.get('qkv_dim')
        n_embd = config.get('n_embd')

        data.append({
            'depth': depth,
            'iter_dt': iter_dt,
            'iterations': iterations,
            'actual_iter': actual_iter,
            'lr': lr,
            'n_head': n_head,
            'qkv_dim': qkv_dim,
            'n_embd': n_embd,
            'run_name': run.name,
            'run_state': run.state
        })

    print(f"Total runs found: {total_runs}")
    print(f"{optimizer_type} runs loaded: {len(data)}")
    print(f"Skipped runs: {skipped_runs}")
    print()

    df = pd.DataFrame(data)
    return df

# =============================================================================
# RUNTIME CALCULATION
# =============================================================================

def calculate_extrapolated_runtime(df):
    """
    Calculate extrapolated total runtime for each run.
    Runtime (seconds) = iter_dt * iterations
    Runtime (hours) = runtime_seconds / 3600
    """
    df['runtime_seconds'] = df['iter_dt'] * df['iterations']
    df['runtime_hours'] = df['runtime_seconds'] / 3600
    df['runtime_days'] = df['runtime_hours'] / 24

    return df

# =============================================================================
# POWER LAW FITTING
# =============================================================================

def power_law(x, A, B):
    """Power law function: y = A * x^B"""
    return A * np.power(x, B)

def fit_runtime_vs_depth(df):
    """
    Fit power law to runtime vs depth.
    For each depth, we'll use the median runtime across different LR multipliers.
    """
    # Group by depth and calculate median runtime
    depth_runtime = df.groupby('depth').agg({
        'runtime_hours': ['median', 'mean', 'std', 'count']
    }).reset_index()

    depth_runtime.columns = ['depth', 'median_runtime_hours', 'mean_runtime_hours',
                             'std_runtime_hours', 'count']

    print("Runtime statistics by depth:")
    print(depth_runtime)
    print()

    # Prepare data for fitting
    depths = depth_runtime['depth'].values
    runtimes = depth_runtime['median_runtime_hours'].values

    # Fit power law
    try:
        # Initial guess: A=1, B=2 (quadratic scaling is common)
        popt, pcov = curve_fit(power_law, depths, runtimes, p0=[1.0, 2.0])
        A_fit, B_fit = popt

        # Calculate R-squared
        residuals = runtimes - power_law(depths, A_fit, B_fit)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((runtimes - np.mean(runtimes))**2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"Power law fit: Runtime (hours) = {A_fit:.6f} * depth^{B_fit:.4f}")
        print(f"R-squared: {r_squared:.6f}")
        print()

        return A_fit, B_fit, r_squared, depth_runtime

    except Exception as e:
        print(f"Error fitting power law: {e}")
        return None, None, None, depth_runtime

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_runtime_scaling(df, depth_runtime, A_fit, B_fit, r_squared, optimizer_name):
    """
    Create visualization of runtime scaling with depth.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left plot: Individual runs and median
    depths = sorted(df['depth'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))
    color_map = {d: colors[i] for i, d in enumerate(depths)}

    # Plot individual runs
    for depth in depths:
        depth_data = df[df['depth'] == depth]
        ax1.scatter([depth] * len(depth_data), depth_data['runtime_hours'],
                   c=[color_map[depth]], alpha=0.5, s=100, edgecolors='black', linewidths=0.5)

    # Plot median values
    ax1.scatter(depth_runtime['depth'], depth_runtime['median_runtime_hours'],
               c='red', marker='D', s=200, edgecolors='black', linewidths=2,
               label='Median runtime', zorder=10)

    # Plot power law fit
    if A_fit is not None and B_fit is not None:
        depth_range = np.linspace(min(depths) * 0.9, max(depths) * 1.5, 100)
        runtime_fit = power_law(depth_range, A_fit, B_fit)
        ax1.plot(depth_range, runtime_fit, '--', color='red', linewidth=3,
                label=f'Power law fit: {A_fit:.3f} × depth$^{{{B_fit:.3f}}}$\n$R^2$ = {r_squared:.4f}')

        # Extrapolate to depths 8-12
        extrapolation_depths = [8, 9, 10, 11, 12]
        for extrap_depth in extrapolation_depths:
            extrap_runtime = power_law(extrap_depth, A_fit, B_fit)
            ax1.scatter([extrap_depth], [extrap_runtime], marker='*', s=300,
                       c='orange', edgecolors='black', linewidths=1.5, zorder=11)
            ax1.text(extrap_depth, extrap_runtime * 1.1, f'{extrap_runtime:.1f}h',
                    ha='center', va='bottom', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.3))

    # Map optimizer names for display
    optimizer_title_map = {'adamw': 'AdamW', 'mk4': 'Dana-Star-MK4', 'dana': 'Dana-Star', 'ademamix': 'AdemaMix'}
    optimizer_title = optimizer_title_map.get(optimizer_name, optimizer_name)

    ax1.set_xlabel('Depth (number of layers)', fontsize=18)
    ax1.set_ylabel('Total Runtime (hours)', fontsize=18)
    ax1.set_title(f'Runtime Scaling with Depth ({optimizer_title})', fontsize=20, fontweight='bold')
    ax1.legend(fontsize=14, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')

    # Right plot: Log-log plot for verification
    ax2.scatter(depth_runtime['depth'], depth_runtime['median_runtime_hours'],
               c='red', marker='D', s=200, edgecolors='black', linewidths=2,
               label='Median runtime')

    if A_fit is not None and B_fit is not None:
        depth_range = np.linspace(min(depths), max(depths) * 1.5, 100)
        runtime_fit = power_law(depth_range, A_fit, B_fit)
        ax2.plot(depth_range, runtime_fit, '--', color='red', linewidth=3,
                label=f'Power law fit: {A_fit:.3f} × depth$^{{{B_fit:.3f}}}$')

    ax2.set_xlabel('Depth (number of layers)', fontsize=18)
    ax2.set_ylabel('Total Runtime (hours)', fontsize=18)
    ax2.set_title('Runtime Scaling (Log-Log)', fontsize=20, fontweight='bold')
    ax2.legend(fontsize=14)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()

    # Save plot
    optimizer_filename_map = {'adamw': 'adamw', 'mk4': 'mk4', 'dana': 'dana', 'ademamix': 'ademamix'}
    opt_filename = optimizer_filename_map.get(optimizer_name, optimizer_name)
    output_file = f'{opt_filename}_runtime_scaling_vs_depth.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Plot saved to: {output_file}")

    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print(f"Runtime Scaling Analysis - Power Law Fit")
    print(f"Optimizer: {args.optimizer} ({optimizer_type})")
    print("="*70)
    print()

    # Load data
    df = load_wandb_runtime_data(args.project, args.group, args.entity, optimizer_type)

    if len(df) == 0:
        print("No data found. Exiting.")
        exit(1)

    # Display sample data
    print("Sample data:")
    print(df.head(10))
    print()

    # Calculate extrapolated runtimes
    df = calculate_extrapolated_runtime(df)

    print("Runtime extrapolations (first 10 runs):")
    print(df[['depth', 'iter_dt', 'iterations', 'runtime_hours', 'runtime_days', 'run_name']].head(10))
    print()

    # Fit power law
    A_fit, B_fit, r_squared, depth_runtime = fit_runtime_vs_depth(df)

    # Create visualization
    if A_fit is not None:
        plot_runtime_scaling(df, depth_runtime, A_fit, B_fit, r_squared, args.optimizer)

        # Print predictions
        print("\nExtrapolated runtimes for larger depths:")
        for depth in [8, 9, 10, 11, 12, 15, 20]:
            runtime_pred = power_law(depth, A_fit, B_fit)
            print(f"  Depth {depth:2d}: {runtime_pred:7.2f} hours ({runtime_pred/24:6.2f} days)")
    else:
        print("Could not fit power law.")

    # Save detailed results to CSV
    optimizer_filename_map = {'adamw': 'adamw', 'mk4': 'mk4', 'dana': 'dana', 'ademamix': 'ademamix'}
    opt_filename = optimizer_filename_map.get(args.optimizer, args.optimizer)
    output_csv = f'{opt_filename}_runtime_data.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed data saved to: {output_csv}")
