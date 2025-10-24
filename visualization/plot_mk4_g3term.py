#!/usr/bin/env python3
"""
Plot optimizer/auto_factor over time for Dana-Star-MK4 sweep runs.

Downloads time series data from WandB and creates a log-log plot showing:
- Individual curves for each run in the group
- Average curve across all runs
- Power law fit to the average
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.optimize import curve_fit

def load_wandb_timeseries(project_name, group_name, metric='optimizer/auto_factor'):
    """
    Load time series data from Weights & Biases for Dana-Star-MK4 runs.

    Args:
        project_name (str): WandB project name
        group_name (str): WandB experiment group name
        metric (str): Metric to extract (default: 'optimizer/auto_factor')

    Returns:
        dict: Dictionary mapping run names to DataFrames with columns [iter, metric_value]
    """
    # Initialize WandB API
    api = wandb.Api()

    print(f"Downloading data from project: {project_name}, group: {group_name}")

    # Get all runs in the specified group
    runs = api.runs(f"ep-rmt-ml-opt/{project_name}", filters={"group": group_name})

    run_data = {}

    for run in runs:
        print(f"Processing run: {run.name}")

        try:
            # Get the history (time series data) for this run
            # We need to get the iter and the metric
            history = run.history(keys=['iter', metric], samples=10000)

            if history.empty:
                print(f"  Skipping run {run.name} - no history data")
                continue

            # Rename columns for consistency
            history = history.rename(columns={'iter': 'iter', metric: 'value'})

            # Drop rows with NaN values
            history = history.dropna()

            if len(history) == 0:
                print(f"  Skipping run {run.name} - no valid data after dropping NaNs")
                continue

            run_data[run.name] = history
            print(f"  Loaded {len(history)} data points")

        except Exception as e:
            print(f"  Error processing run {run.name}: {e}")
            continue

    return run_data

def power_law(x, a, b):
    """Power law function: y = a * x^b"""
    return a * np.power(x, b)

def compute_average_curve(run_data):
    """
    Compute average curve across all runs.

    Interpolates all runs to a common grid of iterations, then averages.

    Args:
        run_data (dict): Dictionary mapping run names to DataFrames

    Returns:
        pd.DataFrame: DataFrame with columns [iter, avg_value]
    """
    if not run_data:
        return pd.DataFrame()

    # Find the range of iterations across all runs
    all_iters = []
    for df in run_data.values():
        all_iters.extend(df['iter'].values)

    min_iter = max(min(all_iters), 1)  # Avoid iter=0 for log scale
    max_iter = max(all_iters)

    # Create a common grid of iterations (log-spaced for better resolution)
    common_iters = np.logspace(np.log10(min_iter), np.log10(max_iter), 200)

    # Interpolate each run to the common grid
    interpolated_values = []

    for run_name, df in run_data.items():
        # Sort by iter
        df_sorted = df.sort_values('iter')

        # Interpolate (log-log scale)
        interp_values = np.interp(
            np.log10(common_iters),
            np.log10(df_sorted['iter'].values),
            np.log10(df_sorted['value'].values)
        )

        interpolated_values.append(10**interp_values)

    # Compute average
    avg_values = np.mean(interpolated_values, axis=0)

    return pd.DataFrame({'iter': common_iters, 'avg_value': avg_values})

def fit_power_law(df, iter_min=None, iter_max=None):
    """
    Fit a power law to the data within a specified iteration window.

    Args:
        df (pd.DataFrame): DataFrame with columns [iter, avg_value]
        iter_min (int, optional): Minimum iteration for fit window
        iter_max (int, optional): Maximum iteration for fit window

    Returns:
        tuple: (a, b) parameters for y = a * x^b, or None if fit fails
    """
    try:
        # Filter data to specified iteration window
        fit_df = df.copy()
        if iter_min is not None:
            fit_df = fit_df[fit_df['iter'] >= iter_min]
        if iter_max is not None:
            fit_df = fit_df[fit_df['iter'] <= iter_max]

        if len(fit_df) < 2:
            print(f"Not enough data points in fit window [{iter_min}, {iter_max}]")
            return None

        print(f"Fitting power law on {len(fit_df)} points in iteration range [{fit_df['iter'].min():.0f}, {fit_df['iter'].max():.0f}]")

        # Fit in log-log space
        popt, _ = curve_fit(
            power_law,
            fit_df['iter'].values,
            fit_df['avg_value'].values,
            p0=[1.0, -0.5],
            maxfev=10000
        )
        return popt
    except Exception as e:
        print(f"Power law fit failed: {e}")
        return None

def plot_timeseries(run_data, output_filename='mk4_g3term.pdf', iter_min=None, iter_max=None):
    """
    Create log-log plot of optimizer/auto_factor over time.

    Args:
        run_data (dict): Dictionary mapping run names to DataFrames
        output_filename (str): Output PDF filename
        iter_min (int, optional): Minimum iteration for power law fit window
        iter_max (int, optional): Maximum iteration for power law fit window
    """
    print(f"\nCreating visualization with {len(run_data)} runs")

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot individual runs with low alpha
    for run_name, df in run_data.items():
        plt.plot(df['iter'], df['value'], alpha=0.3, linewidth=1, color='gray')

    # Compute and plot average
    avg_df = compute_average_curve(run_data)

    if not avg_df.empty:
        plt.plot(avg_df['iter'], avg_df['avg_value'],
                color='blue', linewidth=3, label='Average', zorder=10)

        # Fit and plot power law
        power_law_params = fit_power_law(avg_df, iter_min, iter_max)

        if power_law_params is not None:
            a, b = power_law_params

            # Generate power law curve only within the fit window
            fit_iters = avg_df['iter'].values.copy()
            if iter_min is not None:
                fit_iters = fit_iters[fit_iters >= iter_min]
            if iter_max is not None:
                fit_iters = fit_iters[fit_iters <= iter_max]

            values_fit = power_law(fit_iters, a, b)

            fit_window_str = ""
            if iter_min is not None or iter_max is not None:
                fit_window_str = f" (fit: {iter_min or 'start'}-{iter_max or 'end'})"

            plt.plot(fit_iters, values_fit,
                    color='red', linewidth=2, linestyle='--',
                    label=f'Power law fit: {a:.3e} × iter^{b:.3f}{fit_window_str}', zorder=11)

            print(f"\nPower law fit: y = {a:.3e} × x^{b:.3f}")

    # Formatting
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('optimizer/auto_factor', fontsize=12)
    plt.title(r'norm of $||m||/\sqrt{v}$', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    # Save plot
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as {output_filename}")

    # Print statistics
    print("\nDataset statistics:")
    print(f"Total runs: {len(run_data)}")

    if not avg_df.empty:
        print(f"Iteration range: {avg_df['iter'].min():.0f} to {avg_df['iter'].max():.0f}")
        print(f"Average value range: {avg_df['avg_value'].min():.6e} to {avg_df['avg_value'].max():.6e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot optimizer/auto_factor over time for Dana-Star-MK4 runs')
    parser.add_argument('--project', type=str, default='danastar',
                        help='WandB project name (default: danastar)')
    parser.add_argument('--group', type=str, default='DanaStar_MK4_Small_Sweep_formula9',
                        help='WandB group name (default: DanaStar_MK4_Small_Sweep_formula9)')
    parser.add_argument('--output', type=str, default='mk4_g3term.pdf',
                        help='Output PDF filename (default: mk4_g3term.pdf)')
    parser.add_argument('--metric', type=str, default='optimizer/auto_factor',
                        help='Metric to plot (default: optimizer/auto_factor)')
    parser.add_argument('--iter-min', type=float, default=None,
                        help='Minimum iteration for power law fit window (default: None, uses all data)')
    parser.add_argument('--iter-max', type=float, default=None,
                        help='Maximum iteration for power law fit window (default: None, uses all data)')
    args = parser.parse_args()

    print("Dana-Star-MK4 G3 Term Visualization")
    print("=" * 60)

    if args.iter_min is not None or args.iter_max is not None:
        print(f"Power law fit window: [{args.iter_min or 'start'}, {args.iter_max or 'end'}]")

    # Load data
    run_data = load_wandb_timeseries(args.project, args.group, args.metric)

    if len(run_data) == 0:
        print("ERROR: No data found. Check project and group names.")
    else:
        # Create plot
        plot_timeseries(run_data, args.output, args.iter_min, args.iter_max)
        print("\nVisualization completed successfully!")
