#!/usr/bin/env python3
"""
Plot Auto Factor vs Training Steps with Power Law Fit

This script plots the auto_factor from auto-dana optimizer training runs,
fits a power law to the data in log-log space, and reports the slope.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import warnings
from matplotlib.ticker import FuncFormatter
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS AS NEEDED
# ============================================================================

# WandB project configuration
WANDB_PROJECT = "danastar"
WANDB_ENTITY = "ep-rmt-ml-opt"

# List of run IDs to plot - ADD YOUR RUN IDs HERE
RUN_IDS = ["dvbuhdoo",
"zme347io",
"simwryu2"
    # "abc123def456",  # Example run ID
    # Add your actual run IDs here
]

# Plot configuration
FIGURE_SIZE = (12, 8)
DPI = 300
SAVE_FORMAT = "pdf"  # or "png"

# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def fit_power_law(x, y):
    """
    Fit a power law y = a * x^b in log-log space using numpy.
    
    Args:
        x: Independent variable (e.g., training steps)
        y: Dependent variable (e.g., auto_factor)
    
    Returns:
        slope (b), intercept (log(a)), r_squared
    """
    # Take log of both x and y
    log_x = np.log(x)
    log_y = np.log(y)
    
    # Perform linear regression in log-log space using numpy polyfit
    # polyfit(x, y, 1) returns [slope, intercept] for a degree-1 polynomial
    slope, intercept = np.polyfit(log_x, log_y, 1)
    
    # Calculate R-squared
    y_pred = slope * log_x + intercept
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return slope, intercept, r_squared

def format_tokens(x, pos):
    """
    Format tokens as M (millions) or G (billions).
    
    Args:
        x: The tick value (number of tokens)
        pos: The tick position (unused but required by FuncFormatter)
    
    Returns:
        Formatted string
    """
    if x >= 1e9:
        return f'{x/1e9:.1f}G'
    elif x >= 1e6:
        return f'{x/1e6:.0f}M'
    else:
        return f'{x:.0f}'

def plot_auto_factor(run_ids: List[str], project: str, entity: str):
    """
    Plot auto_factor vs training steps for given run IDs with power law fit.
    
    Args:
        run_ids: List of WandB run IDs to plot
        project: WandB project name
        entity: WandB entity/team name
    """
    print(f"🚀 Plotting auto_factor for {len(run_ids)} runs")
    print(f"📊 Project: {entity}/{project}")
    
    # Connect to WandB
    try:
        api = wandb.Api()
        print("✓ Successfully connected to WandB API")
    except Exception as e:
        print(f"✗ Failed to connect to WandB API: {e}")
        return
    
    # Set up the plot
    plt.figure(figsize=FIGURE_SIZE)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))
    baseline_first_value = None
    baseline_run_name = ""
    
    for i, run_id in enumerate(run_ids):
        try:
            print(f"\n📥 Fetching run: {run_id}")
            
            # Get the run
            run = api.run(f"{entity}/{project}/{run_id}")
            
            # Get run config for legend info
            config = run.config
            opt = config.get('opt', 'unknown')
            lr = config.get('lr', 0)
            
            # Get the history (training data) - full, unsampled
            records = []
            for row in run.scan_history(keys=['_step', 'optimizer/auto_factor']):
                records.append(row)
            history = pd.DataFrame.from_records(records)
            
            if history.empty:
                print(f"  ⚠️  No history data found for run {run_id}")
                continue
            
            # Extract iterations and auto_factor
            if '_step' in history.columns and 'optimizer/auto_factor' in history.columns:
                # Create a mask for rows where both values are non-NaN
                valid_rows = history[['_step', 'optimizer/auto_factor']].notna().all(axis=1)
                
                # Filter to only valid rows
                valid_data = history[valid_rows]
                iterations = valid_data['_step'].values
                auto_factor = valid_data['optimizer/auto_factor'].values
                
                # Remove any remaining invalid values and enforce start at step >= 2
                valid_mask = (iterations >= 2) & (auto_factor > 0) & np.isfinite(iterations) & np.isfinite(auto_factor)
                iterations = iterations[valid_mask]
                auto_factor = auto_factor[valid_mask]
                
                if len(iterations) == 0:
                    print(f"  ⚠️  No valid data points for run {run_id}")
                    continue
                
                print(f"  📊 First step: {iterations[0]}, Last step: {iterations[-1]}, Total points: {len(iterations)}")
                
                # Capture baseline from the first valid run
                if baseline_first_value is None:
                    baseline_first_value = float(auto_factor[0])
                    baseline_run_name = run.name
                
                # Create legend label
                legend_label = f"{opt} | lr={lr:.1e}"
                
                # Plot the curve on log-log scale
                plt.loglog(iterations, auto_factor, 
                          color=colors[i], 
                          linewidth=2, 
                          alpha=0.7,
                          label=legend_label,
                          marker='o',
                          markersize=3,
                          markevery=max(1, len(iterations)//50))
                
                # Explicitly mark first and last points
                plt.loglog(iterations[0], auto_factor[0], 
                          color=colors[i], 
                          marker='o', 
                          markersize=8,
                          markeredgewidth=2,
                          markeredgecolor='black',
                          zorder=5)
                plt.loglog(iterations[-1], auto_factor[-1], 
                          color=colors[i], 
                          marker='s', 
                          markersize=8,
                          markeredgewidth=2,
                          markeredgecolor='black',
                          zorder=5)
                
                # Fit power law using only the last 90% of the data
                start_idx = int(len(iterations) * 0.1)
                iterations_fit = iterations[start_idx:]
                auto_factor_fit = auto_factor[start_idx:]
                slope, intercept, r_squared = fit_power_law(iterations_fit, auto_factor_fit)
                
                # Generate fitted line
                x_fit = np.linspace(iterations.min(), iterations.max(), 100)
                y_fit = np.exp(intercept) * x_fit**slope
                
                # Plot fitted line with slope in legend
                plt.loglog(x_fit, y_fit, 
                          color=colors[i], 
                          linewidth=2, 
                          linestyle='--',
                          alpha=0.5,
                          label=f'Fit: step^{{{slope:.3f}}} (R²={r_squared:.3f})')
                
                print(f"  ✓ Plotted {len(iterations)} points for {run.name}")
                print(f"  📐 Power law fit (using last 90% of data):")
                print(f"     auto_factor ∝ step^{slope:.4f}")
                print(f"     Intercept (log scale): {intercept:.4f}")
                print(f"     R² = {r_squared:.4f}")
                print(f"     Fitted on {len(iterations_fit)} points (from index {start_idx} to {len(iterations)})")
                
            else:
                print(f"  ⚠️  Missing '_step' or 'optimizer/auto_factor' columns for run {run_id}")
                print(f"     Available columns: {history.columns.tolist()}")
                continue
                
        except Exception as e:
            print(f"  ✗ Error processing run {run_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Customize the log-log plot
    plt.xlabel('Iterations (Steps)', fontsize=12)
    plt.ylabel('Auto Factor', fontsize=12)
    plt.title('Auto Factor vs Iterations (Log-Log)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    
    # Plot baseline horizontal line for the first value of the first valid run
    if baseline_first_value is not None:
        plt.axhline(y=baseline_first_value, color='gray', linestyle=':', linewidth=1.8,
                    label=f'Baseline {baseline_run_name} first value = {baseline_first_value:.3g}')
    
    plt.legend(fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"visualization/auto_factor_analysis.{SAVE_FORMAT}"
    plt.savefig(filename, format=SAVE_FORMAT, dpi=DPI, bbox_inches='tight')
    print(f"\n💾 Saved plot: {filename}")
    
    plt.show()
    
    print(f"\n✅ Successfully plotted auto_factor analysis!")

def plot_auto_factor_from_ids(run_ids: List[str]):
    """
    Convenience function to plot auto_factor from a list of run IDs.
    
    Args:
        run_ids: List of WandB run IDs
    """
    if not run_ids:
        print("❌ No run IDs provided. Please add run IDs to the RUN_IDS list or pass them as argument.")
        return
    
    plot_auto_factor(run_ids, WANDB_PROJECT, WANDB_ENTITY)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Main function - modify RUN_IDS list above or call plot_auto_factor_from_ids() with your IDs."""
    
    # Example: Use the RUN_IDS list defined at the top
    if RUN_IDS:
        plot_auto_factor_from_ids(RUN_IDS)
    else:
        print("📝 To use this script:")
        print("1. Add your run IDs to the RUN_IDS list at the top of the file, or")
        print("2. Call plot_auto_factor_from_ids(['run_id_1', 'run_id_2', ...]) with your run IDs")
        print("\nExample:")
        print("plot_auto_factor_from_ids(['abc123def456', 'xyz789uvw012'])")
        
        # Alternative: You can also call it with specific run IDs directly
        # plot_auto_factor_from_ids(['your_run_id_here'])

if __name__ == "__main__":
    main()

