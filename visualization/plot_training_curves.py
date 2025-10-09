#!/usr/bin/env python3
"""
Plot Training Curves from WandB Run IDs

This script takes a list of WandB run IDs and creates a log-log plot showing
training curves with tokens on x-axis and validation loss on y-axis.
Legend shows optimizer, learning rate, and renormalized weight decay.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS AS NEEDED
# ============================================================================

# WandB project configuration
WANDB_PROJECT = "danastar"
WANDB_ENTITY = "ep-rmt-ml-opt"

# List of run IDs to plot - ADD YOUR RUN IDs HERE
RUN_IDS = [
    "76b0qmrg",
    "7cy9z3c1",
    # "abc123def456",  # Example run ID
    # "xyz789uvw012",  # Example run ID
    # Add your actual run IDs here
]

# Plot configuration
FIGURE_SIZE = (12, 8)
DPI = 300
SAVE_FORMAT = "pdf"  # or "png"

# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def plot_training_curves(run_ids: List[str], project: str, entity: str):
    """
    Plot training curves for given run IDs.
    
    Args:
        run_ids: List of WandB run IDs to plot
        project: WandB project name
        entity: WandB entity/team name
    """
    print(f"🚀 Plotting training curves for {len(run_ids)} runs")
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
    plt.style.use('default')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))
    
    for i, run_id in enumerate(run_ids):
        try:
            print(f"\n📥 Fetching run: {run_id}")
            
            # Get the run
            run = api.run(f"{entity}/{project}/{run_id}")
            
            # Get run config for legend info
            config = run.config
            opt = config.get('opt', 'unknown')
            lr = config.get('lr', 0)
            weight_decay = config.get('weight_decay', 0)
            iterations = config.get('iterations', 1)
            
            # Calculate renormalized weight decay
            renorm_wd = weight_decay * lr * iterations if all(x is not None for x in [weight_decay, lr, iterations]) else 0
            
            # Get the history (training data)
            history = run.history()
            
            if history.empty:
                print(f"  ⚠️  No history data found for run {run_id}")
                continue
            
            # Extract tokens and validation loss
            if 'tokens' in history.columns and 'val/loss' in history.columns:
                tokens = history['tokens'].dropna()
                val_loss = history['val/loss'].dropna()
                
                # Align the data (in case of different lengths)
                min_len = min(len(tokens), len(val_loss))
                tokens = tokens.iloc[:min_len]
                val_loss = val_loss.iloc[:min_len]
                
                # Remove any remaining NaN or invalid values
                valid_mask = (tokens > 0) & (val_loss > 0) & np.isfinite(tokens) & np.isfinite(val_loss)
                tokens = tokens[valid_mask]
                val_loss = val_loss[valid_mask]
                
                if len(tokens) == 0:
                    print(f"  ⚠️  No valid data points for run {run_id}")
                    continue
                
                # Create legend label
                legend_label = f"{opt} | lr={lr:.1e} | W={renorm_wd:.1e}"
                
                # Plot the curve
                plt.loglog(tokens, val_loss, 
                          color=colors[i], 
                          linewidth=2, 
                          alpha=0.8,
                          label=legend_label)
                
                print(f"  ✓ Plotted {len(tokens)} points for {run.name}")
                
            else:
                print(f"  ⚠️  Missing 'tokens' or 'val/loss' columns for run {run_id}")
                continue
                
        except Exception as e:
            print(f"  ✗ Error processing run {run_id}: {e}")
            continue
    
    # Customize the plot
    plt.xlabel('Tokens (G)', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Training Curves: Validation Loss vs Tokens', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Format x-axis to show M (millions) or G (billions) as appropriate
    def format_tokens(x, p):
        if x >= 1e9:
            return f'{x/1e9:.1f}G'
        elif x >= 1e6:
            return f'{x/1e6:.0f}M'
        else:
            return f'{x:.0f}'
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_tokens))
    
    # Let the plot auto-scale to show all data
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"visualization/training_curves.{SAVE_FORMAT}"
    plt.savefig(filename, format=SAVE_FORMAT, dpi=DPI, bbox_inches='tight')
    print(f"\n💾 Saved plot: {filename}")
    
    plt.show()
    
    print(f"\n✅ Successfully plotted training curves!")

def plot_curves_from_ids(run_ids: List[str]):
    """
    Convenience function to plot curves from a list of run IDs.
    
    Args:
        run_ids: List of WandB run IDs
    """
    if not run_ids:
        print("❌ No run IDs provided. Please add run IDs to the RUN_IDS list or pass them as argument.")
        return
    
    plot_training_curves(run_ids, WANDB_PROJECT, WANDB_ENTITY)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Main function - modify RUN_IDS list above or call plot_curves_from_ids() with your IDs."""
    
    # Example: Use the RUN_IDS list defined at the top
    if RUN_IDS:
        plot_curves_from_ids(RUN_IDS)
    else:
        print("📝 To use this script:")
        print("1. Add your run IDs to the RUN_IDS list at the top of the file, or")
        print("2. Call plot_curves_from_ids(['run_id_1', 'run_id_2', ...]) with your run IDs")
        print("\nExample:")
        print("plot_curves_from_ids(['abc123def456', 'xyz789uvw012'])")
        
        # Example with dummy IDs (will fail but shows the format)
        # plot_curves_from_ids(['example_run_id_1', 'example_run_id_2'])

if __name__ == "__main__":
    main()
