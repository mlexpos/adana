#!/usr/bin/env python3
"""
Plot Training Curves from WandB Run IDs
"""

import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
WANDB_PROJECT = "danastar"
WANDB_ENTITY = "ep-rmt-ml-opt"

# Add your run IDs here
RUN_IDS = [
    "0j139dq3", "jgoym1f8", "kpp1308q"
]

def get_run_data(run_id):
    """Get all data from a WandB run."""
    try:
        api = wandb.Api()
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        
        # Get configuration
        config = run.config
        opt = config.get('opt', 'unknown')
        lr = config.get('lr', 0)
        weight_decay = config.get('weight_decay', 0)
        iterations = config.get('iterations', 1)
        renorm_wd = weight_decay * lr * iterations
        
        # Try different methods to get ALL data
        print(f"Run {run_id}: Trying different data retrieval methods...")
        
        # Method 1: Standard history()
        history = run.history()
        print(f"Run {run_id}: Method 1 (history()): {len(history)} points")
        
        # Method 2: History with all samples
        history_all = run.history(samples=10000)  # Large number to get all
        print(f"Run {run_id}: Method 2 (history(samples=10000)): {len(history_all)} points")
        
        # Method 3: Scan history
        history_scan = run.scan_history()
        scan_data = list(history_scan)
        print(f"Run {run_id}: Method 3 (scan_history()): {len(scan_data)} points")
        
        # Use the method with most data
        if len(scan_data) > len(history_all) and len(scan_data) > len(history):
            print(f"Run {run_id}: Using scan_history() - most data points")
            df = pd.DataFrame(scan_data)
        elif len(history_all) > len(history):
            print(f"Run {run_id}: Using history(samples=10000) - more data points")
            df = history_all
        else:
            print(f"Run {run_id}: Using standard history()")
            df = history
        
        if df.empty:
            print(f"Run {run_id}: No data found")
            return None
            
        # Check what columns are available
        print(f"Run {run_id}: Available columns: {list(df.columns)}")
        print(f"Run {run_id}: First few rows:")
        print(df.head())
        
        # Get tokens and validation loss
        if 'tokens' in df.columns and 'val/loss' in df.columns:
            tokens = df['tokens'].values
            val_loss = df['val/loss'].values
            
            print(f"Run {run_id}: Raw data - tokens: {len(tokens)}, val_loss: {len(val_loss)}")
            
            # Remove NaN values
            valid_mask = ~(np.isnan(tokens) | np.isnan(val_loss))
            tokens = tokens[valid_mask]
            val_loss = val_loss[valid_mask]
            
            print(f"Run {run_id}: After removing NaN: {len(tokens)} points")
            
            # Remove invalid values (negative or zero)
            valid_mask = (tokens > 0) & (val_loss > 0)
            tokens = tokens[valid_mask]
            val_loss = val_loss[valid_mask]
            
            print(f"Run {run_id}: After removing invalid values: {len(tokens)} points")
            if len(tokens) > 0:
                print(f"Run {run_id}: Token range: {tokens.min():.2e} to {tokens.max():.2e}")
                print(f"Run {run_id}: Loss range: {val_loss.min():.4f} to {val_loss.max():.4f}")
                print(f"Run {run_id}: First 5 points: tokens={tokens[:5]}, loss={val_loss[:5]}")
            
            return {
                'tokens': tokens,
                'val_loss': val_loss,
                'opt': opt,
                'lr': lr,
                'renorm_wd': renorm_wd,
                'run_id': run_id
            }
        else:
            print(f"Run {run_id}: Missing required columns")
            return None
            
    except Exception as e:
        print(f"Error with run {run_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_training_curves(run_ids):
    """Plot training curves for given run IDs."""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))
    
    for i, run_id in enumerate(run_ids):
        print(f"\nProcessing run {run_id}...")
        data = get_run_data(run_id)
        
        if data is None:
            continue
            
        # Create legend label
        legend_label = f"{data['opt']} | lr={data['lr']:.1e} | W={data['renorm_wd']:.1e}"
        
        # Plot with all available points
        print(f"About to plot {len(data['tokens'])} points for {data['opt']}")
        print(f"Token values: {data['tokens'][:10]}... (showing first 10)")
        print(f"Loss values: {data['val_loss'][:10]}... (showing first 10)")
        
        # Plot clean line without markers
        plt.loglog(data['tokens'], data['val_loss'], 
                  color=colors[i], linewidth=2, 
                  label=legend_label)
        
        print(f"Successfully plotted {len(data['tokens'])} points for {data['opt']}")
    
    # Format plot
    plt.xlabel('Tokens', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Training Curves - All Data Points', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save
    plt.savefig("visualization/training_curves.pdf", bbox_inches='tight', dpi=300)
    print("\nPlot saved to visualization/training_curves.pdf")
    plt.show()

if __name__ == "__main__":
    if RUN_IDS:
        print("Starting training curves plot...")
        plot_training_curves(RUN_IDS)
    else:
        print("Add run IDs to the RUN_IDS list")