#!/usr/bin/env python3
"""
Scaling Laws Analysis for Optimizer Comparisons
Plots best learning rate and loss vs compute with power law fits.
Takes optimizer name and wandb group as arguments.
"""

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
import sys
import argparse

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

WANDB_PROJECT = "danastar"
WANDB_ENTITY = "ep-rmt-ml-opt"

# Plot styling
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
})

# ============================================================================
# PARAMETER COMPUTATION
# ============================================================================

def compute_non_embedding_params(n_layer, n_head, n_embd, head_dim, mlp_hidden):
    """Compute non-embedding parameters from architecture config."""
    # Transformer block parameters:
    # - Attention: 3 * (head_dim * n_embd * n_head) for Q, K, V projections
    # - Attention output: n_embd * n_embd
    # - MLP: 2 * (n_embd * mlp_hidden) for up and down projections
    # - LayerNorms: 8 * n_embd (4 per block: 2 LN weights + 2 LN biases, assuming 2 LNs per block)
    # - Final: 2 * n_embd for final LN
    
    non_emb = n_layer * (3 * head_dim * n_embd * n_head + n_embd * n_embd + 
                         2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd
    return int(non_emb)

def compute_total_params(n_layer, n_head, n_embd, head_dim, mlp_hidden, vocab_size):
    """Compute total parameters (including embeddings) from architecture config."""
    non_emb = compute_non_embedding_params(n_layer, n_head, n_embd, head_dim, mlp_hidden)
    emb_params = 2 * n_embd * vocab_size  # token embeddings + position embeddings
    return int(non_emb + emb_params)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_runs_for_group(project, entity, group):
    """Load all runs from a specific WandB group."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    
    data = []
    for run in runs:
        config = run.config
        summary = run.summary
        
        # Extract architecture parameters from config
        n_layer = config.get('n_layer')
        n_head = config.get('n_head')
        n_embd = config.get('n_embd')
        head_dim = config.get('qkv_dim')
        
        # Get MLP hidden size - try multiple possible names
        mlp_hidden = config.get('mlp_hidden_dim')
        
        vocab_size = config.get('vocab_size', 50304)  # Default GPT-2 vocab size
        
        # Training parameters
        lr = config.get('lr')
        val_loss = summary.get('final-val/loss')
        iterations = config.get('iterations')
        
        batch_size = 32
        seq_length = 2048
        
        # Check if we have all required parameters
        required_params = [n_layer, n_head, n_embd, head_dim, mlp_hidden, lr, val_loss, iterations]
        if all(x is not None for x in required_params):
            # Compute parameters
            non_emb_params = compute_non_embedding_params(n_layer, n_head, n_embd, head_dim, mlp_hidden)
            total_params = compute_total_params(n_layer, n_head, n_embd, head_dim, mlp_hidden, vocab_size)
            
            # Compute FLOPs
            compute_non_emb = iterations * non_emb_params * 6 * batch_size * seq_length
            compute_total = iterations * total_params * 6 * batch_size * seq_length
            
            data.append({
                'depth': n_layer,  # Use n_layer as "depth" for plotting
                'lr': lr,
                'val_loss': val_loss,
                'compute_non_emb': compute_non_emb,
                'compute_total': compute_total,
                'n_layer': n_layer,
                'n_head': n_head,
                'n_embd': n_embd,
                'iterations': iterations,
                'non_emb_params': non_emb_params,
                'total_params': total_params,
                'batch_size': batch_size,
                'seq_length': seq_length,
            })
    
    return pd.DataFrame(data)

# ============================================================================
# FITTING FUNCTIONS
# ============================================================================

def power_law_with_offset(x, a, b, c):
    """Power law with offset: y = a + b * x^c"""
    return a + b * np.power(x, c)

def power_law_no_offset(x, b, c):
    """Power law without offset: y = b * x^c"""
    return b * np.power(x, c)

def fit_power_law_logspace(x_data, y_data):
    """
    Fit power law y = b * x^c in log-space using linear regression.
    This is the most stable method: log(y) = log(b) + c*log(x)
    """
    mask = (x_data > 0) & (y_data > 0)
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    
    if len(x_clean) < 2:
        return None
    
    log_x = np.log(x_clean)
    log_y = np.log(y_clean)
    
    # Linear fit: log(y) = c*log(x) + log(b)
    coeffs = np.polyfit(log_x, log_y, 1)
    c_fit = coeffs[0]  # slope = exponent
    b_fit = np.exp(coeffs[1])  # intercept = log(b)
    
    # Compute R² in log-space
    y_pred_log = coeffs[0] * log_x + coeffs[1]
    ss_res = np.sum((log_y - y_pred_log)**2)
    ss_tot = np.sum((log_y - np.mean(log_y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {'b': b_fit, 'c': c_fit, 'r2': r_squared}

def fit_power_law_with_offset_nonlinear(x_data, y_data):
    """
    Fit power law with offset y = a + b * x^c using nonlinear least squares.
    Normalizes data for numerical stability.
    """
    mask = (x_data > 0) & (y_data > 0)
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    
    if len(x_clean) < 4:
        return None
    
    try:
        # Normalize x to avoid numerical issues with very large compute values
        x_scale = np.median(x_clean)
        x_norm = x_clean / x_scale
        
        # Get initial guess from log-space fit (no offset)
        log_fit = fit_power_law_logspace(x_clean, y_clean)
        if log_fit is None:
            return None
        
        # Use log-space fit to initialize parameters
        # For normalized x: y = a + b_norm * x_norm^c
        # Where b_norm = b_original * x_scale^c
        c_init = log_fit['c']
        b_original_init = log_fit['b']
        b_norm_init = b_original_init * (x_scale ** c_init)
        
        # Offset should be smaller than minimum y
        a_init = max(0, y_clean.min() * 0.5)
        
        # Adjusted b_norm to account for offset
        b_norm_init = y_clean.max() - a_init
        
        # Define function with normalized x
        def power_law_normalized(x_n, a, b_n, c):
            return a + b_n * np.power(x_n, c)
        
        # Fit with bounds to ensure reasonable parameters
        bounds = (
            [0, 0, -5],  # lower bounds: a>=0, b_n>=0, c>=-5
            [y_clean.max(), np.inf, 5]  # upper bounds: a<=max(y), b_n unbounded, c<=5
        )
        
        popt, _ = curve_fit(
            power_law_normalized,
            x_norm,
            y_clean,
            p0=[a_init, b_norm_init, c_init],
            bounds=bounds,
            maxfev=50000
        )
        
        # Transform parameters back to original scale
        a_fit = popt[0]
        b_norm_fit = popt[1]
        c_fit = popt[2]
        b_fit = b_norm_fit / (x_scale ** c_fit)
        
        # Compute R² on original scale
        y_pred = power_law_with_offset(x_clean, a_fit, b_fit, c_fit)
        ss_res = np.sum((y_clean - y_pred)**2)
        ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {'a': a_fit, 'b': b_fit, 'c': c_fit, 'r2': r_squared}
    except Exception as e:
        print(f"    Fit with offset failed: {e}")
        return None

def fit_power_law_with_offset_minimize(x_data, y_data):
    """
    Fit y = a + b * x^c using scipy.optimize.minimize with L-BFGS-B.
    All three parameters (a, b, c) are fitted by minimizing sum of squared residuals.
    Returns dict with a, b, c, r2 or None if fit fails.
    """
    from scipy.optimize import minimize
    
    mask = (x_data > 0) & (y_data > 0)
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    
    if len(x_clean) < 4:
        return None
    
    try:
        # Normalize data for better conditioning
        x_scale = np.median(x_clean)
        y_scale = np.median(y_clean)
        x_norm = x_clean / x_scale
        y_norm = y_clean / y_scale
        
        # Get initial guess from no-offset fit
        log_fit = fit_power_law_logspace(x_clean, y_clean)
        if log_fit is None:
            return None
        
        # Initial parameters (normalized)
        c_init = log_fit['c']
        b_init = (log_fit['b'] / y_scale) * (x_scale ** c_init)
        a_init = 0.0
        
        # Objective function: sum of squared residuals
        def objective(params):
            a, b, c = params
            y_pred = a + b * np.power(x_norm, c)
            residuals = y_norm - y_pred
            return np.sum(residuals**2)
        
        # Bounds for normalized parameters
        bounds = [
            (0, 0.95 * y_norm.min()),  # a: 0 to below min(y)
            (0, None),                  # b: positive
            (-5, 5)                     # c: reasonable exponent range
        ]
        
        # Minimize using L-BFGS-B
        result = minimize(
            objective,
            x0=[a_init, b_init, c_init],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 10000}
        )
        
        if not result.success:
            print(f"    Minimization did not converge: {result.message}")
            return None
        
        # Extract and denormalize parameters
        a_norm, b_norm, c_fit = result.x
        a_fit = a_norm * y_scale
        b_fit = b_norm * y_scale / (x_scale ** c_fit)
        
        # Compute R² on original scale
        y_pred = power_law_with_offset(x_clean, a_fit, b_fit, c_fit)
        ss_res = np.sum((y_clean - y_pred)**2)
        ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {'a': float(a_fit), 'b': float(b_fit), 'c': float(c_fit), 'r2': float(r_squared)}
    
    except Exception as e:
        print(f"    Fit with offset failed: {e}")
        return None

# ============================================================================
# PLOTTING
# ============================================================================

def plot_scaling_law(compute_data, y_data, depths, ylabel, title, filename, is_loss=False, top_k_data=None, opt_name='ademamix', group_name='', n_heads=None):
    """Generic plotting function for scaling law analysis.
    
    Args:
        top_k_data: Optional dict mapping depth -> list of (compute, y_value, rank) tuples for top-k results
        opt_name: Optimizer name for legend
        group_name: WandB group name for title
        n_heads: Optional list of n_head values corresponding to each depth
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot data points
    colors = plt.cm.tab10(np.linspace(0, 1, len(depths)))
    for idx, (c, y, d) in enumerate(zip(compute_data, y_data, depths)):
        # Create label with depth and n_head if available
        if n_heads is not None and idx < len(n_heads):
            label = f'Depth {d}, {n_heads[idx]} heads'
        else:
            label = f'Depth {d}'
        
        # Plot best point (largest)
        ax.scatter(c, y, s=100, alpha=0.8, color=colors[idx], 
                  edgecolors='black', linewidths=1.5, zorder=3, label=label)
        
        # Plot top-k points if provided (with decreasing sizes)
        if top_k_data is not None and d in top_k_data:
            for compute_val, y_val, rank in top_k_data[d]:
                # Skip the best one (already plotted)
                if rank == 1:
                    continue
                # Size decreases linearly from 90 to 10 for ranks 2-10
                size = max(10, 100 - (rank - 1) * 10)
                alpha = max(0.3, 0.8 - (rank - 1) * 0.05)
                ax.scatter(compute_val, y_val, s=size, alpha=alpha, color=colors[idx],
                          edgecolors='gray', linewidths=0.8, zorder=2)
    
    # Fit power laws
    compute_array = np.array(compute_data, dtype=np.float64)
    y_array = np.array(y_data, dtype=np.float64)
    
    x_fit = np.logspace(np.log10(min(compute_data) * 0.8), 
                       np.log10(max(compute_data) * 1.2), 100)
    
    # Fit 1: Without offset (log-space fit - most stable)
    print(f"\n  Fit 1: NO offset (b*C^c) - log-space linear regression")
    result = fit_power_law_logspace(compute_array, y_array)
    if result:
        b, c, r2 = result['b'], result['c'], result['r2']
        print(f"    {ylabel} = {b:.6e} × C^{c:.4f}")
        print(f"    R² = {r2:.6f}")
        y_pred = power_law_no_offset(x_fit, b, c)
        ax.plot(x_fit, y_pred, 'b--', linewidth=2.5, alpha=0.8, 
               label=f'No offset: {b:.2e}×$C^{{{c:.3f}}}$ (R²={r2:.3f})', zorder=2)
    
    # Fit 2: With offset (scipy minimize with L-BFGS-B) - only for loss plots
    if is_loss:
        print(f"\n  Fit 2: WITH offset (a + b*C^c) - scipy minimize (L-BFGS-B)")
        result = fit_power_law_with_offset_minimize(compute_array, y_array)
        if result:
            a, b, c, r2 = result['a'], result['b'], result['c'], result['r2']
            print(f"    {ylabel} = {a:.6e} + {b:.6e} × C^{c:.4f}")
            print(f"    R² = {r2:.6f}")
            y_pred = power_law_with_offset(x_fit, a, b, c)
            # Ensure positive predictions for log scale
            y_pred = np.maximum(y_pred, np.finfo(float).tiny)
            ax.plot(
                x_fit,
                y_pred,
                'r-',
                linewidth=2.5,
                alpha=0.8,
                label=f'With offset: {a:.2e}+{b:.2e}×$C^{{{c:.3f}}}$ (R²={r2:.3f})',
                zorder=2,
            )
    
    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute (FLOPs)', fontweight='bold', fontsize=13)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"\n✓ Saved: {filename}")

def create_plots(df, depths, opt_name, title_suffix, group_name):
    """Create all scaling law plots."""
    
    # Extract data for each depth
    compute_non_emb = []
    compute_total = []
    best_lrs = []
    best_losses = []
    depths_with_data = []  # Track which depths actually have data
    n_heads_list = []  # Track n_head for each depth
    
    # For top-10 LR visualization
    top_10_lr_non_emb = {}
    top_10_lr_total = {}
    top_10_loss_non_emb = {}
    top_10_loss_total = {}
    
    for depth in depths:
        depth_data = df[df['depth'] == depth]
        if len(depth_data) == 0:
            continue
        
        # Use modal compute value
        c_non_emb = depth_data['compute_non_emb'].mode()[0]
        c_total = depth_data['compute_total'].mode()[0]
        
        # Find best run (minimum val_loss)
        best_run = depth_data.loc[depth_data['val_loss'].idxmin()]
        
        compute_non_emb.append(c_non_emb)
        compute_total.append(c_total)
        best_lrs.append(best_run['lr'])
        best_losses.append(best_run['val_loss'])
        depths_with_data.append(depth)  # Track this depth has data
        n_heads_list.append(int(best_run['n_head']))  # Track n_head for this depth
        
        # Get top-10 runs by loss for this depth
        top_10_runs = depth_data.nsmallest(10, 'val_loss')
        
        # Store top-10 for LR plots (compute, lr, rank)
        top_10_lr_non_emb[depth] = [(c_non_emb, row['lr'], rank+1) 
                                     for rank, (_, row) in enumerate(top_10_runs.iterrows())]
        top_10_lr_total[depth] = [(c_total, row['lr'], rank+1)
                                  for rank, (_, row) in enumerate(top_10_runs.iterrows())]
        
        # Store top-10 for loss plots (compute, loss, rank)
        top_10_loss_non_emb[depth] = [(c_non_emb, row['val_loss'], rank+1)
                                       for rank, (_, row) in enumerate(top_10_runs.iterrows())]
        top_10_loss_total[depth] = [(c_total, row['val_loss'], rank+1)
                                     for rank, (_, row) in enumerate(top_10_runs.iterrows())]
    
    # Create output directory if it doesn't exist
    import os
    output_dir = 'visualization/BigHead_dfe'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Loss vs Compute (non-embedding)
    print("\n" + "="*70)
    print("LOSS vs COMPUTE (Non-embedding)")
    print("="*70)
    plot_scaling_law(
        compute_non_emb, best_losses, depths_with_data,
        ylabel='Best Validation Loss',
        title=f'Best Validation Loss vs Compute (Non-embedding)\n{opt_name} | {group_name}',
        filename=f'{output_dir}/{opt_name}_{group_name}_loss_vs_compute_nonemb.pdf',
        is_loss=True,
        top_k_data=top_10_loss_non_emb,
        opt_name=opt_name,
        group_name=group_name,
        n_heads=n_heads_list
    )
    
    # Plot 2: Loss vs Compute (total)
    print("\n" + "="*70)
    print("LOSS vs COMPUTE (Total)")
    print("="*70)
    plot_scaling_law(
        compute_total, best_losses, depths_with_data,
        ylabel='Best Validation Loss',
        title=f'Best Validation Loss vs Compute (Total)\n{opt_name} | {group_name}',
        filename=f'{output_dir}/{opt_name}_{group_name}_loss_vs_compute_total.pdf',
        is_loss=True,
        top_k_data=top_10_loss_total,
        opt_name=opt_name,
        group_name=group_name,
        n_heads=n_heads_list
    )
    
    # Plot 3: Learning Rate vs Compute (non-embedding)
    print("\n" + "="*70)
    print("LEARNING RATE vs COMPUTE (Non-embedding)")
    print("="*70)
    plot_scaling_law(
        compute_non_emb, best_lrs, depths_with_data,
        ylabel='Best Learning Rate',
        title=f'Best Learning Rate vs Compute (Non-embedding)\n{opt_name} | {group_name}',
        filename=f'{output_dir}/{opt_name}_{group_name}_best_lr_vs_compute_nonemb.pdf',
        is_loss=False,
        top_k_data=top_10_lr_non_emb,
        opt_name=opt_name,
        group_name=group_name,
        n_heads=n_heads_list
    )
    
    # Plot 4: Learning Rate vs Compute (total)
    print("\n" + "="*70)
    print("LEARNING RATE vs COMPUTE (Total)")
    print("="*70)
    plot_scaling_law(
        compute_total, best_lrs, depths_with_data,
        ylabel='Best Learning Rate',
        title=f'Best Learning Rate vs Compute (Total)\n{opt_name} | {group_name}',
        filename=f'{output_dir}/{opt_name}_{group_name}_best_lr_vs_compute_total.pdf',
        is_loss=False,
        top_k_data=top_10_lr_total,
        opt_name=opt_name,
        group_name=group_name,
        n_heads=n_heads_list
    )

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Scaling Laws Analysis - Plot best LR and loss vs compute'
    )
    parser.add_argument(
        '--opt',
        type=str,
        required=True,
        help='Optimizer name (e.g., ademamix, adamw, d-muon)'
    )
    parser.add_argument(
        '--group',
        type=str,
        required=True,
        help='WandB group name'
    )
    parser.add_argument(
        '--depths',
        type=int,
        nargs='+',
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        help='List of depths to analyze (default: 2-20)'
    )
    
    args = parser.parse_args()
    
    # Get configuration
    opt_name = args.opt.lower()
    wandb_group = args.group
    depths = args.depths
    
    print("="*70)
    print(f"Scaling Laws Analysis")
    print(f"Optimizer: {opt_name}")
    print(f"Group: {wandb_group}")
    print(f"Depths: {depths}")
    print("="*70)
    
    print("\nLoading runs from WandB...")
    df = load_runs_for_group(WANDB_PROJECT, WANDB_ENTITY, wandb_group)
    
    if df.empty:
        print("ERROR: No data found")
        return
    
    print(f"Loaded {len(df)} runs")
    print(f"Depths found: {sorted(df['depth'].unique())}")
    
    # Filter to requested depths
    df = df[df['depth'].isin(depths)]
    print(f"Runs after depth filter: {len(df)}")
    
    # Summary
    print("\nSummary per depth:")
    for depth in depths:
        depth_data = df[df['depth'] == depth]
        if len(depth_data) > 0:
            best_loss = depth_data['val_loss'].min()
            best_idx = depth_data['val_loss'].idxmin()
            best_lr = depth_data.loc[best_idx, 'lr']
            
            # Get architecture details from first run (should be same for all runs at this depth)
            first_run = depth_data.iloc[0]
            iterations = first_run['iterations']
            n_embd = first_run['n_embd']
            n_head = first_run['n_head']
            non_emb_params = first_run['non_emb_params']
            compute_non_emb = first_run['compute_non_emb']
            batch_size = first_run['batch_size']
            seq_length = first_run['seq_length']
            
            print(f"  Depth {depth}: {len(depth_data)} runs")
            print(f"    Best: loss={best_loss:.4f}, lr={best_lr:.4e}")
            print(f"    Config: n_embd={n_embd}, n_head={n_head}")
            print(f"    Iterations: {iterations:,}")
            print(f"    Batch: batch_size={batch_size:.0f}, seq_length={seq_length:.0f}")
            print(f"    Non-emb params: {non_emb_params:,.0f} ({non_emb_params:.2e})")
            print(f"    Compute (non-emb): {compute_non_emb:.2e} FLOPs")
    
    # Create plots
    print("\nCreating visualizations...")
    create_plots(df, depths, opt_name, wandb_group, wandb_group)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()

