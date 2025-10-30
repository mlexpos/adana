#!/usr/bin/env python3
"""
Scaling Rules Analysis for AdamW Universal Scaling Search
Analyzes runs with different scaling rules (constant lr with varying omega, or constant omega with varying lr)
Plots loss vs compute with power law fits (no offset) for each scaling rule.
"""

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

WANDB_PROJECT = "danastar"
WANDB_ENTITY = "ep-rmt-ml-opt"
WANDB_GROUP = "enoki_adamw_universal_scaling_search"

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
    non_emb = n_layer * (3 * head_dim * n_embd * n_head + n_embd * n_embd + 
                         2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd
    return int(non_emb)

def compute_total_params(n_layer, n_head, n_embd, head_dim, mlp_hidden, vocab_size):
    """Compute total parameters (including embeddings) from architecture config."""
    non_emb = compute_non_embedding_params(n_layer, n_head, n_embd, head_dim, mlp_hidden)
    emb_params = 2 * n_embd * vocab_size
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
        
        # Extract architecture parameters
        n_layer = config.get('n_layer')
        n_head = config.get('n_head')
        n_embd = config.get('n_embd')
        head_dim = config.get('qkv_dim')
        mlp_hidden = config.get('mlp_hidden_dim')
        vocab_size = config.get('vocab_size', 50304)
        
        # Training parameters
        lr = config.get('lr')
        weight_decay = config.get('weight_decay')
        renorm_weight_decay = config.get('renorm_weight_decay')  # Omega scaling parameter
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
                'run_id': run.id,
                'run_name': run.name,
                'lr': lr,
                'weight_decay': weight_decay if weight_decay is not None else 0.0,
                'renorm_weight_decay': renorm_weight_decay if renorm_weight_decay is not None else None,
                'val_loss': val_loss,
                'compute_non_emb': compute_non_emb,
                'compute_total': compute_total,
                'n_layer': n_layer,
                'n_head': n_head,
                'n_embd': n_embd,
                'iterations': iterations,
                'non_emb_params': non_emb_params,
                'total_params': total_params,
            })
    
    return pd.DataFrame(data)

# ============================================================================
# SCALING RULE CATEGORIZATION
# ============================================================================

def categorize_scaling_rules(df):
    """
    Categorize runs into different scaling rules:
    - renorm_weight_decay=4, lr=0.001
    - renorm_weight_decay=4, lr=0.00001
    - lr=0.001, varying weight_decay
    """
    scaling_rules = {}
    
    # Show what we have in the data
    print("\nRun distribution by (lr, renorm_weight_decay, weight_decay):")
    grouped = df.groupby(['lr', 'renorm_weight_decay', 'weight_decay']).size()
    print(grouped)
    
    # Rule 1: renorm_weight_decay=4, lr=0.001
    mask1 = (df['renorm_weight_decay'] == 4) & (df['lr'] == 0.001)
    if mask1.any():
        scaling_rules['ω=4, lr=0.001'] = df[mask1].copy()
        print(f"\nRule 1 (ω=4, lr=0.001): {mask1.sum()} runs")
    
    # Rule 2: renorm_weight_decay=4, lr=0.00001
    mask2 = (df['renorm_weight_decay'] == 4) & (df['lr'] == 0.00001)
    if mask2.any():
        scaling_rules['ω=4, lr=0.00001'] = df[mask2].copy()
        print(f"Rule 2 (ω=4, lr=0.00001): {mask2.sum()} runs")
    
    # Rule 3: lr=0.001, varying weight_decay (renorm_weight_decay may vary or be None)
    mask3 = (df['lr'] == 0.001) & ~mask1  # Exclude rule 1 to avoid duplicates
    if mask3.any():
        scaling_rules['lr=0.001, varying wd'] = df[mask3].copy()
        print(f"Rule 3 (lr=0.001, varying wd): {mask3.sum()} runs")
    
    return scaling_rules

# ============================================================================
# FITTING FUNCTIONS
# ============================================================================

def power_law_no_offset(x, b, c):
    """Power law without offset: y = b * x^c"""
    return b * np.power(x, c)

def fit_power_law_logspace(x_data, y_data):
    """
    Fit power law y = b * x^c in log-space using linear regression.
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

# ============================================================================
# PLOTTING
# ============================================================================

def plot_scaling_rules(scaling_rules, use_total_compute=False):
    """
    Plot loss vs compute for each scaling rule with power law fits.
    Each rule gets its own color and fit line.
    """
    compute_type = 'compute_total' if use_total_compute else 'compute_non_emb'
    compute_label = 'Total' if use_total_compute else 'Non-embedding'
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Define distinct colors for each rule
    color_map = {
        'ω=4, lr=0.001': '#1f77b4',        # blue
        'ω=4, lr=0.00001': '#ff7f0e',      # orange
        'lr=0.001, varying wd': '#2ca02c'  # green
    }
    
    # Track overall min/max for fit line range
    all_compute = []
    
    for rule_name, rule_data in scaling_rules.items():
        if len(rule_data) == 0:
            continue
        
        color = color_map.get(rule_name, '#333333')
        
        # Extract compute and loss values
        compute_vals = rule_data[compute_type].values
        loss_vals = rule_data['val_loss'].values
        
        all_compute.extend(compute_vals)
        
        # Plot data points
        ax.scatter(compute_vals, loss_vals, s=120, alpha=0.7, color=color,
                  edgecolors='black', linewidths=1.5, zorder=3, label=f'{rule_name}')
        
        # Fit power law (no offset)
        result = fit_power_law_logspace(compute_vals, loss_vals)
        if result:
            b, c, r2 = result['b'], result['c'], result['r2']
            
            # Generate fit line
            x_min, x_max = compute_vals.min(), compute_vals.max()
            x_fit = np.logspace(np.log10(x_min * 0.8), np.log10(x_max * 1.2), 100)
            y_fit = power_law_no_offset(x_fit, b, c)
            
            # Plot fit with same color
            ax.plot(x_fit, y_fit, '-', linewidth=3, alpha=0.9, color=color,
                   label=f'{rule_name} fit: {b:.2e}×C^{{{c:.3f}}} (R²={r2:.3f})', zorder=2)
            
            print(f"\n{rule_name}:")
            print(f"  Loss = {b:.6e} × C^{c:.4f}")
            print(f"  R² = {r2:.6f}")
            print(f"  N_runs = {len(rule_data)}")
    
    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(f'Compute ({compute_label}) [FLOPs]', fontweight='bold', fontsize=14)
    ax.set_ylabel('Validation Loss', fontweight='bold', fontsize=15)
    ax.set_title(f'Loss vs Compute - Different Scaling Rules\nAdamW Universal Scaling Search',
                fontweight='bold', fontsize=17)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    ax.legend(loc='best', framealpha=0.95, fontsize=11, ncol=1)
    plt.tight_layout()
    
    # Save figure
    output_dir = 'visualization/BigHead_dfe'
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/adamw_scaling_rules_{compute_type}.pdf'
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"\n✓ Saved: {filename}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print(f"Scaling Rules Analysis - AdamW")
    print(f"Group: {WANDB_GROUP}")
    print("="*70)
    
    print("\nLoading runs from WandB...")
    df = load_runs_for_group(WANDB_PROJECT, WANDB_ENTITY, WANDB_GROUP)
    
    if df.empty:
        print("ERROR: No data found")
        return
    
    print(f"Loaded {len(df)} runs")
    
    # Show summary of parameters
    print("\nParameter ranges:")
    print(f"  Learning rates: {sorted(df['lr'].unique())}")
    print(f"  Weight decays: {sorted(df['weight_decay'].unique())}")
    print(f"  Renorm weight decay (ω) values: {sorted(df['renorm_weight_decay'].dropna().unique())}")
    
    # Show all unique combinations
    print("\nAll unique parameter combinations:")
    print(df[['lr', 'weight_decay', 'renorm_weight_decay', 'n_layer']].drop_duplicates().sort_values(['lr', 'renorm_weight_decay', 'weight_decay']))
    
    # Categorize into scaling rules
    print("\n" + "="*70)
    print("CATEGORIZING SCALING RULES")
    print("="*70)
    scaling_rules = categorize_scaling_rules(df)
    
    print(f"\nFound {len(scaling_rules)} scaling rules:")
    for rule_name, rule_data in scaling_rules.items():
        print(f"  {rule_name}: {len(rule_data)} runs")
    
    # Plot for non-embedding compute
    print("\n" + "="*70)
    print("PLOTTING: Non-embedding Compute")
    print("="*70)
    plot_scaling_rules(scaling_rules, use_total_compute=False)
    
    # Plot for total compute
    print("\n" + "="*70)
    print("PLOTTING: Total Compute")
    print("="*70)
    plot_scaling_rules(scaling_rules, use_total_compute=True)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()

