#!/usr/bin/env python3
"""
WandB Hyperparameter Analysis and Visualization Script

This script connects to Weights & Biases, filters runs from specified groups,
and creates visualizations showing hyperparameters with the best performance metrics.
All configuration is done within the script - no command line arguments needed.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS AS NEEDED
# ============================================================================

# WandB project configuration
WANDB_PROJECT = "danastar"  # Change to your project name
WANDB_ENTITY = "ep-rmt-ml-opt"  # Change to your entity/team name

# Group and filtering configuration
TARGET_GROUPS = [

    #"AdamW_small_lr_weight_decay_sweeps",
    #"AdamW_35M_lr_weight_decay_sweeps",
    #"AdamW_90M_lr_weight_decay_sweeps",
    #"AdamW_180M_lr_weight_decay_sweeps",
    #"AdamW_330M_lr_weight_decay_sweeps",
    #"Ademamix_dana_small_lr_wd_sweep",
    #"Ademamix_dana_35M_lr_wd_gamma3factor_sweep"
    #"Ademamix_dana_90M_lr_wd_sweep_new"
    #"Ademamix_dana_180M_lr_wd_sweep_new"
<<<<<<< Updated upstream
    #"Ademamix_dana_180M_gamma3factor_0_5_lr_weight_decay_sweep",
    #"Ademamix_dana_180M_lr_weight_decay_gamma3factor_0_25_sweeps",
    "Ademamix_dana_330M_lr_weight_decay_gamma3factor_sweeps"
=======
    "Ademamix_small_lr_wd_delta_gamma3factor_sweeps",
>>>>>>> Stashed changes
    # Add more groups as needed
]

# Additional filters (optional)
ADDITIONAL_FILTERS = {
    "config.dataset": "fineweb_100",
<<<<<<< Updated upstream
    "config.opt": "dana",
    "config.gamma_3_factor": .5,
=======
    "config.opt": "ademamix",
    #"config.gamma_3_factor": 0.5,
>>>>>>> Stashed changes
    # "config.dataset": "fineweb",  # Uncomment to filter by dataset
    # "state": "finished",  # Only finished runs
    # Add more filters as needed
}

# Exclusion filters - runs matching these will be EXCLUDED
EXCLUSION_FILTERS = {
    "tag": "old_dataset",    
    # "config.opt": "adamw",  # Exclude adamw optimizer
    # "config.dataset": "fineweb_100",  # Exclude fineweb_100 dataset
    # "state": "failed",  # Exclude failed runs
    # Add more exclusion filters as needed
}

# Metric configuration
PRIMARY_METRIC = "final-val/loss"  # The metric to optimize (lower is better)
MINIMIZE_METRIC = True  # Set to False if higher is better

# Hyperparameters to analyze and visualize
HYPERPARAMS_TO_ANALYZE = [
    "lr",           # Learning rate
    "weight_decay", # Weight decay
    "batch_size",   # Batch size
    "opt",          # Optimizer
    "dataset",      # Dataset
    #"gamma_3_factor",      # Gamma 3 factor
    "beta1",      # Beta 1
    "beta2",      # Beta 2
    "delta",      # Delta
    "gamma_3_factor",      # Gamma 3 factor
    # "User",         # User
    # Add more hyperparameters as needed
]

# Additional computed parameters (these are calculated, not extracted from config)
COMPUTED_PARAMS = [
    "renorm_weight_decay",  # W = weight_decay * lr * iterations
]

# Number of best runs to highlight
TOP_N_RUNS = 3

# Visualization settings
FIGURE_SIZE = (15, 10)
DPI = 300
SAVE_FORMAT = "pdf"  # or "png"

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def connect_to_wandb():
    """Initialize wandb API connection."""
    try:
        api = wandb.Api()
        print("✓ Successfully connected to WandB API")
        return api
    except Exception as e:
        print(f"✗ Failed to connect to WandB API: {e}")
        return None

def fetch_runs_data(api, project: str, entity: str, groups: List[str], 
                   additional_filters: Dict = None, exclusion_filters: Dict = None) -> pd.DataFrame:
    """
    Fetch runs data from WandB for specified groups and filters.
    
    Args:
        api: WandB API object
        project: WandB project name
        entity: WandB entity/team name
        groups: List of group names to fetch
        additional_filters: Additional filters to apply
        exclusion_filters: Filters to exclude runs (NOT filters)
    
    Returns:
        DataFrame with runs data
    """
    all_data = []
    
    for group in groups:
        print(f"\n📥 Fetching data for group: {group}")
        
        # Build filters
        filters = {"group": group}
        if additional_filters:
            filters.update(additional_filters)
        
        try:
            # Get runs from the group
            runs = api.runs(f"{entity}/{project}", filters=filters)
            
            group_data = []
            for run in runs:
                try:
                    # Get run config and summary
                    config = run.config
                    summary = run.summary
                    
                    # Apply exclusion filters
                    if exclusion_filters:
                        skip_run = False
                        for filter_key, filter_value in exclusion_filters.items():
                            if filter_key.startswith('config.'):
                                config_key = filter_key.replace('config.', '')
                                if config.get(config_key) == filter_value:
                                    skip_run = True
                                    break
                            elif filter_key == 'state':
                                if run.state == filter_value:
                                    skip_run = True
                                    break
                            elif hasattr(run, filter_key):
                                if getattr(run, filter_key) == filter_value:
                                    skip_run = True
                                    break
                        
                        if skip_run:
                            continue
                    
                    # Extract basic info
                    run_data = {
                        'run_id': run.id,
                        'run_name': run.name,
                        'group': group,
                        'state': run.state,
                        'created_at': run.created_at,
                    }
                    
                    # Extract hyperparameters
                    for param in HYPERPARAMS_TO_ANALYZE:
                        run_data[f'config_{param}'] = config.get(param)
                    
                    # Compute renormalized weight decay: W = weight_decay * lr * iterations
                    weight_decay = config.get('weight_decay')
                    lr = config.get('lr')
                    iterations = config.get('iterations')
                    
                    if all(x is not None for x in [weight_decay, lr, iterations]):
                        run_data['config_renorm_weight_decay'] = weight_decay * lr * iterations
                    else:
                        run_data['config_renorm_weight_decay'] = None
                    
                    # Extract the primary metric
                    run_data[PRIMARY_METRIC] = summary.get(PRIMARY_METRIC)
                    
                    # Extract additional metrics that might be useful
                    for key, value in summary.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            run_data[f'metric_{key}'] = value
                    
                    group_data.append(run_data)
                    
                except Exception as e:
                    print(f"  ⚠️  Error processing run {run.name}: {e}")
                    continue
            
            print(f"  ✓ Fetched {len(group_data)} runs from group {group}")
            all_data.extend(group_data)
            
        except Exception as e:
            print(f"  ✗ Error fetching group {group}: {e}")
            continue
    
    df = pd.DataFrame(all_data)
    print(f"\n📊 Total runs fetched: {len(df)}")
    
    return df

def clean_and_filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and filter the data, removing runs with missing critical information.
    
    Args:
        df: Raw DataFrame from WandB
    
    Returns:
        Cleaned DataFrame
    """
    print("\n🧹 Cleaning and filtering data...")
    
    initial_count = len(df)
    
    # Remove runs without the primary metric
    df = df.dropna(subset=[PRIMARY_METRIC])
    print(f"  ✓ Removed {initial_count - len(df)} runs missing {PRIMARY_METRIC}")
    
    # Convert primary metric to numeric, coercing errors to NaN
    df[PRIMARY_METRIC] = pd.to_numeric(df[PRIMARY_METRIC], errors='coerce')
    
    # Remove any runs that couldn't be converted to numeric
    initial_count_after_dropna = len(df)
    df = df.dropna(subset=[PRIMARY_METRIC])
    print(f"  ✓ Removed {initial_count_after_dropna - len(df)} runs with non-numeric {PRIMARY_METRIC}")
    
    # Remove runs without key hyperparameters
    key_params = [f'config_{param}' for param in HYPERPARAMS_TO_ANALYZE]
    df = df.dropna(subset=key_params, how='any')
    print(f"  ✓ Removed runs missing key hyperparameters, {len(df)} runs remaining")
    
    # Remove outliers (optional - you can modify this logic)
    if PRIMARY_METRIC == "val/loss":
        # Remove validation losses that are too high (likely failed runs)
        q99 = df[PRIMARY_METRIC].quantile(0.99)
        outlier_threshold = min(q99, 10.0)  # Cap at 10.0 for loss
        initial_len = len(df)
        df = df[df[PRIMARY_METRIC] < outlier_threshold]
        print(f"  ✓ Removed {initial_len - len(df)} outliers (metric > {outlier_threshold:.2f})")
    
    print(f"  ✓ Final dataset: {len(df)} runs")
    return df

def find_best_hyperparams(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Find the best hyperparameter combinations based on the primary metric.
    
    Args:
        df: Cleaned DataFrame
        top_n: Number of top runs to return
    
    Returns:
        DataFrame with top performing runs
    """
    print(f"\n🏆 Finding top {top_n} hyperparameter combinations...")
    
    # Sort by primary metric
    ascending = MINIMIZE_METRIC  # True for loss (minimize), False for accuracy (maximize)
    best_runs = df.nsmallest(top_n, PRIMARY_METRIC) if ascending else df.nlargest(top_n, PRIMARY_METRIC)
    
    print(f"  ✓ Best {PRIMARY_METRIC}: {best_runs[PRIMARY_METRIC].iloc[0]:.4f}")
    print(f"  ✓ Worst in top {top_n}: {best_runs[PRIMARY_METRIC].iloc[-1]:.4f}")
    
    return best_runs

def create_hyperparameter_visualization(df: pd.DataFrame, best_runs: pd.DataFrame):
    """
    Create comprehensive visualizations of hyperparameters vs performance.
    
    Args:
        df: Full dataset
        best_runs: Top performing runs
    """
    print("\n📈 Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Calculate number of subplots needed (include computed parameters)
    all_params = HYPERPARAMS_TO_ANALYZE + COMPUTED_PARAMS
    n_params = len(all_params)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIGURE_SIZE)
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Create individual hyperparameter plots
    for i, param in enumerate(all_params):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        param_col = f'config_{param}'
        
        if param_col in df.columns:
            # Scatter plot of all runs
            scatter = ax.scatter(df[param_col], df[PRIMARY_METRIC], 
                               alpha=0.6, s=30, c='lightblue', label='All runs')
            
            # Highlight best runs
            ax.scatter(best_runs[param_col], best_runs[PRIMARY_METRIC], 
                      alpha=0.8, s=60, c='red', label=f'Top {len(best_runs)}', zorder=5)
            
            ax.set_xlabel(param)
            ax.set_ylabel(PRIMARY_METRIC)
            ax.set_title(f'{param} vs {PRIMARY_METRIC}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Use log scale for learning rate if applicable
            if param == 'lr' and df[param_col].min() > 0:
                ax.set_xscale('log')
    
    # Remove empty subplots
    for i in range(n_params, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            fig.delaxes(axes[row, col])
        else:
            fig.delaxes(axes[col])
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"visualization/hyperparameter_analysis.{SAVE_FORMAT}"
    plt.savefig(filename, format=SAVE_FORMAT, dpi=DPI, bbox_inches='tight')
    print(f"  ✓ Saved hyperparameter analysis plot: {filename}")
    
    # Create a correlation heatmap
    create_correlation_heatmap(df)
    
    
    plt.show()

def create_correlation_heatmap(df: pd.DataFrame):
    """Create a correlation heatmap of hyperparameters and metrics."""
    print("  📊 Creating correlation heatmap...")
    
    # Select numeric columns for correlation
    numeric_cols = []
    all_params = HYPERPARAMS_TO_ANALYZE + COMPUTED_PARAMS
    for param in all_params:
        param_col = f'config_{param}'
        if param_col in df.columns and df[param_col].dtype in ['float64', 'int64']:
            numeric_cols.append(param_col)
    
    # Add primary metric
    numeric_cols.append(PRIMARY_METRIC)
    
    # Add other numeric metrics
    metric_cols = [col for col in df.columns if col.startswith('metric_') and df[col].dtype in ['float64', 'int64']]
    numeric_cols.extend(metric_cols[:5])  # Limit to first 5 additional metrics
    
    # Skip if no numeric columns
    if len(numeric_cols) < 2:
        print("  ⚠️  Not enough numeric columns for correlation heatmap")
        return
    
    # Create correlation matrix
    corr_data = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Hyperparameter and Metric Correlations')
    plt.tight_layout()
    
    filename = f"visualization/correlation_heatmap.{SAVE_FORMAT}"
    plt.savefig(filename, format=SAVE_FORMAT, dpi=DPI, bbox_inches='tight')
    print(f"  ✓ Saved correlation heatmap: {filename}")


def print_best_hyperparams_summary(best_runs: pd.DataFrame):
    """Print a summary of the best hyperparameter combinations."""
    print(f"\n🎯 TOP {len(best_runs)} HYPERPARAMETER COMBINATIONS:")
    print("=" * 80)
    
    for i, (idx, run) in enumerate(best_runs.iterrows(), 1):
        print(f"\nRank {i}: {run['run_name']} (Group: {run['group']})")
        print(f"  Run ID: {run['run_id']}")
        print(f"  {PRIMARY_METRIC}: {run[PRIMARY_METRIC]:.4f}")
        
        all_params = HYPERPARAMS_TO_ANALYZE + COMPUTED_PARAMS
        for param in all_params:
            param_col = f'config_{param}'
            if param_col in run and pd.notna(run[param_col]):
                value = run[param_col]
                if param in ['lr', 'renorm_weight_decay']:
                    print(f"  {param}: {value:.2e}")
                else:
                    print(f"  {param}: {value}")
    
    print("\n" + "=" * 80)

def main():
    """Main execution function."""
    print("🚀 Starting WandB Hyperparameter Analysis")
    print("=" * 50)
    
    # Connect to WandB
    api = connect_to_wandb()
    if not api:
        return
    
    # Fetch data
    df = fetch_runs_data(api, WANDB_PROJECT, WANDB_ENTITY, TARGET_GROUPS, ADDITIONAL_FILTERS, EXCLUSION_FILTERS)
    if df.empty:
        print("❌ No data fetched. Check your configuration.")
        return
    
    # Clean and filter data
    df_clean = clean_and_filter_data(df)
    if df_clean.empty:
        print("❌ No data remaining after cleaning. Check your filters.")
        return
    
    # Find best hyperparameters
    best_runs = find_best_hyperparams(df_clean, TOP_N_RUNS)
    
    # Print summary
    print_best_hyperparams_summary(best_runs)
    
    # Create visualizations
    create_hyperparameter_visualization(df_clean, best_runs)
    
    print(f"\n✅ Analysis complete! Check the generated {SAVE_FORMAT} files for visualizations.")

if __name__ == "__main__":
    main()
