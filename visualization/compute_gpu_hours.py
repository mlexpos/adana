#!/usr/bin/env python3
"""
Compute GPU Hours from WandB Data

This script computes actual GPU hours from WandB experiments by:
1. Fetching runs from a specified group/optimizer
2. Extracting iter_dt (time per iteration) from wandb history
3. Computing total wall time = mean(iter_dt) * total_iterations
4. Computing GPU hours = wall_time_hours * num_gpus

Usage:
    python compute_gpu_hours.py --group Enoki_ScaledGPT --optimizer adamw
    python compute_gpu_hours.py --group Enoki_ScaledGPT --optimizer adamw --output gpu_hours_table.csv
    python compute_gpu_hours.py --group Enoki_ScaledGPT --optimizer adamw --detailed
"""

import wandb
import numpy as np
import pandas as pd
import argparse
import json
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SCALING_RULE_CONFIG = {
    'BigHead': {
        'group': 'DanaStar_MK4_BigHead_Sweep',
        'size_param': 'n_layer',  # depth
    },
    'EggHead': {
        'group': 'DanaStar_MK4_EggHead_Sweep',
        'size_param': 'n_head',  # heads
    },
    'Enoki': {
        'group': 'DanaStar_MK4_Enoki_Sweep',
        'size_param': 'n_head',  # heads
    },
    'Enoki_ScaledGPT': {
        'group': 'Enoki_ScaledGPT',
        'size_param': 'n_head',  # heads
    },
    'Eryngii': {
        'group': 'eryngii_sweeps',
        'size_param': 'n_head',  # heads
    },
    'Eryngii_Scaled': {
        'group': 'Eryngii_ScaledGPT',
        'size_param': 'n_head',  # heads
    },
    'Qwen3_Scaled': {
        'group': 'Qwen3_ScaledGPT',
        'size_param': 'n_head',  # heads
    },
    'Qwen3_Hoyer': {
        'group': 'Qwen3_Hoyer',
        'size_param': 'n_head',  # heads
    }
}

# =============================================================================
# PARAMETER CALCULATION FUNCTIONS
# =============================================================================

def compute_model_size(size, scaling_rule):
    """
    Compute total model parameters for a given size and scaling rule.

    Args:
        size: For BigHead, this is depth. For others, this is heads.
        scaling_rule: The scaling rule name

    Returns:
        int: Total number of parameters
    """
    vocab_size = 50304

    if scaling_rule == 'BigHead':
        depth = size
        n_embd = 16 * depth * depth
        head_dim = 16 * depth
        mlp_hidden = 32 * depth * depth
        n_head = depth
        n_layer = depth
        non_emb = depth * (3 * head_dim * n_embd * n_head + n_embd * n_embd +
                          2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd

    elif scaling_rule == 'EggHead':
        heads = size
        n_embd = 16 * heads * heads
        head_dim = 16 * heads
        mlp_hidden = 32 * heads * heads
        n_head = heads
        n_layer = int(heads * (heads - 1) / 2)
        non_emb = n_layer * (3 * head_dim * n_embd * n_head + n_embd * n_embd +
                            2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd

    elif scaling_rule in ('Enoki', 'Enoki_ScaledGPT'):
        heads = size
        n_embd = heads * 64
        n_layer = int(3 * heads // 4)
        non_emb = 12 * n_embd * n_embd * n_layer

    elif scaling_rule in ('Eryngii', 'Eryngii_Scaled'):
        heads = size
        head_dim = int(round(32 * heads / 3 / 8) * 8)
        n_head = heads
        n_layer = int(heads * heads // 8)
        n_embd = n_head * head_dim
        non_emb = 12 * n_embd * n_embd * n_layer

    elif scaling_rule in ('Qwen3_Scaled', 'Qwen3_Hoyer'):
        heads = size
        head_dim = 128
        n_head = heads
        n_layer = 2 * heads
        n_embd = 128 * heads
        total_qkv_dim = n_head * head_dim
        per_layer = 5 * n_embd * total_qkv_dim + 2 * head_dim + 9 * n_embd * n_embd + 2 * n_embd
        non_emb = n_layer * per_layer + n_embd

    else:
        raise ValueError(f"Unknown scaling rule: {scaling_rule}")

    total_params = non_emb + 2 * n_embd * vocab_size
    return int(total_params)


def format_params(params):
    """Format parameter count in human-readable form (e.g., 45.7M, 1.41B)."""
    if params >= 1e9:
        return f"{params/1e9:.2f}B"
    elif params >= 1e6:
        return f"{params/1e6:.1f}M"
    elif params >= 1e3:
        return f"{params/1e3:.1f}K"
    else:
        return str(params)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_config_value(config_dict):
    """Extract values from nested config structure (some wandb versions nest values)."""
    result = {}
    for key, val in config_dict.items():
        if isinstance(val, dict) and 'value' in val:
            result[key] = val['value']
        else:
            result[key] = val
    return result


def get_gpu_info_from_run(run):
    """
    Try to extract GPU type from wandb run metadata.

    Wandb stores system metadata that may include GPU info.
    """
    gpu_info = None

    # Try different locations where GPU info might be stored
    try:
        # Check run.metadata
        if hasattr(run, 'metadata') and run.metadata:
            metadata = run.metadata
            if 'gpu' in metadata:
                gpu_info = metadata['gpu']
            elif 'gpu_type' in metadata:
                gpu_info = metadata['gpu_type']

        # Check run.system_metrics
        if gpu_info is None and hasattr(run, 'system_metrics') and run.system_metrics:
            system_metrics = run.system_metrics
            for key in system_metrics:
                if 'gpu' in key.lower():
                    gpu_info = str(system_metrics[key])
                    break

        # Check summary for gpu info
        if gpu_info is None:
            summary = run.summary
            if hasattr(summary, '_json_dict'):
                summary_dict = summary._json_dict if isinstance(summary._json_dict, dict) else {}
            else:
                summary_dict = dict(summary) if summary else {}

            for key in summary_dict:
                if 'gpu' in key.lower():
                    gpu_info = str(summary_dict[key])
                    break

        # Check config for gpu info
        if gpu_info is None:
            config = run.config
            if isinstance(config, str):
                try:
                    config = json.loads(config)
                except:
                    config = {}
            config = extract_config_value(config) if isinstance(config, dict) else {}

            for key in config:
                if 'gpu' in key.lower():
                    gpu_info = str(config[key])
                    break

    except Exception as e:
        pass

    return gpu_info

# =============================================================================
# MAIN DATA LOADING AND COMPUTATION
# =============================================================================

def compute_gpu_hours_from_wandb(project, entity, group, optimizer_type, scaling_rule=None, detailed=False):
    """
    Compute GPU hours from WandB runs.

    Args:
        project: WandB project name
        entity: WandB entity name
        group: WandB group name
        optimizer_type: Optimizer to filter for (e.g., 'adamw')
        scaling_rule: Optional scaling rule name (for computing model size)
        detailed: If True, return per-run details

    Returns:
        DataFrame with GPU hours by model size
    """
    api = wandb.Api()

    print(f"Loading runs from {entity}/{project}")
    print(f"  Group: {group}")
    print(f"  Optimizer: {optimizer_type}")
    print()

    runs = api.runs(f"{entity}/{project}", filters={"group": group})

    # Determine size parameter from scaling rule
    if scaling_rule is None:
        # Try to infer from group name
        for rule_name, rule_config in SCALING_RULE_CONFIG.items():
            if rule_config['group'] == group:
                scaling_rule = rule_name
                break

    if scaling_rule and scaling_rule in SCALING_RULE_CONFIG:
        size_param = SCALING_RULE_CONFIG[scaling_rule]['size_param']
    else:
        size_param = 'n_head'  # Default

    print(f"  Size parameter: {size_param}")
    print(f"  Scaling rule: {scaling_rule}")
    print()

    # Collect data for each run
    run_data = []
    gpu_types_found = set()

    total_runs = 0
    processed_runs = 0
    skipped_runs = 0

    for run in runs:
        total_runs += 1

        # Parse config
        config = run.config
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except (json.JSONDecodeError, TypeError):
                skipped_runs += 1
                continue
        if hasattr(config, 'as_dict'):
            config = config.as_dict()
        elif not isinstance(config, dict):
            try:
                config = dict(config)
            except (TypeError, ValueError):
                skipped_runs += 1
                continue
        config = extract_config_value(config)

        # Parse summary
        summary = run.summary
        if hasattr(summary, '_json_dict') and isinstance(summary._json_dict, str):
            try:
                summary = json.loads(summary._json_dict)
            except (json.JSONDecodeError, TypeError):
                skipped_runs += 1
                continue

        # Filter by optimizer
        opt = config.get('opt', '')
        if opt != optimizer_type:
            skipped_runs += 1
            continue

        # Check completion
        # Handle summary as either dict or wandb Summary object
        if isinstance(summary, dict):
            actual_iter = summary.get('iter', 0)
        else:
            try:
                actual_iter = summary['iter'] if 'iter' in summary else 0
            except (KeyError, TypeError):
                actual_iter = 0

        iterations_config = config.get('iterations', 0)
        if actual_iter > 0 and iterations_config > 0 and actual_iter < iterations_config:
            print(f"  Skipping incomplete run: {run.name} ({actual_iter}/{iterations_config})")
            skipped_runs += 1
            continue

        # Get model size
        size = config.get(size_param)
        if size is None:
            print(f"  Skipping run without {size_param}: {run.name}")
            skipped_runs += 1
            continue

        # Get world_size (number of GPUs)
        world_size = config.get('world_size', 1)

        # Get GPU type
        gpu_type = get_gpu_info_from_run(run)
        if gpu_type:
            gpu_types_found.add(gpu_type)

        # Try to get iter_dt from history
        iter_dt_mean = None
        iter_dt_sum = None
        iter_dt_count = 0

        try:
            # Fetch history with iter_dt
            history = run.history(keys=['iter_dt', '_step'], pandas=True)

            if 'iter_dt' in history.columns:
                iter_dt_values = history['iter_dt'].dropna().values
                if len(iter_dt_values) > 0:
                    iter_dt_mean = np.mean(iter_dt_values)
                    iter_dt_sum = np.sum(iter_dt_values)
                    iter_dt_count = len(iter_dt_values)

        except Exception as e:
            print(f"  Warning: Could not fetch iter_dt for {run.name}: {e}")

        # If we don't have iter_dt, try to estimate from run duration
        if iter_dt_mean is None:
            try:
                # Get run duration from summary or metadata
                if hasattr(run, 'runtime'):
                    runtime_seconds = run.runtime
                elif '_runtime' in summary:
                    runtime_seconds = summary['_runtime']
                elif 'runtime' in summary:
                    runtime_seconds = summary['runtime']
                else:
                    runtime_seconds = None

                if runtime_seconds is not None and actual_iter > 0:
                    iter_dt_mean = runtime_seconds / actual_iter
                    iter_dt_sum = runtime_seconds
                    iter_dt_count = actual_iter
            except Exception as e:
                pass

        if iter_dt_mean is None:
            print(f"  Skipping run without timing info: {run.name}")
            skipped_runs += 1
            continue

        # Compute GPU hours
        # Method 1: sum(iter_dt) * num_gpus / 3600
        # Method 2: mean(iter_dt) * total_iters * num_gpus / 3600
        # We'll use method 2 if we don't have complete iter_dt data

        total_iterations = actual_iter if actual_iter > 0 else iterations_config

        # Wall time in seconds
        wall_time_seconds = iter_dt_mean * total_iterations

        # GPU hours = wall_time * num_gpus / 3600
        gpu_hours = (wall_time_seconds * world_size) / 3600.0

        # Compute model size
        if scaling_rule:
            total_params = compute_model_size(size, scaling_rule)
        else:
            total_params = None

        # Get final validation loss
        if isinstance(summary, dict):
            final_loss = summary.get('final-val/loss')
        else:
            try:
                final_loss = summary['final-val/loss'] if 'final-val/loss' in summary else None
            except (KeyError, TypeError):
                final_loss = None

        run_info = {
            'run_name': run.name,
            'size': size,
            'total_params': total_params,
            'world_size': world_size,
            'total_iterations': total_iterations,
            'iter_dt_mean': iter_dt_mean,
            'iter_dt_count': iter_dt_count,
            'wall_time_hours': wall_time_seconds / 3600.0,
            'gpu_hours': gpu_hours,
            'gpu_type': gpu_type,
            'final_loss': final_loss,
        }

        run_data.append(run_info)
        processed_runs += 1

        if detailed:
            print(f"  Processed: {run.name}")
            print(f"    Size={size}, world_size={world_size}, iters={total_iterations}")
            print(f"    iter_dt_mean={iter_dt_mean:.4f}s, gpu_hours={gpu_hours:.2f}")

    print()
    print(f"Total runs: {total_runs}")
    print(f"Processed: {processed_runs}")
    print(f"Skipped: {skipped_runs}")
    print()

    if gpu_types_found:
        print(f"GPU types found: {gpu_types_found}")
    else:
        print("GPU type: Not found in metadata (likely H100 based on portal info)")
    print()

    if not run_data:
        print("No valid runs found!")
        return None

    # Create DataFrame
    df = pd.DataFrame(run_data)

    return df


def aggregate_by_model_size(df):
    """
    Aggregate GPU hours by model size, taking the best (lowest loss) run for each size.

    Args:
        df: DataFrame with run data

    Returns:
        DataFrame aggregated by model size
    """
    # Group by size and select the run with the lowest final loss
    # (or highest GPU hours if loss is not available, as proxy for completed run)

    results = []

    for size, group in df.groupby('size'):
        # Filter for runs with final_loss
        group_with_loss = group[group['final_loss'].notna()]

        if len(group_with_loss) > 0:
            # Select best run (lowest loss)
            best_idx = group_with_loss['final_loss'].idxmin()
            best_run = group_with_loss.loc[best_idx]
        else:
            # Fall back to first run
            best_run = group.iloc[0]

        results.append({
            'size': size,
            'total_params': best_run['total_params'],
            'model_size_formatted': format_params(best_run['total_params']) if best_run['total_params'] else f"size={size}",
            'world_size': best_run['world_size'],
            'total_iterations': best_run['total_iterations'],
            'iter_dt_mean': best_run['iter_dt_mean'],
            'wall_time_hours': best_run['wall_time_hours'],
            'gpu_hours': best_run['gpu_hours'],
            'gpu_type': best_run['gpu_type'],
            'final_loss': best_run['final_loss'],
            'run_name': best_run['run_name'],
            'num_runs': len(group),
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('total_params' if result_df['total_params'].notna().any() else 'size')

    return result_df

# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_table(df, title="GPU Hours by Model Size"):
    """Print a formatted table of GPU hours."""
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print()

    # Header
    print(f"{'Model Size':<15} {'GPU Hours':<12} {'Wall Time (h)':<15} {'GPUs':<8} {'Iters':<12} {'iter_dt (s)':<12} {'GPU Type':<20}")
    print("-" * 100)

    for _, row in df.iterrows():
        model_size = row['model_size_formatted']
        gpu_hours = row['gpu_hours']
        wall_time = row['wall_time_hours']
        gpus = row['world_size']
        iters = row['total_iterations']
        iter_dt = row['iter_dt_mean']
        gpu_type = row['gpu_type'] if pd.notna(row['gpu_type']) else 'N/A'
        # Shorten GPU type for display
        gpu_type_short = gpu_type[:18] if len(str(gpu_type)) > 18 else gpu_type

        print(f"{model_size:<15} {gpu_hours:<12.1f} {wall_time:<15.2f} {gpus:<8} {iters:<12} {iter_dt:<12.4f} {gpu_type_short:<20}")

    print("-" * 100)
    print(f"{'Total':<15} {df['gpu_hours'].sum():<12.1f}")
    print()

    # Print GPU type summary if available
    gpu_types = df['gpu_type'].dropna().unique()
    if len(gpu_types) > 0:
        print(f"GPU Types found: {', '.join(gpu_types)}")
    else:
        print("GPU Type: H100 (inferred from WandB portal)")
    print()


def export_to_csv(df, filename):
    """Export results to CSV file."""
    df.to_csv(filename, index=False)
    print(f"Results saved to: {filename}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute GPU hours from WandB data')
    parser.add_argument('--group', type=str, default='Enoki_ScaledGPT',
                       help='WandB group name (default: Enoki_ScaledGPT)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       help='Optimizer to filter for (default: adamw)')
    parser.add_argument('--project', type=str, default='danastar',
                       help='WandB project name (default: danastar)')
    parser.add_argument('--entity', type=str, default='ep-rmt-ml-opt',
                       help='WandB entity name (default: ep-rmt-ml-opt)')
    parser.add_argument('--scaling-rule', type=str, default=None,
                       choices=['BigHead', 'EggHead', 'Enoki', 'Enoki_ScaledGPT',
                               'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', 'Qwen3_Hoyer'],
                       help='Scaling rule for parameter calculation (auto-detected if not specified)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV filename (optional)')
    parser.add_argument('--detailed', action='store_true',
                       help='Print detailed per-run information')
    parser.add_argument('--gpu-filter', type=str, default=None,
                       help='Filter by GPU type (e.g., "H100" or "A100")')
    parser.add_argument('--show-all-runs', action='store_true',
                       help='Show all runs instead of best per size')
    args = parser.parse_args()

    print()
    print("=" * 80)
    print("GPU Hours Computation from WandB Data")
    print("=" * 80)
    print()

    # Load and process data
    df = compute_gpu_hours_from_wandb(
        project=args.project,
        entity=args.entity,
        group=args.group,
        optimizer_type=args.optimizer,
        scaling_rule=args.scaling_rule,
        detailed=args.detailed
    )

    if df is not None and len(df) > 0:
        # Apply GPU filter if specified
        if args.gpu_filter:
            original_len = len(df)
            df = df[df['gpu_type'].str.contains(args.gpu_filter, case=False, na=False)]
            print(f"Filtered by GPU type '{args.gpu_filter}': {original_len} -> {len(df)} runs")
            print()

        if len(df) == 0:
            print("No data after filtering!")
            exit(1)

        # Aggregate by model size
        aggregated_df = aggregate_by_model_size(df)

        # Print table
        gpu_filter_str = f" ({args.gpu_filter})" if args.gpu_filter else ""
        print_table(aggregated_df, f"GPU Hours by Model Size ({args.group} / {args.optimizer}{gpu_filter_str})")

        # Export to CSV if requested
        if args.output:
            export_to_csv(aggregated_df, args.output)

        # Also show the reference table format requested
        print()
        print("Reference Table Format:")
        print("-" * 50)
        print(f"{'Model Size':<15} | {'GPU Hours':<12} | {'GPU Type':<15}")
        print("-" * 50)
        for _, row in aggregated_df.iterrows():
            gpu_type_short = 'H100' if 'H100' in str(row['gpu_type']) else ('A100' if 'A100' in str(row['gpu_type']) else 'N/A')
            print(f"{row['model_size_formatted']:<15} | {row['gpu_hours']:<12.0f} | {gpu_type_short:<15}")
        print("-" * 50)

        # If we have both A100 and H100 runs, also show H100-equivalent hours
        # Using approximate ratio: H100 ~1.5-2x faster than A100 for transformer training
        gpu_types = aggregated_df['gpu_type'].dropna().unique()
        has_both = any('H100' in str(gt) for gt in gpu_types) and any('A100' in str(gt) for gt in gpu_types)

        if has_both:
            print()
            print("H100-Equivalent GPU Hours (A100 hours * 2.0 for rough comparison):")
            print("-" * 50)
            print(f"{'Model Size':<15} | {'H100-eq Hours':<15}")
            print("-" * 50)
            A100_TO_H100_RATIO = 2.0  # A100 is roughly half the speed of H100 for this workload
            for _, row in aggregated_df.iterrows():
                is_h100 = 'H100' in str(row['gpu_type'])
                h100_eq_hours = row['gpu_hours'] if is_h100 else row['gpu_hours'] / A100_TO_H100_RATIO
                print(f"{row['model_size_formatted']:<15} | {h100_eq_hours:<15.0f}")
            print("-" * 50)

    else:
        print("No data to display.")
