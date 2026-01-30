#!/usr/bin/env python3
"""
Loss Curves Visualization for Different Scaling Rules

For each model size (head/depth), this script:
1. Finds the run with the lowest final validation loss
2. Plots the full validation loss curve as log(loss) vs log(compute)

Supported Scaling Rules:
1. BigHead: depth-based scaling
2. EggHead: heads-based quadratic depth scaling
3. Enoki: heads-based DiLoco scaling
4. Eryngii: heads-based scaling with increased head dimension and depth
5. Qwen3: Qwen3-style scaling with fixed head_dim=128

Usage:
    python bighead_loss_curves.py --scaling-rule BigHead --optimizer adamw
    python bighead_loss_curves.py --scaling-rule Enoki --optimizer mk4
    python bighead_loss_curves.py --scaling-rule EggHead --optimizer d-muon
    python bighead_loss_curves.py --scaling-rule Qwen3 --optimizer mk4 adamw
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

# =============================================================================
# SCALING RULE CONFIGURATION
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
    'Qwen3_Scaled': {
        'group': 'Qwen3_ScaledGPT',
        'size_param': 'n_head',  # heads
    },
    'Qwen3_Hoyer': {
        'group': 'Qwen3_Hoyer',
        'size_param': 'n_head',  # heads
    }
}

# Optimizer display configuration (name, color)
OPTIMIZER_COLORS = {
    'adamw': '#1f77b4',           # blue
    'dana-star-mk4': '#ff7f0e',    # orange
    'dana': '#2ca02c',             # green
    'ademamix': '#d62728',         # red
    'd-muon': '#9467bd',           # purple
    'manau': '#8c564b',            # brown
    'dana-star-no-tau-kappa-0-85': '#228B22',  # forest green
}

# =============================================================================
# COMPUTE CALCULATION FUNCTIONS
# =============================================================================

def compute_non_embedding_params(size, scaling_rule):
    """
    Compute non-embedding parameters based on scaling rule.

    Args:
        size: For BigHead, this is depth. For others, this is heads.
        scaling_rule: One of 'BigHead', 'EggHead', 'Enoki', 'Eryngii', or 'Qwen3'

    Returns:
        int: Number of non-embedding parameters
    """
    if scaling_rule == 'BigHead':
        # From BigHead.sh:
        # head_dim = 16 * depth, n_embd = 16 * depth^2, mlp = 32 * depth^2
        # n_head = depth, n_layer = depth
        # non_emb = n_layer * (3 * head_dim * n_embd * n_head + n_embd^2 + 2 * n_embd * mlp + 8 * n_embd) + 2 * n_embd
        depth = size
        head_dim = 16 * depth
        n_embd = 16 * depth * depth
        mlp_hidden = 32 * depth * depth
        n_head = depth
        n_layer = depth
        non_emb = n_layer * (3 * head_dim * n_embd * n_head + n_embd * n_embd + 2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd

    elif scaling_rule == 'EggHead':
        # From EggHead.sh:
        # head_dim = 16 * heads, n_embd = 16 * heads^2, mlp = 32 * heads^2
        # n_head = heads, n_layer = heads * (heads - 1) / 2
        # Same non-emb formula as BigHead
        heads = size
        head_dim = 16 * heads
        n_embd = 16 * heads * heads
        mlp_hidden = 32 * heads * heads
        n_head = heads
        n_layer = int(heads * (heads - 1) / 2)
        non_emb = n_layer * (3 * head_dim * n_embd * n_head + n_embd * n_embd + 2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd

    elif scaling_rule in ('Enoki', 'Enoki_ScaledGPT'):
        # From Enoki.sh:
        # head_dim = 64 (fixed), n_embd = heads * 64, mlp = 4 * n_embd
        # n_head = heads, n_layer = 3 * heads / 4
        # non_emb = 12 * n_embd^2 * n_layer
        heads = size
        head_dim = 64  # Fixed for Enoki
        n_embd = heads * 64
        mlp_hidden = 4 * n_embd
        n_head = heads
        n_layer = int(3 * heads // 4)
        non_emb = 12 * n_embd * n_embd * n_layer

    elif scaling_rule == 'Eryngii':
        # From Eryngii.sh:
        # head_dim = round(32 * heads / 3 / 8) * 8
        # n_head = heads, n_layer = heads^2 / 8
        # n_embd = n_head * head_dim, mlp = 4 * n_embd
        # non_emb = 12 * n_embd^2 * n_layer
        heads = size
        head_dim = int(round(32 * heads / 3 / 8) * 8)
        n_head = heads
        n_layer = int(heads * heads // 8)
        n_embd = n_head * head_dim
        mlp_hidden = 4 * n_embd
        non_emb = 12 * n_embd * n_embd * n_layer

    elif scaling_rule in ('Qwen3_Scaled', 'Qwen3_Hoyer'):
        # From Qwen3.sh:
        # head_dim = 128 (fixed), n_embd = heads * 128, mlp = 3 * n_embd
        # n_head = heads, n_layer = 2 * heads
        # With gating (default): non_emb = n_layer * (5 * n_embd * total_qkv_dim + 2 * head_dim + 9 * n_embd^2 + 2 * n_embd) + n_embd
        # where total_qkv_dim = n_head * head_dim
        heads = size
        head_dim = 128  # Fixed for Qwen3
        n_head = heads
        n_layer = int(2 * heads)
        n_embd = heads * 128
        mlp_hidden = 3 * n_embd
        total_qkv_dim = n_head * head_dim

        # Per layer (with gating)
        attn = 5 * n_embd * total_qkv_dim  # q_proj (2x) + k_proj + v_proj + o_proj
        qk_norm = 2 * head_dim
        mlp_params = 9 * n_embd * n_embd  # gate_proj + up_proj + down_proj (mlp_hidden = 3 * n_embd)
        layer_norms = 2 * n_embd

        per_layer = attn + qk_norm + mlp_params + layer_norms
        non_emb = n_layer * per_layer + n_embd  # +n_embd for final norm

    else:
        raise ValueError(f"Unknown scaling rule: {scaling_rule}")

    return int(non_emb)

def compute_total_params(size, scaling_rule):
    """Compute total parameters including embeddings."""
    non_emb = compute_non_embedding_params(size, scaling_rule)

    if scaling_rule == 'BigHead':
        n_embd = 16 * size * size
    elif scaling_rule == 'EggHead':
        n_embd = 16 * size * size
    elif scaling_rule in ('Enoki', 'Enoki_ScaledGPT'):
        n_embd = size * 64
    elif scaling_rule == 'Eryngii':
        heads = size
        head_dim = int(round(32 * heads / 3 / 8) * 8)
        n_embd = heads * head_dim
    elif scaling_rule in ('Qwen3_Scaled', 'Qwen3_Hoyer'):
        n_embd = size * 128
    else:
        raise ValueError(f"Unknown scaling rule: {scaling_rule}")

    vocab_size = 50304
    total_params = non_emb + 2 * n_embd * vocab_size
    return int(total_params)

def compute_flops_per_iteration(size, scaling_rule, batch_size=1, seq_len=512):
    """
    Compute FLOPs per iteration based on model size and configuration.

    Returns:
        float: FLOPs per iteration
    """
    non_emb = float(compute_non_embedding_params(size, scaling_rule))
    # Approximate: 6 * params * seq_len for forward + backward pass
    # Use float to avoid overflow
    flops = 6.0 * non_emb * float(seq_len) * float(batch_size)
    return flops

# =============================================================================
# DATA LOADING
# =============================================================================

def load_wandb_data_with_curves(project_name, group_name, entity, optimizer_type, scaling_rule,
                                target_clipsnr=None, clipsnr_tolerance=0.1, wd_decaying_filter=False):
    """
    Load experiment data from WandB for loss curve visualization.

    Returns:
        dict: Dictionary mapping model sizes to best run data
    """
    api = wandb.Api()

    print(f"Loading data from {group_name}...")
    print(f"Scaling rule: {scaling_rule}")
    print(f"Optimizer: {optimizer_type}")

    runs = api.runs(f"{entity}/{project_name}", filters={"group": group_name})

    # Dictionary to store best run for each size
    best_runs = {}
    size_param = SCALING_RULE_CONFIG[scaling_rule]['size_param']

    total_runs = 0
    skipped = 0

    for run in runs:
        total_runs += 1
        print(f"Processing run: {run.name}")

        # Handle different wandb API versions
        config = run.config
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except (json.JSONDecodeError, TypeError):
                print(f"  Skipping - could not parse config")
                skipped += 1
                continue

        # Extract value from nested dict structure
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
                print(f"  Skipping - could not parse summary")
                skipped += 1
                continue

        # Filter by optimizer
        opt = config.get('opt', '')
        if opt != optimizer_type:
            skipped += 1
            continue

        # Filter by clipsnr if specified
        if target_clipsnr is not None:
            clipsnr = config.get('clipsnr')
            if clipsnr is None or abs(clipsnr - target_clipsnr) > clipsnr_tolerance:
                skipped += 1
                continue

        # Filter by wd_decaying if requested
        if wd_decaying_filter:
            wd_decaying = config.get('wd_decaying', False)
            if not wd_decaying:
                skipped += 1
                continue

        # Check if run completed
        iterations_config = config.get('iterations')
        actual_iter = summary.get('iter')

        if actual_iter is None or iterations_config is None:
            print(f"  Skipping - missing iteration info")
            skipped += 1
            continue

        if actual_iter < iterations_config:
            print(f"  Skipping - incomplete ({actual_iter}/{iterations_config})")
            skipped += 1
            continue

        # Get model size
        size = config.get(size_param)
        if size is None:
            print(f"  Skipping - missing {size_param}")
            skipped += 1
            continue

        # Get final validation loss
        val_loss = summary.get('final-val/loss')
        if val_loss is None:
            print(f"  Skipping - missing final-val/loss")
            skipped += 1
            continue

        # Check if this is the best run for this size
        if size not in best_runs or val_loss < best_runs[size]['final_val_loss']:
            print(f"  -> New best for size={size} (val_loss={val_loss:.4f})")

            # Fetch the validation loss history
            try:
                history = run.history(keys=["val/loss"], samples=10000)
                val_losses = history["val/loss"].dropna().values

                if len(val_losses) == 0:
                    print(f"  -> No validation data available")
                    continue

                # Get eval_interval from config (defaults to 115 if not found)
                eval_interval = config.get('eval_interval', 115)

                # Generate iteration numbers
                iterations = np.arange(len(val_losses)) * eval_interval

                # Get batch size and seq_len for compute calculation
                local_batch_size = config.get('batch_size', 1)
                world_size = config.get('world_size', 1)
                acc_steps = config.get('acc_steps', 1)
                # Try both seq_len and sequence_length
                seq_len = config.get('sequence_length', config.get('seq_len', 512))

                # Calculate effective batch size (includes gradient accumulation)
                effective_batch_size = local_batch_size * world_size * acc_steps

                print(f"     eval_interval={eval_interval}, num_val_points={len(val_losses)}, max_iter={iterations[-1] if len(iterations) > 0 else 0}")
                print(f"     local_batch_size={local_batch_size}, world_size={world_size}, acc_steps={acc_steps}, effective_batch_size={effective_batch_size}, seq_len={seq_len}")

                best_runs[size] = {
                    'config': config,
                    'final_val_loss': val_loss,
                    'val_losses': val_losses,
                    'iterations': iterations,
                    'run_name': run.name,
                    'batch_size': effective_batch_size,
                    'seq_len': seq_len,
                }
            except Exception as e:
                print(f"  -> Failed to fetch history: {e}")
                continue

    print(f"\nProcessed {total_runs} runs, skipped {skipped}, kept {len(best_runs)} best runs")
    return best_runs

def load_wandb_data_multi_optimizer(project_name, group_name, entity, optimizer_types, scaling_rule,
                                    target_clipsnr=None, clipsnr_tolerance=0.1, wd_decaying_filter=False):
    """
    Load experiment data from WandB for multiple optimizers.

    Returns:
        dict: Dictionary mapping optimizer -> {size -> best run data}
    """
    all_runs = {}
    for optimizer_type in optimizer_types:
        print(f"\n{'='*60}")
        print(f"Loading data for optimizer: {optimizer_type}")
        print(f"{'='*60}")
        best_runs = load_wandb_data_with_curves(
            project_name, group_name, entity, optimizer_type, scaling_rule,
            target_clipsnr=target_clipsnr,
            clipsnr_tolerance=clipsnr_tolerance,
            wd_decaying_filter=wd_decaying_filter
        )
        if best_runs:
            all_runs[optimizer_type] = best_runs
    return all_runs

# =============================================================================
# PLOTTING
# =============================================================================

def plot_loss_curves(best_runs, scaling_rule, optimizer_type, output_filename):
    """
    Create plot of validation loss curves for best runs (single optimizer).
    """
    print(f"\nCreating visualization with {len(best_runs)} loss curves")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort sizes for consistent coloring
    sizes = sorted(best_runs.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))

    # Plot each loss curve
    for i, size in enumerate(sizes):
        run_data = best_runs[size]
        val_losses = run_data['val_losses']
        iterations = run_data['iterations']

        # Calculate compute (FLOPs)
        flops_per_iter = compute_flops_per_iteration(
            size, scaling_rule,
            batch_size=run_data['batch_size'],
            seq_len=run_data['seq_len']
        )
        # Use float64 arrays to avoid overflow
        compute_flops = np.float64(iterations) * np.float64(flops_per_iter)

        # Convert to petaflop-days
        # 1 petaflop = 10^15 flops
        # 1 day = 86400 seconds
        flops_per_petaflop_day = 1e15 * 86400
        compute_petaflop_days = compute_flops / flops_per_petaflop_day

        # Get total params for label
        total_params = compute_total_params(size, scaling_rule)

        # Create label
        size_param = SCALING_RULE_CONFIG[scaling_rule]['size_param']
        if size_param == 'n_layer':
            label = f'Depth={size} ({total_params/1e6:.1f}M params)'
        else:
            label = f'Heads={size} ({total_params/1e6:.1f}M params)'

        # Plot
        ax.plot(compute_petaflop_days, val_losses,
               color=colors[i],
               alpha=0.8, linewidth=2.5,
               label=label)

    # Formatting
    ax.set_xlabel('Compute (PetaFLOP-days)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=20, fontweight='bold')

    title = f'{scaling_rule} Loss Curves - {optimizer_type.upper()}'
    ax.set_title(title, fontsize=24, fontweight='bold')

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    ax.legend(loc='best', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    plt.tight_layout()

    # Save plot
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as {output_filename}")

    # Print statistics
    print("\nRun statistics:")
    for size in sorted(best_runs.keys()):
        run_data = best_runs[size]
        print(f"\nSize {size}:")
        print(f"  Run: {run_data['run_name']}")
        print(f"  Final val loss: {run_data['final_val_loss']:.4f}")
        print(f"  Number of val measurements: {len(run_data['val_losses'])}")
        print(f"  Total params: {compute_total_params(size, scaling_rule)/1e6:.1f}M")


def plot_loss_curves_multi_optimizer(all_runs, scaling_rule, output_filename):
    """
    Create plot of validation loss curves for multiple optimizers on the same axis.

    Args:
        all_runs: dict mapping optimizer -> {size -> best run data}
        scaling_rule: Name of the scaling rule
        output_filename: Output file path for the plot
    """
    # Count total curves
    total_curves = sum(len(runs) for runs in all_runs.values())
    print(f"\nCreating visualization with {total_curves} loss curves across {len(all_runs)} optimizers")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 9))

    # Get all unique sizes across all optimizers
    all_sizes = set()
    for runs in all_runs.values():
        all_sizes.update(runs.keys())
    all_sizes = sorted(all_sizes)

    # Create a colormap for sizes
    size_colors = plt.cm.viridis(np.linspace(0, 1, len(all_sizes)))
    size_to_color = {size: size_colors[i] for i, size in enumerate(all_sizes)}

    # Line styles for different optimizers
    line_styles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]
    markers = ['o', 's', '^', 'D', 'v', 'p']

    optimizer_list = list(all_runs.keys())

    # Track labels to avoid duplicates
    size_labels_added = set()
    optimizer_labels_added = set()

    # Plot each optimizer
    for opt_idx, (optimizer_type, best_runs) in enumerate(all_runs.items()):
        line_style = line_styles[opt_idx % len(line_styles)]
        marker = markers[opt_idx % len(markers)]

        for size in sorted(best_runs.keys()):
            run_data = best_runs[size]
            val_losses = run_data['val_losses']
            iterations = run_data['iterations']

            # Calculate compute (FLOPs)
            flops_per_iter = compute_flops_per_iteration(
                size, scaling_rule,
                batch_size=run_data['batch_size'],
                seq_len=run_data['seq_len']
            )
            # Use float64 arrays to avoid overflow
            compute_flops = np.float64(iterations) * np.float64(flops_per_iter)

            # Convert to petaflop-days
            flops_per_petaflop_day = 1e15 * 86400
            compute_petaflop_days = compute_flops / flops_per_petaflop_day

            # Get total params for label
            total_params = compute_total_params(size, scaling_rule)

            # Create label
            size_param = SCALING_RULE_CONFIG[scaling_rule]['size_param']
            if size_param == 'n_layer':
                size_label = f'Depth={size} ({total_params/1e6:.1f}M)'
            else:
                size_label = f'Heads={size} ({total_params/1e6:.1f}M)'

            # Combined label
            label = f'{optimizer_type} - {size_label}'

            # Use color based on optimizer if few optimizers, otherwise based on size
            if len(all_runs) <= 3:
                color = OPTIMIZER_COLORS.get(optimizer_type, f'C{opt_idx}')
            else:
                color = size_to_color[size]

            # Plot
            ax.plot(compute_petaflop_days, val_losses,
                   color=color,
                   linestyle=line_style,
                   alpha=0.8, linewidth=2.0,
                   label=label,
                   marker=marker,
                   markevery=max(1, len(val_losses)//10),
                   markersize=5)

    # Formatting
    ax.set_xlabel('Compute (PetaFLOP-days)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=20, fontweight='bold')

    optimizer_names = ', '.join([opt.upper() for opt in all_runs.keys()])
    title = f'{scaling_rule} Loss Curves\n{optimizer_names}'
    ax.set_title(title, fontsize=22, fontweight='bold')

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    # Create legend - may need to adjust based on number of curves
    if total_curves <= 12:
        ax.legend(loc='best', fontsize=11, framealpha=0.9, ncol=1)
    else:
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)

    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    plt.tight_layout()

    # Save plot
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as {output_filename}")

    # Print statistics
    print("\nRun statistics:")
    for optimizer_type, best_runs in all_runs.items():
        print(f"\n{'='*40}")
        print(f"Optimizer: {optimizer_type}")
        print(f"{'='*40}")
        for size in sorted(best_runs.keys()):
            run_data = best_runs[size]
            print(f"\n  Size {size}:")
            print(f"    Run: {run_data['run_name']}")
            print(f"    Final val loss: {run_data['final_val_loss']:.4f}")
            print(f"    Number of val measurements: {len(run_data['val_losses'])}")
            print(f"    Total params: {compute_total_params(size, scaling_rule)/1e6:.1f}M")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot loss curves for different scaling rules')
    parser.add_argument('--scaling-rule', type=str, required=True,
                       choices=['BigHead', 'EggHead', 'Enoki', 'Enoki_ScaledGPT', 'Eryngii', 'Qwen3', 'Qwen3_Scaled', 'Qwen3_Hoyer'],
                       help='Model scaling rule')
    parser.add_argument('--optimizer', type=str, nargs='+', required=True,
                       choices=['adamw', 'mk4', 'dana', 'ademamix', 'd-muon', 'manau', 'dana-star-no-tau-kappa-0-85'],
                       help='Optimizer type(s) - can specify multiple')
    parser.add_argument('--project', type=str, default='danastar',
                       help='WandB project name (default: danastar)')
    parser.add_argument('--group', type=str, default=None,
                       help='WandB group name (default: auto-determined from scaling-rule)')
    parser.add_argument('--entity', type=str, default='ep-rmt-ml-opt',
                       help='WandB entity name (default: ep-rmt-ml-opt)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename for plot (default: auto-generated)')
    parser.add_argument('--target-clipsnr', type=float, default=None,
                       help='Target clipsnr value for MK4 optimizer (default: None)')
    parser.add_argument('--clipsnr-tolerance', type=float, default=0.1,
                       help='Tolerance for clipsnr matching (default: 0.1)')
    parser.add_argument('--wd-decaying', action='store_true',
                       help='For Manau optimizer: filter for runs with wd_decaying=True')
    args = parser.parse_args()

    # Map optimizer abbreviations
    optimizer_map = {
        'adamw': 'adamw',
        'mk4': 'dana-star-mk4',
        'dana': 'dana',
        'ademamix': 'ademamix',
        'd-muon': 'd-muon',
        'manau': 'manau',
        'dana-star-no-tau-kappa-0-85': 'dana-star-no-tau-kappa-0-85'
    }
    optimizer_types = [optimizer_map[opt] for opt in args.optimizer]

    # Get scaling rule configuration
    scaling_config = SCALING_RULE_CONFIG[args.scaling_rule]

    # Determine WandB group
    if args.group is None:
        wandb_group = scaling_config['group']
    else:
        wandb_group = args.group

    # Determine output filename
    if args.output is None:
        opt_str = '_'.join(args.optimizer)
        output_filename = f'{args.scaling_rule}_loss_curves_{opt_str}.pdf'
    else:
        output_filename = args.output

    print(f"Loss Curves Visualization")
    print("=" * 60)
    print(f"Scaling Rule: {args.scaling_rule}")
    print(f"Optimizers: {optimizer_types}")
    print(f"WandB Group: {wandb_group}")
    print("=" * 60)

    # Load data
    if len(optimizer_types) == 1:
        # Single optimizer - use original function
        best_runs = load_wandb_data_with_curves(
            args.project, wandb_group, args.entity, optimizer_types[0], args.scaling_rule,
            target_clipsnr=args.target_clipsnr,
            clipsnr_tolerance=args.clipsnr_tolerance,
            wd_decaying_filter=args.wd_decaying
        )

        if len(best_runs) == 0:
            print("ERROR: No data found. Check parameters.")
        else:
            # Create plot
            plot_loss_curves(best_runs, args.scaling_rule, optimizer_types[0], output_filename)
            print("\nVisualization completed successfully!")
    else:
        # Multiple optimizers - use multi-optimizer function
        all_runs = load_wandb_data_multi_optimizer(
            args.project, wandb_group, args.entity, optimizer_types, args.scaling_rule,
            target_clipsnr=args.target_clipsnr,
            clipsnr_tolerance=args.clipsnr_tolerance,
            wd_decaying_filter=args.wd_decaying
        )

        if len(all_runs) == 0:
            print("ERROR: No data found. Check parameters.")
        else:
            # Create plot
            plot_loss_curves_multi_optimizer(all_runs, args.scaling_rule, output_filename)
            print("\nVisualization completed successfully!")
