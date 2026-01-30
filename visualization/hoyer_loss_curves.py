#!/usr/bin/env python3
"""
Hoyer Loss Curves Visualization for Qwen3 Models

This script visualizes cross-entropy loss and hoyer loss over time for experiments
in the Qwen3_Hoyer wandb group.

Key features:
1. Plots cross-entropy loss (val/loss - hoyer_loss_coeff * val/hoyer_loss) on log-log axes
2. Different line styles for each optimizer (adamw, dana-star-mk4, dana-mk4)
3. Color based on log(hoyer_loss_coeff) using plasma colormap (0.0 to 0.8 range)
4. Separate PDF for hoyer loss values (hoyer_loss - 1) on log-log axes
5. Command line filtering by heads, optimizers, and hoyer_loss_coeff

Qwen3 scaling:
    head_dim = 128 (fixed)
    n_layer = 2 * heads
    n_embd = 128 * heads
    mlp_hidden = 3 * n_embd

Usage:
    python hoyer_loss_curves.py
    python hoyer_loss_curves.py --heads 6 8 10 --optimizers adamw mk4
    python hoyer_loss_curves.py --hoyer-coeff-range 1e-4 1e-2
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style, rc, rcParams
import argparse
import json

# Matplotlib formatting
style.use('seaborn-v0_8-darkgrid')
rc('font', family='sans-serif')
rcParams['font.weight'] = 'normal'
rcParams['font.size'] = 14
rcParams['figure.figsize'] = (14, 10)
rcParams['axes.linewidth'] = 1.5
rcParams['axes.edgecolor'] = '#333333'
rcParams['grid.alpha'] = 0.3
rcParams['grid.linestyle'] = '--'
rcParams['legend.framealpha'] = 0.95
rcParams['legend.edgecolor'] = '#333333'

# =============================================================================
# CONFIGURATION
# =============================================================================

WANDB_GROUP = 'Qwen3_Hoyer'

# Optimizer to line style mapping
OPTIMIZER_LINESTYLES = {
    'adamw': '-',
    'dana-star-mk4': '--',
    'dana-mk4': ':',
}

# Optimizer display names
OPTIMIZER_NAMES = {
    'adamw': 'AdamW',
    'dana-star-mk4': 'Dana-Star-MK4',
    'dana-mk4': 'Dana-MK4',
}

# =============================================================================
# COMPUTE CALCULATION FUNCTIONS (Qwen3)
# =============================================================================

def compute_qwen3_params(heads):
    """
    Compute parameters for Qwen3 architecture.

    Args:
        heads: Number of attention heads

    Returns:
        dict with non_emb, total_params, n_layer, n_embd
    """
    head_dim = 128
    n_head = heads
    n_layer = 2 * heads
    n_embd = 128 * heads
    total_qkv_dim = n_head * head_dim

    # Qwen3 with gating:
    # per_layer = 5 * n_embd * total_qkv_dim + 2 * head_dim + 9 * n_embd^2 + 2 * n_embd
    per_layer = 5 * n_embd * total_qkv_dim + 2 * head_dim + 9 * n_embd * n_embd + 2 * n_embd
    non_emb = float(n_layer * per_layer + n_embd)

    vocab_size = 50304
    total_params = float(non_emb + 2 * n_embd * vocab_size)

    return {
        'non_emb': int(non_emb),
        'total_params': int(total_params),
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd
    }

def compute_flops_per_iteration(heads, batch_size=1, seq_len=512):
    """Compute FLOPs per iteration for Qwen3."""
    params = compute_qwen3_params(heads)
    non_emb = float(params['non_emb'])
    flops = 6.0 * non_emb * float(seq_len) * float(batch_size)
    return flops

# =============================================================================
# DATA LOADING
# =============================================================================

def load_hoyer_data(project_name, entity,
                    heads_filter=None,
                    optimizers_filter=None,
                    hoyer_coeff_min=None,
                    hoyer_coeff_max=None,
                    allow_incomplete=False,
                    no_qknorm=False):
    """
    Load experiment data from WandB for Hoyer loss visualization.

    Args:
        project_name: WandB project name
        entity: WandB entity name
        heads_filter: List of head counts to include (None = all)
        optimizers_filter: List of optimizers to include (None = all)
        hoyer_coeff_min: Minimum hoyer_loss_coeff (None = no min)
        hoyer_coeff_max: Maximum hoyer_loss_coeff (None = no max)
        allow_incomplete: If True, include runs that haven't completed (default: False)
        no_qknorm: If True, only include runs with no_qknorm=True; if False, exclude those runs (default: False)

    Returns:
        list: List of run data dictionaries
    """
    api = wandb.Api()

    print(f"Loading data from {WANDB_GROUP}...")
    runs = api.runs(f"{entity}/{project_name}", filters={"group": WANDB_GROUP})

    run_data_list = []
    total_runs = 0
    skipped = 0

    for run in runs:
        total_runs += 1

        # Handle config
        config = run.config
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except (json.JSONDecodeError, TypeError):
                skipped += 1
                continue

        # Extract values from nested dict structure
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
                skipped += 1
                continue

        # Get key parameters
        opt = config.get('opt', '')
        heads = config.get('n_head')
        hoyer_coeff = config.get('hoyer_loss_coeff')

        if heads is None or hoyer_coeff is None:
            skipped += 1
            continue

        # Apply filters
        if heads_filter is not None and heads not in heads_filter:
            skipped += 1
            continue

        if optimizers_filter is not None and opt not in optimizers_filter:
            skipped += 1
            continue

        if hoyer_coeff_min is not None and hoyer_coeff < hoyer_coeff_min:
            skipped += 1
            continue

        if hoyer_coeff_max is not None and hoyer_coeff > hoyer_coeff_max:
            skipped += 1
            continue

        # Filter by no_qknorm flag
        run_no_qknorm = config.get('no_qknorm', False)
        if no_qknorm:
            # Only include runs WITH no_qknorm=True
            if not run_no_qknorm:
                skipped += 1
                continue
        else:
            # Exclude runs with no_qknorm=True (default behavior)
            if run_no_qknorm:
                skipped += 1
                continue

        # Check completion (unless allow_incomplete is True)
        iterations_config = config.get('iterations')
        actual_iter = summary.get('iter')

        if actual_iter is None or iterations_config is None:
            skipped += 1
            continue

        # Allow runs that completed at least 80% of iterations (unless allow_incomplete)
        if not allow_incomplete and actual_iter < iterations_config * 0.8:
            print(f"  Skipping {run.name} - incomplete ({actual_iter}/{iterations_config})")
            skipped += 1
            continue

        # Fetch validation loss and hoyer loss history with step information
        try:
            # Use scan_history to get actual step values along with loss data
            scan_history = list(run.scan_history(keys=["val/loss", "val/hoyer_loss", "_step"]))

            if len(scan_history) == 0:
                print(f"  Skipping {run.name} - no validation data")
                skipped += 1
                continue

            # Extract aligned data (only rows where all three values are present)
            aligned_data = []
            for row in scan_history:
                step = row.get('_step')
                val_loss = row.get('val/loss')
                hoyer_loss = row.get('val/hoyer_loss')
                if step is not None and val_loss is not None and hoyer_loss is not None:
                    aligned_data.append((step, val_loss, hoyer_loss))

            if len(aligned_data) == 0:
                print(f"  Skipping {run.name} - no aligned validation data")
                skipped += 1
                continue

            # Unpack aligned data
            iterations = np.array([d[0] for d in aligned_data])
            val_losses = np.array([d[1] for d in aligned_data])
            hoyer_losses = np.array([d[2] for d in aligned_data])

            # Calculate cross-entropy loss: val/loss - hoyer_coeff * val/hoyer_loss
            ce_losses = val_losses - hoyer_coeff * hoyer_losses

            # Get batch size and seq_len for compute calculation
            local_batch_size = config.get('batch_size', 1)
            world_size = config.get('world_size', 1)
            acc_steps = config.get('acc_steps', 1)
            seq_len = config.get('sequence_length', config.get('seq_len', 512))
            global_batch_size = local_batch_size * world_size * acc_steps

            print(f"  Loaded {run.name}: heads={heads}, opt={opt}, hoyer_coeff={hoyer_coeff:.1e}, points={len(val_losses)}")

            # Get learning rate
            lr = config.get('lr', config.get('learning_rate', None))

            run_data_list.append({
                'config': config,
                'heads': heads,
                'optimizer': opt,
                'hoyer_coeff': hoyer_coeff,
                'lr': lr,
                'val_losses': val_losses,
                'hoyer_losses': hoyer_losses,
                'ce_losses': ce_losses,
                'iterations': iterations,
                'run_name': run.name,
                'batch_size': global_batch_size,
                'seq_len': seq_len,
            })

        except Exception as e:
            print(f"  Failed to fetch history for {run.name}: {e}")
            skipped += 1
            continue

    print(f"\nProcessed {total_runs} runs, skipped {skipped}, kept {len(run_data_list)} runs")
    return run_data_list

# =============================================================================
# PLOTTING
# =============================================================================

def get_log_color(value, val_min, val_max):
    """
    Get color from plasma colormap based on log(value).
    Uses 0.0 to 0.8 of colormap range.
    """
    if val_min == val_max:
        return plt.cm.plasma(0.4)

    log_val = np.log10(value)
    log_min = np.log10(val_min)
    log_max = np.log10(val_max)

    # Normalize to [0, 1]
    normalized = (log_val - log_min) / (log_max - log_min)

    # Map to [0.0, 0.8] of colormap
    cmap_val = normalized * 0.8

    return plt.cm.plasma(cmap_val)


def get_hoyer_color(hoyer_coeff, coeff_min, coeff_max):
    """
    Get color from plasma colormap based on log(hoyer_coeff).
    Uses 0.0 to 0.8 of colormap range.
    """
    return get_log_color(hoyer_coeff, coeff_min, coeff_max)

def plot_ce_loss_curves(run_data_list, output_filename, show_inset=False, color_by='hoyer'):
    """
    Create plot of cross-entropy loss curves on log-log axes.

    Args:
        run_data_list: List of run data dictionaries
        output_filename: Output PDF filename
        show_inset: If True, add inset showing final [0.1, 1.0] portion of compute axis
        color_by: 'hoyer' to color by hoyer_loss_coeff, 'lr' to color by learning rate
    """
    print(f"\nCreating cross-entropy loss visualization with {len(run_data_list)} curves (color_by={color_by})")

    if len(run_data_list) == 0:
        print("No data to plot!")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Get range of values for coloring
    if color_by == 'lr':
        all_values = [r['lr'] for r in run_data_list if r['lr'] is not None]
        if len(all_values) == 0:
            print("Warning: No learning rate data available, falling back to hoyer")
            color_by = 'hoyer'
            all_values = [r['hoyer_coeff'] for r in run_data_list]
        color_label = 'Learning rate'
    else:
        all_values = [r['hoyer_coeff'] for r in run_data_list]
        color_label = 'Hoyer loss coefficient'

    val_min = min(all_values)
    val_max = max(all_values)

    # Sort by heads, then optimizer, then hoyer_coeff for consistent ordering
    sorted_runs = sorted(run_data_list, key=lambda x: (x['heads'], x['optimizer'], x['hoyer_coeff']))

    # Plot each curve
    for run_data in sorted_runs:
        heads = run_data['heads']
        opt = run_data['optimizer']
        hoyer_coeff = run_data['hoyer_coeff']
        ce_losses = run_data['ce_losses']
        iterations = run_data['iterations']

        # Calculate compute (FLOPs)
        flops_per_iter = compute_flops_per_iteration(
            heads,
            batch_size=run_data['batch_size'],
            seq_len=run_data['seq_len']
        )
        compute_flops = np.float64(iterations) * np.float64(flops_per_iter)

        # Convert to petaflop-days
        flops_per_petaflop_day = 1e15 * 86400
        compute_petaflop_days = compute_flops / flops_per_petaflop_day

        # Skip zero compute points
        valid_idx = compute_petaflop_days > 0
        compute_petaflop_days = compute_petaflop_days[valid_idx]
        ce_losses = ce_losses[valid_idx]

        if len(compute_petaflop_days) == 0:
            continue

        # Get styling
        if color_by == 'lr' and run_data['lr'] is not None:
            color = get_log_color(run_data['lr'], val_min, val_max)
        else:
            color = get_log_color(hoyer_coeff, val_min, val_max)
        linestyle = OPTIMIZER_LINESTYLES.get(opt, '-')
        opt_name = OPTIMIZER_NAMES.get(opt, opt)

        # Get total params for label
        params = compute_qwen3_params(heads)
        total_params = params['total_params']

        if color_by == 'lr' and run_data['lr'] is not None:
            label = f'h={heads} ({total_params/1e6:.0f}M), {opt_name}, lr={run_data["lr"]:.0e}'
        else:
            label = f'h={heads} ({total_params/1e6:.0f}M), {opt_name}, coeff={hoyer_coeff:.0e}'

        ax.plot(compute_petaflop_days, ce_losses,
                color=color,
                linestyle=linestyle,
                alpha=0.8, linewidth=2.0,
                label=label)

    # Formatting
    ax.set_xlabel('Compute (PetaFLOP-days)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Cross-Entropy Loss', fontsize=18, fontweight='bold')
    ax.set_title('Qwen3 Hoyer Experiments - Cross-Entropy Loss Curves', fontsize=20, fontweight='bold')

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set y-axis ticks at every 0.1 (2.9, 3.0, 3.1, etc.)
    from matplotlib.ticker import FixedLocator, FixedFormatter

    # Get y-axis limits and create ticks at 0.1 intervals
    y_min, y_max = ax.get_ylim()
    # Round to nearest 0.1
    tick_min = np.floor(y_min * 10) / 10
    tick_max = np.ceil(y_max * 10) / 10
    y_ticks = np.arange(tick_min, tick_max + 0.05, 0.1)
    # Filter to only include ticks within the data range
    y_ticks = y_ticks[(y_ticks >= y_min * 0.95) & (y_ticks <= y_max * 1.05)]

    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f'{t:.1f}' for t in y_ticks]))

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    # Create custom legend entries for line styles
    from matplotlib.lines import Line2D

    # Add line style legend
    style_handles = []
    for opt, ls in OPTIMIZER_LINESTYLES.items():
        if any(r['optimizer'] == opt for r in run_data_list):
            line = Line2D([0], [0], color='gray', linestyle=ls, linewidth=2)
            style_handles.append((line, OPTIMIZER_NAMES.get(opt, opt)))

    # Add colorbar with scientific notation labels
    log_min = np.log10(val_min)
    log_max = np.log10(val_max)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                               norm=plt.Normalize(vmin=log_min, vmax=log_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(color_label, fontsize=14)

    # Create scientific notation labels (1e0, 1e-1, etc.)
    tick_positions = np.arange(np.ceil(log_min), np.floor(log_max) + 1)
    tick_labels = [f'1e{int(t)}' for t in tick_positions]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)

    # Add horizontal lines for actual data points
    if color_by == 'lr':
        actual_log_vals = sorted(set(np.log10(r['lr']) for r in run_data_list if r['lr'] is not None))
    else:
        actual_log_vals = sorted(set(np.log10(r['hoyer_coeff']) for r in run_data_list))
    for log_val in actual_log_vals:
        # Normalize to colorbar position
        if log_max > log_min:
            pos = (log_val - log_min) / (log_max - log_min)
            cbar.ax.axhline(y=pos, color='white', linewidth=1.5, alpha=0.8)
            cbar.ax.axhline(y=pos, color='black', linewidth=0.5, alpha=0.8)

    # Legend with line styles only - place outside plot (reduced whitespace)
    if style_handles:
        legend_handles, legend_labels = zip(*style_handles)
        ax.legend(legend_handles, legend_labels, loc='upper left',
                 bbox_to_anchor=(1.02, 1.0), fontsize=12,
                 title='Optimizer', title_fontsize=12, framealpha=0.9)

    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    # Add inset if requested
    if show_inset:
        # Create inset axes - half width and half height, positioned in upper right
        ax_inset = fig.add_axes([0.32, 0.42, 0.50, 0.50])  # [left, bottom, width, height]

        # Find the compute range for inset (last 10% of compute range, normalized 0.1-1.0)
        all_max_compute = []
        all_inset_losses = []  # Track losses in the inset region for y-axis limits
        for run_data in sorted_runs:
            heads = run_data['heads']
            iterations = run_data['iterations']
            flops_per_iter = compute_flops_per_iteration(
                heads,
                batch_size=run_data['batch_size'],
                seq_len=run_data['seq_len']
            )
            compute_flops = np.float64(iterations) * np.float64(flops_per_iter)
            flops_per_petaflop_day = 1e15 * 86400
            compute_petaflop_days = compute_flops / flops_per_petaflop_day
            valid_idx = compute_petaflop_days > 0
            if np.any(valid_idx):
                all_max_compute.append(np.max(compute_petaflop_days[valid_idx]))

        if all_max_compute:
            max_compute = max(all_max_compute)
            inset_x_min = 0.1 * max_compute
            inset_x_max = max_compute

            # Plot each curve in the inset and collect y values for limits
            for run_data in sorted_runs:
                heads = run_data['heads']
                opt = run_data['optimizer']
                hoyer_coeff = run_data['hoyer_coeff']
                ce_losses = run_data['ce_losses']
                iterations = run_data['iterations']

                flops_per_iter = compute_flops_per_iteration(
                    heads,
                    batch_size=run_data['batch_size'],
                    seq_len=run_data['seq_len']
                )
                compute_flops = np.float64(iterations) * np.float64(flops_per_iter)
                flops_per_petaflop_day = 1e15 * 86400
                compute_petaflop_days = compute_flops / flops_per_petaflop_day

                valid_idx = compute_petaflop_days > 0
                compute_petaflop_days = compute_petaflop_days[valid_idx]
                ce_losses_plot = ce_losses[valid_idx]

                if len(compute_petaflop_days) == 0:
                    continue

                # Collect losses in the inset x-range for y-axis limits
                inset_mask = (compute_petaflop_days >= inset_x_min) & (compute_petaflop_days <= inset_x_max)
                if np.any(inset_mask):
                    all_inset_losses.extend(ce_losses_plot[inset_mask])

                if color_by == 'lr' and run_data['lr'] is not None:
                    color = get_log_color(run_data['lr'], val_min, val_max)
                else:
                    color = get_log_color(hoyer_coeff, val_min, val_max)
                linestyle = OPTIMIZER_LINESTYLES.get(opt, '-')

                ax_inset.plot(compute_petaflop_days, ce_losses_plot,
                             color=color,
                             linestyle=linestyle,
                             alpha=1.0, linewidth=0.5)

            ax_inset.set_xlim(inset_x_min, inset_x_max)
            ax_inset.set_xscale('log')
            ax_inset.set_yscale('log')

            # Set y-axis limits to just fit the data in the inset region
            if all_inset_losses:
                y_min = min(all_inset_losses)
                y_max = max(all_inset_losses)
                # Add small padding (2%)
                y_range = y_max - y_min
                ax_inset.set_ylim(y_min - 0.02 * y_range, y_max + 0.02 * y_range)
                inset_y_min = y_min - 0.02 * y_range
                inset_y_max = y_max + 0.02 * y_range
            else:
                inset_y_min, inset_y_max = ax_inset.get_ylim()

            ax_inset.tick_params(axis='both', which='major', labelsize=8)
            ax_inset.grid(True, alpha=0.3, which='both', linestyle='--')

            # Draw black rectangle on main plot around the region being enlarged
            from matplotlib.patches import Rectangle
            rect = Rectangle((inset_x_min, inset_y_min),
                            inset_x_max - inset_x_min,
                            inset_y_max - inset_y_min,
                            fill=False, edgecolor='black', linewidth=1.5,
                            linestyle='-', zorder=10)
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Cross-entropy loss plot saved as {output_filename}")

    plt.close()

def plot_hoyer_loss_curves(run_data_list, output_filename, show_inset=False, color_by='hoyer'):
    """
    Create plot of (hoyer_loss - 1) curves on log-log axes.

    Args:
        run_data_list: List of run data dictionaries
        output_filename: Output PDF filename
        show_inset: If True, add inset showing final [0.1, 1.0] portion of compute axis
        color_by: 'hoyer' to color by hoyer_loss_coeff, 'lr' to color by learning rate
    """
    print(f"\nCreating hoyer loss visualization with {len(run_data_list)} curves (color_by={color_by})")

    if len(run_data_list) == 0:
        print("No data to plot!")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Get range of values for coloring
    if color_by == 'lr':
        all_values = [r['lr'] for r in run_data_list if r['lr'] is not None]
        if len(all_values) == 0:
            print("Warning: No learning rate data available, falling back to hoyer")
            color_by = 'hoyer'
            all_values = [r['hoyer_coeff'] for r in run_data_list]
        color_label = 'Learning rate'
    else:
        all_values = [r['hoyer_coeff'] for r in run_data_list]
        color_label = 'Hoyer loss coefficient'

    val_min = min(all_values)
    val_max = max(all_values)

    # Sort by heads, then optimizer, then hoyer_coeff
    sorted_runs = sorted(run_data_list, key=lambda x: (x['heads'], x['optimizer'], x['hoyer_coeff']))

    # Plot each curve
    for run_data in sorted_runs:
        heads = run_data['heads']
        opt = run_data['optimizer']
        hoyer_coeff = run_data['hoyer_coeff']
        hoyer_losses = run_data['hoyer_losses']
        iterations = run_data['iterations']

        # Calculate (hoyer_loss - 1)
        hoyer_minus_one = hoyer_losses - 1.0

        # Calculate compute (FLOPs)
        flops_per_iter = compute_flops_per_iteration(
            heads,
            batch_size=run_data['batch_size'],
            seq_len=run_data['seq_len']
        )
        compute_flops = np.float64(iterations) * np.float64(flops_per_iter)

        # Convert to petaflop-days
        flops_per_petaflop_day = 1e15 * 86400
        compute_petaflop_days = compute_flops / flops_per_petaflop_day

        # Skip zero/negative values for log scale
        valid_idx = (compute_petaflop_days > 0) & (hoyer_minus_one > 0)
        compute_petaflop_days = compute_petaflop_days[valid_idx]
        hoyer_minus_one = hoyer_minus_one[valid_idx]

        if len(compute_petaflop_days) == 0:
            continue

        # Get styling
        if color_by == 'lr' and run_data['lr'] is not None:
            color = get_log_color(run_data['lr'], val_min, val_max)
        else:
            color = get_log_color(hoyer_coeff, val_min, val_max)
        linestyle = OPTIMIZER_LINESTYLES.get(opt, '-')
        opt_name = OPTIMIZER_NAMES.get(opt, opt)

        # Get total params for label
        params = compute_qwen3_params(heads)
        total_params = params['total_params']

        if color_by == 'lr' and run_data['lr'] is not None:
            label = f'h={heads} ({total_params/1e6:.0f}M), {opt_name}, lr={run_data["lr"]:.0e}'
        else:
            label = f'h={heads} ({total_params/1e6:.0f}M), {opt_name}, coeff={hoyer_coeff:.0e}'

        ax.plot(compute_petaflop_days, hoyer_minus_one,
                color=color,
                linestyle=linestyle,
                alpha=0.8, linewidth=2.0,
                label=label)

    # Formatting
    ax.set_xlabel('Compute (PetaFLOP-days)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Hoyer Loss - 1', fontsize=18, fontweight='bold')
    ax.set_title('Qwen3 Hoyer Experiments - Hoyer Loss Curves', fontsize=20, fontweight='bold')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    # Create custom legend entries for line styles
    from matplotlib.lines import Line2D

    style_handles = []
    for opt, ls in OPTIMIZER_LINESTYLES.items():
        if any(r['optimizer'] == opt for r in run_data_list):
            line = Line2D([0], [0], color='gray', linestyle=ls, linewidth=2)
            style_handles.append((line, OPTIMIZER_NAMES.get(opt, opt)))

    # Add colorbar with scientific notation labels
    log_min = np.log10(val_min)
    log_max = np.log10(val_max)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                               norm=plt.Normalize(vmin=log_min, vmax=log_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(color_label, fontsize=14)

    # Create scientific notation labels (1e0, 1e-1, etc.)
    tick_positions = np.arange(np.ceil(log_min), np.floor(log_max) + 1)
    tick_labels = [f'1e{int(t)}' for t in tick_positions]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)

    # Add horizontal lines for actual data points
    if color_by == 'lr':
        actual_log_vals = sorted(set(np.log10(r['lr']) for r in run_data_list if r['lr'] is not None))
    else:
        actual_log_vals = sorted(set(np.log10(r['hoyer_coeff']) for r in run_data_list))
    for log_val in actual_log_vals:
        # Normalize to colorbar position
        if log_max > log_min:
            pos = (log_val - log_min) / (log_max - log_min)
            cbar.ax.axhline(y=pos, color='white', linewidth=1.5, alpha=0.8)
            cbar.ax.axhline(y=pos, color='black', linewidth=0.5, alpha=0.8)

    # Legend with line styles only - place outside plot (reduced whitespace)
    if style_handles:
        legend_handles, legend_labels = zip(*style_handles)
        ax.legend(legend_handles, legend_labels, loc='upper left',
                 bbox_to_anchor=(1.02, 1.0), fontsize=12,
                 title='Optimizer', title_fontsize=12, framealpha=0.9)

    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    # Add inset if requested
    if show_inset:
        # Create inset axes - half width and half height, positioned in upper right
        ax_inset = fig.add_axes([0.52, 0.52, 0.35, 0.35])  # [left, bottom, width, height]

        # Find the compute range for inset (last 10% of compute range, normalized 0.1-1.0)
        all_max_compute = []
        all_inset_losses = []  # Track losses in the inset region for y-axis limits
        for run_data in sorted_runs:
            heads = run_data['heads']
            iterations = run_data['iterations']
            flops_per_iter = compute_flops_per_iteration(
                heads,
                batch_size=run_data['batch_size'],
                seq_len=run_data['seq_len']
            )
            compute_flops = np.float64(iterations) * np.float64(flops_per_iter)
            flops_per_petaflop_day = 1e15 * 86400
            compute_petaflop_days = compute_flops / flops_per_petaflop_day
            valid_idx = compute_petaflop_days > 0
            if np.any(valid_idx):
                all_max_compute.append(np.max(compute_petaflop_days[valid_idx]))

        if all_max_compute:
            max_compute = max(all_max_compute)
            inset_x_min = 0.1 * max_compute
            inset_x_max = max_compute

            # Plot each curve in the inset and collect y values for limits
            for run_data in sorted_runs:
                heads = run_data['heads']
                opt = run_data['optimizer']
                hoyer_coeff = run_data['hoyer_coeff']
                hoyer_losses = run_data['hoyer_losses']
                iterations = run_data['iterations']

                # Calculate (hoyer_loss - 1)
                hoyer_minus_one = hoyer_losses - 1.0

                flops_per_iter = compute_flops_per_iteration(
                    heads,
                    batch_size=run_data['batch_size'],
                    seq_len=run_data['seq_len']
                )
                compute_flops = np.float64(iterations) * np.float64(flops_per_iter)
                flops_per_petaflop_day = 1e15 * 86400
                compute_petaflop_days = compute_flops / flops_per_petaflop_day

                valid_idx = (compute_petaflop_days > 0) & (hoyer_minus_one > 0)
                compute_petaflop_days = compute_petaflop_days[valid_idx]
                hoyer_minus_one = hoyer_minus_one[valid_idx]

                if len(compute_petaflop_days) == 0:
                    continue

                # Collect losses in the inset x-range for y-axis limits
                inset_mask = (compute_petaflop_days >= inset_x_min) & (compute_petaflop_days <= inset_x_max)
                if np.any(inset_mask):
                    all_inset_losses.extend(hoyer_minus_one[inset_mask])

                if color_by == 'lr' and run_data['lr'] is not None:
                    color = get_log_color(run_data['lr'], val_min, val_max)
                else:
                    color = get_log_color(hoyer_coeff, val_min, val_max)
                linestyle = OPTIMIZER_LINESTYLES.get(opt, '-')

                ax_inset.plot(compute_petaflop_days, hoyer_minus_one,
                             color=color,
                             linestyle=linestyle,
                             alpha=0.8, linewidth=1.5)

            ax_inset.set_xlim(inset_x_min, inset_x_max)
            ax_inset.set_xscale('log')
            ax_inset.set_yscale('log')

            # Set y-axis limits to just fit the data in the inset region
            if all_inset_losses:
                y_min = min(all_inset_losses)
                y_max = max(all_inset_losses)
                # Add small padding (2%)
                y_range = y_max - y_min
                ax_inset.set_ylim(y_min - 0.02 * y_range, y_max + 0.02 * y_range)
                inset_y_min = y_min - 0.02 * y_range
                inset_y_max = y_max + 0.02 * y_range
            else:
                inset_y_min, inset_y_max = ax_inset.get_ylim()

            ax_inset.tick_params(axis='both', which='major', labelsize=8)
            ax_inset.grid(True, alpha=0.3, which='both', linestyle='--')

            # Draw black rectangle on main plot around the region being enlarged
            from matplotlib.patches import Rectangle
            rect = Rectangle((inset_x_min, inset_y_min),
                            inset_x_max - inset_x_min,
                            inset_y_max - inset_y_min,
                            fill=False, edgecolor='black', linewidth=1.5,
                            linestyle='-', zorder=10)
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Hoyer loss plot saved as {output_filename}")

    plt.close()

# =============================================================================
# MAIN
# =============================================================================

def generate_output_filename(base_name, heads, optimizers):
    """Generate output filename including heads and optimizers chosen."""
    parts = [base_name]

    if heads:
        heads_str = '_'.join(str(h) for h in sorted(heads))
        parts.append(f'h{heads_str}')

    if optimizers:
        # Use short names for optimizers
        opt_short = {
            'adamw': 'adamw',
            'dana-star-mk4': 'starmk4',
            'dana-mk4': 'mk4'
        }
        opts_str = '_'.join(opt_short.get(o, o) for o in sorted(optimizers))
        parts.append(opts_str)

    return '_'.join(parts) + '.pdf'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Hoyer loss curves for Qwen3 experiments')
    parser.add_argument('--project', type=str, default='danastar',
                       help='WandB project name (default: danastar)')
    parser.add_argument('--entity', type=str, default='ep-rmt-ml-opt',
                       help='WandB entity name (default: ep-rmt-ml-opt)')
    parser.add_argument('--heads', type=int, nargs='+', default=None,
                       help='Head counts to include (default: all)')
    parser.add_argument('--optimizers', type=str, nargs='+', default=None,
                       choices=['adamw', 'dana-star-mk4', 'dana-mk4'],
                       help='Optimizers to include (default: all)')
    parser.add_argument('--hoyer-coeff-min', type=float, default=None,
                       help='Minimum hoyer_loss_coeff to include (default: no min)')
    parser.add_argument('--hoyer-coeff-max', type=float, default=None,
                       help='Maximum hoyer_loss_coeff to include (default: no max)')
    parser.add_argument('--allow-incomplete', action='store_true',
                       help='Include runs that have not completed (default: require 80%% completion)')
    parser.add_argument('--no-qknorm', action='store_true',
                       help='Include only runs with no_qknorm=True (default: exclude those runs)')
    parser.add_argument('--inset', action='store_true',
                       help='Add inset showing final [0.1, 1.0] portion of compute axis')
    parser.add_argument('--output-ce', type=str, default=None,
                       help='Output filename for cross-entropy loss plot (default: auto-generated)')
    parser.add_argument('--output-hoyer', type=str, default=None,
                       help='Output filename for hoyer loss plot (default: auto-generated)')
    parser.add_argument('--color-by-lr', action='store_true',
                       help='Color curves by learning rate instead of hoyer_loss_coeff')
    args = parser.parse_args()

    # Generate default output filenames if not specified
    if args.output_ce is None:
        args.output_ce = generate_output_filename('hoyer_ce_loss', args.heads, args.optimizers)
    if args.output_hoyer is None:
        args.output_hoyer = generate_output_filename('hoyer_loss', args.heads, args.optimizers)

    print("=" * 70)
    print("Hoyer Loss Curves Visualization")
    print("=" * 70)
    print(f"WandB Group: {WANDB_GROUP}")
    print(f"Heads filter: {args.heads if args.heads else 'all'}")
    print(f"Optimizers filter: {args.optimizers if args.optimizers else 'all'}")
    print(f"Hoyer coeff range: [{args.hoyer_coeff_min}, {args.hoyer_coeff_max}]")
    print(f"Allow incomplete runs: {args.allow_incomplete}")
    print(f"No QK normalization filter: {args.no_qknorm}")
    print(f"Show inset: {args.inset}")
    print(f"Color by: {'learning rate' if args.color_by_lr else 'hoyer coefficient'}")
    print(f"Output CE file: {args.output_ce}")
    print(f"Output Hoyer file: {args.output_hoyer}")
    print("=" * 70)

    # Load data
    run_data_list = load_hoyer_data(
        args.project,
        args.entity,
        heads_filter=args.heads,
        optimizers_filter=args.optimizers,
        hoyer_coeff_min=args.hoyer_coeff_min,
        hoyer_coeff_max=args.hoyer_coeff_max,
        allow_incomplete=args.allow_incomplete,
        no_qknorm=args.no_qknorm
    )

    if len(run_data_list) == 0:
        print("ERROR: No data found. Check parameters.")
    else:
        # Determine color_by value
        color_by = 'lr' if args.color_by_lr else 'hoyer'

        # Create cross-entropy loss plot
        plot_ce_loss_curves(run_data_list, args.output_ce, show_inset=args.inset, color_by=color_by)

        # Create hoyer loss plot
        plot_hoyer_loss_curves(run_data_list, args.output_hoyer, show_inset=args.inset, color_by=color_by)

        print("\n" + "=" * 70)
        print("Visualization completed successfully!")
        print("=" * 70)
