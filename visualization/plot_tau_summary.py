#!/usr/bin/env python
"""
Plot Tau Statistics Summary

Creates a 2x2 summary figure showing tau CDF evolution for key parameter types:
- Embedding (wte)
- Attention
- MLP
- Output (lm_head)

Usage:
    python plot_tau_summary.py <tau_stats_dir> [--output output.pdf]
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import argparse
from pathlib import Path
import json
import re
from collections import defaultdict


def regularize_tau(tau, t):
    """Apply tau regularization formula from optimizers.py."""
    tau = np.asarray(tau)
    tau_clipped = np.minimum(tau, 0.5)
    tau_reg = np.maximum(
        tau_clipped / (1.0 - tau_clipped),
        1.0 / (1.0 + t)
    )
    return tau_reg


def transform_prob_to_yaxis(prob):
    """Transform probability to y-axis value using logit transformation with tail compression."""
    prob = np.asarray(prob)
    upper_threshold = 0.999
    lower_threshold = 0.001
    y_upper_threshold = np.log10(upper_threshold / (1 - upper_threshold))
    y_lower_threshold = np.log10(lower_threshold / (1 - lower_threshold))

    y = np.zeros_like(prob, dtype=float)

    mask_lower_tail = prob < lower_threshold
    y[mask_lower_tail] = y_lower_threshold - 1 + (prob[mask_lower_tail] / lower_threshold)

    mask_middle = (prob >= lower_threshold) & (prob <= upper_threshold)
    prob_middle = np.clip(prob[mask_middle], 1e-10, 1 - 1e-10)
    y[mask_middle] = np.log10(prob_middle / (1 - prob_middle))

    mask_upper_tail = prob > upper_threshold
    y[mask_upper_tail] = y_upper_threshold + (prob[mask_upper_tail] - upper_threshold) / (1 - upper_threshold)

    return y


def format_yaxis_label(y, pos):
    """Format y-axis labels to show percentages."""
    upper_threshold = 0.999
    lower_threshold = 0.001
    y_upper_threshold = np.log10(upper_threshold / (1 - upper_threshold))
    y_lower_threshold = np.log10(lower_threshold / (1 - lower_threshold))

    if y < y_lower_threshold:
        prob = (y - (y_lower_threshold - 1)) * lower_threshold
    elif y > y_upper_threshold:
        prob = upper_threshold + (y - y_upper_threshold) * (1 - upper_threshold)
    else:
        prob = 1.0 / (1.0 + 10**(-y))

    prob = np.clip(prob, 0.0, 1.0)

    if prob >= 0.9999:
        return '100%'
    elif prob < 0.01 or prob > 0.99:
        return f'{prob*100:.2f}%'
    else:
        return f'{prob*100:.1f}%'


def reconstruct_single_cdf(smallest, largest, n):
    """Reconstruct CDF from order statistics."""
    if len(smallest) == 0 and len(largest) == 0:
        return np.array([]), np.array([])

    num_stats = max(len(smallest), len(largest))
    max_k = int(np.ceil(np.log(n * 10) / np.log(1.1)))
    indices = np.int32(1.1 ** np.arange(max_k + 1)) - 1
    indices = np.unique(indices)
    indices = indices[indices < n]
    indices = indices[:num_stats]

    tau_values = []
    cdf_values = []

    for i in range(len(smallest)):
        if i >= len(indices):
            break
        tau = smallest[i]
        position = indices[i]
        cdf_prob = (position + 1) / n
        tau_values.append(tau)
        cdf_values.append(cdf_prob)

    for i in range(len(largest)):
        if i >= len(indices):
            break
        tau = largest[i]
        reversed_position = n - 1 - indices[i]
        cdf_prob = (reversed_position + 1) / n
        tau_values.append(tau)
        cdf_values.append(cdf_prob)

    tau_values = np.array(tau_values)
    cdf_values = np.array(cdf_values)

    sort_idx = np.argsort(tau_values)
    return tau_values[sort_idx], cdf_values[sort_idx]


def merge_cdfs(cdfs_list):
    """Merge multiple CDFs by interpolating and averaging."""
    if len(cdfs_list) == 0:
        return {'tau_values': np.array([]), 'cdf_values': np.array([])}

    all_tau_values = []
    for tau_vals, _ in cdfs_list:
        all_tau_values.extend(tau_vals)
    all_tau_values = np.sort(np.unique(all_tau_values))

    interpolated_cdfs = []
    for tau_vals, cdf_vals in cdfs_list:
        interp_cdf = np.interp(
            all_tau_values,
            tau_vals,
            cdf_vals,
            left=0.0,
            right=1.0
        )
        interpolated_cdfs.append(interp_cdf)

    averaged_cdf = np.mean(interpolated_cdfs, axis=0)

    return {
        'tau_values': all_tau_values,
        'cdf_values': averaged_cdf
    }


def normalize_param_name(param_name):
    """Normalize parameter name by removing transformer block numbers."""
    pattern = r'\.h\.\d+\.'

    if re.search(pattern, param_name):
        normalized = re.sub(pattern, '.', param_name)
        normalized = normalized.replace('module.transformer.', '')
        return f"Transformer {normalized}"
    else:
        normalized = param_name.replace('module.transformer.', '')
        return f"Transformer {normalized}"


def group_parameters_by_type(param_names):
    """Group parameter names by their normalized type."""
    groups = defaultdict(list)
    for param_name in param_names:
        normalized = normalize_param_name(param_name)
        groups[normalized].append(param_name)
    return groups


def load_tau_stats_directory(tau_stats_dir):
    """Load all tau statistics from a directory."""
    tau_stats_dir = Path(tau_stats_dir)

    metadata_file = tau_stats_dir / "tau_stats_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    pkl_files = sorted(tau_stats_dir.glob("tau_stats_iter_*.pkl"))

    tau_stats_data = {}
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        iteration = data['metadata']['iteration']
        tau_stats_data[iteration] = data

    available_iterations = sorted(tau_stats_data.keys())

    return metadata, tau_stats_data, available_iterations


def plot_cdf_for_group(ax, tau_stats_data, param_group, timesteps_data, title):
    """Plot aggregated dual-tail CDF for a parameter group across selected timesteps."""
    for iteration, eval_step, color in timesteps_data:
        if iteration not in tau_stats_data:
            continue

        cdfs_list = []
        for param_name in param_group:
            param_data = tau_stats_data[iteration]['tau_statistics'].get(param_name)
            if param_data is None:
                continue

            largest = param_data['largest_order_stats']
            smallest = param_data['smallest_order_stats']
            n = param_data['num_elements']

            tau_values, cdf_values = reconstruct_single_cdf(smallest, largest, n)

            if len(tau_values) > 0:
                cdfs_list.append((tau_values, cdf_values))

        if len(cdfs_list) == 0:
            continue

        merged_cdf = merge_cdfs(cdfs_list)
        tau_values = merged_cdf['tau_values']
        cdf_values = merged_cdf['cdf_values']

        if len(tau_values) == 0:
            continue

        # Apply tau regularization
        tau_values = regularize_tau(tau_values, iteration)

        # Transform probabilities to y-axis values
        y_vals = transform_prob_to_yaxis(cdf_values)

        # Plot
        tokens = iteration * 2048 * 8  # batch_size=8, seq_len=2048
        tokens_str = f"{tokens/1e9:.1f}B" if tokens >= 1e9 else f"{tokens/1e6:.0f}M"
        ax.plot(tau_values, y_vals,
                label=f"t={tokens_str}",
                color=color, alpha=0.8, linewidth=1.5)

    ax.set_xlabel(r'$\tilde{\tau}$ (regularized)', fontsize=14)
    ax.set_ylabel('CDF', fontsize=14)
    ax.set_xscale('log')
    ax.set_title(title, fontsize=16)

    # Set up custom y-axis ticks
    ytick_probs = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    ytick_vals = transform_prob_to_yaxis(ytick_probs)
    ax.set_yticks(ytick_vals)
    ax.yaxis.set_major_formatter(FuncFormatter(format_yaxis_label))

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=10, loc='upper left')
    ax.tick_params(axis='both', which='major', labelsize=12)


def main():
    parser = argparse.ArgumentParser(description="Plot tau statistics summary")
    parser.add_argument("tau_stats_dir", type=str, help="Path to tau_stats directory")
    parser.add_argument("--output", type=str, default="results/tau_stats_summary.pdf",
                       help="Output PDF file")
    parser.add_argument("--figsize", type=float, nargs=2, default=[12, 10],
                       help="Figure size (width height)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")

    args = parser.parse_args()

    tau_stats_dir = Path(args.tau_stats_dir)

    if not tau_stats_dir.exists():
        print(f"Error: {tau_stats_dir} does not exist")
        return

    print(f"Loading tau statistics from: {tau_stats_dir}")
    metadata, tau_stats_data, available_iterations = load_tau_stats_directory(tau_stats_dir)

    print(f"Loaded {len(tau_stats_data)} timesteps")
    print(f"Available iterations: {available_iterations}")

    # Get parameter groups
    first_iter = available_iterations[0]
    all_param_names = sorted(tau_stats_data[first_iter]['tau_statistics'].keys())
    param_groups = group_parameters_by_type(all_param_names)

    # Select key groups for 2x2 figure
    # Map to display names
    group_mapping = {
        'Embedding': 'Transformer wte.weight',
        'Attention': 'Transformer attn.c_attn.weight',
        'MLP': 'Transformer mlp.c_fc.weight',
        'Output': 'Transformer module.lm_head.weight',
    }

    # Select subset of timesteps for cleaner visualization
    # Use early, mid, and late training
    selected_iterations = [
        available_iterations[0],   # early
        available_iterations[len(available_iterations)//3],  # early-mid
        available_iterations[2*len(available_iterations)//3],  # mid-late
        available_iterations[-1],  # late
    ]
    print(f"Plotting iterations: {selected_iterations}")

    # Create color map
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(selected_iterations)))

    timesteps_data = []
    for iteration, color in zip(selected_iterations, colors):
        eval_step = tau_stats_data[iteration]['metadata']['eval_step_number']
        timesteps_data.append((iteration, eval_step, color))

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=tuple(args.figsize))
    axes = axes.flatten()

    for idx, (display_name, group_key) in enumerate(group_mapping.items()):
        if group_key not in param_groups:
            print(f"Warning: {group_key} not found in parameter groups")
            continue

        param_group = param_groups[group_key]
        n_tensors = len(param_group)
        title = f"{display_name} ({n_tensors} tensors)"

        plot_cdf_for_group(axes[idx], tau_stats_data, param_group, timesteps_data, title)

    # Add overall title with metadata
    kappa = metadata.get('optimizer_params', {}).get('kappa', 'N/A')
    n_head = metadata.get('model_params', {}).get('n_head', 'N/A')
    fig.suptitle(rf'Dana-Star-MK4 $\tilde{{\tau}}$ Statistics (Enoki-{n_head}H, $\kappa$={kappa})',
                 fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
