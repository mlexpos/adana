#!/usr/bin/env python
"""
Plot Aggregated Tau Statistics

Creates a multi-page PDF with one CDF plot per parameter type, aggregating across
transformer blocks. For example, all "module.transformer.h.{N}.mlp.gate_proj.weight"
tensors are merged into a single "Transformer mlp.gate_proj.weight" plot.

Usage:
    python plot_aggregated_tensors.py <tau_stats_dir> [--output output.pdf] [--timesteps t1,t2,t3] [--raw-tau]
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from pathlib import Path
import json
import re
from collections import defaultdict


def regularize_tau(tau, t):
    """Apply tau regularization formula from optimizers.py.

    tau_reg = max(clip_tohalf(tau) / (1 - clip_tohalf(tau)), 1/(1+t))
    where clip_tohalf(tau) = min(tau, 0.5)

    Args:
        tau: Raw tau value(s)
        t: Timestep

    Returns:
        Regularized tau value(s)
    """
    tau = np.asarray(tau)
    tau_clipped = np.minimum(tau, 0.5)
    tau_reg = np.maximum(
        tau_clipped / (1.0 - tau_clipped),
        1.0 / (1.0 + t)
    )
    return tau_reg


def transform_prob_to_yaxis(prob):
    """Transform probability to y-axis value using logit transformation with tail compression.

    Uses logit transformation below 99.9%, then compresses the tail from 99.9% to 100%
    to exactly 1 unit, making the upper tail finite and easier to visualize.

    Key points:
    - prob < 0.001: compressed tail
    - 0.001 ≤ prob ≤ 0.999: standard logit transformation
    - prob > 0.999: compressed to fit in 1 unit above logit(0.999)

    Args:
        prob: Probability value(s) between 0 and 1

    Returns:
        Transformed y-axis value(s)
    """
    prob = np.asarray(prob)

    # Threshold for tail compression
    upper_threshold = 0.999
    lower_threshold = 0.001

    # Compute logit value at thresholds
    y_upper_threshold = np.log10(upper_threshold / (1 - upper_threshold))  # ≈ 3.0
    y_lower_threshold = np.log10(lower_threshold / (1 - lower_threshold))  # ≈ -3.0

    # Initialize output array
    y = np.zeros_like(prob, dtype=float)

    # Lower tail compression: prob < 0.001
    # Map [0, 0.001] to [y_lower_threshold - 1, y_lower_threshold]
    mask_lower_tail = prob < lower_threshold
    y[mask_lower_tail] = y_lower_threshold - 1 + (prob[mask_lower_tail] / lower_threshold)

    # Middle region: standard logit
    # Map [0.001, 0.999] using logit
    mask_middle = (prob >= lower_threshold) & (prob <= upper_threshold)
    prob_middle = np.clip(prob[mask_middle], 1e-10, 1 - 1e-10)
    y[mask_middle] = np.log10(prob_middle / (1 - prob_middle))

    # Upper tail compression: prob > 0.999
    # Map [0.999, 1.0] to [y_upper_threshold, y_upper_threshold + 1]
    mask_upper_tail = prob > upper_threshold
    y[mask_upper_tail] = y_upper_threshold + (prob[mask_upper_tail] - upper_threshold) / (1 - upper_threshold)

    return y


def format_yaxis_label(y, pos):
    """Format y-axis labels to show percentages.

    Inverts the modified logit transformation with tail compression.

    Args:
        y: Y-axis value (from transform_prob_to_yaxis)
        pos: Position (unused, required by FuncFormatter)

    Returns:
        Formatted string label
    """
    # Thresholds (same as in transform_prob_to_yaxis)
    upper_threshold = 0.999
    lower_threshold = 0.001
    y_upper_threshold = np.log10(upper_threshold / (1 - upper_threshold))  # ≈ 3.0
    y_lower_threshold = np.log10(lower_threshold / (1 - lower_threshold))  # ≈ -3.0

    # Inverse transformation
    if y < y_lower_threshold:
        # Lower tail: [y_lower_threshold - 1, y_lower_threshold] → [0, 0.001]
        prob = (y - (y_lower_threshold - 1)) * lower_threshold
    elif y > y_upper_threshold:
        # Upper tail: [y_upper_threshold, y_upper_threshold + 1] → [0.999, 1.0]
        prob = upper_threshold + (y - y_upper_threshold) * (1 - upper_threshold)
    else:
        # Middle region: inverse logit
        # prob = 1 / (1 + 10^(-y))
        prob = 1.0 / (1.0 + 10**(-y))

    # Clip to valid range
    prob = np.clip(prob, 0.0, 1.0)

    # Special case for 100%
    if prob >= 0.9999:
        return '100%'
    # Format based on magnitude
    elif prob < 0.01 or prob > 0.99:
        return f'{prob*100:.2f}%'
    else:
        return f'{prob*100:.1f}%'


def add_axis_break_markers(ax):
    """Add visual indicators for axis breaks at the compressed tails.

    Draws diagonal lines directly on the y-axis to indicate scale changes at:
    - Between 0% and 0.1% (lower tail)
    - Between 99.9% and 100% (upper tail)

    Args:
        ax: Matplotlib axis object
    """
    # Get y-axis positions for the breaks
    upper_threshold = 0.999
    lower_threshold = 0.001
    y_upper_threshold = np.log10(upper_threshold / (1 - upper_threshold))  # ≈ 3.0
    y_lower_threshold = np.log10(lower_threshold / (1 - lower_threshold))  # ≈ -3.0

    # Upper break: between 99.9% and 100%
    y_upper_break = y_upper_threshold + 0.5  # Middle of the compressed region

    # Lower break: between 0% and 0.1%
    y_lower_break = y_lower_threshold - 0.5  # Middle of the compressed region

    # Convert data coordinates to display coordinates for y-values
    # Use axis coordinates (0 to 1) for x-position on the axis
    trans = ax.get_yaxis_transform()  # This transform uses axis coords for x, data coords for y

    # Height of diagonal lines in data coordinates
    dy = 0.15

    # Width in axis coordinates (fraction of plot width)
    dx_axis = 0.02  # 2% of the plot width

    # Draw upper break markers (two parallel diagonal lines on the y-axis)
    for offset in [-0.1, 0.1]:
        y_start = y_upper_break + offset - dy/2
        y_end = y_upper_break + offset + dy/2
        ax.plot([0, dx_axis], [y_start, y_end],
                transform=trans, color='k', linewidth=2,
                clip_on=False, zorder=100, solid_capstyle='butt')

    # Draw lower break markers (two parallel diagonal lines on the y-axis)
    for offset in [-0.1, 0.1]:
        y_start = y_lower_break + offset - dy/2
        y_end = y_lower_break + offset + dy/2
        ax.plot([0, dx_axis], [y_start, y_end],
                transform=trans, color='k', linewidth=2,
                clip_on=False, zorder=100, solid_capstyle='butt')


def reconstruct_single_cdf(smallest, largest, n):
    """Reconstruct CDF from order statistics.

    Args:
        smallest: Array of smallest order statistics
        largest: Array of largest order statistics
        n: Total number of parameters

    Returns:
        tau_values, cdf_values arrays (both sorted by tau)
    """
    if len(smallest) == 0 and len(largest) == 0:
        return np.array([]), np.array([])

    # Generate indices used in compute_tau_order_statistics
    num_stats = max(len(smallest), len(largest))
    max_k = int(np.ceil(np.log(n * 10) / np.log(1.1)))
    indices = np.int32(1.1 ** np.arange(max_k + 1)) - 1
    indices = np.unique(indices)
    indices = indices[indices < n]  # Keep only valid indices
    indices = indices[:num_stats]   # Truncate to actual number of stats

    tau_values = []
    cdf_values = []

    # Process smallest order statistics
    for i in range(len(smallest)):
        if i >= len(indices):
            break
        tau = smallest[i]
        position = indices[i]  # 0-indexed position in sorted array
        cdf_prob = (position + 1) / n  # CDF: P(tau <= tau_value)
        tau_values.append(tau)
        cdf_values.append(cdf_prob)

    # Process largest order statistics
    for i in range(len(largest)):
        if i >= len(indices):
            break
        tau = largest[i]
        reversed_position = n - 1 - indices[i]  # Position in sorted array
        cdf_prob = (reversed_position + 1) / n  # CDF: P(tau <= tau_value)
        tau_values.append(tau)
        cdf_values.append(cdf_prob)

    tau_values = np.array(tau_values)
    cdf_values = np.array(cdf_values)

    # Sort by tau for proper CDF
    sort_idx = np.argsort(tau_values)
    return tau_values[sort_idx], cdf_values[sort_idx]


def merge_cdfs(cdfs_list):
    """Merge multiple CDFs by interpolating and averaging.

    Given a list of (tau_values, cdf_values) tuples, creates a common grid
    and averages the interpolated CDFs.

    Args:
        cdfs_list: List of (tau_values, cdf_values) tuples

    Returns:
        Dictionary with merged 'tau_values' and 'cdf_values' arrays
    """
    if len(cdfs_list) == 0:
        return {'tau_values': np.array([]), 'cdf_values': np.array([])}

    # Step 1: Collect all unique tau values to form common grid
    all_tau_values = []
    for tau_vals, _ in cdfs_list:
        all_tau_values.extend(tau_vals)
    all_tau_values = np.sort(np.unique(all_tau_values))

    # Step 2: Interpolate each CDF on common grid
    interpolated_cdfs = []
    for tau_vals, cdf_vals in cdfs_list:
        # Interpolate with extrapolation handling
        # CDF should be 0 for t < min(tau) and 1 for t > max(tau)
        interp_cdf = np.interp(
            all_tau_values,
            tau_vals,
            cdf_vals,
            left=0.0,   # CDF = 0 for tau < minimum observed
            right=1.0   # CDF = 1 for tau > maximum observed
        )
        interpolated_cdfs.append(interp_cdf)

    # Step 3: Average across CDFs
    averaged_cdf = np.mean(interpolated_cdfs, axis=0)

    return {
        'tau_values': all_tau_values,
        'cdf_values': averaged_cdf
    }


def normalize_param_name(param_name):
    """Normalize parameter name by removing transformer block numbers.

    Examples:
        module.transformer.h.3.mlp.gate_proj.weight -> Transformer mlp.gate_proj.weight
        module.transformer.h.15.attn.k_norm.weight -> Transformer attn.k_norm.weight
        module.transformer.wte.weight -> Transformer wte.weight
        module.transformer.ln_f.weight -> Transformer ln_f.weight

    Args:
        param_name: Original parameter name

    Returns:
        Normalized parameter name
    """
    # Pattern to match transformer block numbers: h.{digits}.
    pattern = r'\.h\.\d+\.'

    if re.search(pattern, param_name):
        # Replace h.{N}. with just the part after it
        normalized = re.sub(pattern, '.', param_name)
        # Remove module.transformer prefix and clean up
        normalized = normalized.replace('module.transformer.', '')
        return f"Transformer {normalized}"
    else:
        # Handle non-block parameters (wte, ln_f, etc.)
        normalized = param_name.replace('module.transformer.', '')
        return f"Transformer {normalized}"


def group_parameters_by_type(param_names):
    """Group parameter names by their normalized type.

    Args:
        param_names: List of parameter names

    Returns:
        Dictionary mapping normalized name -> list of original names
    """
    groups = defaultdict(list)
    for param_name in param_names:
        normalized = normalize_param_name(param_name)
        groups[normalized].append(param_name)
    return groups


def plot_aggregated_cdf(ax, tau_stats_data, param_group, timesteps_data, use_raw_tau=False):
    """Plot aggregated dual-tail CDF for a parameter group across multiple timesteps.

    Args:
        ax: Matplotlib axis
        tau_stats_data: Dictionary mapping iteration to tau statistics
        param_group: List of parameter names to aggregate
        timesteps_data: List of (iteration, eval_step, color) tuples
        use_raw_tau: If True, plot raw tau; if False, plot regularized tau (default: False)
    """
    for iteration, eval_step, color in timesteps_data:
        if iteration not in tau_stats_data:
            continue

        # Collect CDFs for all parameters in this group at this timestep
        cdfs_list = []
        for param_name in param_group:
            param_data = tau_stats_data[iteration]['tau_statistics'].get(param_name)
            if param_data is None:
                continue

            largest = param_data['largest_order_stats']
            smallest = param_data['smallest_order_stats']
            n = param_data['num_elements']

            # Reconstruct CDF for this parameter
            tau_values, cdf_values = reconstruct_single_cdf(smallest, largest, n)

            if len(tau_values) > 0:
                cdfs_list.append((tau_values, cdf_values))

        if len(cdfs_list) == 0:
            continue

        # Merge CDFs across parameters in the group
        merged_cdf = merge_cdfs(cdfs_list)
        tau_values = merged_cdf['tau_values']
        cdf_values = merged_cdf['cdf_values']

        if len(tau_values) == 0:
            continue

        # Apply tau regularization if requested
        if not use_raw_tau:
            tau_values = regularize_tau(tau_values, iteration)

        # Transform probabilities to y-axis values
        y_vals = transform_prob_to_yaxis(cdf_values)

        # Plot
        ax.plot(tau_values, y_vals,
                label=f"iter={iteration} (eval {eval_step}, n={len(param_group)} tensors)",
                color=color, alpha=0.8)

    # Format plot
    tau_label = r'$\tau$' if use_raw_tau else r'$\tau_{reg}$'
    ax.set_xlabel(f'{tau_label} (log scale)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_xscale('log')

    # Set up custom y-axis ticks and labels
    # Show: 0.1%, 1%, 10%, 50%, 90%, 99%, 99.9%, 100%
    ytick_probs = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]
    ytick_vals = transform_prob_to_yaxis(ytick_probs)
    ax.set_yticks(ytick_vals)
    ax.yaxis.set_major_formatter(FuncFormatter(format_yaxis_label))

    # Add horizontal line at 50%
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    # Grid
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=8, loc='best')

    # Add axis break markers
    add_axis_break_markers(ax)


def load_tau_stats_directory(tau_stats_dir):
    """Load all tau statistics from a directory.

    Args:
        tau_stats_dir: Path to tau_stats directory

    Returns:
        Tuple of (metadata, tau_stats_data, available_iterations)
        where tau_stats_data maps iteration -> data dict
    """
    tau_stats_dir = Path(tau_stats_dir)

    # Load metadata
    metadata_file = tau_stats_dir / "tau_stats_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load all pickle files
    pkl_files = sorted(tau_stats_dir.glob("tau_stats_iter_*.pkl"))

    tau_stats_data = {}
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        iteration = data['metadata']['iteration']
        tau_stats_data[iteration] = data

    available_iterations = sorted(tau_stats_data.keys())

    return metadata, tau_stats_data, available_iterations


def main():
    parser = argparse.ArgumentParser(description="Plot aggregated tau statistics by parameter type")
    parser.add_argument("tau_stats_dir", type=str, help="Path to tau_stats directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Output PDF file (default: <tau_stats_dir>/aggregated_tau_stats.pdf)")
    parser.add_argument("--timesteps", type=str, default=None,
                       help="Comma-separated list of iterations to plot (default: all)")
    parser.add_argument("--raw-tau", action="store_true",
                       help="Plot raw tau values instead of regularized tau (default: plot regularized tau)")

    args = parser.parse_args()

    tau_stats_dir = Path(args.tau_stats_dir)

    if not tau_stats_dir.exists():
        print(f"Error: {tau_stats_dir} does not exist")
        return

    # Load data
    print(f"Loading tau statistics from: {tau_stats_dir}")
    metadata, tau_stats_data, available_iterations = load_tau_stats_directory(tau_stats_dir)

    print(f"Loaded {len(tau_stats_data)} timesteps")
    print(f"Available iterations: {available_iterations}")

    # Determine which iterations to plot
    if args.timesteps:
        selected_iterations = [int(t.strip()) for t in args.timesteps.split(',')]
        # Validate
        invalid = [t for t in selected_iterations if t not in available_iterations]
        if invalid:
            print(f"Warning: Invalid iterations {invalid} not in available data")
            selected_iterations = [t for t in selected_iterations if t in available_iterations]
    else:
        selected_iterations = available_iterations

    print(f"Plotting {len(selected_iterations)} timesteps: {selected_iterations}")

    # Get list of all parameters from first timestep
    first_iter = available_iterations[0]
    all_param_names = sorted(tau_stats_data[first_iter]['tau_statistics'].keys())
    print(f"Found {len(all_param_names)} total parameters")

    # Group parameters by type
    param_groups = group_parameters_by_type(all_param_names)
    grouped_names = sorted(param_groups.keys())
    print(f"Grouped into {len(grouped_names)} parameter types")

    # Print grouping summary
    print("\nParameter grouping:")
    for group_name in grouped_names:
        group = param_groups[group_name]
        print(f"  {group_name}: {len(group)} tensors")

    # Create color map for different timesteps
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_iterations)))

    # Create timesteps_data: (iteration, eval_step, color)
    timesteps_data = []
    for iteration, color in zip(selected_iterations, colors):
        eval_step = tau_stats_data[iteration]['metadata']['eval_step_number']
        timesteps_data.append((iteration, eval_step, color))

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = tau_stats_dir / "aggregated_tau_stats.pdf"

    # Get use_raw_tau from args
    use_raw_tau = getattr(args, 'raw_tau', False)
    tau_mode = "raw tau" if use_raw_tau else "regularized tau"
    print(f"\nPlotting {tau_mode} values")

    print(f"\nCreating PDF: {output_file}")
    print(f"This will create {len(grouped_names)} pages (one per parameter type)")

    # Create multi-page PDF
    with PdfPages(output_file) as pdf:
        for i, group_name in enumerate(grouped_names):
            if (i + 1) % 5 == 0:
                print(f"  Processing parameter type {i+1}/{len(grouped_names)}: {group_name}")

            param_group = param_groups[group_name]

            # Create figure for this parameter type
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))

            # Plot aggregated CDF
            plot_aggregated_cdf(ax, tau_stats_data, param_group, timesteps_data, use_raw_tau=use_raw_tau)

            # Set title with parameter type and count
            ax.set_title(f"{group_name} (aggregated over {len(param_group)} tensors)",
                        fontsize=10, pad=10)

            # Add overall title with metadata
            architecture = metadata.get('architecture', 'unknown')
            kappa = metadata.get('optimizer_params', {}).get('kappa', 'N/A')
            fig.suptitle(f"Tau Statistics: {architecture} (kappa={kappa})",
                        fontsize=14, y=0.98)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"\n✓ PDF created successfully: {output_file}")
    print(f"  Total pages: {len(grouped_names)}")
    print(f"  Timesteps plotted: {len(selected_iterations)}")


if __name__ == "__main__":
    main()
