#!/usr/bin/env python
"""
Tau Statistics Plotting: Visualize tau order statistics from saved data.

This script loads tau statistics data from pickle files and creates visualizations
showing cumulative distribution functions with log-tau on x-axis and a clever y-axis
that shows both tails: 10%, 1%, 0.1% and 90%, 99%, 99.9% cumulative probability.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import argparse
import os
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot tau order statistics from saved data")

    parser.add_argument("data_file", type=str, help="Path to the pickle file with tau statistics data")
    parser.add_argument("--output_dir", type=str, default="jax/taustats_plots", help="Directory to save plots")
    parser.add_argument("--output_prefix", type=str, default=None, help="Prefix for output plot files (default: use input filename)")
    parser.add_argument("--timesteps", type=str, default=None, help="Comma-separated list of timesteps to plot (e.g., '1000,8000,64000'). If not specified, plots all available timesteps.")
    parser.add_argument("--raw-tau", action="store_true", help="Plot raw tau values instead of regularized tau (default: plot regularized tau)")

    return parser.parse_args()


def load_data(data_file):
    """Load tau statistics data from pickle file.

    Args:
        data_file: Path to the pickle file

    Returns:
        Dictionary with metadata and results
    """
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data


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


def reconstruct_single_expert_cdf(smallest, largest, n):
    """Reconstruct CDF from order statistics for a single expert.

    Args:
        smallest: Array of smallest order statistics
        largest: Array of largest order statistics
        n: Total number of parameters for this expert

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


def merge_expert_cdfs(tau_stats_tree, n_per_expert):
    """Merge per-expert tau order statistics to estimate global CDF.

    Given per-expert CDFs, estimates Pr(X* <= t) where X* is uniformly
    randomly chosen from all experts' parameters.

    Pr(X* <= t) = (1/m) * sum_i Pr(X_i <= t)

    Args:
        tau_stats_tree: Tree with per-expert (largest, smallest) order stats
        n_per_expert: Number of parameters per expert (d)

    Returns:
        Dictionary with merged 'tau_values' and 'cdf_values' arrays
    """
    # Step 1: Collect per-expert CDFs
    expert_cdfs = []  # List of (tau_values, cdf_values) tuples

    def collect_expert_cdf(node):
        if node is None:
            return
        # Check if this is a list of per-expert (largest, smallest) tuples
        if isinstance(node, list):
            for item in node:
                if isinstance(item, tuple) and len(item) == 2:
                    largest, smallest = item
                    if largest is not None and smallest is not None:
                        # Reconstruct CDF for this expert
                        tau_vals, cdf_vals = reconstruct_single_expert_cdf(
                            smallest, largest, n_per_expert
                        )
                        if len(tau_vals) > 0:
                            expert_cdfs.append((tau_vals, cdf_vals))
                else:
                    # Recursively process if not a tuple
                    collect_expert_cdf(item)
        elif isinstance(node, tuple) and len(node) == 2:
            # Legacy format: single (largest, smallest) tuple
            # This might be from old data or non-2D arrays
            largest, smallest = node
            if largest is not None and smallest is not None:
                # Check if largest/smallest are arrays (not lists of arrays)
                if isinstance(largest, np.ndarray) and largest.ndim == 1:
                    tau_vals, cdf_vals = reconstruct_single_expert_cdf(
                        smallest, largest, n_per_expert
                    )
                    if len(tau_vals) > 0:
                        expert_cdfs.append((tau_vals, cdf_vals))
        elif isinstance(node, dict):
            for v in node.values():
                collect_expert_cdf(v)

    collect_expert_cdf(tau_stats_tree)

    if len(expert_cdfs) == 0:
        return {'tau_values': np.array([]), 'cdf_values': np.array([])}

    print(f"  Collected {len(expert_cdfs)} expert CDFs")

    # Step 2: Collect all unique tau values to form common grid
    all_tau_values = []
    for tau_vals, _ in expert_cdfs:
        all_tau_values.extend(tau_vals)
    all_tau_values = np.sort(np.unique(all_tau_values))

    print(f"  Common grid has {len(all_tau_values)} unique tau values")

    # Step 3: Interpolate each expert's CDF on common grid
    interpolated_cdfs = []
    for tau_vals, cdf_vals in expert_cdfs:
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

    # Step 4: Average across experts to get Pr(X* <= t)
    averaged_cdf = np.mean(interpolated_cdfs, axis=0)

    return {
        'tau_values': all_tau_values,
        'cdf_values': averaged_cdf
    }


def transform_prob_to_yaxis(prob):
    """Transform probability to y-axis value using logit transformation with tail compression.

    Uses logit transformation below 99.9%, then compresses the tail from 99.9% to 100%
    to exactly 1 unit, making the upper tail finite and easier to visualize.

    Key points:
    - prob < 0.001: logit scale (symmetric with upper tail)
    - 0.001 ≤ prob ≤ 0.999: standard logit transformation
    - prob > 0.999: compressed to fit in 1 unit above logit(0.999)

    The transformation is:
    - For prob ≤ 0.999: y = log10(prob / (1 - prob))
    - For prob > 0.999: y = logit(0.999) + (prob - 0.999) / (1 - 0.999) * 1
                            = logit(0.999) + (prob - 0.999) / 0.001

    Examples:
    - prob=0.001: y ≈ -3.0
    - prob=0.5: y = 0
    - prob=0.999: y ≈ 3.0
    - prob=0.9999: y ≈ 3.0 + 0.9
    - prob=1.0: y = 3.0 + 1.0 = 4.0

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

    # Get the figure and axis coordinates
    fig = ax.get_figure()

    # Convert data coordinates to display coordinates for y-values
    # Use axis coordinates (0 to 1) for x-position on the axis
    trans = ax.get_yaxis_transform()  # This transform uses axis coords for x, data coords for y

    # Height of diagonal lines in data coordinates
    dy = 0.15

    # Width in axis coordinates (fraction of plot width)
    dx_axis = 0.02  # 2% of the plot width

    # Draw upper break markers (two parallel diagonal lines on the y-axis)
    # Each marker goes from left edge to slightly right
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


def plot_dual_tail_cdf(merged_cdf, label, ax, timestep, use_raw_tau=False, linestyle='-', color=None, alpha=1.0):
    """Plot CDF with dual-tail y-axis showing both small and large probabilities.

    Args:
        merged_cdf: Dictionary with 'tau_values' and 'cdf_values' arrays from merge_expert_cdfs
        label: Label for the plot
        ax: Matplotlib axis
        timestep: Current timestep (used for tau regularization)
        use_raw_tau: If True, plot raw tau values; if False, apply regularization (default: False)
        linestyle: Line style
        color: Line color (optional)
        alpha: Transparency (optional)
    """
    tau_values = merged_cdf['tau_values']
    cdf_values = merged_cdf['cdf_values']

    if len(tau_values) == 0:
        return

    # Apply tau regularization if requested
    if not use_raw_tau:
        tau_values = regularize_tau(tau_values, timestep)

    # Print CDF data to stdout
    print(f"\n{label}:")
    print(f"  Total data points: {len(tau_values)}")
    print(f"  CDF range: [{np.min(cdf_values):.6f}, {np.max(cdf_values):.6f}]")
    print(f"  Tau range: [{np.min(tau_values):.8e}, {np.max(tau_values):.8e}]")
    print(f"  Sample of (tau, CDF) pairs:")
    # Print first 10, middle 10, and last 10
    n = len(tau_values)
    indices_to_print = list(range(min(10, n)))
    if n > 20:
        indices_to_print += list(range(n//2 - 5, n//2 + 5))
    if n > 10:
        indices_to_print += list(range(max(0, n - 10), n))
    indices_to_print = sorted(set(indices_to_print))

    for i in indices_to_print:
        print(f"    {i:4d}: tau={tau_values[i]:.8e}, CDF={cdf_values[i]:.8e}")

    # Transform probabilities to y-axis values
    y_vals = transform_prob_to_yaxis(cdf_values)

    # Plot
    plot_kwargs = {'label': label, 'linestyle': linestyle, 'alpha': alpha}
    if color is not None:
        plot_kwargs['color'] = color

    ax.plot(tau_values, y_vals, **plot_kwargs)


def plot_combined(data, timesteps, output_path, use_raw_tau=False):
    """Create a combined plot with all timesteps using dual-tail CDF.

    Args:
        data: Dictionary with metadata and results
        timesteps: List of timesteps to plot (or None for all)
        output_path: Path to save the plot
        use_raw_tau: If True, plot raw tau values; if False, apply regularization
    """
    tau_statistics = data['results']['tau_statistics']
    available_timesteps = tau_statistics['timestamps']
    tau_stats_list = tau_statistics['tau_stats']
    metadata = data['metadata']

    # Get total number of parameters (d * m)
    d = metadata['model_params']['d']
    m = metadata['model_params']['m']
    n_total = d * m

    # Filter timesteps if specified
    if timesteps is not None:
        indices = [i for i, t in enumerate(available_timesteps) if t in timesteps]
    else:
        indices = list(range(len(available_timesteps)))

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Define color map for different timesteps
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

    # Plot each timestep
    for idx, color in zip(indices, colors):
        timestep = available_timesteps[idx]
        tau_stats_tree = tau_stats_list[idx]

        print(f"\nProcessing timestep {timestep}...")
        # Merge per-expert CDFs
        merged_cdf = merge_expert_cdfs(tau_stats_tree, d)

        # Plot dual-tail CDF
        plot_dual_tail_cdf(
            merged_cdf,
            f"t={timestep}",
            ax,
            timestep,
            use_raw_tau=use_raw_tau,
            color=color
        )

    # Format plot
    tau_label = r'$\tau$' if use_raw_tau else r'$\tau_{reg}$'
    ax.set_xlabel(f'{tau_label} (log scale)', fontsize=14)
    ax.set_ylabel('Cumulative Probability', fontsize=14)
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
    ax.legend(fontsize=10, loc='best')

    # Add axis break markers to indicate compressed tails
    add_axis_break_markers(ax)

    # Add metadata
    metadata = data['metadata']
    tau_type = "Raw Tau" if use_raw_tau else "Regularized Tau"
    ax.set_title(f"{tau_type} Statistics CDF (alpha={metadata['model_params']['alpha']}, "
                f"m={metadata['model_params']['m']}, kappa={metadata['optimizer_params']['kappa']})",
                fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to: {output_path}")
    plt.close()


def plot_individual(data, timesteps, output_dir, output_prefix, use_raw_tau=False):
    """Create individual plots for each timestep using dual-tail CDF.

    Args:
        data: Dictionary with metadata and results
        timesteps: List of timesteps to plot (or None for all)
        output_dir: Directory to save plots
        output_prefix: Prefix for output files
        use_raw_tau: If True, plot raw tau values; if False, apply regularization
    """
    tau_statistics = data['results']['tau_statistics']
    available_timesteps = tau_statistics['timestamps']
    tau_stats_list = tau_statistics['tau_stats']
    metadata = data['metadata']

    # Get total number of parameters (d * m)
    d = metadata['model_params']['d']
    m = metadata['model_params']['m']
    n_total = d * m

    # Filter timesteps if specified
    if timesteps is not None:
        indices = [i for i, t in enumerate(available_timesteps) if t in timesteps]
    else:
        indices = list(range(len(available_timesteps)))

    # Plot each timestep individually
    for idx in indices:
        timestep = available_timesteps[idx]
        tau_stats_tree = tau_stats_list[idx]

        print(f"\nProcessing timestep {timestep}...")
        # Merge per-expert CDFs
        merged_cdf = merge_expert_cdfs(tau_stats_tree, d)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Plot dual-tail CDF
        plot_dual_tail_cdf(
            merged_cdf,
            f"t={timestep}",
            ax,
            timestep,
            use_raw_tau=use_raw_tau,
            color='blue'
        )

        # Format plot
        tau_label = r'$\tau$' if use_raw_tau else r'$\tau_{reg}$'
        ax.set_xlabel(f'{tau_label} (log scale)', fontsize=14)
        ax.set_ylabel('Cumulative Probability', fontsize=14)
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
        ax.legend(fontsize=10, loc='best')

        # Add axis break markers to indicate compressed tails
        add_axis_break_markers(ax)

        # Add metadata
        tau_type = "Raw Tau" if use_raw_tau else "Regularized Tau"
        ax.set_title(f"{tau_type} Statistics CDF at t={timestep} (alpha={metadata['model_params']['alpha']}, "
                    f"m={metadata['model_params']['m']}, kappa={metadata['optimizer_params']['kappa']})",
                    fontsize=16)

        plt.tight_layout()

        # Save plot
        output_path = os.path.join(output_dir, f"{output_prefix}_t{timestep}.pdf")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved individual plot to: {output_path}")
        plt.close()


def main():
    """Main function."""
    args = parse_args()

    # Load data
    print(f"Loading data from: {args.data_file}")
    data = load_data(args.data_file)

    # Print metadata
    metadata = data['metadata']
    print("\nMetadata:")
    print(f"  Optimizer: {metadata['optimizer']}")
    print(f"  Model: alpha={metadata['model_params']['alpha']}, m={metadata['model_params']['m']}, "
          f"zeta={metadata['model_params']['zeta']}, beta={metadata['model_params']['beta']}")
    print(f"  Optimizer params: kappa={metadata['optimizer_params']['kappa']}, "
          f"clipsnr={metadata['optimizer_params']['clipsnr']}, delta={metadata['optimizer_params']['delta']}")
    print(f"  Training: {metadata['training_params']['steps']} steps, "
          f"batch_size={metadata['training_params']['batch_size']}")

    # Get available timesteps
    tau_statistics = data['results']['tau_statistics']
    available_timesteps = tau_statistics['timestamps']
    print(f"\nAvailable timesteps: {available_timesteps}")

    # Parse timesteps argument
    if args.timesteps is not None:
        timesteps = [int(t.strip()) for t in args.timesteps.split(',')]
        # Validate timesteps
        invalid = [t for t in timesteps if t not in available_timesteps]
        if invalid:
            print(f"Warning: Invalid timesteps {invalid} not in available timesteps")
            timesteps = [t for t in timesteps if t in available_timesteps]
    else:
        timesteps = None

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine output prefix
    if args.output_prefix is None:
        output_prefix = Path(args.data_file).stem
    else:
        output_prefix = args.output_prefix

    # Get use_raw_tau from args
    use_raw_tau = getattr(args, 'raw_tau', False)
    tau_mode = "raw tau" if use_raw_tau else "regularized tau"
    print(f"\nPlotting {tau_mode} values")

    # Create combined plot (always)
    output_path = os.path.join(args.output_dir, f"{output_prefix}_combined.pdf")
    plot_combined(data, timesteps, output_path, use_raw_tau=use_raw_tau)

    # Create individual plots if timesteps are specified
    if timesteps is not None:
        plot_individual(data, timesteps, args.output_dir, output_prefix, use_raw_tau=use_raw_tau)

    print("\nDone!")


if __name__ == "__main__":
    main()
