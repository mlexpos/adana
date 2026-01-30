#!/usr/bin/env python
"""
SNR Clipping Plots: Visualize optimizer statistics for DANA-STAR-MK4.

Plots optimizer/m_norm, optimizer/kappa_factor_schedule, optimizer/auto_factor,
optimizer/alpha_schedule in log-log scale vs tokens.

Includes theoretical (1+t)^{1-kappa} line for comparison.
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from wandb_cache import CachedWandBApi


def parse_args():
    parser = argparse.ArgumentParser(description="SNR Clipping Plots from WandB data")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save plots")
    parser.add_argument("--output_prefix", type=str, default="snr_clipping_stats",
                       help="Prefix for output plot file")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved figures")
    parser.add_argument("--figsize", type=float, nargs=2, default=[14, 10],
                       help="Figure size (width height)")
    parser.add_argument("--project", type=str, default="danastar",
                       help="WandB project name")
    parser.add_argument("--entity", type=str, default="ep-rmt-ml-opt",
                       help="WandB entity name")
    parser.add_argument("--run_id", type=str, default="3eq8c06j",
                       help="WandB run ID to fetch")
    parser.add_argument("--force_refresh", action="store_true",
                       help="Force refresh WandB cache (bypass cached data)")
    parser.add_argument("--cache_dir", type=str, default="wandb_cache",
                       help="Directory for WandB cache")
    return parser.parse_args()


def fetch_run_data(api, entity, project, run_id, cache_dir="wandb_cache", force_refresh=False):
    """Fetch run history and config from WandB with caching.

    Returns:
        Tuple of (data_dict, config_dict)
    """
    import json
    from pathlib import Path

    cache_path = Path(cache_dir) / "history" / f"{run_id}_aligned.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to load from cache
    if cache_path.exists() and not force_refresh:
        print(f"Loading cached data for run {run_id}")
        with open(cache_path, 'r') as f:
            cached = json.load(f)
        data = {k: np.array(v) for k, v in cached['data'].items()}
        config = cached['config']
        print(f"  Loaded {len(data['_step'])} aligned rows from cache")
        return data, config

    print(f"Fetching run {run_id} from {entity}/{project}")

    run = api.run(f"{entity}/{project}/{run_id}")

    # Get config
    config = dict(run.config)
    print(f"  Config: n_head={config.get('n_head')}, kappa={config.get('kappa')}, clipsnr={config.get('clipsnr')}")

    # Metrics we want
    opt_metrics = [
        'optimizer/m_norm',
        'optimizer/kappa_factor_schedule',
        'optimizer/auto_factor',
        'optimizer/alpha_schedule',
    ]

    print(f"  Fetching history (this may take a while)...")
    history = list(run.scan_history())

    # Build aligned data - only keep rows where optimizer metrics are logged
    aligned_data = {
        '_step': [],
        'tokens': [],
    }
    for key in opt_metrics:
        aligned_data[key] = []

    for h in history:
        # Check if this row has optimizer metrics
        if h.get('optimizer/m_norm') is not None:
            aligned_data['_step'].append(h.get('_step', 0))
            aligned_data['tokens'].append(h.get('tokens', h.get('_step', 0)))
            for key in opt_metrics:
                aligned_data[key].append(h.get(key))

    print(f"  Got {len(history)} history entries")
    print(f"  Aligned {len(aligned_data['_step'])} rows with optimizer metrics")

    # Save to cache
    with open(cache_path, 'w') as f:
        json.dump({'data': aligned_data, 'config': config}, f)
    print(f"  Cached to {cache_path}")

    # Convert to numpy arrays
    data = {k: np.array(v) for k, v in aligned_data.items()}

    for key in data:
        print(f"    {key}: {len(data[key])} values")

    return data, config


def plot_snr_clipping_stats(data, config, args):
    """Create log-log plots of optimizer statistics."""

    # Get tokens for x-axis and step for theoretical line
    x = data['tokens']
    t = data['_step']
    x_label = 'Tokens'

    # Get kappa for theoretical line
    kappa = config.get('kappa', 0.85)
    clipsnr = config.get('clipsnr', 2.0)
    n_head = config.get('n_head', 'unknown')

    # Metrics to plot with LaTeX titles
    metrics = [
        ('optimizer/m_norm', r'$\|m_t\|$', 'tab:blue'),
        ('optimizer/kappa_factor_schedule', r'$\alpha_{\mathrm{fac}} / \mathrm{mfac}$', 'tab:orange'),
        ('optimizer/auto_factor', r'$\mathrm{mfac} = \frac{|m_t| \cdot \mathrm{norm}}{\tilde{\tau}}$', 'tab:green'),
        ('optimizer/alpha_schedule', r'$\alpha_{\mathrm{fac}}$', 'tab:red'),
    ]

    # Get mfac data for alpha_schedule theoretical line
    mfac_data = data.get('optimizer/auto_factor')

    fig, axes = plt.subplots(2, 2, figsize=tuple(args.figsize))
    axes = axes.flatten()

    for idx, (metric_key, metric_name, color) in enumerate(metrics):
        ax = axes[idx]

        if metric_key not in data:
            ax.set_title(f'{metric_name} (no data)')
            continue

        y = data[metric_key]

        # Filter out zeros/negatives/NaNs for log scale
        mask = (x > 0) & (y > 0) & (t > 0) & np.isfinite(y) & np.isfinite(x)
        x_plot = x[mask]
        y_plot = y[mask]
        t_plot = t[mask]

        # Sort by x to avoid line artifacts from out-of-order points
        sort_idx = np.argsort(x_plot)
        x_plot = x_plot[sort_idx]
        y_plot = y_plot[sort_idx]
        t_plot = t_plot[sort_idx]

        if len(x_plot) == 0:
            ax.set_title(f'{metric_name} (no positive data)')
            continue

        # Plot the actual data with descriptive legend labels
        if metric_key == 'optimizer/m_norm':
            legend_label = r'$\|m_t\|$ (momentum norm)'
        elif metric_key == 'optimizer/kappa_factor_schedule':
            legend_label = r'$\alpha_{\mathrm{fac}} / \mathrm{mfac}$ (effective $\kappa$ factor)'
        elif metric_key == 'optimizer/auto_factor':
            legend_label = r'mfac (SNR proxy)'
        elif metric_key == 'optimizer/alpha_schedule':
            legend_label = r'$\alpha_{\mathrm{fac}}$ (clipped)'
        else:
            legend_label = metric_name

        ax.loglog(x_plot, y_plot, color=color, linewidth=2, label=legend_label, alpha=0.8)

        # Add theoretical (1+t)^{1-kappa} line for kappa_factor_schedule
        if metric_key == 'optimizer/kappa_factor_schedule':
            theoretical = (1 + t_plot) ** (1 - kappa)
            ax.loglog(x_plot, theoretical, 'k--', linewidth=2,
                     label=r'$(1+t)^{1-\kappa}$ (unclipped)')

        # Add theoretical mfac * (1+t)^{1-kappa} line for alpha_schedule
        if metric_key == 'optimizer/alpha_schedule' and mfac_data is not None:
            mfac_plot = mfac_data[mask][sort_idx]
            theoretical_alpha = mfac_plot * (1 + t_plot) ** (1 - kappa)
            ax.loglog(x_plot, theoretical_alpha, 'k--', linewidth=2,
                     label=r'$\mathrm{mfac} \cdot (1+t)^{1-\kappa}$ (unclipped)')
            # Add annotation explaining the gap
            ax.annotate('Gap due to\nSNR clipping',
                       xy=(3e9, 0.15), fontsize=14,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel(x_label, fontsize=18)
        ax.set_ylabel(metric_name, fontsize=18)
        ax.set_title(metric_name, fontsize=20)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Add overall title
    fig.suptitle(r'Dana-Star-MK4 Optimizer Statistics', fontsize=32)

    plt.tight_layout()

    # Save plot
    os.makedirs(args.output_dir, exist_ok=True)
    filepath = os.path.join(args.output_dir, f"{args.output_prefix}.pdf")
    plt.savefig(filepath, dpi=args.dpi, bbox_inches='tight')
    print(f"\nPlot saved to: {filepath}")
    plt.close()


def main():
    args = parse_args()

    print("="*70)
    print("SNR Clipping Statistics Plot")
    print("="*70)

    api = CachedWandBApi(cache_dir=args.cache_dir, force_refresh=args.force_refresh)

    # Fetch run data (with caching)
    data, config = fetch_run_data(api, args.entity, args.project, args.run_id,
                                   cache_dir=args.cache_dir, force_refresh=args.force_refresh)

    # Create plot
    print("\nGenerating plot...")
    plot_snr_clipping_stats(data, config, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
