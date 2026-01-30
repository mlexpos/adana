#!/usr/bin/env python
"""
Hoyer vs Regularization Figure: Cross-entropy loss vs Hoyer regularization parameter.

Plots final CE loss (y-axis) vs hoyer_loss_coeff (x-axis) for different optimizers.
Different line styles for different model sizes (heads).

Panel size/title/legend format from llm_hummingbird_slice_plot.py.
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from opt_colors import OPT_COLORS, OPT_DISPLAY_NAMES
from wandb_cache import CachedWandBApi


WANDB_GROUP = 'Qwen3_Hoyer'

# Line styles for different head counts
HEAD_LINESTYLES = {
    10: ':',
    12: '-.',
    14: '--',
    16: '-',
}

HEAD_MARKERS = {
    10: 's',    # square
    12: '^',    # triangle
    14: 'D',    # diamond
    16: 'o',    # circle
}


def get_head_linestyle(heads):
    """Get line style for a given head count."""
    return HEAD_LINESTYLES.get(heads, '-')


def get_head_marker(heads):
    """Get marker for a given head count."""
    return HEAD_MARKERS.get(heads, 'o')


def parse_args():
    parser = argparse.ArgumentParser(description="Hoyer vs Regularization Plot")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save plots")
    parser.add_argument("--output_prefix", type=str, default="hoyer_vs_regularization",
                       help="Prefix for output plot file")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved figures")
    parser.add_argument("--figsize", type=float, nargs=2, default=[14, 10],
                       help="Figure size (width height)")
    parser.add_argument("--project", type=str, default="danastar",
                       help="WandB project name")
    parser.add_argument("--entity", type=str, default="ep-rmt-ml-opt",
                       help="WandB entity name")
    parser.add_argument("--heads", type=int, nargs='+', default=None,
                       help="Head counts to include (default: all)")
    parser.add_argument("--optimizers", type=str, nargs='+', default=None,
                       help="Optimizers to include (default: all)")
    parser.add_argument("--no-qknorm", action="store_true",
                       help="Include only runs with no_qknorm=True")
    parser.add_argument("--allow-incomplete", action="store_true",
                       help="Include runs that haven't completed")
    parser.add_argument("--force_refresh", action="store_true",
                       help="Force refresh WandB cache")
    parser.add_argument("--cache_dir", type=str, default="wandb_cache",
                       help="Directory for WandB cache")
    return parser.parse_args()


def fetch_hoyer_data(api, entity, project, heads_filter=None, optimizers_filter=None,
                     no_qknorm=False, allow_incomplete=False):
    """
    Fetch final validation data from WandB for Hoyer experiments.

    Returns:
        list: List of dicts with keys: optimizer, heads, hoyer_coeff, ce_loss, run_name
    """
    print(f"Fetching data from {entity}/{project}, group={WANDB_GROUP}")

    runs = api.runs(f"{entity}/{project}", filters={"group": WANDB_GROUP})

    data = []
    total_runs = 0
    skipped = 0

    for run in runs:
        total_runs += 1
        config = run.config
        summary = run.summary

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

        # Filter by no_qknorm flag
        run_no_qknorm = config.get('no_qknorm', False)
        if no_qknorm:
            if not run_no_qknorm:
                skipped += 1
                continue
        else:
            if run_no_qknorm:
                skipped += 1
                continue

        # Check completion
        iterations_config = config.get('iterations')
        actual_iter = summary.get('iter')

        if actual_iter is None or iterations_config is None:
            skipped += 1
            continue

        if not allow_incomplete and actual_iter < iterations_config * 0.8:
            skipped += 1
            continue

        # Get final validation metrics from summary
        final_loss = summary.get('final-val/loss')
        final_hoyer_loss = summary.get('val/hoyer_loss')

        if final_loss is None:
            skipped += 1
            continue

        # Compute cross-entropy loss
        if final_hoyer_loss is not None:
            ce_loss = final_loss - hoyer_coeff * final_hoyer_loss
        else:
            ce_loss = final_loss

        data.append({
            'optimizer': opt,
            'heads': heads,
            'hoyer_coeff': hoyer_coeff,
            'ce_loss': ce_loss,
            'run_name': run.name,
        })

    print(f"Processed {total_runs} runs, skipped {skipped}, kept {len(data)} runs")
    return data


def plot_hoyer_vs_regularization(data, args):
    """Create the Hoyer vs regularization plot (CE loss vs hoyer_coeff)."""
    fig, ax = plt.subplots(1, 1, figsize=tuple(args.figsize))

    # Get unique optimizers and head counts
    optimizers = sorted(set(d['optimizer'] for d in data))
    head_counts = sorted(set(d['heads'] for d in data))

    print(f"Optimizers: {optimizers}")
    print(f"Head counts: {head_counts}")

    # Group data by optimizer and heads
    for opt in optimizers:
        for heads in head_counts:
            group_data = [d for d in data if d['optimizer'] == opt and d['heads'] == heads]

            if not group_data:
                continue

            # Sort by hoyer_coeff
            group_data.sort(key=lambda x: x['hoyer_coeff'])

            hoyer_coeffs = [d['hoyer_coeff'] for d in group_data]
            losses = [d['ce_loss'] for d in group_data]

            # Get styling
            color = OPT_COLORS.get(opt, '#333333')
            linestyle = get_head_linestyle(heads)
            marker = get_head_marker(heads)
            opt_name = OPT_DISPLAY_NAMES.get(opt, opt)

            label = f'{opt_name} (h={heads})'

            print(f"  Plotting {label}: {len(group_data)} points")

            ax.plot(hoyer_coeffs, losses,
                    color=color,
                    linestyle=linestyle,
                    linewidth=4,
                    marker=marker,
                    markersize=10,
                    label=label)

    ax.set_xlabel('Hoyer Regularization (λ)', fontsize=28)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=28)

    title = 'Qwen3: CE Loss vs Hoyer Regularization'
    ax.set_title(title, fontsize=42)

    ax.set_xscale('log')

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=16, ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=12)

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
    print("Hoyer vs Regularization Plot")
    print("="*70)
    print(f"Heads filter: {args.heads if args.heads else 'all'}")
    print(f"Optimizers filter: {args.optimizers if args.optimizers else 'all'}")
    print(f"No QK normalization filter: {args.no_qknorm}")
    print(f"Allow incomplete runs: {args.allow_incomplete}")
    print("="*70)

    api = CachedWandBApi(cache_dir=args.cache_dir, force_refresh=args.force_refresh)

    # Fetch data
    data = fetch_hoyer_data(
        api, args.entity, args.project,
        heads_filter=args.heads,
        optimizers_filter=args.optimizers,
        no_qknorm=args.no_qknorm,
        allow_incomplete=args.allow_incomplete,
    )

    if not data:
        print("ERROR: No data found. Check parameters.")
        return

    # Create plot
    print("\nGenerating plot...")
    plot_hoyer_vs_regularization(data, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
