#!/usr/bin/env python
"""
LLM Hummingbird Plot: Visualize kappa sweep results from WandB Enoki experiments.

Fetches data from WandB group "Enoki_ScaledGPT_kappa0" for 16-head models,
plotting validation loss vs training step for different kappa and clipsnr values.

Color scheme:
- clipsnr=100 → adana color (olive/yellow-green)
- clipsnr=0.5 → mk4 color (red)
- clipsnr=2 → interpolate between mk4 and adana
- clipsnr=0.25 → darker red

Also includes AdamW baseline from Enoki_ScaledGPT group.
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from opt_colors import OPT_COLORS
from wandb_cache import CachedWandBApi


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Hummingbird Plot from WandB data")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save plots")
    parser.add_argument("--output_prefix", type=str, default="llm_hummingbird",
                       help="Prefix for output plot file")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved figures")
    parser.add_argument("--figsize", type=float, nargs=2, default=[14, 10],
                       help="Figure size (width height)")
    parser.add_argument("--kappa_min", type=float, default=0.0,
                       help="Minimum kappa value to plot")
    parser.add_argument("--project", type=str, default="danastar",
                       help="WandB project name")
    parser.add_argument("--entity", type=str, default="ep-rmt-ml-opt",
                       help="WandB entity name")
    parser.add_argument("--n_head", type=int, default=16,
                       help="Number of heads to filter for")
    parser.add_argument("--force_refresh", action="store_true",
                       help="Force refresh WandB cache (bypass cached data)")
    parser.add_argument("--cache_dir", type=str, default="wandb_cache",
                       help="Directory for WandB cache")
    return parser.parse_args()


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-1 scale)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def interpolate_color(color1_hex, color2_hex, t):
    """Interpolate between two colors. t=0 gives color1, t=1 gives color2."""
    c1 = hex_to_rgb(color1_hex)
    c2 = hex_to_rgb(color2_hex)
    return tuple(c1[i] + t * (c2[i] - c1[i]) for i in range(3))


def get_clipsnr_color(clipsnr):
    """Get color for a given clipsnr value.

    Color scheme:
    - clipsnr=100 → adana color (olive/yellow-green)
    - clipsnr=0.5 → mk4 color (red)
    - clipsnr=2 → interpolate between mk4 and adana
    - clipsnr=0.25 → darker red
    """
    adana_color = OPT_COLORS.get('adana', '#bcbd22')  # Olive/yellow-green
    mk4_color = OPT_COLORS.get('mk4', '#d62728')      # Red

    if clipsnr >= 100:
        return adana_color
    elif clipsnr <= 0.25:
        return '#8B0000'  # Dark red
    elif clipsnr <= 0.5:
        return mk4_color
    elif clipsnr <= 2:
        # Interpolate between mk4 (clipsnr=0.5) and adana (clipsnr=100)
        # Use log scale for interpolation
        t = (np.log10(clipsnr) - np.log10(0.5)) / (np.log10(100) - np.log10(0.5))
        return interpolate_color(mk4_color, adana_color, t)
    else:
        # Between 2 and 100, interpolate
        t = (np.log10(clipsnr) - np.log10(0.5)) / (np.log10(100) - np.log10(0.5))
        return interpolate_color(mk4_color, adana_color, t)


def fetch_kappa_sweep_data(api, entity, project, n_head):
    """Fetch kappa sweep data from WandB.

    Returns:
        Dict mapping (clipsnr, kappa) -> {'steps': [], 'losses': [], 'run_name': str}
    """
    print(f"Fetching data from {entity}/{project}, group=Enoki_ScaledGPT_kappa0, n_head={n_head}")

    runs = api.runs(f"{entity}/{project}", filters={"group": "Enoki_ScaledGPT_kappa0"})

    data = {}
    for run in runs:
        config = run.config
        if config.get('n_head') != n_head:
            continue

        kappa = config.get('kappa')
        clipsnr = config.get('clipsnr')

        if kappa is None or clipsnr is None:
            continue

        # Fetch history
        history = list(run.scan_history(keys=["val/loss", "_step"]))
        if not history:
            continue

        steps = [h['_step'] for h in history if 'val/loss' in h and '_step' in h]
        losses = [h['val/loss'] for h in history if 'val/loss' in h and '_step' in h]

        if len(steps) == 0:
            continue

        key = (clipsnr, kappa)
        data[key] = {
            'steps': np.array(steps),
            'losses': np.array(losses),
            'run_name': run.name
        }

    print(f"  Found {len(data)} runs")
    return data


def fetch_adamw_baseline(api, entity, project, n_head):
    """Fetch best AdamW baseline from Enoki_ScaledGPT group.

    Returns:
        Dict with 'steps', 'losses', 'run_name', 'final_loss'
    """
    print(f"Fetching AdamW baseline from {entity}/{project}, group=Enoki_ScaledGPT, n_head={n_head}")

    runs = api.runs(f"{entity}/{project}", filters={"group": "Enoki_ScaledGPT"})

    best_run = None
    best_loss = float('inf')

    for run in runs:
        config = run.config
        if config.get('n_head') != n_head:
            continue
        if config.get('opt') != 'adamw':
            continue

        final_loss = run.summary.get('final-val/loss')
        if final_loss is not None and final_loss < best_loss:
            best_loss = final_loss
            best_run = run

    if best_run is None:
        print("  No AdamW baseline found")
        return None

    print(f"  Best AdamW run: {best_run.name}, final_loss={best_loss:.4f}")

    # Fetch history
    history = list(best_run.scan_history(keys=["val/loss", "_step"]))
    steps = [h['_step'] for h in history if 'val/loss' in h and '_step' in h]
    losses = [h['val/loss'] for h in history if 'val/loss' in h and '_step' in h]

    return {
        'steps': np.array(steps),
        'losses': np.array(losses),
        'run_name': best_run.name,
        'final_loss': best_loss
    }


def plot_llm_hummingbird(kappa_data, adamw_baseline, args):
    """Create the LLM hummingbird plot."""
    fig, ax = plt.subplots(1, 1, figsize=tuple(args.figsize))

    # Get unique clipsnr values
    clipsnr_values = sorted(set(k[0] for k in kappa_data.keys()))
    print(f"ClipSNR values: {clipsnr_values}")

    # Plot each clipsnr group
    legend_handles = []
    legend_labels = []

    for clipsnr in clipsnr_values:
        base_color = get_clipsnr_color(clipsnr)

        # Get kappa values for this clipsnr
        kappa_values = sorted([k[1] for k in kappa_data.keys() if k[0] == clipsnr])
        kappa_values = [k for k in kappa_values if k >= args.kappa_min]

        if not kappa_values:
            continue

        print(f"  ClipSNR={clipsnr}: {len(kappa_values)} kappa values")

        first_for_clipsnr = True
        for kappa in kappa_values:
            key = (clipsnr, kappa)
            if key not in kappa_data:
                continue

            steps = kappa_data[key]['steps']
            losses = kappa_data[key]['losses']

            # Alpha varies with kappa (0.25 at kappa_min, 1.0 at kappa=1.0)
            if args.kappa_min < 1.0:
                alpha = 0.25 + 0.75 * (kappa - args.kappa_min) / (1.0 - args.kappa_min)
            else:
                alpha = 1.0
            alpha = max(0.25, min(1.0, alpha))

            # Linewidth varies with kappa
            linewidth = 1.0 + 2.0 * (kappa - args.kappa_min) / (1.0 - args.kappa_min) if args.kappa_min < 1.0 else 2.0

            line, = ax.semilogy(steps, losses,
                               color=base_color,
                               alpha=alpha,
                               linewidth=linewidth)

            # Legend entry for first and last kappa of each clipsnr
            if first_for_clipsnr:
                legend_handles.append(line)
                legend_labels.append(f'clipsnr={clipsnr} (κ={kappa:.1f})')
                first_for_clipsnr = False
            elif kappa == max(kappa_values):
                legend_handles.append(line)
                legend_labels.append(f'clipsnr={clipsnr} (κ={kappa:.1f})')

    # Plot AdamW baseline
    if adamw_baseline is not None:
        line, = ax.semilogy(adamw_baseline['steps'], adamw_baseline['losses'],
                           color='black',
                           alpha=0.9,
                           linewidth=3,
                           linestyle='--')
        legend_handles.append(line)
        legend_labels.append(f'AdamW (baseline)')

    ax.set_xlabel('Training Step', fontsize=16)
    ax.set_ylabel('Validation Loss', fontsize=16)
    ax.set_ylim(2.5, 6.0)

    title = f'LLM Hummingbird Plot (Enoki, {args.n_head} heads)\n'
    title += f'ClipSNR sweep with κ ∈ [{args.kappa_min}, 1.0]'
    ax.set_title(title, fontsize=14)

    ax.grid(True, alpha=0.3)
    ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
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
    print("LLM Hummingbird Plot")
    print("="*70)

    api = CachedWandBApi(cache_dir=args.cache_dir, force_refresh=args.force_refresh)

    # Fetch kappa sweep data
    kappa_data = fetch_kappa_sweep_data(api, args.entity, args.project, args.n_head)

    # Fetch AdamW baseline
    adamw_baseline = fetch_adamw_baseline(api, args.entity, args.project, args.n_head)

    # Create plot
    print("\nGenerating plot...")
    plot_llm_hummingbird(kappa_data, adamw_baseline, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
