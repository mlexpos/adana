#!/usr/bin/env python
"""
LLM Hummingbird Slice Plot: Final validation loss vs kappa for different clipsnr values.

Fetches data from WandB group "Enoki_ScaledGPT_kappa0" for 16-head models,
plotting final-val/loss vs kappa for different clipsnr values.

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
    parser = argparse.ArgumentParser(description="LLM Hummingbird Slice Plot from WandB data")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save plots")
    parser.add_argument("--output_prefix", type=str, default="llm_hummingbird_slice",
                       help="Prefix for output plot file")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved figures")
    parser.add_argument("--figsize", type=float, nargs=2, default=[14, 10],
                       help="Figure size (width height)")
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
        Dict mapping (clipsnr, kappa) -> {'final_loss': float, 'run_name': str}
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

        # Ignore clipsnr=8
        if clipsnr == 8:
            continue

        # For clipsnr=0.25, only select learning rates approximately equal to 0.000163
        if clipsnr == 0.25:
            lr = config.get('lr', config.get('learning_rate'))
            if lr is None or abs(lr - 0.000163) > 1e-5:
                continue

        # Get final validation loss from summary
        final_loss = run.summary.get('final-val/loss')
        if final_loss is None:
            continue

        key = (clipsnr, kappa)
        data[key] = {
            'final_loss': final_loss,
            'run_name': run.name
        }

    print(f"  Found {len(data)} runs with final-val/loss")
    return data


def fetch_adamw_baseline(api, entity, project, n_head):
    """Fetch best AdamW baseline from Enoki_ScaledGPT group.

    Returns:
        Dict with 'final_loss', 'run_name' or None
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

    return {
        'final_loss': best_loss,
        'run_name': best_run.name
    }


def is_reliable_kappa(kappa):
    """Filter out unreliable kappa values (those ending in .X5 like 0.05, 0.15, 0.75, etc.)."""
    # Round to avoid floating point issues, then check if it's a multiple of 0.1
    kappa_rounded = round(kappa * 100) / 100
    # Check if the hundredths digit is 5 (e.g., 0.05, 0.15, 0.25, ...)
    hundredths = round(kappa_rounded * 100) % 10
    return hundredths != 5


def plot_llm_hummingbird_slice(kappa_data, adamw_baseline, args):
    """Create the LLM hummingbird slice plot (final loss vs kappa)."""
    # Use sans-serif font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']

    fig, ax = plt.subplots(1, 1, figsize=tuple(args.figsize))

    # Get unique clipsnr values
    clipsnr_values = sorted(set(k[0] for k in kappa_data.keys()))
    print(f"ClipSNR values: {clipsnr_values}")

    # Plot each clipsnr group
    for clipsnr in clipsnr_values:
        color = get_clipsnr_color(clipsnr)

        # Get kappa values and final losses for this clipsnr (filter out unreliable 0.X5 values)
        kappa_loss_pairs = [(k[1], kappa_data[k]['final_loss'])
                           for k in kappa_data.keys()
                           if k[0] == clipsnr and is_reliable_kappa(k[1])]

        if not kappa_loss_pairs:
            continue

        # Sort by kappa
        kappa_loss_pairs.sort(key=lambda x: x[0])
        kappas = [p[0] for p in kappa_loss_pairs]
        losses = [p[1] for p in kappa_loss_pairs]

        print(f"  ClipSNR={clipsnr}: {len(kappas)} points, loss range [{min(losses):.4f}, {max(losses):.4f}]")

        # Get paper-style label
        if clipsnr >= 100:
            label = 'ADANA'
        else:
            label = f'DANA-MK4 (c={clipsnr})'

        # Plot line with markers (thicker lines)
        ax.plot(kappas, losses,
                color=color,
                linewidth=7,
                marker='o',
                markersize=12,
                label=label)

    # Plot AdamW baseline as horizontal line (thicker)
    if adamw_baseline is not None:
        ax.axhline(y=adamw_baseline['final_loss'],
                   color='black',
                   linestyle='--',
                   linewidth=7,
                   label=f'AdamW (baseline)')

    ax.set_xlabel('κ (kappa)', fontsize=36, fontfamily='sans-serif')
    ax.set_ylabel('Final Validation Loss', fontsize=36, fontfamily='sans-serif')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(None, 3.25)

    title = f'LLM Final Loss vs κ (Enoki, {args.n_head} heads)'
    ax.set_title(title, fontsize=48, fontfamily='sans-serif')

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=24)
    # Increased tick label size to 32
    ax.tick_params(axis='both', which='major', labelsize=32)

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
    print("LLM Hummingbird Slice Plot")
    print("="*70)

    api = CachedWandBApi(cache_dir=args.cache_dir, force_refresh=args.force_refresh)

    # Fetch kappa sweep data
    kappa_data = fetch_kappa_sweep_data(api, args.entity, args.project, args.n_head)

    # Fetch AdamW baseline
    adamw_baseline = fetch_adamw_baseline(api, args.entity, args.project, args.n_head)

    # Create plot
    print("\nGenerating plot...")
    plot_llm_hummingbird_slice(kappa_data, adamw_baseline, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
