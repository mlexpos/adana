#!/usr/bin/env python
"""
Hummingbird Plot Multi-Optimizer: Visualize training results from multiple data files.

This script loads training data from multiple files (one per optimizer) and creates
a combined visualization plot with color gradients based on kappa values.

Color scheme options:
1. Default: For each optimizer, alpha varies from 0.25 (kappa_min) to 1.0 (kappa=1.0)
   using the optimizer's color from opt_colors.py.
2. --color_by_clipsnr: Color by clipsnr value (like LLM hummingbird slice plot).
   'adana' is treated as clipsnr=16 for color purposes.
"""
import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import os

from opt_colors import OPT_COLORS, OPT_DISPLAY_NAMES


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
    - clipsnr>=16 → adana color (olive/yellow-green)
    - clipsnr=0.5 → mk4 color (red)
    - clipsnr in between → interpolate on log scale
    - clipsnr<=0.25 → darker red
    """
    adana_color = OPT_COLORS.get('adana', '#bcbd22')  # Olive/yellow-green
    mk4_color = OPT_COLORS.get('mk4', '#d62728')      # Red

    if clipsnr >= 16:
        return adana_color
    elif clipsnr <= 0.25:
        return '#8B0000'  # Dark red
    elif clipsnr <= 0.5:
        return mk4_color
    else:
        # Interpolate between mk4 (clipsnr=0.5) and adana (clipsnr=16)
        # Use log scale for interpolation
        t = (np.log10(clipsnr) - np.log10(0.5)) / (np.log10(16) - np.log10(0.5))
        return interpolate_color(mk4_color, adana_color, t)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hummingbird Plot Multi-Optimizer: Visualize optimizer training results"
    )

    parser.add_argument("--data_files", type=str, nargs='+', required=True,
                       help="Paths to pickle files containing training data (one per optimizer)")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save plots")
    parser.add_argument("--output_prefix", type=str, default="hummingbird_multiopt",
                       help="Prefix for output plot file")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved figures")
    parser.add_argument("--figsize", type=float, nargs=2, default=[14, 10],
                       help="Figure size (width height)")
    parser.add_argument("--no_adam", action="store_true",
                       help="Don't plot Adam baseline")
    parser.add_argument("--kappa_min", type=float, default=0.5,
                       help="Minimum kappa value to plot (default: 0.5)")
    parser.add_argument("--title", type=str, default=None,
                       help="Custom title for the plot (supports LaTeX, e.g., 'PLRF $\\kappa$ sweep')")
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                       help="Y-axis limits (min max), e.g., --ylim 0.05 3.0")
    parser.add_argument("--color_by_clipsnr", action="store_true",
                       help="Color by clipsnr value instead of optimizer. 'adana' treated as clipsnr=16")

    return parser.parse_args()


def load_data(data_file):
    """Load training data from pickle file.

    Args:
        data_file: Path to pickle file

    Returns:
        metadata: Dict containing all metadata
        results: Dict mapping optimizer name to training results
    """
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    return data['metadata'], data['results']


def interpolate_alpha(kappa, kappa_min):
    """Interpolate alpha from 0.25 (kappa_min) to 1.0 (kappa=1.0).

    Args:
        kappa: Current kappa value
        kappa_min: Minimum kappa value (maps to alpha=0.25)

    Returns:
        Alpha value between 0.25 and 1.0
    """
    # Interpolation factor: 0 at kappa_min, 1 at kappa=1.0
    if kappa_min >= 1.0:
        t = 1.0
    else:
        t = (kappa - kappa_min) / (1.0 - kappa_min)
        t = max(0.0, min(1.0, t))  # Clamp to [0, 1]

    # Linear interpolation from 0.25 to 1.0
    return 0.25 + t * (1.0 - 0.25)


def get_optimizer_color(optimizer_name):
    """Get the base color for an optimizer from OPT_COLORS.

    Args:
        optimizer_name: Name of the optimizer (e.g., 'dana-star-mk4', 'ademamix')

    Returns:
        Hex color string
    """
    # Direct lookup
    if optimizer_name in OPT_COLORS:
        return OPT_COLORS[optimizer_name]

    # Try common variations
    variations = [
        optimizer_name.lower(),
        optimizer_name.replace('_', '-'),
        optimizer_name.replace('-', '_'),
    ]

    for var in variations:
        if var in OPT_COLORS:
            return OPT_COLORS[var]

    # Default to a distinct color if not found
    print(f"Warning: No color found for optimizer '{optimizer_name}', using default")
    return '#333333'


def get_clipsnr_from_metadata(metadata):
    """Extract clipsnr value from metadata.

    For 'adana' optimizer, returns 16 (treated as high clipsnr).
    """
    optimizer = metadata.get('optimizer', '')
    if optimizer == 'adana':
        return 16.0

    tp = metadata.get('training_params', {})
    return tp.get('clipsnr', 1.0)


def plot_hummingbird_multiopt(all_data, args):
    """Create the multi-optimizer hummingbird plot.

    Args:
        all_data: List of (metadata, results) tuples, one per data file
        args: Command line arguments
    """
    # Use sans-serif font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']

    fig, ax = plt.subplots(1, 1, figsize=tuple(args.figsize))

    # Track which Adam baselines we've plotted to avoid duplicates
    adam_plotted = False

    # Track legend entries
    legend_handles = []
    legend_labels = []

    # Process each data file
    for metadata, results in all_data:
        optimizer = metadata['optimizer']
        kappa_values = np.array(metadata['kappa_values'])

        # Get display name for legend
        display_name = OPT_DISPLAY_NAMES.get(optimizer, optimizer)

        # Get color based on mode
        if args.color_by_clipsnr:
            clipsnr = get_clipsnr_from_metadata(metadata)
            opt_color = get_clipsnr_color(clipsnr)
            # Use readable names based on optimizer type
            if optimizer == 'ademamix':
                color_label = 'Ademamix'
            elif clipsnr >= 16 or optimizer == 'adana':
                color_label = 'ADANA'
            elif optimizer == 'dana-star-mk4':
                color_label = 'DANA-Star-MK4'
            elif optimizer == 'dana-mk4':
                color_label = 'DANA-MK4'
            else:
                color_label = f'DANA-MK4 (c={clipsnr})'
        else:
            opt_color = get_optimizer_color(optimizer)
            color_label = display_name

        # Filter kappa values based on kappa_min
        valid_kappas = [k for k in kappa_values if k >= args.kappa_min]

        if len(valid_kappas) == 0:
            print(f"Warning: No kappa values >= {args.kappa_min} for {optimizer}")
            continue

        # Plot kappa sweep curves for this optimizer
        legend_line = None
        for kappa in valid_kappas:
            opt_name = f"{optimizer}_kappa{kappa:.1f}"
            if opt_name in results:
                timestamps = results[opt_name]['timestamps']
                losses = results[opt_name]['losses']

                # Get interpolated alpha (0.25 at kappa_min, 1.0 at kappa=1.0)
                alpha = interpolate_alpha(kappa, args.kappa_min)

                # Linewidth varies with kappa (thicker for higher kappa)
                linewidth = 3.0 + 4.0 * (kappa - args.kappa_min) / (1.0 - args.kappa_min) if args.kappa_min < 1.0 else 5.0

                line, = ax.loglog(timestamps, losses,
                         color=opt_color,
                         alpha=alpha,
                         linewidth=linewidth)

                # Keep the line with highest kappa for legend (most visible)
                if kappa == max(valid_kappas):
                    legend_line = line

        # Add single legend entry per optimizer
        if legend_line is not None:
            legend_handles.append(legend_line)
            legend_labels.append(color_label)

        # Plot Adam baseline (only once, using first data file that has it)
        if 'adam' in results and not args.no_adam and not adam_plotted:
            timestamps = results['adam']['timestamps']
            losses = results['adam']['losses']
            line, = ax.loglog(timestamps, losses,
                     color='black',
                     alpha=0.9,
                     linewidth=7,
                     linestyle='--')
            legend_handles.append(line)
            legend_labels.append('AdamW (baseline)')
            adam_plotted = True

    ax.set_xlabel('Training Iteration', fontsize=36, fontfamily='sans-serif')
    ax.set_ylabel('Population Risk', fontsize=36, fontfamily='sans-serif')

    # Set y-axis limits
    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])
    else:
        ax.set_ylim(0.1, 2.0)

    # Set title
    if args.title is not None:
        ax.set_title(args.title, fontsize=48, fontfamily='sans-serif')
    elif len(all_data) > 0:
        # Build title from first metadata (assuming same model/training params)
        metadata = all_data[0][0]
        mp = metadata['model_params']
        tp = metadata['training_params']

        # List all optimizers
        opt_names = [d[0]['optimizer'] for d in all_data]
        opt_str = ', '.join(opt_names)

        title = f'Hummingbird Plot (Multi-Optimizer)\n'
        title += f'Optimizers: {opt_str}\n'
        title += f'α={mp["alpha"]}, m={mp["m"]}, ζ={mp["zeta"]}, d={mp["d"]}, β={mp["beta"]}\n'
        title += f'batch={tp["batch_size"]}, steps={tp["steps"]}, κ_min={args.kappa_min}'
        ax.set_title(title, fontsize=14)

    ax.grid(True, alpha=0.3)
    ax.legend(legend_handles, legend_labels, loc='lower left', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=32)

    plt.tight_layout()

    # Save plot
    os.makedirs(args.output_dir, exist_ok=True)
    filepath = os.path.join(args.output_dir, f"{args.output_prefix}.pdf")
    plt.savefig(filepath, dpi=args.dpi, bbox_inches='tight')
    print(f"Hummingbird plot saved to: {filepath}")
    plt.close()


def print_metadata(metadata, data_file):
    """Print metadata summary for a single data file.

    Args:
        metadata: Dict containing metadata
        data_file: Path to the data file
    """
    print(f"\n--- {os.path.basename(data_file)} ---")
    print(f"Optimizer: {metadata['optimizer']}")
    print(f"Kappa values: {metadata['kappa_values']}")
    print(f"Steps: {metadata['training_params']['steps']}")


def main():
    """Main function."""
    args = parse_args()

    # Load all data files
    all_data = []

    print("="*80)
    print("LOADING DATA FILES")
    print("="*80)

    for data_file in args.data_files:
        if not os.path.exists(data_file):
            print(f"Warning: Data file not found: {data_file}, skipping...")
            continue

        print(f"Loading: {data_file}")
        metadata, results = load_data(data_file)
        all_data.append((metadata, results))
        print_metadata(metadata, data_file)

    if len(all_data) == 0:
        print("Error: No valid data files loaded")
        return

    print(f"\n{'='*80}")
    print(f"Loaded {len(all_data)} data files")
    print(f"Kappa minimum: {args.kappa_min}")
    print("="*80)

    # Print results summary
    print("\nTraining Results Summary:")
    for metadata, results in all_data:
        optimizer = metadata['optimizer']
        print(f"\n  {optimizer}:")
        for name, res in results.items():
            if name != 'adam':
                final_loss = res['losses'][-1]
                print(f"    {name}: final loss = {final_loss:.6e}")
    print()

    # Create plot
    print("Generating multi-optimizer hummingbird plot...")
    plot_hummingbird_multiopt(all_data, args)

    print("Done!")


if __name__ == "__main__":
    main()
