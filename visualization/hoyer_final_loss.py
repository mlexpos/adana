#!/usr/bin/env python
"""
Hoyer Final Loss Figure: Cross-entropy loss vs compute for different Hoyer coefficients.

Fetches final-val/loss and final-val/hoyer_loss from WandB Qwen3_Hoyer group,
computes cross-entropy loss, and plots vs compute (petaflop-days).

Key features:
- Uses final validation loss only (not full loss history)
- Colors by optimizer (from opt_colors.py)
- Line styles by hoyer_loss_coeff level
- Panel size/title/legend format from llm_hummingbird_slice_plot.py
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from opt_colors import OPT_COLORS, OPT_DISPLAY_NAMES
from wandb_cache import CachedWandBApi


WANDB_GROUP = 'Qwen3_Hoyer'

# Line styles for different Hoyer coefficient levels
HOYER_LINESTYLES = {
    0.0: '-',       # Solid for no Hoyer
    1e-4: '--',     # Dashed
    1e-3: '-.',     # Dash-dot
    1e-2: ':',      # Dotted
}


def get_hoyer_linestyle(hoyer_coeff):
    """Get line style for a given hoyer coefficient.

    Falls back to finding closest match if exact value not in dict.
    """
    if hoyer_coeff in HOYER_LINESTYLES:
        return HOYER_LINESTYLES[hoyer_coeff]

    # Find closest match
    if hoyer_coeff == 0 or hoyer_coeff < 1e-5:
        return '-'
    elif hoyer_coeff < 5e-4:
        return '--'
    elif hoyer_coeff < 5e-3:
        return '-.'
    else:
        return ':'


def get_hoyer_label(hoyer_coeff):
    """Get display label for hoyer coefficient."""
    if hoyer_coeff == 0 or hoyer_coeff < 1e-6:
        return 'λ=0'
    else:
        # Format as scientific notation
        exp = int(np.floor(np.log10(hoyer_coeff)))
        mantissa = hoyer_coeff / (10 ** exp)
        if abs(mantissa - 1.0) < 0.01:
            return f'λ=1e{exp}'
        else:
            return f'λ={mantissa:.1f}e{exp}'


def parse_args():
    parser = argparse.ArgumentParser(description="Hoyer Final Loss Plot from WandB data")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save plots")
    parser.add_argument("--output_prefix", type=str, default="hoyer_final_loss",
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
    parser.add_argument("--hoyer-coeff-min", type=float, default=None,
                       help="Minimum hoyer_loss_coeff to include")
    parser.add_argument("--hoyer-coeff-max", type=float, default=None,
                       help="Maximum hoyer_loss_coeff to include")
    parser.add_argument("--no-qknorm", action="store_true",
                       help="Include only runs with no_qknorm=True")
    parser.add_argument("--allow-incomplete", action="store_true",
                       help="Include runs that haven't completed")
    parser.add_argument("--force_refresh", action="store_true",
                       help="Force refresh WandB cache")
    parser.add_argument("--cache_dir", type=str, default="wandb_cache",
                       help="Directory for WandB cache")
    return parser.parse_args()


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


def compute_to_petaflop_days(flops):
    """Convert FLOPs to petaflop-days."""
    flops_per_petaflop_day = 1e15 * 86400
    return flops / flops_per_petaflop_day


def fetch_hoyer_final_data(api, entity, project, heads_filter=None, optimizers_filter=None,
                           hoyer_coeff_min=None, hoyer_coeff_max=None,
                           no_qknorm=False, allow_incomplete=False):
    """
    Fetch final validation data from WandB for Hoyer experiments.

    Returns:
        list: List of dicts with keys: optimizer, heads, hoyer_coeff, ce_loss, compute_pflop_days, run_name
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

        if hoyer_coeff_min is not None and hoyer_coeff < hoyer_coeff_min:
            skipped += 1
            continue

        if hoyer_coeff_max is not None and hoyer_coeff > hoyer_coeff_max:
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
            print(f"  Skipping {run.name} - incomplete ({actual_iter}/{iterations_config})")
            skipped += 1
            continue

        # Get final validation metrics from summary
        # Note: final-val/loss exists but hoyer_loss is stored as val/hoyer_loss
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

        # Compute total FLOPs
        local_batch_size = config.get('batch_size', 1)
        world_size = config.get('world_size', 1)
        acc_steps = config.get('acc_steps', 1)
        seq_len = config.get('sequence_length', config.get('seq_len', 512))
        global_batch_size = local_batch_size * world_size * acc_steps

        flops_per_iter = compute_flops_per_iteration(heads, batch_size=global_batch_size, seq_len=seq_len)
        total_flops = float(actual_iter) * flops_per_iter
        compute_pflop_days = compute_to_petaflop_days(total_flops)

        params = compute_qwen3_params(heads)

        data.append({
            'optimizer': opt,
            'heads': heads,
            'hoyer_coeff': hoyer_coeff,
            'ce_loss': ce_loss,
            'compute_pflop_days': compute_pflop_days,
            'total_params': params['total_params'],
            'run_name': run.name,
        })

        print(f"  {run.name}: heads={heads}, opt={opt}, hoyer={hoyer_coeff:.1e}, CE={ce_loss:.4f}, C={compute_pflop_days:.3f} PF-days")

    print(f"\nProcessed {total_runs} runs, skipped {skipped}, kept {len(data)} runs")
    return data


def plot_hoyer_final_loss(data, args):
    """Create the Hoyer final loss plot (CE loss vs compute)."""
    fig, ax = plt.subplots(1, 1, figsize=tuple(args.figsize))

    # Get unique optimizers and hoyer coefficients
    optimizers = sorted(set(d['optimizer'] for d in data))
    hoyer_coeffs = sorted(set(d['hoyer_coeff'] for d in data))

    print(f"Optimizers: {optimizers}")
    print(f"Hoyer coefficients: {hoyer_coeffs}")

    # Group data by optimizer and hoyer_coeff
    # For each group, we'll plot compute vs CE loss across different head sizes
    for opt in optimizers:
        for hoyer_coeff in hoyer_coeffs:
            group_data = [d for d in data if d['optimizer'] == opt and d['hoyer_coeff'] == hoyer_coeff]

            if not group_data:
                continue

            # Sort by compute
            group_data.sort(key=lambda x: x['compute_pflop_days'])

            computes = [d['compute_pflop_days'] for d in group_data]
            losses = [d['ce_loss'] for d in group_data]

            # Get styling
            color = OPT_COLORS.get(opt, '#333333')
            linestyle = get_hoyer_linestyle(hoyer_coeff)
            opt_name = OPT_DISPLAY_NAMES.get(opt, opt)
            hoyer_label = get_hoyer_label(hoyer_coeff)

            label = f'{opt_name} ({hoyer_label})'

            print(f"  Plotting {label}: {len(group_data)} points, loss range [{min(losses):.4f}, {max(losses):.4f}]")

            ax.plot(computes, losses,
                    color=color,
                    linestyle=linestyle,
                    linewidth=4,
                    marker='o',
                    markersize=8,
                    label=label)

    ax.set_xlabel('Compute (PetaFLOP-days)', fontsize=28)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=28)

    title = 'Qwen3 Hoyer Experiments - Final CE Loss vs Compute'
    ax.set_title(title, fontsize=42)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=20)
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
    print("Hoyer Final Loss Plot")
    print("="*70)
    print(f"Heads filter: {args.heads if args.heads else 'all'}")
    print(f"Optimizers filter: {args.optimizers if args.optimizers else 'all'}")
    print(f"Hoyer coeff range: [{args.hoyer_coeff_min}, {args.hoyer_coeff_max}]")
    print(f"No QK normalization filter: {args.no_qknorm}")
    print(f"Allow incomplete runs: {args.allow_incomplete}")
    print("="*70)

    api = CachedWandBApi(cache_dir=args.cache_dir, force_refresh=args.force_refresh)

    # Fetch data
    data = fetch_hoyer_final_data(
        api, args.entity, args.project,
        heads_filter=args.heads,
        optimizers_filter=args.optimizers,
        hoyer_coeff_min=args.hoyer_coeff_min,
        hoyer_coeff_max=args.hoyer_coeff_max,
        no_qknorm=args.no_qknorm,
        allow_incomplete=args.allow_incomplete,
    )

    if not data:
        print("ERROR: No data found. Check parameters.")
        return

    # Create plot
    print("\nGenerating plot...")
    plot_hoyer_final_loss(data, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
