#!/usr/bin/env python3
"""
Simple WandB data plot for Dana-Star-MK4 sweep results

Plots final validation loss vs learning rate (log-log scale):
- Tab coloring by omega bins (following softplus_fitting_jax.py binning logic)
- Marker size based on log(clipsnr)
- No curve fitting
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def group_nearby_omegas(omega_values, tolerance_percent=1.0):
    """
    Group omega values that are within a specified percentage of each other.

    This function clusters omega values that are very close (within tolerance_percent)
    into groups and assigns each group a representative value.

    Args:
        omega_values (list): List of calculated omega values
        tolerance_percent (float): Tolerance as percentage (default: 1.0%)

    Returns:
        list: List of grouped omega values (same length as input)
    """
    if not omega_values:
        return []

    # Convert to numpy array and sort
    omegas = np.array(omega_values)
    sorted_indices = np.argsort(omegas)
    sorted_omegas = omegas[sorted_indices]

    # Group nearby values
    groups = []
    current_group = [sorted_omegas[0]]

    for i in range(1, len(sorted_omegas)):
        current_omega = sorted_omegas[i]
        group_mean = np.mean(current_group)

        # Check if current omega is within tolerance of the group mean
        relative_diff = abs(current_omega - group_mean) / group_mean * 100

        if relative_diff <= tolerance_percent:
            current_group.append(current_omega)
        else:
            groups.append(current_group)
            current_group = [current_omega]

    # Add the last group
    groups.append(current_group)

    # Create mapping from original omega to group representative
    omega_to_group = {}
    for group in groups:
        # Use the mean of the group as the representative value
        group_representative = round(np.mean(group), 3)
        for omega in group:
            omega_to_group[omega] = group_representative

    # Map back to original order
    grouped_omegas = []
    for original_omega in omega_values:
        grouped_omegas.append(omega_to_group[original_omega])

    return grouped_omegas

def load_wandb_data(project_name, group_name):
    """
    Load experiment data from Weights & Biases for Dana-Star-MK4 runs.

    Args:
        project_name (str): WandB project name
        group_name (str): WandB experiment group name

    Returns:
        pd.DataFrame: Processed experiment data with columns:
            - lr: learning rate
            - wd_ts: weight decay timescale
            - weight_decay: weight decay
            - clipsnr: clip SNR value
            - dataset: dataset name
            - val_loss: final validation loss
            - omega_3digits: grouped omega value (wd_ts * lr * weight_decay)
    """
    # Initialize WandB API
    api = wandb.Api()

    print(f"Downloading data from project: {project_name}, group: {group_name}")

    # Get all runs in the specified group
    runs = api.runs(f"ep-rmt-ml-opt/{project_name}", filters={"group": group_name})

    # First pass: collect all raw data with calculated omega values
    raw_data = []
    raw_omega_values = []

    for run in runs:
        print(f"Processing run: {run.name}")

        # Extract configuration and summary from run
        config = run.config
        summary = run.summary

        # Handle case where config is a JSON string (needs parsing)
        if isinstance(config, str):
            try:
                import json
                config = json.loads(config)
            except json.JSONDecodeError as e:
                print(f"Skipping run {run.name} - failed to parse config JSON: {e}")
                continue

        # Handle case where config might not be a dictionary after parsing
        if not isinstance(config, dict):
            print(f"Skipping run {run.name} - config is not a dictionary (type: {type(config)})")
            continue

        # Handle case where summary is an HTTPSummary object with _json_dict string attribute
        if hasattr(summary, '_json_dict') and isinstance(summary._json_dict, str):
            try:
                import json
                summary = json.loads(summary._json_dict)
            except json.JSONDecodeError as e:
                print(f"Skipping run {run.name} - failed to parse summary._json_dict: {e}")
                continue
        # Handle case where summary is a plain string (needs parsing)
        elif isinstance(summary, str):
            try:
                import json
                summary = json.loads(summary)
            except json.JSONDecodeError as e:
                print(f"Skipping run {run.name} - failed to parse summary JSON: {e}")
                continue

        # Get fields - WandB config format is {"field": {"desc": ..., "value": actual_value}}
        lr = config.get('lr', {}).get('value') if isinstance(config.get('lr'), dict) else config.get('lr')
        wd_ts = config.get('wd_ts', {}).get('value') if isinstance(config.get('wd_ts'), dict) else config.get('wd_ts')
        weight_decay = config.get('weight_decay', {}).get('value') if isinstance(config.get('weight_decay'), dict) else config.get('weight_decay')
        clipsnr = config.get('clipsnr', {}).get('value') if isinstance(config.get('clipsnr'), dict) else config.get('clipsnr')
        dataset = config.get('dataset', {}).get('value') if isinstance(config.get('dataset'), dict) else config.get('dataset')
        val_loss = summary.get('final-val/loss')

        # Get iteration counts to check if run completed
        iterations_config = config.get('iterations', {}).get('value') if isinstance(config.get('iterations'), dict) else config.get('iterations')
        actual_iter = summary.get('iter')

        required_fields = [lr, wd_ts, weight_decay, clipsnr, dataset, val_loss, iterations_config, actual_iter]

        if all(x is not None for x in required_fields):
            # Check if run completed all iterations
            if actual_iter < iterations_config:
                print(f"Skipping run {run.name} - incomplete: {actual_iter}/{iterations_config} iterations ({actual_iter/iterations_config*100:.1f}%)")
                continue

            # Calculate raw omega = wd_ts * lr * weight_decay (not rounded yet)
            raw_omega = wd_ts * lr * weight_decay

            raw_data.append({
                'lr': lr,
                'wd_ts': wd_ts,
                'weight_decay': weight_decay,
                'clipsnr': clipsnr,
                'dataset': dataset,
                'val_loss': val_loss,
                'raw_omega': raw_omega,
            })
            raw_omega_values.append(raw_omega)
        else:
            print(f"Skipping run {run.name} - missing data: lr={lr}, wd_ts={wd_ts}, weight_decay={weight_decay}, clipsnr={clipsnr}, dataset={dataset}, val_loss={val_loss}, iterations={iterations_config}, iter={actual_iter}")

    # Second pass: group nearby omega values (within 1% tolerance)
    print(f"Grouping omega values within 1% tolerance...")
    grouped_omega_values = group_nearby_omegas(raw_omega_values, tolerance_percent=1.0)

    # Third pass: create final data with grouped omega values
    data = []
    for i, raw_entry in enumerate(raw_data):
        grouped_omega = grouped_omega_values[i]

        # Create final data entry with grouped omega
        final_entry = raw_entry.copy()
        final_entry['omega_3digits'] = grouped_omega
        del final_entry['raw_omega']  # Remove temporary field

        data.append(final_entry)

    return pd.DataFrame(data)

def plot_sweep_results(df, output_filename='mk4_sweep_results.pdf'):
    """
    Create scatter plot of validation loss vs learning rate.

    Args:
        df (pd.DataFrame): Experiment data
        output_filename (str): Output PDF filename
    """
    print(f"\nCreating visualization with {len(df)} data points")

    # Get unique clipsnr values and assign colors
    unique_clipsnr = sorted(df['clipsnr'].unique())
    print(f"Unique clipsnr values: {unique_clipsnr}")

    # Use tab10 colormap for clipsnr values
    tab10_cmap = plt.cm.tab10
    clipsnr_to_color = {clipsnr: tab10_cmap(i % 10) for i, clipsnr in enumerate(unique_clipsnr)}

    # Get unique omega values for sizing
    unique_omegas = sorted(df['omega_3digits'].unique())
    print(f"Unique omega values: {unique_omegas}")

    # Map omega values to marker sizes (linearly spaced)
    min_size = 50
    max_size = 200
    if len(unique_omegas) > 1:
        omega_to_size = {omega: min_size + (max_size - min_size) * i / (len(unique_omegas) - 1)
                        for i, omega in enumerate(unique_omegas)}
    else:
        omega_to_size = {unique_omegas[0]: (min_size + max_size) / 2}

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot each clipsnr group separately for legend
    for clipsnr in unique_clipsnr:
        clipsnr_data = df[df['clipsnr'] == clipsnr]

        # Get marker sizes based on omega values
        sizes = [omega_to_size[omega] for omega in clipsnr_data['omega_3digits']]

        plt.scatter(clipsnr_data['lr'], clipsnr_data['val_loss'],
                   c=[clipsnr_to_color[clipsnr]], s=sizes, alpha=0.7,
                   label=f'clipsnr={clipsnr:.1f}', edgecolors='black', linewidths=0.5)

    # Formatting
    plt.xlabel('Learning Rate (lr)', fontsize=12)
    plt.ylabel('Validation Loss (final-val/loss)', fontsize=12)
    plt.title('Dana-Star-MK4 Sweep: Final Validation Loss vs Learning Rate', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    # Add note about marker size and color
    note_text = f'Color = clipsnr\nMarker size = ω (wd_ts×lr×weight_decay)'
    plt.text(0.02, 0.98, note_text,
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save plot
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as {output_filename}")

    # Print statistics
    print("\nDataset statistics:")
    print(f"Total runs: {len(df)}")
    print(f"Number of clipsnr values: {len(unique_clipsnr)}")
    print(f"Number of omega (ω) bins: {len(unique_omegas)}")
    print(f"Learning rate range: {df['lr'].min():.6f} to {df['lr'].max():.6f}")
    print(f"ClipSNR range: {df['clipsnr'].min():.2f} to {df['clipsnr'].max():.2f}")
    print(f"Omega (ω) range: {df['omega_3digits'].min():.3f} to {df['omega_3digits'].max():.3f}")
    print(f"Validation loss range: {df['val_loss'].min():.4f} to {df['val_loss'].max():.4f}")

    # Print best results
    print("\nTop 5 runs by validation loss:")
    best_runs = df.nsmallest(5, 'val_loss')[['lr', 'wd_ts', 'clipsnr', 'omega_3digits', 'val_loss']]
    for idx, row in best_runs.iterrows():
        print(f"  lr={row['lr']:.6f}, wd_ts={row['wd_ts']:.1f}, clipsnr={row['clipsnr']:.2f}, ω={row['omega_3digits']:.3f}, val_loss={row['val_loss']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Dana-Star-MK4 sweep results')
    parser.add_argument('--project', type=str, default='danastar',
                        help='WandB project name (default: danastar)')
    parser.add_argument('--group', type=str, default='DanaStar_MK4_Small_Sweep',
                        help='WandB group name (default: DanaStar_MK4_Small_Sweep)')
    parser.add_argument('--output', type=str, default='mk4_sweep_results.pdf',
                        help='Output PDF filename (default: mk4_sweep_results.pdf)')
    args = parser.parse_args()

    print("Dana-Star-MK4 Sweep Visualization")
    print("=" * 60)

    # Load data
    df = load_wandb_data(args.project, args.group)

    if len(df) == 0:
        print("ERROR: No data found. Check project and group names.")
    else:
        # Create plot
        plot_sweep_results(df, args.output)
        print("\nVisualization completed successfully!")
