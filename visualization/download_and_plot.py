import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Download and plot DanaStar experiment data')
parser.add_argument('--model', type=str, default='35M',
                    choices=['35M', '90M'],
                    help='Model size (default: 35M)')
args = parser.parse_args()

# Configuration based on model size
model_size = args.model
configs = {
    '35M': {
        'group': 'DanaStar_35M_LR_WD_Sweep',
        'title': 'DanaStar 35M: Validation Loss vs Learning Rate',
        'filename': '35m_lrwd_sweep.pdf'
    },
    '90M': {
        'group': 'DanaStar_90M_LR_WD_Sweep',
        'title': 'DanaStar 90M: Validation Loss vs Learning Rate',
        'filename': '90m_lrwd_sweep.pdf'
    }
}

config = configs[model_size]
print(f"Using configuration for {model_size}: {config}")

# Initialize wandb API
api = wandb.Api()

# Get runs from the specific group
project = "danastar"
group = config['group']

print(f"Downloading data for {model_size} model from project: {project}, group: {group}")

# Get all runs in the group
runs = api.runs(f"ep-rmt-ml-opt/{project}", filters={"group": group})

data = []
for run in runs:
    print(f"Processing run: {run.name}")

    # Get run config and summary
    config = run.config
    summary = run.summary

    # Extract the required fields
    lr = config.get('lr')
    wd_ts = config.get('wd_ts')
    dataset = config.get('dataset')
    val_loss = summary.get('val/loss')

    if all(x is not None for x in [lr, wd_ts, dataset, val_loss]):
        # Calculate omega = wd_ts * lr, rounded to 2 decimal places for color
        omega = round(wd_ts * lr, 2)
        # Also calculate omega rounded to 3 decimal places for fitting
        omega_3digits = round(wd_ts * lr, 3)

        data.append({
            'lr': lr,
            'wd_ts': wd_ts,
            'dataset': dataset,
            'val_loss': val_loss,
            'omega': omega,
            'omega_3digits': omega_3digits,
            'log_omega': np.log(omega) if omega > 0 else np.nan
        })
    else:
        print(f"Skipping run {run.name} - missing data: lr={lr}, wd_ts={wd_ts}, dataset={dataset}, val_loss={val_loss}")

# Convert to DataFrame
df = pd.DataFrame(data)
print(f"Downloaded {len(df)} runs with complete data")
print(f"Datasets found: {df['dataset'].unique()}")

# Remove outlier with validation loss around 4.1
print(f"Before filtering outliers: {len(df)} runs")
df = df[df['val_loss'] < 4.0]  # Exclude the large outlier
print(f"After filtering outliers: {len(df)} runs")

# Verify omega values rounded to 3 digits
unique_omega_3digits = sorted(df['omega_3digits'].unique())
print(f"Unique omega values (3 digits): {unique_omega_3digits}")
print(f"Number of unique omega values: {len(unique_omega_3digits)}")

# Create the plot
plt.figure(figsize=(10, 8))

# Create clipped plasma colormap (0 to 0.85 range)
plasma_cmap = plt.cm.plasma
plasma_colors = plasma_cmap(np.linspace(0, 0.85, 256))
clipped_plasma = ListedColormap(plasma_colors)

# Separate data by dataset
fineweb_data = df[df['dataset'] == 'fineweb']
fineweb_100_data = df[df['dataset'] == 'fineweb_100']

# Create scatter plots with different markers
if len(fineweb_data) > 0:
    scatter1 = plt.scatter(fineweb_data['lr'], fineweb_data['val_loss'],
                          c=fineweb_data['log_omega'], cmap=clipped_plasma,
                          marker='o', s=50, alpha=0.7, label='fineweb')

if len(fineweb_100_data) > 0:
    scatter2 = plt.scatter(fineweb_100_data['lr'], fineweb_100_data['val_loss'],
                          c=fineweb_100_data['log_omega'], cmap=clipped_plasma,
                          marker='x', s=50, alpha=0.7, label='fineweb_100')


# Add colorbar
if len(df) > 0:
    cbar = plt.colorbar(label='log(ω) where ω = wd_ts × lr')

# Set labels and title
plt.xlabel('Learning Rate (lr)')
plt.ylabel('Validation Loss (val/loss)')
title = config.get('title', f'DanaStar {model_size}: Validation Loss vs Learning Rate')
plt.title(title)
plt.legend()

# Use log scale for x-axis if needed (learning rates are often small)
plt.xscale('log')

# Save as PDF
plt.tight_layout()
filename = config.get('filename', f'{model_size.lower()}_lrwd_sweep.pdf')
plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
print(f"Plot saved as {filename}")

# Print summary statistics
print(f"\nSummary:")
print(f"Total runs plotted: {len(df)}")
print(f"fineweb runs: {len(fineweb_data)}")
print(f"fineweb_100 runs: {len(fineweb_100_data)}")
print(f"Learning rate range: {df['lr'].min():.2e} to {df['lr'].max():.2e}")
print(f"Validation loss range: {df['val_loss'].min():.4f} to {df['val_loss'].max():.4f}")
print(f"Omega range: {df['omega'].min():.4f} to {df['omega'].max():.4f}")