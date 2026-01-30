#!/usr/bin/env python3
"""
Compare Scaling Rules Performance (BigHead vs EggHead vs Enoki vs Eryngii)

This script compares the scaling performance between different model architectures:
1. BigHead: depth-based scaling (n_layer = depth)
2. EggHead: quadratic depth scaling (n_layer = heads * (heads-1) / 2)
3. Enoki: DiLoco scaling (n_layer = 3 * heads / 4)
4. Enoki_Scaled: DiLoco scaling with ScaledGPT initialization
5. Eryngii: increased head dimension and depth scaling (n_layer = heads^2 / 8)
6. Eryngii_Scaled: increased head dimension and depth scaling with ScaledGPT initialization

For each architecture and model size, it takes the best final-val/loss achieved,
plots loss vs compute (or non-emb params), and fits saturated power laws: loss = a + b * X^c

Joint Fitting Approach:
- All curves are fit simultaneously with a SHARED saturation level 'a'
- Uses JAX + Adagrad optimization
- Log-space fitting for numerical stability
- Weighted MSE loss (larger models get more weight)
- Constraint: 0 < a < min(observed losses) via sigmoid transformation

Usage:
    python compare_scaling_rules.py --scaling-rules BigHead Enoki --optimizers adamw
    python compare_scaling_rules.py --scaling-rules BigHead Enoki --optimizers adamw mk4
    python compare_scaling_rules.py --scaling-rules BigHead EggHead Enoki --optimizers mk4 d-muon manau
    python compare_scaling_rules.py --scaling-rules BigHead Enoki Eryngii --optimizers adamw --fit-metric non_emb
"""

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style, rc, rcParams
import argparse
import warnings
import os
import json
import jax
import jax.numpy as jnp
from jax import grad, jit
import optax
from scipy.optimize import brentq

from opt_colors import OPT_COLORS, OPT_LINESTYLES, OPT_DISPLAY_NAMES, OPT_COLORS_KAPPA, OPT_DISPLAY_NAMES_KAPPA

warnings.filterwarnings('ignore')

# =============================================================================
# SCALING RULE CONFIGURATION
# =============================================================================

SCALING_RULE_CONFIG = {
    'BigHead': {
        'group': 'DanaStar_MK4_BigHead_Sweep',
        'color': 'tab:blue',
        'marker': 'o',
        'linestyle': '-',
    },
    'EggHead': {
        'group': 'DanaStar_MK4_EggHead_Sweep',
        'color': 'tab:green',
        'marker': 's',
        'linestyle': '--',
    },
    'Enoki': {
        'group': 'DanaStar_MK4_Enoki_Sweep',
        'color': 'tab:orange',
        'marker': 'D',
        'linestyle': '-.',
    },
    'Enoki_Scaled': {
        'group': 'Enoki_ScaledGPT',
        'color': 'tab:cyan',
        'marker': 'D',
        'linestyle': '-',
    },
    'Enoki_Scaled_adana_no_gradient': {
        'group': 'Enoki_ScaledGPT_short_average_impact',
        'color': 'tab:cyan',
        'marker': 'D',
        'linestyle': '-',
    },
    'Eryngii': {
        'group': 'eryngii_sweeps',
        'color': 'tab:purple',
        'marker': '^',
        'linestyle': ':',
    },
    'Eryngii_Scaled': {
        'group': 'Eryngii_ScaledGPT',
        'color': 'tab:pink',
        'marker': '^',
        'linestyle': '-',
    },
    'Qwen3_Scaled': {
        'group': 'Qwen3_ScaledGPT',
        'color': 'tab:purple',
        'marker': 's',
        'linestyle': '-',
    },
    'Qwen3_Hoyer': {
        'group': 'Qwen3_Hoyer',
        'color': 'tab:red',
        'marker': 's',
        'linestyle': '--',
    }
}

# Matplotlib formatting - improved aesthetics
style.use('seaborn-v0_8-darkgrid')
rc('font', family='sans-serif')
rcParams['font.weight'] = 'normal'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (16, 10)
rcParams['axes.linewidth'] = 1.5
rcParams['axes.edgecolor'] = '#333333'
rcParams['grid.alpha'] = 0.3
rcParams['grid.linestyle'] = '--'
rcParams['legend.framealpha'] = 0.95
rcParams['legend.edgecolor'] = '#333333'

# =============================================================================
# TICK AND LABEL SIZE CONFIGURATION
# =============================================================================

# Tick label sizes (the numbers on the axes)
TICK_LABELSIZE_X = 50
TICK_LABELSIZE_VALLOSS = 50
TICK_LABELSIZE_GAIN = 50
TICK_LABELSIZE_HEADS = 50  # For the top x-axis (heads/depth numbers)

# Tick width and length for major ticks
TICK_WIDTH_MAJOR = 2.5
TICK_LENGTH_MAJOR = 12

# Tick width and length for minor ticks
TICK_WIDTH_MINOR = 1.5
TICK_LENGTH_MINOR = 6

# Axis label font sizes (the axis titles like "Compute", "Val Loss", etc.)
AXIS_LABEL_FONTSIZE = 30
AXIS_LABEL_FONTSIZE_SECONDARY = 30  # For relative plot or when showing both loss and gain
AXIS_LABEL_FONTSIZE_GAIN = 30  # For compute gain axis
AXIS_LABEL_FONTSIZE_GAIN_SECONDARY = 30  # For compute gain axis when not primary
AXIS_LABEL_FONTSIZE_HEADS = 30  # For heads/depth axis label on top

# Legend font size
LEGEND_FONTSIZE = 35

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description='Compare scaling rules performance')
parser.add_argument('--scaling-rules', type=str, nargs='+', required=True,
                    choices=['BigHead', 'EggHead', 'Enoki', 'Enoki_Scaled', 'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', 'Qwen3_Hoyer'],
                    help='Scaling rules to compare (can specify multiple)')
parser.add_argument('--optimizers', type=str, nargs='+', required=True,
                    choices=['adamw', 'mk4', 'dana', 'ademamix', 'd-muon', 'manau', 'manau-hard', 'adamw-decaying-wd', 'dana-mk4', 'ademamix-decaying-wd', 'dana-star-no-tau', 'dana-star', 'dana-star-no-tau-kappa-0-8', 'dana-star-no-tau-kappa-0-85', 'dana-star-no-tau-kappa-0-9', 'dana-mk4-kappa-0-85', 'dana-star-mk4-kappa-0-85', 'dana-star-no-tau-dana-constant', 'dana-star-no-tau-beta2-constant', 'dana-star-no-tau-beta1', 'dana-star-no-tau-dana-constant-beta2-constant', 'dana-star-no-tau-dana-constant-beta1', 'dana-star-no-tau-dana-constant-beta2-constant-beta1', 'dana-star-no-tau-kappa-1-0', 'logadam', 'logadam-nesterov', 'adana-no-gradient'],
                    help='Optimizer types to analyze (can specify multiple, e.g., --optimizers adamw mk4)')
parser.add_argument('--project', type=str, default='danastar',
                    help='WandB project name (default: danastar)')
parser.add_argument('--entity', type=str, default='ep-rmt-ml-opt',
                    help='WandB entity name (default: ep-rmt-ml-opt)')
parser.add_argument('--output', type=str, default=None,
                    help='Output filename for plot (default: auto-generated)')
parser.add_argument('--n-steps', type=int, default=50000,
                    help='Number of optimization steps for joint fitting (default: 50000)')
parser.add_argument('--learning-rate', type=float, default=0.1,
                    help='Learning rate for Adagrad optimizer (default: 0.1)')
parser.add_argument('--min-compute', type=float, default=None,
                    help='Minimum compute threshold in PetaFlop-Hours (default: None)')
parser.add_argument('--fit-metric', type=str, default='compute',
                    choices=['compute', 'non_emb'],
                    help='Metric to use for fitting: compute (PFH) or non_emb (parameters) (default: compute)')
parser.add_argument('--a-lower-bound', type=float, default=0.0,
                    help='Lower bound constant for saturation parameter a (default: 0.0)')
parser.add_argument('--equal-weight', action='store_true',
                    help='Use equal weights for all data points instead of weighting by compute (default: False)')
parser.add_argument('--fit-relative-to-adamw', action='store_true',
                    help='Plot relative to AdamW baseline (AdamW appears as horizontal line at 0)')
parser.add_argument('--plot-compute-gain', type=str, nargs='+', default=None,
                    choices=['adamw', 'mk4', 'dana', 'ademamix', 'd-muon', 'manau', 'manau-hard', 'adamw-decaying-wd', 'dana-mk4', 'ademamix-decaying-wd', 'dana-star-no-tau', 'dana-star', 'dana-star-no-tau-kappa-0-8', 'dana-star-no-tau-kappa-0-85', 'dana-star-no-tau-kappa-0-9', 'dana-mk4-kappa-0-85', 'dana-star-mk4-kappa-0-85', 'dana-star-no-tau-dana-constant', 'dana-star-no-tau-beta2-constant', 'dana-star-no-tau-beta1', 'dana-star-no-tau-dana-constant-beta2-constant', 'dana-star-no-tau-dana-constant-beta1', 'dana-star-no-tau-dana-constant-beta2-constant-beta1', 'logadam', 'logadam-nesterov', 'adana-no-gradient'],
                    help='Plot compute gain relative to specified baseline optimizer(s). Can specify one or more baselines (will be included in fitting even if not in --optimizers)')
parser.add_argument('--affine-by-part', action='store_true',
                    help='Use piecewise affine interpolation of baseline data instead of fitted curve for compute gain calculation (requires --plot-compute-gain)')
parser.add_argument('--head-min', type=int, default=None,
                    help='Minimum head/depth size to include in fitting (still plots all points)')
parser.add_argument('--head-max', type=int, default=None,
                    help='Maximum head/depth size to include in fitting (still plots all points)')
parser.add_argument('--no-title', action='store_true',
                    help='Do not show title on plot')
parser.add_argument('--no-loss', action='store_true',
                    help='Hide loss points and fitted curves, show only compute gains (requires --plot-compute-gain)')
parser.add_argument('--no-heads-axis', action='store_true',
                    help='Hide the top x-axis showing head/depth counts')
parser.add_argument('--no-fit', action='store_true',
                    help='Skip plotting fitted curve compute gains, only show affine-by-part (requires --plot-compute-gain)')
parser.add_argument('--vertical-bar', action='store_true',
                    help='Show vertical grey line at 24 heads')
parser.add_argument('--kappa-ablation', action='store_true',
                    help='Use kappa-based colors and display names (show only kappa values in legend)')
args = parser.parse_args()

# Map optimizer names
optimizer_map = {'adamw': 'adamw', 'mk4': 'dana-star-mk4', 'dana': 'dana', 'ademamix': 'ademamix',
                 'd-muon': 'd-muon', 'manau': 'manau', 'manau-hard': 'manau-hard', 'adamw-decaying-wd': 'adamw-decaying-wd', 'dana-mk4': 'dana-mk4', 'ademamix-decaying-wd': 'ademamix-decaying-wd', 'dana-star-no-tau': 'dana-star-no-tau', 'dana-star': 'dana-star', 'dana-star-no-tau-kappa-0-8': 'dana-star-no-tau-kappa-0-8', 'dana-star-no-tau-kappa-0-85': 'dana-star-no-tau-kappa-0-85', 'dana-star-no-tau-kappa-0-9': 'dana-star-no-tau-kappa-0-9', 'dana-mk4-kappa-0-85': 'dana-mk4-kappa-0-85', 'dana-star-mk4-kappa-0-85': 'dana-star-mk4', 'dana-star-no-tau-dana-constant': 'dana-star-no-tau-dana-constant', 'dana-star-no-tau-beta2-constant': 'dana-star-no-tau-beta2-constant', 'dana-star-no-tau-beta1': 'dana-star-no-tau-beta1', 'dana-star-no-tau-dana-constant-beta2-constant': 'dana-star-no-tau-dana-constant-beta2-constant', 'dana-star-no-tau-dana-constant-beta1': 'dana-star-no-tau-dana-constant-beta1', 'dana-star-no-tau-dana-constant-beta2-constant-beta1': 'dana-star-no-tau-dana-constant-beta2-constant-beta1', 'dana-star-no-tau-kappa-1-0': 'dana-star-no-tau-kappa-1-0', 'logadam': 'logadam', 'logadam-nesterov': 'logadam-nesterov', 'adana-no-gradient': 'dana-star-no-tau-short-average-impact'}
optimizer_types = [optimizer_map[opt] for opt in args.optimizers]

# =============================================================================
# PARAMETER CALCULATION FUNCTIONS
# =============================================================================

def compute_params(size, scaling_rule):
    """
    Compute parameters for a given size and scaling rule.

    Args:
        size: For BigHead, this is depth. For EggHead/Enoki/Eryngii/Qwen3_Scaled/Qwen3_Hoyer, this is heads.
        scaling_rule: One of 'BigHead', 'EggHead', 'Enoki', 'Enoki_Scaled', 'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', 'Qwen3_Hoyer'

    Returns:
        dict with non_emb, total_params, compute (PFH), etc.
    """
    if scaling_rule == 'BigHead':
        # BigHead: depth-based scaling
        depth = size
        head_dim = 16 * depth
        n_embd = 16 * depth * depth
        mlp_hidden = 32 * depth * depth
        n_head = depth
        n_layer = depth

        # Non-embedding params
        non_emb = float(depth * (3 * head_dim * n_embd * n_head + n_embd * n_embd +
                                 2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd)

        # Total params
        vocab_size = 50304
        total_params = float(non_emb + 2 * n_embd * vocab_size)

    elif scaling_rule == 'EggHead':
        # EggHead: quadratic depth scaling
        heads = size
        head_dim = 16 * heads
        n_embd = 16 * heads * heads
        mlp_hidden = 32 * heads * heads
        n_head = heads
        n_layer = int(heads * (heads - 1) / 2)

        # Non-embedding params
        non_emb = float(n_layer * (3 * head_dim * n_embd * n_head + n_embd * n_embd +
                                    2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd)

        # Total params
        vocab_size = 50304
        total_params = float(non_emb + 2 * n_embd * vocab_size)

    elif scaling_rule == 'Enoki' or scaling_rule == 'Enoki_Scaled':
        # Enoki and Enoki_Scaled: DiLoco scaling
        heads = size
        head_dim = 64  # Fixed
        n_embd = heads * 64
        mlp_hidden = 4 * n_embd
        n_head = heads
        n_layer = int(3 * heads // 4)

        # Non-embedding params (DiLoco formula)
        non_emb = float(12 * n_embd * n_embd * n_layer)

        # Total params
        vocab_size = 50304
        total_params = float(non_emb + 2 * n_embd * vocab_size)

    elif scaling_rule == 'Eryngii' or scaling_rule == 'Eryngii_Scaled':
        # Eryngii and Eryngii_Scaled: increased head dimension and depth scaling
        heads = size
        head_dim = int(round(32 * heads / 3 / 8) * 8)  # Rounded to multiple of 8
        n_head = heads
        n_layer = int(heads * heads // 8)
        n_embd = n_head * head_dim
        mlp_hidden = 4 * n_embd

        # Non-embedding params (DiLoco formula)
        non_emb = float(12 * n_embd * n_embd * n_layer)

        # Total params
        vocab_size = 50304
        total_params = float(non_emb + 2 * n_embd * vocab_size)

    elif scaling_rule == 'Qwen3_Scaled' or scaling_rule == 'Qwen3_Hoyer':
        # Qwen3_Scaled / Qwen3_Hoyer: heads-based Qwen3 scaling with elementwise gating
        # head_dim=128, n_layer=2*heads, n_embd=128*heads, mlp_hidden=3*n_embd
        heads = size
        head_dim = 128
        n_head = heads
        n_layer = 2 * heads
        n_embd = 128 * heads
        total_qkv_dim = n_head * head_dim

        # Qwen3 with gating: non_emb = n_layer * (5 * n_embd * total_qkv_dim + 2 * head_dim + 9 * n_embd^2 + 2 * n_embd) + n_embd
        # Per layer:
        # - attn = 5 * n_embd * total_qkv_dim  # q_proj (2x with gating) + k_proj + v_proj + o_proj
        # - qk_norm = 2 * head_dim
        # - mlp = 9 * n_embd^2  # SwiGLU: gate_proj + up_proj + down_proj (mlp_hidden = 3 * n_embd)
        # - layer_norms = 2 * n_embd
        per_layer = 5 * n_embd * total_qkv_dim + 2 * head_dim + 9 * n_embd * n_embd + 2 * n_embd
        non_emb = float(n_layer * per_layer + n_embd)  # +n_embd for final norm

        # Total params
        vocab_size = 50304
        total_params = float(non_emb + 2 * n_embd * vocab_size)

    else:
        raise ValueError(f"Unknown scaling rule: {scaling_rule}")

    # Compute in FLOPs
    compute_flops = 6.0 * non_emb * total_params * 20.0

    # Convert to PetaFlop-Hours: 1 PFH = 3600e15 FLOPs
    compute_pfh = compute_flops / (3600e15)

    return {
        'non_emb': int(non_emb),
        'total_params': int(total_params),
        'compute': compute_pfh,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd
    }

# =============================================================================
# DATA LOADING
# =============================================================================

def load_scaling_rule_data(scaling_rule, project, entity, optimizer_type, min_compute=None, head_min=None, head_max=None, kappa_filter=None):
    """Load data for a scaling rule and get best loss for each size.

    Args:
        scaling_rule: One of 'BigHead', 'EggHead', 'Enoki'
        project: WandB project name
        entity: WandB entity name
        optimizer_type: Type of optimizer to filter for
        min_compute: Minimum compute threshold in PFH (optional)
        head_min: Minimum head/depth size to include in fitting (optional)
        head_max: Maximum head/depth size to include in fitting (optional)
    """
    api = wandb.Api()

    # Special handling for adana-no-gradient: use different group
    effective_scaling_rule = scaling_rule
    if optimizer_type == 'dana-star-no-tau-short-average-impact':
        effective_scaling_rule = scaling_rule + '_adana_no_gradient'
        if effective_scaling_rule not in SCALING_RULE_CONFIG:
            effective_scaling_rule = scaling_rule  # fallback to original

    config = SCALING_RULE_CONFIG[effective_scaling_rule]
    group = config['group']

    print(f"Loading {scaling_rule} data from {group}...")
    runs = api.runs(f"{entity}/{project}", filters={"group": group})

    data = []
    for run in runs:
        # Handle different wandb API versions where config might be a string or dict
        run_config = run.config
        if isinstance(run_config, str):
            # In wandb version 0.22.2, config is returned as a JSON string
            # Parse it to get a dictionary
            try:
                run_config = json.loads(run_config)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  Warning: Could not parse config JSON for run {run.name}, skipping...")
                continue

        # Convert to dict if it's a wandb Config object
        if hasattr(run_config, 'as_dict'):
            run_config = run_config.as_dict()
        elif not isinstance(run_config, dict):
            # If it's not a dict and doesn't have as_dict, try to convert
            try:
                run_config = dict(run_config)
            except (TypeError, ValueError):
                print(f"  Warning: Could not convert config for run {run.name}, skipping...")
                continue

        # Extract value from nested dict structure if needed
        # In some versions, config has structure {"key": {"value": actual_value}}
        def extract_value(config_dict):
            """Extract values from nested config structure."""
            result = {}
            for key, val in config_dict.items():
                if isinstance(val, dict) and 'value' in val:
                    result[key] = val['value']
                else:
                    result[key] = val
            return result

        run_config = extract_value(run_config)

        # Handle summary (also may be a JSON string in some wandb versions)
        summary = run.summary
        if hasattr(summary, '_json_dict') and isinstance(summary._json_dict, str):
            try:
                summary = json.loads(summary._json_dict)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  Warning: Could not parse summary JSON for run {run.name}, skipping...")
                continue
        elif not isinstance(summary, dict) and hasattr(summary, '__getitem__'):
            # If it's a wandb Summary object with dict-like access, use it as-is
            pass

        # Filter by optimizer
        opt = run_config.get('opt', '')
        if opt != optimizer_type:
            continue

        # Special filter for adana-no-gradient: require g2_factor = 0
        if optimizer_type == 'dana-star-no-tau-short-average-impact':
            g2_factor = run_config.get('g2_factor', None)
            if g2_factor != 0:
                continue

        # Special filter for dana-star-mk4-kappa-0-85: require kappa = 0.85
        if kappa_filter is not None:
            kappa = run_config.get('kappa', None)
            if kappa != kappa_filter:
                continue

        # Check completion
        actual_iter = summary.get('iter', 0)
        iterations_config = run_config.get('iterations', 0)
        if actual_iter > 0 and iterations_config > 0 and actual_iter < iterations_config:
            continue

        # Get size parameter based on scaling rule
        if scaling_rule == 'BigHead':
            size = run_config.get('n_layer')  # depth
        else:  # EggHead, Enoki, Enoki_Scaled, Eryngii, Eryngii_Scaled, Qwen3_Scaled, or Qwen3_Hoyer
            size = run_config.get('n_head')  # heads

        val_loss = summary.get('final-val/loss')

        if size is None or val_loss is None:
            continue

        data.append({
            'size': size,
            'val_loss': val_loss,
            'run_name': run.name
        })

    df = pd.DataFrame(data)
    print(f"  Loaded {len(df)} {scaling_rule} runs")

    if len(df) == 0:
        return df

    # Get best loss for each size
    best_by_size = df.groupby('size')['val_loss'].min().reset_index()

    # Add compute information
    results = []
    for _, row in best_by_size.iterrows():
        size = row['size']
        params = compute_params(size, scaling_rule)

        # Apply minimum compute filter if specified
        if min_compute is not None and params['compute'] < min_compute:
            continue

        result_row = {
            'size': size,
            'val_loss': row['val_loss'],
            'compute': params['compute'],
            'non_emb': params['non_emb'],
            'total_params': params['total_params'],
            'scaling_rule': scaling_rule
        }
        
        # Only add use_for_fit column if head_min or head_max is specified
        if head_min is not None or head_max is not None:
            use_for_fit = True
            if head_min is not None:
                use_for_fit = use_for_fit and (size >= head_min)
            if head_max is not None:
                use_for_fit = use_for_fit and (size <= head_max)
            result_row['use_for_fit'] = use_for_fit
        
        results.append(result_row)

    result_df = pd.DataFrame(results)
    if min_compute is not None and len(result_df) > 0:
        print(f"  Filtered to {len(result_df)} data points with compute >= {min_compute:.4e} PFH")
    if (head_min is not None or head_max is not None) and len(result_df) > 0 and 'use_for_fit' in result_df.columns:
        n_fit = result_df['use_for_fit'].sum()
        range_desc = []
        if head_min is not None:
            range_desc.append(f"head/depth >= {head_min}")
        if head_max is not None:
            range_desc.append(f"head/depth <= {head_max}")
        print(f"  Using {n_fit}/{len(result_df)} data points for fitting ({' and '.join(range_desc)})")

    return result_df

# =============================================================================
# JOINT FITTING FUNCTIONS (JAX + Adagrad)
# =============================================================================

@jit
def saturated_power_law(x, a, b, c):
    """Saturated power law: y = a + b * x^c"""
    return a + b * jnp.power(x, c)


@jit
def broken_power_law(x, a, b, c, e, f):
    """Broken power law: y = a + b*x^{-c} + e*x^{-f}

    Parameters:
        x: compute (or other metric)
        a: shared saturation level (asymptotic loss)
        b: coefficient for first power law term
        c: exponent for first power law term (positive, used as -c)
        e: coefficient for second power law term
        f: exponent for second power law term (positive, used as -f)
    """
    return a + b * jnp.power(x, -c) + e * jnp.power(x, -f)

def joint_fit_saturated_power_laws(datasets, n_steps=50000, lr=0.1, a_lower_bound=0.0):
    """
    Fit saturated power laws to multiple datasets with a SHARED saturation level 'a'.

    Args:
        datasets: List of dicts, each with 'x' and 'y' arrays
        n_steps: Number of optimization steps
        lr: Learning rate for Adagrad
        a_lower_bound: Non-trainable lower bound constant for parameter a (default: 0.0)

    Returns:
        Fitted parameters for each curve
    """
    n_curves = len(datasets)

    # Get min observed loss across all datasets (for constraint)
    min_loss = min(float(jnp.min(d['y'])) for d in datasets)

    # Initialize parameters
    # Use log-parameterization for numerical stability
    # a_raw will be transformed: a = min_loss * sigmoid(a_raw)
    params = {
        'a_raw': jnp.array(0.0),  # Shared saturation level (raw)
        'log_b': jnp.array([jnp.log(0.1) for _ in range(n_curves)]),
        'c': jnp.array([-0.5 for _ in range(n_curves)])
    }

    # Prepare data
    x_data = [jnp.array(d['x'], dtype=jnp.float32) for d in datasets]
    y_data = [jnp.array(d['y'], dtype=jnp.float32) for d in datasets]
    weights_data = [jnp.array(d['weights'], dtype=jnp.float32) for d in datasets]

    def loss_fn(params):
        """Weighted MSE loss in log space"""
        a = a_lower_bound + jax.nn.sigmoid(params['a_raw']) * (min_loss - a_lower_bound)
        total_loss = 0.0

        for i in range(n_curves):
            b = jnp.exp(params['log_b'][i])
            c = params['c'][i]

            y_pred = saturated_power_law(x_data[i], a, b, c)

            # Log-space loss for numerical stability
            log_y_true = jnp.log(y_data[i])
            log_y_pred = jnp.log(jnp.maximum(y_pred, 1e-10))

            residuals = log_y_pred - log_y_true
            weighted_mse = jnp.sum(weights_data[i] * residuals**2) / jnp.sum(weights_data[i])

            total_loss += weighted_mse

        return total_loss / n_curves

    # Optimize with Adagrad
    optimizer = optax.adagrad(learning_rate=lr)
    opt_state = optimizer.init(params)

    @jit
    def step(params, opt_state):
        loss_value, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    # Training loop
    print(f"\nJoint fitting with {n_curves} curves...")
    for i in range(n_steps):
        params, opt_state, loss_value = step(params, opt_state)

        if i % 10000 == 0 or i == n_steps - 1:
            print(f"  Step {i:6d}: Loss = {float(loss_value):.6f}")

    # Extract final parameters
    a_fit = float(a_lower_bound + jax.nn.sigmoid(params['a_raw']) * (min_loss - a_lower_bound))
    results = []

    for i in range(n_curves):
        b_fit = float(jnp.exp(params['log_b'][i]))
        c_fit = float(params['c'][i])

        results.append({
            'a': a_fit,
            'b': b_fit,
            'c': c_fit
        })

    return results

def fit_all_saturated_power_laws_joint(data_list, n_steps=50000, learning_rate=0.1, a_lower_bound=0.0):
    """
    Fit BROKEN POWER LAWS to multiple datasets with a SHARED saturation level 'a'.

    Broken power law: loss = a + b*C^{-c} + e*C^{-f}
    where:
        - 'a' (saturation level) is shared across all curves
        - 'b', 'c', 'e', 'f' are fit per optimizer

    Initial values: a=0.11, b=0.50, c=0.050, e=2.50, f=0.045

    Args:
        data_list: List of dicts, each with 'compute', 'loss', and 'name' keys
        n_steps: Number of optimization steps
        learning_rate: Learning rate for Adagrad optimizer
        a_lower_bound: Non-trainable lower bound constant for parameter a (default: 0.0)

    Returns:
        Dict with 'a' (shared saturation) and 'curves' (dict mapping name -> params)
    """
    print(f"\nPreparing {len(data_list)} curves for joint BROKEN POWER LAW fitting...")
    print(f"Model: loss = a + b*C^{{-c}} + e*C^{{-f}}")
    print(f"  - 'a' shared across all optimizers")
    print(f"  - 'b', 'c', 'e', 'f' fit per optimizer")

    # Convert data to JAX arrays
    jax_data = []
    for i, data in enumerate(data_list):
        compute_arr = jnp.array(data['compute'], dtype=jnp.float32)
        loss_arr = jnp.array(data['loss'], dtype=jnp.float32)
        name = data['name']

        print(f"  Curve {i}: {name}")
        print(f"    Data points: {len(compute_arr)}")
        print(f"    Compute range: {float(jnp.min(compute_arr)):.4e} to {float(jnp.max(compute_arr)):.4e}")
        print(f"    Loss range: {float(jnp.min(loss_arr)):.4f} to {float(jnp.max(loss_arr)):.4f}")

        # Set weights: equal weights if requested, otherwise weight by compute
        if args.equal_weight:
            weights_arr = jnp.ones_like(compute_arr)
        else:
            weights_arr = compute_arr  # Weight by compute (larger models matter more)

        jax_data.append({
            'compute': compute_arr,
            'loss': loss_arr,
            'name': name,
            'weights': weights_arr
        })

    # Initialize parameters for broken power law
    # Initial values: a=0.11, b=0.50, c=0.050, e=2.50, f=0.045
    # Params: [a_raw, log(b_0), c_0, log(e_0), f_0, log(b_1), c_1, log(e_1), f_1, ...]
    n_curves = len(jax_data)

    # Initialize a_raw to give a ~ 0.11 when min_loss ~ 2.5
    # Using sigmoid: a = a_lower_bound + sigmoid(a_raw) * (min_loss * 0.99 - a_lower_bound)
    # For a ~ 0.11 with min_loss ~ 2.5, we need sigmoid(a_raw) ~ 0.11/2.475 ~ 0.044
    # sigmoid^{-1}(0.044) ~ -3.1
    init_a_raw = -3.1

    init_params = [init_a_raw]  # a_raw
    for i in range(n_curves):
        # [log(b_i), c_i, log(e_i), f_i] with initial values b=0.50, c=0.050, e=2.50, f=0.045
        init_params.extend([
            jnp.log(0.50),  # log(b)
            0.050,          # c (positive, will be used as -c in power)
            jnp.log(2.50),  # log(e)
            0.045           # f (positive, will be used as -f in power)
        ])

    fit_params = jnp.array(init_params, dtype=jnp.float32)

    @jit
    def loss_fn(params):
        """Joint loss function for all curves with shared saturation (broken power law)."""
        # Extract shared saturation
        a_raw = params[0]
        min_loss = jnp.min(jnp.array([jnp.min(d['loss']) for d in jax_data]))
        a = a_lower_bound + jax.nn.sigmoid(a_raw) * (min_loss * 0.99 - a_lower_bound)

        total_loss = 0.0
        total_weight = 0.0

        for i in range(n_curves):
            # Extract per-curve parameters: [log(b), c, log(e), f]
            log_b = params[1 + 4*i]
            c = params[1 + 4*i + 1]
            log_e = params[1 + 4*i + 2]
            f = params[1 + 4*i + 3]

            b = jnp.exp(log_b)
            e = jnp.exp(log_e)

            compute_i = jax_data[i]['compute']
            loss_i = jax_data[i]['loss']
            weights_i = jax_data[i]['weights']

            # Broken power law: loss = a + b*C^{-c} + e*C^{-f}
            pred_loss = a + b * jnp.power(compute_i, -c) + e * jnp.power(compute_i, -f)

            # Log-space fitting for numerical stability
            log_loss_true = jnp.log(loss_i)
            log_loss_pred = jnp.log(jnp.maximum(pred_loss, 1e-10))

            # Weighted MSE in log space
            residuals = (log_loss_pred - log_loss_true) ** 2
            curve_loss = jnp.sum(weights_i * residuals)
            curve_weight = jnp.sum(weights_i)

            total_loss += curve_loss
            total_weight += curve_weight

        return total_loss / total_weight

    # Optimize
    optimizer = optax.adagrad(learning_rate)
    opt_state = optimizer.init(fit_params)
    grad_fn = jit(grad(loss_fn))

    best_loss = float('inf')
    best_params = fit_params

    print(f"\nStarting optimization (broken power law)...")
    for step in range(n_steps):
        grads = grad_fn(fit_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        fit_params = optax.apply_updates(fit_params, updates)

        current_loss = float(loss_fn(fit_params))
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = fit_params

        if step % 10000 == 0 or step == n_steps - 1:
            a_raw = best_params[0]
            min_loss = float(jnp.min(jnp.array([jnp.min(d['loss']) for d in jax_data])))
            a = float(a_lower_bound + jax.nn.sigmoid(a_raw) * (min_loss * 0.99 - a_lower_bound))
            print(f"  Step {step:5d}: loss={best_loss:.6e}, a={a:.4f}")

    # Extract final parameters
    a_raw = best_params[0]
    min_loss = float(jnp.min(jnp.array([jnp.min(d['loss']) for d in jax_data])))
    a = float(a_lower_bound + jax.nn.sigmoid(a_raw) * (min_loss * 0.99 - a_lower_bound))

    results = {
        'a': a,
        'curves': {}
    }

    for i in range(n_curves):
        log_b = float(best_params[1 + 4*i])
        c = float(best_params[1 + 4*i + 1])
        log_e = float(best_params[1 + 4*i + 2])
        f = float(best_params[1 + 4*i + 3])

        b = float(jnp.exp(log_b))
        e = float(jnp.exp(log_e))

        name = jax_data[i]['name']
        compute_vals = np.array(jax_data[i]['compute'])
        loss_vals = np.array(jax_data[i]['loss'])

        # Compute R-squared using broken power law
        predictions = a + b * np.power(compute_vals, -c) + e * np.power(compute_vals, -f)
        residuals = loss_vals - predictions
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((loss_vals - np.mean(loss_vals))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        results['curves'][name] = {
            'b': b,
            'c': c,
            'e': e,
            'f': f,
            'r_squared': r_squared
        }

    return results

# =============================================================================
# PLOTTING
# =============================================================================

def _display_name(opt_short):
    """Convert optimizer short name to display name for plots."""
    return OPT_DISPLAY_NAMES.get(opt_short, opt_short)

def _filename_safe_name(opt_short):
    """Convert optimizer short name to filename-safe version."""
    display = OPT_DISPLAY_NAMES.get(opt_short, opt_short)
    # Replace special characters with safe alternatives for filenames
    return (display
            .replace('*', '-Star')
            .replace('κ', '-kappa-')
            .replace('β₁', '-beta1')
            .replace('β₂', '-beta2')
            .replace(' ', '')
            .replace('(', '')
            .replace(')', ''))

def _display_rule(rule):
    """Convert rule name to display name for plots."""
    return rule.replace('Enoki_Scaled', '').replace('_Scaled', '').strip()

def plot_comparison_multi_optimizer(data_dict, fit_results, scaling_rules, optimizer_shorts, optimizer_types, fit_metric, show_title=True, show_heads_axis=True, head_max=None, show_vertical_bar=False):
    """
    Plot comparison of multiple optimizers and scaling rules on a single plot.

    Args:
        data_dict: Dict {optimizer_type: {scaling_rule: DataFrame}}
        fit_results: Dict with 'a' and 'curves' from joint fitting
        scaling_rules: List of scaling rule names
        optimizer_shorts: List of short optimizer names
        optimizer_types: List of full optimizer type names
        fit_metric: 'compute' or 'non_emb'
        show_title: Whether to show the plot title (default: True)
        head_max: Maximum head/depth size to limit x-axis range (default: None)
    """
    fig, ax = plt.subplots(figsize=(18, 10))

    # Scaling rule markers
    rule_markers = {
        'BigHead': 'D',
        'EggHead': 's',
        'Enoki': 'o',
        'Eryngii': '^'
    }

    # Scaling rule line styles
    rule_linestyles = {
        'BigHead': '-',
        'EggHead': '--',
        'Enoki': ':',
        'Eryngii': '-.'
    }

    # Collect all metric values for plot range
    all_metric_vals = []
    for opt_type in optimizer_types:
        for rule in scaling_rules:
            df = data_dict[opt_type][rule]
            if len(df) > 0:
                all_metric_vals.extend(df[fit_metric].values)

    if len(all_metric_vals) > 0:
        metric_min = np.min(all_metric_vals)
        metric_max = np.max(all_metric_vals)
        
        # If head_max is specified, limit the upper bound of the plot range
        if head_max is not None:
            # Find the metric value corresponding to head_max for any scaling rule
            max_metric_at_head_max = None
            for rule in scaling_rules:
                params_at_head_max = compute_params(head_max, rule)
                rule_metric = params_at_head_max[fit_metric]
                if max_metric_at_head_max is None or rule_metric > max_metric_at_head_max:
                    max_metric_at_head_max = rule_metric
            
            # Use the smaller of the actual max or head_max metric
            if max_metric_at_head_max is not None:
                metric_max = min(metric_max, max_metric_at_head_max)
        
        # Determine the plot range multiplier based on whether head_max is specified
        upper_multiplier = 1.15 if head_max is not None else 2.0
        plot_range = np.logspace(np.log10(metric_min * 0.5), np.log10(metric_max * upper_multiplier), 200)
    else:
        plot_range = None

    # Collect handles and labels for custom legend ordering
    # We'll plot without labels first, then create custom legend
    fit_handles = {}
    obs_handles = {}

    metric_symbol = 'C' if fit_metric == 'compute' else 'P'

    # Plot each optimizer x scaling_rule combination
    for opt_idx, opt_type in enumerate(optimizer_types):
        opt_short = optimizer_shorts[opt_idx]
        color = OPT_COLORS.get(opt_short, 'black')

        for rule in scaling_rules:
            df = data_dict[opt_type][rule]

            if len(df) == 0:
                continue

            marker = rule_markers.get(rule, 'x')
            linestyle = rule_linestyles.get(rule, '-')

            # Separate points used for fitting vs. only for display
            if 'use_for_fit' in df.columns:
                df_fit = df[df['use_for_fit']]
                df_no_fit = df[~df['use_for_fit']]
            else:
                df_fit = df
                df_no_fit = pd.DataFrame()

            # Plot observed data used for fitting (solid) - improved styling
            if len(df_fit) > 0:
                scatter = ax.scatter(df_fit[fit_metric], df_fit['val_loss'],
                          s=150, marker=marker, c=color, edgecolors='white', linewidths=2.0,
                          zorder=10, alpha=0.85)

                obs_key = (rule, opt_short)
                obs_handles[obs_key] = (scatter, f'{_display_name(opt_short)} {_display_rule(rule)}')
            
            # Plot observed data NOT used for fitting (hollow/faded)
            if len(df_no_fit) > 0:
                ax.scatter(df_no_fit[fit_metric], df_no_fit['val_loss'],
                          s=150, marker=marker, facecolors='none', edgecolors=color, linewidths=2.0,
                          zorder=10, alpha=0.5)

            # Plot fitted curve if available (no label yet)
            curve_name = f'{opt_short}_{rule}'
            if curve_name in fit_results['curves'] and plot_range is not None:
                a = fit_results['a']
                b = fit_results['curves'][curve_name]['b']
                c = fit_results['curves'][curve_name]['c']
                e = fit_results['curves'][curve_name]['e']
                f = fit_results['curves'][curve_name]['f']
                r2 = fit_results['curves'][curve_name]['r_squared']

                # Broken power law: loss = a + b*C^{-c} + e*C^{-f}
                loss_fit = a + b * np.power(plot_range, -c) + e * np.power(plot_range, -f)

                line, = ax.plot(plot_range, loss_fit, linestyle=linestyle, color=color, linewidth=3.0,
                       zorder=9, alpha=0.9)

                fit_key = (rule, opt_short)
                # Print fit formula to output
                print(f"  Fit: {_display_name(opt_short)} {_display_rule(rule)}: {a:.3f}+{b:.2e}{metric_symbol}^{{-{c:.3f}}}+{e:.2e}{metric_symbol}^{{-{f:.3f}}} (R²={r2:.3f})")
                fit_handles[fit_key] = (line, f'{_display_name(opt_short)} {_display_rule(rule)}')

    # Create custom ordered legend
    # Order: For each scaling rule, show fit curves (or observed if no fit)
    legend_handles = []
    legend_labels = []

    for rule in scaling_rules:
        for opt_short in optimizer_shorts:
            fit_key = (rule, opt_short)
            obs_key = (rule, opt_short)
            
            # Prefer fit curve with equation, otherwise just show optimizer name
            if fit_key in fit_handles:
                handle, label = fit_handles[fit_key]
                legend_handles.append(handle)
                legend_labels.append(label)
            elif obs_key in obs_handles:
                handle, label = obs_handles[obs_key]
                legend_handles.append(handle)
                legend_labels.append(label)

    # Get metric info
    if fit_metric == 'compute':
        xlabel = 'Compute (PFH)'
    else:  # non_emb
        xlabel = 'Non-embedding Parameters'

    # Formatting - improved aesthetics with larger axis titles
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Val Loss', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Set y-axis limit for val loss
    ax.set_ylim(top=4.5)
    
    # Set x-axis limits when head_max is specified
    if head_max is not None and len(all_metric_vals) > 0:
        ax.set_xlim(metric_min * 0.5, metric_max * 1.15)
    
    # Configure tick appearance - all axes
    ax.tick_params(axis='x', which='major', labelsize=TICK_LABELSIZE_X, width=TICK_WIDTH_MAJOR, length=TICK_LENGTH_MAJOR)
    ax.tick_params(axis='x', which='minor', width=TICK_WIDTH_MINOR, length=TICK_LENGTH_MINOR)
    ax.tick_params(axis='y', which='major', labelsize=TICK_LABELSIZE_VALLOSS, width=TICK_WIDTH_MAJOR, length=TICK_LENGTH_MAJOR)
    ax.tick_params(axis='y', which='minor', labelsize=TICK_LABELSIZE_VALLOSS, width=TICK_WIDTH_MINOR, length=TICK_LENGTH_MINOR)
    
    # Ensure y-axis minor tick labels use the same font size as major ticks
    from matplotlib.ticker import LogFormatterSciNotation
    ax.yaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(2, 0.4)))
    for label in ax.yaxis.get_minorticklabels():
        label.set_fontsize(TICK_LABELSIZE_VALLOSS)

    if show_title:
        opts_str = ', '.join([_display_name(opt) for opt in optimizer_shorts])
        rules_str = ' vs '.join(scaling_rules)
        ax.set_title(f'Scaling Laws Comparison: {rules_str}\nOptimizers: {opts_str} (Shared saturation a = {fit_results["a"]:.4f})',
                    fontsize=20, fontweight='bold', pad=20)

    ax.legend(legend_handles, legend_labels, fontsize=LEGEND_FONTSIZE, loc='best', framealpha=0.95, ncol=2, 
              edgecolor='#333333', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Remove spines for cleaner look
    # Hide top spine when heads axis is not shown
    if not show_heads_axis:
        ax.spines['top'].set_visible(False)
        ax.tick_params(top=False)
    else:
        # When heads axis is shown, hide top spine of main axis (ax2 will have its own)
        ax.spines['top'].set_visible(False)
        ax.tick_params(top=False)
    
    # Always hide right spine (no compute gains in this plot)
    ax.spines['right'].set_visible(False)
    ax.tick_params(right=False)
    
    # Add vertical line at 24 heads (if requested)
    if show_vertical_bar:
        # Calculate metric value for 24 heads from any available scaling rule
        metric_24_heads = None
        for rule in scaling_rules:
            if rule in ['Enoki', 'Enoki_Scaled', 'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', 'Qwen3_Hoyer']:
                # For head-based scaling rules, compute metric for 24 heads
                params_24 = compute_params(24, rule)
                metric_24_heads = params_24[fit_metric]
                break
        
        if metric_24_heads is not None:
            ax.axvline(x=metric_24_heads, color='gray', linestyle='-', linewidth=2.0, alpha=0.5, zorder=1)

    # Add second x-axis showing size (heads or depth) on top
    if show_heads_axis:
        ax2 = ax.twiny()
        ax2.set_xscale('log')
        ax2.set_xlim(ax.get_xlim())
        
        # Collect all sizes from data
        all_sizes = set()
        size_to_metric = {}
        for opt_type in optimizer_types:
            for rule in scaling_rules:
                df = data_dict[opt_type][rule]
                if len(df) > 0:
                    for _, row in df.iterrows():
                        size = row['size']
                        metric_val = row[fit_metric]
                        all_sizes.add(size)
                        if size not in size_to_metric:
                            size_to_metric[size] = metric_val
        
        if len(all_sizes) > 0:
            all_sizes_sorted = sorted(all_sizes)
            
            # Show fewer ticks - select every other one or every third one depending on count
            if len(all_sizes_sorted) > 10:
                # Show every third tick
                selected_indices = range(0, len(all_sizes_sorted), 3)
            elif len(all_sizes_sorted) > 6:
                # Show every other tick
                selected_indices = range(0, len(all_sizes_sorted), 2)
            else:
                # Show all ticks
                selected_indices = range(len(all_sizes_sorted))
            
            selected_sizes = [all_sizes_sorted[i] for i in selected_indices]
            ax2.set_xticks([size_to_metric[size] for size in selected_sizes])
            ax2.set_xticklabels([str(size) for size in selected_sizes], fontweight='bold', fontsize=TICK_LABELSIZE_HEADS)
            
            # Set tick parameters for top x-axis
            ax2.tick_params(axis='x', which='major', labelsize=TICK_LABELSIZE_HEADS, width=TICK_WIDTH_MAJOR, length=TICK_LENGTH_MAJOR, pad=10)
            ax2.tick_params(axis='x', which='minor', width=TICK_WIDTH_MINOR, length=TICK_LENGTH_MINOR)
            
            # Hide bottom, left, and right spines of ax2 (only show top spine)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # Determine label based on scaling rules (use 'Heads' for head-based, 'Depth' for BigHead)
            if 'BigHead' in scaling_rules and len(scaling_rules) == 1:
                size_label = 'Depth'
            else:
                size_label = 'Heads'
            ax2.set_xlabel(size_label, fontsize=AXIS_LABEL_FONTSIZE_HEADS, fontweight='bold')

    plt.tight_layout()

    return fig

def plot_comparison_relative_to_adamw(data_dict, fit_results, scaling_rules, optimizer_shorts, optimizer_types, fit_metric, show_title=True, show_heads_axis=True, head_max=None, show_vertical_bar=False):
    """
    Plot comparison with AdamW normalized to horizontal line (slope 0 in log-log space).
    
    In log-log space, we plot log(loss) vs log(metric).
    The fit is: log(loss) = log(a + b * metric^c)
    
    To make AdamW have slope 0, we subtract its log-loss from all curves:
    y_normalized = log(loss) - log(loss_adamw)
    
    This makes AdamW a horizontal line at y=0, and other optimizers' slopes 
    become relative to AdamW's scaling behavior.

    Args:
        data_dict: Dict {optimizer_type: {scaling_rule: DataFrame}}
        fit_results: Dict with 'a' and 'curves' from joint fitting
        scaling_rules: List of scaling rule names
        optimizer_shorts: List of short optimizer names
        optimizer_types: List of full optimizer type names
        fit_metric: 'compute' or 'non_emb'
        show_title: Whether to show the plot title (default: True)
        head_max: Maximum head/depth size to limit x-axis range (default: None)
    """
    fig, ax = plt.subplots(figsize=(18, 10))

    # Scaling rule markers
    rule_markers = {
        'BigHead': 'D',
        'EggHead': 's',
        'Enoki': 'o',
        'Eryngii': '^'
    }

    # Scaling rule line styles
    rule_linestyles = {
        'BigHead': '-',
        'EggHead': '--',
        'Enoki': ':',
        'Eryngii': '-.'
    }

    # Collect all metric values for plot range
    all_metric_vals = []
    for opt_type in optimizer_types:
        for rule in scaling_rules:
            df = data_dict[opt_type][rule]
            if len(df) > 0:
                all_metric_vals.extend(df[fit_metric].values)

    if len(all_metric_vals) > 0:
        metric_min = np.min(all_metric_vals)
        metric_max = np.max(all_metric_vals)
        
        # If head_max is specified, limit the upper bound of the plot range
        if head_max is not None:
            # Find the metric value corresponding to head_max for any scaling rule
            max_metric_at_head_max = None
            for rule in scaling_rules:
                params_at_head_max = compute_params(head_max, rule)
                rule_metric = params_at_head_max[fit_metric]
                if max_metric_at_head_max is None or rule_metric > max_metric_at_head_max:
                    max_metric_at_head_max = rule_metric
            
            # Use the smaller of the actual max or head_max metric
            if max_metric_at_head_max is not None:
                metric_max = min(metric_max, max_metric_at_head_max)
        
        # Determine the plot range multiplier based on whether head_max is specified
        upper_multiplier = 1.15 if head_max is not None else 2.0
        plot_range = np.logspace(np.log10(metric_min * 0.5), np.log10(metric_max * upper_multiplier), 200)
    else:
        plot_range = None

    # Collect handles and labels for custom legend ordering
    fit_handles = {}
    obs_handles = {}

    metric_symbol = 'C' if fit_metric == 'compute' else 'P'

    # Get AdamW fitted curves for each scaling rule (for normalization)
    adamw_fits = {}
    a = fit_results['a']
    
    for rule in scaling_rules:
        curve_name = f'adamw_{rule}'
        if curve_name in fit_results['curves']:
            adamw_fits[rule] = {
                'a': a,
                'b': fit_results['curves'][curve_name]['b'],
                'c': fit_results['curves'][curve_name]['c'],
                'e': fit_results['curves'][curve_name]['e'],
                'f': fit_results['curves'][curve_name]['f']
            }

    # Plot each optimizer x scaling_rule combination
    for opt_idx, opt_type in enumerate(optimizer_types):
        opt_short = optimizer_shorts[opt_idx]
        color = OPT_COLORS.get(opt_short, 'black')

        for rule in scaling_rules:
            df = data_dict[opt_type][rule]

            if len(df) == 0:
                continue

            marker = rule_markers.get(rule, 'x')
            linestyle = rule_linestyles.get(rule, '-')

            # Get AdamW fit for this scaling rule
            if rule not in adamw_fits:
                continue  # Skip if no AdamW baseline

            adamw_a = adamw_fits[rule]['a']
            adamw_b = adamw_fits[rule]['b']
            adamw_c = adamw_fits[rule]['c']
            adamw_e = adamw_fits[rule]['e']
            adamw_f = adamw_fits[rule]['f']

            # Separate points used for fitting vs. only for display
            if 'use_for_fit' in df.columns:
                df_fit = df[df['use_for_fit']]
                df_no_fit = df[~df['use_for_fit']]
            else:
                df_fit = df
                df_no_fit = pd.DataFrame()

            # Calculate normalized observed losses for fitting points
            if len(df_fit) > 0:
                metric_vals = df_fit[fit_metric].values
                observed_losses = df_fit['val_loss'].values
                # Broken power law: loss = a + b*C^{-c} + e*C^{-f}
                adamw_baseline = adamw_a + adamw_b * np.power(metric_vals, -adamw_c) + adamw_e * np.power(metric_vals, -adamw_f)
                normalized_losses = observed_losses / adamw_baseline

                # Plot normalized observed data (log scale for y) - improved styling
                scatter = ax.scatter(metric_vals, normalized_losses,
                          s=150, marker=marker, c=color, edgecolors='white', linewidths=2.0,
                          zorder=10, alpha=0.85)

                obs_key = (rule, opt_short)
                obs_handles[obs_key] = (scatter, f'{_display_name(opt_short)} {_display_rule(rule)}')
            
            # Calculate normalized observed losses for non-fitting points (hollow/faded)
            if len(df_no_fit) > 0:
                metric_vals_no_fit = df_no_fit[fit_metric].values
                observed_losses_no_fit = df_no_fit['val_loss'].values
                adamw_baseline_no_fit = adamw_a + adamw_b * np.power(metric_vals_no_fit, -adamw_c) + adamw_e * np.power(metric_vals_no_fit, -adamw_f)
                normalized_losses_no_fit = observed_losses_no_fit / adamw_baseline_no_fit

                ax.scatter(metric_vals_no_fit, normalized_losses_no_fit,
                          s=150, marker=marker, facecolors='none', edgecolors=color, linewidths=2.0,
                          zorder=10, alpha=0.5)

            # Plot normalized fitted curve if available
            curve_name = f'{opt_short}_{rule}'
            if curve_name in fit_results['curves'] and plot_range is not None:
                opt_a = fit_results['a']
                opt_b = fit_results['curves'][curve_name]['b']
                opt_c = fit_results['curves'][curve_name]['c']
                opt_e = fit_results['curves'][curve_name]['e']
                opt_f = fit_results['curves'][curve_name]['f']
                r2 = fit_results['curves'][curve_name]['r_squared']

                # Calculate normalized fit using broken power law
                # y_norm = loss / loss_adamw
                opt_fit = opt_a + opt_b * np.power(plot_range, -opt_c) + opt_e * np.power(plot_range, -opt_f)
                adamw_baseline_curve = adamw_a + adamw_b * np.power(plot_range, -adamw_c) + adamw_e * np.power(plot_range, -adamw_f)
                normalized_fit = opt_fit / adamw_baseline_curve

                line, = ax.plot(plot_range, normalized_fit, linestyle=linestyle, color=color, linewidth=3.0,
                       zorder=9, alpha=0.9)

                fit_key = (rule, opt_short)
                
                # For AdamW, it will be a constant at 1.0 (log scale makes this a horizontal line)
                if opt_short == 'adamw':
                    print(f"  Fit: {_display_name(opt_short)} {_display_rule(rule)}: baseline (ratio=1)")
                    fit_handles[fit_key] = (line, f'{_display_name(opt_short)} {_display_rule(rule)}')
                else:
                    print(f"  Fit: {_display_name(opt_short)} {_display_rule(rule)}: broken power law / adamw (R²={r2:.3f})")
                    fit_handles[fit_key] = (line, f'{_display_name(opt_short)} {_display_rule(rule)}')

    # Create custom ordered legend
    legend_handles = []
    legend_labels = []

    for rule in scaling_rules:
        for opt_short in optimizer_shorts:
            fit_key = (rule, opt_short)
            obs_key = (rule, opt_short)
            
            # Prefer fit curve with equation, otherwise just show optimizer name
            if fit_key in fit_handles:
                handle, label = fit_handles[fit_key]
                legend_handles.append(handle)
                legend_labels.append(label)
            elif obs_key in obs_handles:
                handle, label = obs_handles[obs_key]
                legend_handles.append(handle)
                legend_labels.append(label)

    # Get metric info
    if fit_metric == 'compute':
        xlabel = 'Compute (PFH)'
    else:  # non_emb
        xlabel = 'Non-embedding Parameters'

    # Formatting - both axes in log scale, improved aesthetics with larger axis titles
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE_SECONDARY, fontweight='bold')
    ax.set_ylabel('Validation Loss Ratio (Loss / Loss$_{AdamW}$)', fontsize=AXIS_LABEL_FONTSIZE_SECONDARY, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Set y-axis limit for val loss ratio
    ax.set_ylim(top=4.5)
    
    # Set x-axis limits when head_max is specified
    if head_max is not None and len(all_metric_vals) > 0:
        ax.set_xlim(metric_min * 0.5, metric_max * 1.15)
    
    # Configure tick appearance - all axes
    ax.tick_params(axis='x', which='major', labelsize=TICK_LABELSIZE_X, width=TICK_WIDTH_MAJOR, length=TICK_LENGTH_MAJOR)
    ax.tick_params(axis='x', which='minor', width=TICK_WIDTH_MINOR, length=TICK_LENGTH_MINOR)
    ax.tick_params(axis='y', which='major', labelsize=TICK_LABELSIZE_VALLOSS, width=TICK_WIDTH_MAJOR, length=TICK_LENGTH_MAJOR)
    ax.tick_params(axis='y', which='minor', labelsize=TICK_LABELSIZE_VALLOSS, width=TICK_WIDTH_MINOR, length=TICK_LENGTH_MINOR)
    
    # Ensure y-axis minor tick labels use the same font size as major ticks
    from matplotlib.ticker import LogFormatterSciNotation
    ax.yaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(2, 0.4)))
    for label in ax.yaxis.get_minorticklabels():
        label.set_fontsize(TICK_LABELSIZE_VALLOSS)
    
    # Horizontal line at y=1 represents AdamW baseline (slope 0 in log-log)
    ax.axhline(y=1.0, color='#333333', linestyle='--', linewidth=2.5, alpha=0.8, label='AdamW baseline (ratio=1)', zorder=5)

    if show_title:
        opts_str = ', '.join([_display_name(opt) for opt in optimizer_shorts])
        rules_str = ' vs '.join(scaling_rules)
        ax.set_title(f'Scaling Laws Comparison (Relative to AdamW, Log-Log): {rules_str}\nOptimizers: {opts_str}',
                    fontsize=20, fontweight='bold', pad=20)

    ax.legend(legend_handles, legend_labels, fontsize=LEGEND_FONTSIZE, loc='best', framealpha=0.95, ncol=2,
              edgecolor='#333333', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Remove spines for cleaner look
    # Hide top spine when heads axis is not shown
    if not show_heads_axis:
        ax.spines['top'].set_visible(False)
        ax.tick_params(top=False)
    else:
        # When heads axis is shown, hide top spine of main axis (ax2 will have its own)
        ax.spines['top'].set_visible(False)
        ax.tick_params(top=False)
    
    # Always hide right spine (no compute gains in this plot)
    ax.spines['right'].set_visible(False)
    ax.tick_params(right=False)
    
    # Add vertical line at 24 heads (if requested)
    if show_vertical_bar:
        # Calculate metric value for 24 heads from any available scaling rule
        metric_24_heads = None
        for rule in scaling_rules:
            if rule in ['Enoki', 'Enoki_Scaled', 'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', 'Qwen3_Hoyer']:
                # For head-based scaling rules, compute metric for 24 heads
                params_24 = compute_params(24, rule)
                metric_24_heads = params_24[fit_metric]
                break
        
        if metric_24_heads is not None:
            ax.axvline(x=metric_24_heads, color='gray', linestyle='-', linewidth=2.0, alpha=0.5, zorder=1)

    # Add second x-axis showing size (heads or depth) on top
    if show_heads_axis:
        ax2 = ax.twiny()
        ax2.set_xscale('log')
        ax2.set_xlim(ax.get_xlim())
        
        # Collect all sizes from data
        all_sizes = set()
        size_to_metric = {}
        for opt_type in optimizer_types:
            for rule in scaling_rules:
                df = data_dict[opt_type][rule]
                if len(df) > 0:
                    for _, row in df.iterrows():
                        size = row['size']
                        metric_val = row[fit_metric]
                        all_sizes.add(size)
                        if size not in size_to_metric:
                            size_to_metric[size] = metric_val
        
        if len(all_sizes) > 0:
            all_sizes_sorted = sorted(all_sizes)
            
            # Show fewer ticks - select every other one or every third one depending on count
            if len(all_sizes_sorted) > 10:
                # Show every third tick
                selected_indices = range(0, len(all_sizes_sorted), 3)
            elif len(all_sizes_sorted) > 6:
                # Show every other tick
                selected_indices = range(0, len(all_sizes_sorted), 2)
            else:
                # Show all ticks
                selected_indices = range(len(all_sizes_sorted))
            
            selected_sizes = [all_sizes_sorted[i] for i in selected_indices]
            ax2.set_xticks([size_to_metric[size] for size in selected_sizes])
            ax2.set_xticklabels([str(size) for size in selected_sizes], fontweight='bold', fontsize=TICK_LABELSIZE_HEADS)
            
            # Set tick parameters for top x-axis
            ax2.tick_params(axis='x', which='major', labelsize=TICK_LABELSIZE_HEADS, width=TICK_WIDTH_MAJOR, length=TICK_LENGTH_MAJOR, pad=10)
            ax2.tick_params(axis='x', which='minor', width=TICK_WIDTH_MINOR, length=TICK_LENGTH_MINOR)
            
            # Hide bottom, left, and right spines of ax2 (only show top spine)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # Determine label based on scaling rules (use 'Depth' for BigHead, 'Heads' for head-based)
            if 'BigHead' in scaling_rules and len(scaling_rules) == 1:
                size_label = 'Depth'
            else:
                size_label = 'Heads'
            ax2.set_xlabel(size_label, fontsize=AXIS_LABEL_FONTSIZE_HEADS, fontweight='bold')

    plt.tight_layout()

    return fig

def plot_compute_gain(data_dict, fit_results, scaling_rules, optimizer_shorts, optimizer_types, fit_metric, baseline_optimizers, use_affine_by_part=False, show_title=True, no_loss=False, show_heads_axis=True, head_max=None, no_fit=False, show_vertical_bar=False, kappa_ablation=False):
    """
    Plot compute gain percentage relative to specified baseline optimizer(s).
    Same as plot_comparison_multi_optimizer but with added right y-axis for compute gain.
    
    Now plots BOTH affine-by-part and fitted curve compute gains simultaneously:
    - Affine-by-part: scatter points + connecting lines (solid/dashed based on optimizer)
    - Fitted curve: smooth dashed lines with lower opacity
    
    Args:
        baseline_optimizers: List of short names of baseline optimizer(s) (e.g., ['adamw-decaying-wd'])
        use_affine_by_part: If True, plot affine-by-part compute gain (always plotted now)
        show_title: Whether to show the plot title (default: True)
        no_loss: If True, hide loss points and fitted curves, show only compute gains (default: False)
        head_max: Maximum head/depth size to limit x-axis range (default: None)
    """
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Scaling rule markers
    rule_markers = {
        'BigHead': 'D',
        'EggHead': 's',
        'Enoki': 'o',
        'Enoki_Scaled': 'o',
        'Eryngii': '^',
        'Eryngii_Scaled': '^',
        'Qwen3_Scaled': 's',
        'Qwen3_Hoyer': 's'
    }
    
    # Scaling rule line styles
    rule_linestyles = {
        'BigHead': '-',
        'EggHead': '--',
        'Enoki': ':',
        'Enoki_Scaled': '-',
        'Eryngii': '-.',
        'Eryngii_Scaled': '-',
        'Qwen3_Scaled': '-',
        'Qwen3_Hoyer': '--'
    }
    
    # Collect all metric values for plot range
    all_metric_vals = []
    for opt_type in optimizer_types:
        for rule in scaling_rules:
            df = data_dict[opt_type][rule]
            if len(df) > 0:
                all_metric_vals.extend(df[fit_metric].values)
    
    if len(all_metric_vals) > 0:
        metric_min = np.min(all_metric_vals)
        metric_max = np.max(all_metric_vals)
        
        # If head_max is specified, limit the upper bound of the plot range
        if head_max is not None:
            # Find the metric value corresponding to head_max for any scaling rule
            max_metric_at_head_max = None
            for rule in scaling_rules:
                params_at_head_max = compute_params(head_max, rule)
                rule_metric = params_at_head_max[fit_metric]
                if max_metric_at_head_max is None or rule_metric > max_metric_at_head_max:
                    max_metric_at_head_max = rule_metric
            
            # Use the smaller of the actual max or head_max metric
            if max_metric_at_head_max is not None:
                metric_max = min(metric_max, max_metric_at_head_max)
        
        # Determine the plot range multiplier based on whether head_max is specified
        upper_multiplier = 1.15 if head_max is not None else 2.0
        plot_range = np.logspace(np.log10(metric_min * 0.5), np.log10(metric_max * upper_multiplier), 200)
    else:
        plot_range = None
    
    # Collect handles and labels for custom legend ordering
    fit_handles = {}
    obs_handles = {}
    gain_handles = {}  # For compute gain plots when no_loss=True
    
    metric_symbol = 'C' if fit_metric == 'compute' else 'P'
    a = fit_results['a']
    
    # Get baseline data/fits for compute gain calculation
    # Now we ALWAYS prepare both affine-by-part and fitted curves
    # Support multiple baselines: each baseline is a dict indexed by rule
    baseline_fits = {}  # baseline_fits[baseline_opt][rule] = {...}
    baseline_affine_funcs = {}  # baseline_affine_funcs[baseline_opt][rule] = {...}
    
    # Create piecewise affine interpolation from baseline optimizer's actual data
    print(f"\nPreparing baseline optimizer(s) data: {', '.join(baseline_optimizers)}")
    
    for baseline_optimizer in baseline_optimizers:
        # Get baseline optimizer's full name
        baseline_opt_type = optimizer_map[baseline_optimizer]
        
        baseline_fits[baseline_optimizer] = {}
        baseline_affine_funcs[baseline_optimizer] = {}
        
        for rule in scaling_rules:
            # Get baseline optimizer's data for this scaling rule
            if baseline_opt_type in data_dict and rule in data_dict[baseline_opt_type]:
                baseline_df = data_dict[baseline_opt_type][rule]
                
                if len(baseline_df) > 0:
                    # Sort by metric (compute or non_emb)
                    baseline_df_sorted = baseline_df.sort_values(by=fit_metric)
                    baseline_metric = baseline_df_sorted[fit_metric].values
                    baseline_loss = baseline_df_sorted['val_loss'].values
                    
                    print(f"  {baseline_optimizer} {rule}: {len(baseline_metric)} affine points from {baseline_metric[0]:.4e} to {baseline_metric[-1]:.4e}")
                    
                    # Create interpolation function (linear interpolation in log-log space)
                    # This creates an affine-by-part function: loss = f(metric)
                    # We'll store both forward (metric -> loss) and need inverse (loss -> metric)
                    baseline_affine_funcs[baseline_optimizer][rule] = {
                        'metric': baseline_metric,
                        'loss': baseline_loss
                    }
            
            # Also get fitted curves
            baseline_curve_name = f'{baseline_optimizer}_{rule}'
            if baseline_curve_name in fit_results['curves']:
                baseline_fits[baseline_optimizer][rule] = {
                    'b': fit_results['curves'][baseline_curve_name]['b'],
                    'c': fit_results['curves'][baseline_curve_name]['c'],
                    'e': fit_results['curves'][baseline_curve_name]['e'],
                    'f': fit_results['curves'][baseline_curve_name]['f']
                }
                print(f"  {baseline_optimizer} {rule}: fitted curve available")
    
    # Create second y-axis for compute gain
    ax_gain = ax.twinx()
    
    # Plot each optimizer x scaling_rule combination
    for opt_idx, opt_type in enumerate(optimizer_types):
        opt_short = optimizer_shorts[opt_idx]
        
        # Use kappa-based colors and names if kappa_ablation is enabled
        if kappa_ablation:
            color = OPT_COLORS_KAPPA.get(opt_short, OPT_COLORS.get(opt_short, 'black'))
            gain_color = color
        else:
            color = OPT_COLORS.get(opt_short, 'black')
            # For compute gain plots: override mk4 colors to match their non-mk4 counterparts
            gain_color = color
            if opt_short == 'dana-mk4':
                gain_color = OPT_COLORS.get('dana-star-no-tau', color)
            elif opt_short == 'dana-mk4-kappa-0-85':
                gain_color = OPT_COLORS.get('dana-star-no-tau-kappa-0-85', color)
        
        for rule in scaling_rules:
            df = data_dict[opt_type][rule]
            
            if len(df) == 0:
                continue
            
            marker = rule_markers.get(rule, 'x')
            linestyle = rule_linestyles.get(rule, '-')
            
            # Separate points used for fitting vs. only for display
            if 'use_for_fit' in df.columns:
                df_fit = df[df['use_for_fit']]
                df_no_fit = df[~df['use_for_fit']]
            else:
                df_fit = df
                df_no_fit = pd.DataFrame()
            
            # Plot observed data used for fitting - improved styling
            if not no_loss:
                if len(df_fit) > 0:
                    scatter = ax.scatter(df_fit[fit_metric], df_fit['val_loss'],
                              s=150, marker=marker, c=color, edgecolors='white', linewidths=2.0,
                              zorder=10, alpha=0.85)
                    
                    obs_key = (rule, opt_short)
                    if kappa_ablation:
                        display_name = OPT_DISPLAY_NAMES_KAPPA.get(opt_short, OPT_DISPLAY_NAMES.get(opt_short, opt_short))
                        label = f'{display_name} {_display_rule(rule)}' if display_name.startswith('κ') else f'{_display_name(opt_short)} {_display_rule(rule)}'
                    else:
                        label = f'{_display_name(opt_short)} {_display_rule(rule)}'
                    obs_handles[obs_key] = (scatter, label)
                
                # Plot observed data NOT used for fitting (hollow/faded)
                if len(df_no_fit) > 0:
                    ax.scatter(df_no_fit[fit_metric], df_no_fit['val_loss'],
                              s=150, marker=marker, facecolors='none', edgecolors=color, linewidths=2.0,
                              zorder=10, alpha=0.5)
                
                # Plot fitted curve if available
                curve_name = f'{opt_short}_{rule}'
                if curve_name in fit_results['curves'] and plot_range is not None:
                    b = fit_results['curves'][curve_name]['b']
                    c = fit_results['curves'][curve_name]['c']
                    e = fit_results['curves'][curve_name]['e']
                    f = fit_results['curves'][curve_name]['f']
                    r2 = fit_results['curves'][curve_name]['r_squared']
                    
                    # Broken power law: loss = a + b*C^{-c} + e*C^{-f}
                    loss_fit = a + b * np.power(plot_range, -c) + e * np.power(plot_range, -f)
                    
                    line, = ax.plot(plot_range, loss_fit, linestyle=linestyle, color=color, linewidth=3.0,
                           zorder=9, alpha=0.9)
                    
                    fit_key = (rule, opt_short)
                    # Print fit formula to output
                    print(f"  Fit: {_display_name(opt_short)} {_display_rule(rule)}: {a:.3f}+{b:.2e}{metric_symbol}^{{-{c:.3f}}}+{e:.2e}{metric_symbol}^{{-{f:.3f}}} (R²={r2:.3f})")
                    if kappa_ablation:
                        display_name = OPT_DISPLAY_NAMES_KAPPA.get(opt_short, OPT_DISPLAY_NAMES.get(opt_short, opt_short))
                        label = f'{display_name} {_display_rule(rule)}' if display_name.startswith('κ') else f'{_display_name(opt_short)} {_display_rule(rule)}'
                    else:
                        label = f'{_display_name(opt_short)} {_display_rule(rule)}'
                    fit_handles[fit_key] = (line, label)
            
            # Plot compute gain on right axis
            # Define curve_name for use in compute gain calculations
            curve_name = f'{opt_short}_{rule}'
            
            # Determine which baseline to use for this optimizer
            # If there's only one baseline: compute gain relative to it (skip if optimizer IS the baseline)
            # If k baselines are given: match optimizer by index (1st opt vs 1st baseline, 2nd vs 2nd, etc.)
            baselines_for_this_opt = []
            if len(baseline_optimizers) == 1:
                if opt_short != baseline_optimizers[0]:
                    baselines_for_this_opt = [baseline_optimizers[0]]
            else:
                # Multiple baselines: match by position in the optimizer list
                # Find the index of this optimizer in the args.optimizers list
                if opt_short in args.optimizers:
                    opt_index = args.optimizers.index(opt_short)
                    # Check if there's a corresponding baseline at the same index
                    if opt_index < len(baseline_optimizers):
                        baseline_for_opt = baseline_optimizers[opt_index]
                        # Only compute gain if optimizer is not the same as its baseline
                        if opt_short != baseline_for_opt:
                            baselines_for_this_opt = [baseline_for_opt]
            
            for baseline_optimizer in baselines_for_this_opt:
                    # Plot affine-by-part compute gain (if available)
                    if baseline_optimizer in baseline_affine_funcs and rule in baseline_affine_funcs[baseline_optimizer]:
                        # Use piecewise affine interpolation
                        baseline_metric = baseline_affine_funcs[baseline_optimizer][rule]['metric']
                        baseline_loss = baseline_affine_funcs[baseline_optimizer][rule]['loss']
                        
                        # Get optimizer's actual observed data points for this rule
                        opt_type_full = optimizer_map[opt_short]
                        opt_df = data_dict[opt_type_full][rule]
                        
                        if len(opt_df) > 0:
                            # For each observed point of this optimizer
                            opt_metrics = opt_df[fit_metric].values
                            opt_losses = opt_df['val_loss'].values
                            
                            # For each point, find where baseline achieves the same loss
                            compute_gains = []
                            valid_opt_metrics = []
                            
                            for opt_metric, opt_loss in zip(opt_metrics, opt_losses):
                                # Find the compute needed by baseline to achieve opt_loss
                                # The baseline affine function is: loss = f(metric)
                                # We need to invert it: metric = f^(-1)(loss)
                                
                                # Check if this loss is in the range of baseline losses
                                if opt_loss >= baseline_loss.min() and opt_loss <= baseline_loss.max():
                                    # Interpolate in log-log space for better behavior
                                    log_baseline_metric = np.log(baseline_metric)
                                    log_baseline_loss = np.log(baseline_loss)
                                    log_opt_loss = np.log(opt_loss)
                                    
                                    # Since baseline_loss might not be monotonic, we need to handle this carefully
                                    # For scaling laws, loss typically decreases with compute
                                    # So we interpolate: given a loss, find the metric
                                    # We need to reverse the arrays if loss is decreasing
                                    if baseline_loss[0] > baseline_loss[-1]:
                                        # Loss decreases with metric (typical case)
                                        log_metric_baseline = np.interp(log_opt_loss, log_baseline_loss[::-1], log_baseline_metric[::-1])
                                    else:
                                        # Loss increases with metric (unusual)
                                        log_metric_baseline = np.interp(log_opt_loss, log_baseline_loss, log_baseline_metric)
                                    
                                    metric_baseline = np.exp(log_metric_baseline)
                                    gain = metric_baseline / opt_metric
                                    
                                    compute_gains.append(gain)
                                    valid_opt_metrics.append(opt_metric)
                                elif opt_loss < baseline_loss.min():
                                    # Optimizer achieves better loss than baseline's best
                                    # Extrapolate using last two baseline points
                                    if len(baseline_loss) >= 2:
                                        log_m1, log_m2 = np.log(baseline_metric[-2]), np.log(baseline_metric[-1])
                                        log_l1, log_l2 = np.log(baseline_loss[-2]), np.log(baseline_loss[-1])
                                        log_opt_loss = np.log(opt_loss)
                                        
                                        # Linear extrapolation in log-log space
                                        # slope = d(log_metric) / d(log_loss)
                                        slope = (log_m2 - log_m1) / (log_l2 - log_l1)
                                        log_metric_baseline = log_m2 + slope * (log_opt_loss - log_l2)
                                        metric_baseline = np.exp(log_metric_baseline)
                                        gain = metric_baseline / opt_metric
                                        
                                        compute_gains.append(gain)
                                        valid_opt_metrics.append(opt_metric)
                                elif opt_loss > baseline_loss.max():
                                    # Optimizer achieves worse loss than baseline's worst
                                    # Extrapolate using first two baseline points
                                    if len(baseline_loss) >= 2:
                                        log_m1, log_m2 = np.log(baseline_metric[0]), np.log(baseline_metric[1])
                                        log_l1, log_l2 = np.log(baseline_loss[0]), np.log(baseline_loss[1])
                                        log_opt_loss = np.log(opt_loss)
                                        
                                        # Linear extrapolation in log-log space
                                        slope = (log_m2 - log_m1) / (log_l2 - log_l1)
                                        log_metric_baseline = log_m1 + slope * (log_opt_loss - log_l1)
                                        metric_baseline = np.exp(log_metric_baseline)
                                        gain = metric_baseline / opt_metric
                                        
                                        compute_gains.append(gain)
                                        valid_opt_metrics.append(opt_metric)
                            
                            if len(compute_gains) > 0:
                                # Plot the gains as scatter points connected by lines
                                # Make them more prominent when no_loss is True
                                scatter_size = 200 if no_loss else 150
                                line_alpha = 1.0  # Full color for affine-by-part lines
                                line_width = 11.0 if no_loss else 9.0
                                # Affine-by-part lines are always solid
                                line_style = '-'
                                
                                scatter_gain = ax_gain.scatter(valid_opt_metrics, compute_gains, 
                                              s=scatter_size, marker=marker, c=gain_color, edgecolors='white',
                                              linewidths=1.5, zorder=11, alpha=0.85)
                                
                                # Also plot connecting lines for visualization
                                sorted_indices = np.argsort(valid_opt_metrics)
                                sorted_metrics = np.array(valid_opt_metrics)[sorted_indices]
                                sorted_gains = np.array(compute_gains)[sorted_indices]
                                # Affine-by-part lines: always solid and fully opaque
                                line_gain, = ax_gain.plot(sorted_metrics, sorted_gains, linestyle='-', color=gain_color,
                                           linewidth=line_width, alpha=1.0, zorder=8)
                                
                                # Store handle for legend when no_loss
                                if no_loss:
                                    gain_key = (rule, opt_short, baseline_optimizer, 'affine')
                                    if kappa_ablation:
                                        display_name = OPT_DISPLAY_NAMES_KAPPA.get(opt_short, OPT_DISPLAY_NAMES.get(opt_short, opt_short))
                                        label = display_name if display_name.startswith('κ') else f'{_display_name(opt_short)} {_display_rule(rule)}'
                                    else:
                                        label_suffix = f' vs {_display_name(baseline_optimizer)}' if len(baseline_optimizers) > 1 else ''
                                        label = f'{_display_name(opt_short)} {_display_rule(rule)} {label_suffix}'
                                    gain_handles[gain_key] = (line_gain, label)
                    
                    # Plot fitted curve compute gain (if available and not disabled by --no-fit)
                    if not no_fit and baseline_optimizer in baseline_fits and rule in baseline_fits[baseline_optimizer] and curve_name in fit_results['curves']:
                        # Use fitted curve with broken power law
                        baseline_b = baseline_fits[baseline_optimizer][rule]['b']
                        baseline_c = baseline_fits[baseline_optimizer][rule]['c']
                        baseline_e = baseline_fits[baseline_optimizer][rule]['e']
                        baseline_f = baseline_fits[baseline_optimizer][rule]['f']
                        
                        # Get optimizer fitted parameters
                        opt_b_fit = fit_results['curves'][curve_name]['b']
                        opt_c_fit = fit_results['curves'][curve_name]['c']
                        opt_e_fit = fit_results['curves'][curve_name]['e']
                        opt_f_fit = fit_results['curves'][curve_name]['f']
                        
                        # Calculate compute gain using broken power law:
                        # For a given compute C used by optimizer, it reaches loss L = a + b*C^{-c} + e*C^{-f}
                        # Find compute C_baseline needed by baseline optimizer to reach same loss L
                        # L = a + baseline_b*C_baseline^{-baseline_c} + baseline_e*C_baseline^{-baseline_f}
                        # This requires numerical solution
                        
                        optimizer_loss = a + opt_b_fit * np.power(plot_range, -opt_c_fit) + opt_e_fit * np.power(plot_range, -opt_f_fit)
                        compute_gain_ratios_fit = []
                        valid_plot_range_fit = []
                        
                        for C, L in zip(plot_range, optimizer_loss):
                            if L <= a:
                                continue
                            
                            def baseline_loss_fn(log_compute):
                                compute = np.exp(log_compute)
                                return a + baseline_b * np.power(compute, -baseline_c) + baseline_e * np.power(compute, -baseline_f) - L
                            
                            try:
                                log_compute_baseline = brentq(baseline_loss_fn, -10, 30)
                                compute_baseline = np.exp(log_compute_baseline)
                                ratio = compute_baseline / C
                                compute_gain_ratios_fit.append(ratio)
                                valid_plot_range_fit.append(C)
                            except ValueError:
                                continue
                        
                        if len(compute_gain_ratios_fit) > 0:
                            compute_gain_ratio_fit = np.array(compute_gain_ratios_fit)
                            valid_plot_range_fit = np.array(valid_plot_range_fit)
                            
                            # Fitted curve: use dashed line with lower opacity
                            line_alpha_fit = 0.8 if no_loss else 0.5
                            line_width_fit = 11.0 if no_loss else 9.0
                            line_style_fit = '--'  # Always dashed for fitted curves
                            
                            line_gain_fit, = ax_gain.plot(valid_plot_range_fit, compute_gain_ratio_fit, linestyle=line_style_fit, color=gain_color,
                                       linewidth=line_width_fit, alpha=line_alpha_fit, zorder=7)
                            
                            # Don't store handle for legend (fitted curves are shown but not in legend)
    
    # Add baseline line at 1.0 (ratio = 1 means same efficiency) on gain axis
    # Skip this line when using affine-by-part to focus on other optimizers
    # Only show if there's a single baseline
    if not use_affine_by_part and len(baseline_optimizers) == 1:
        baseline_color = OPT_COLORS.get(baseline_optimizers[0], '#00CED1')
        ax_gain.axhline(y=1.0, color=baseline_color, linestyle='--', linewidth=1.5, alpha=0.6, zorder=7)
    
    # Add vertical line at 24 heads (if requested)
    if show_vertical_bar:
        # Calculate metric value for 24 heads from any available scaling rule
        metric_24_heads = None
        for rule in scaling_rules:
            if rule in ['Enoki', 'Enoki_Scaled', 'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', 'Qwen3_Hoyer']:
                # For head-based scaling rules, compute metric for 24 heads
                params_24 = compute_params(24, rule)
                metric_24_heads = params_24[fit_metric]
                break
        
        if metric_24_heads is not None:
            ax.axvline(x=metric_24_heads, color='gray', linestyle='-', linewidth=2.0, alpha=0.5, zorder=1)
    
    # Create custom ordered legend (same as standard plot, or only compute gains if no_loss)
    legend_handles = []
    legend_labels = []
    
    if not no_loss:
        for rule in scaling_rules:
            for opt_short in optimizer_shorts:
                fit_key = (rule, opt_short)
                obs_key = (rule, opt_short)
                
                # Prefer fit curve with equation, otherwise just show optimizer name
                if fit_key in fit_handles:
                    handle, label = fit_handles[fit_key]
                    legend_handles.append(handle)
                    legend_labels.append(label)
                elif obs_key in obs_handles:
                    handle, label = obs_handles[obs_key]
                    legend_handles.append(handle)
                    legend_labels.append(label)
    else:
        # When no_loss=True, create legend only for compute gains
        for rule in scaling_rules:
            for opt_short in optimizer_shorts:
                # Check all baselines for this optimizer
                for baseline_opt in baseline_optimizers:
                    if opt_short != baseline_opt:
                        # Add only affine-by-part gain (fitted curves are shown but not in legend)
                        gain_key_affine = (rule, opt_short, baseline_opt, 'affine')
                        if gain_key_affine in gain_handles:
                            handle, label = gain_handles[gain_key_affine]
                            legend_handles.append(handle)
                            legend_labels.append(label)
    
    # Get metric info
    if fit_metric == 'compute':
        xlabel = 'Compute (PFH)'
    else:
        xlabel = 'Non-embedding Parameters'
    
    # Formatting - improved aesthetics with larger axis titles
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE_SECONDARY, fontweight='bold')
    if no_loss:
        # Hide left y-axis when showing only compute gains
        ax.set_ylabel('', fontsize=AXIS_LABEL_FONTSIZE_SECONDARY, fontweight='bold')
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    else:
        ax.set_ylabel('Val Loss', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax.set_yscale('log')
    
    ax.set_xscale('log')
    
    # Set x-axis limits when head_max is specified
    if head_max is not None and len(all_metric_vals) > 0:
        ax.set_xlim(metric_min * 0.5, metric_max * 1.15)
    
    # Configure tick appearance - all axes
    ax.tick_params(axis='x', which='major', labelsize=TICK_LABELSIZE_X, width=TICK_WIDTH_MAJOR, length=TICK_LENGTH_MAJOR)
    ax.tick_params(axis='x', which='minor', width=TICK_WIDTH_MINOR, length=TICK_LENGTH_MINOR)
    ax.tick_params(axis='y', which='major', labelsize=TICK_LABELSIZE_VALLOSS, width=TICK_WIDTH_MAJOR, length=TICK_LENGTH_MAJOR)
    ax.tick_params(axis='y', which='minor', labelsize=TICK_LABELSIZE_VALLOSS, width=TICK_WIDTH_MINOR, length=TICK_LENGTH_MINOR)
    
    # Right axis for compute gain (ratio) - make it primary when no_loss
    if no_loss:
        # Only show baseline name if there's a single baseline
        if len(baseline_optimizers) == 1:
            ylabel = f'Compute Efficiency vs {_display_name(baseline_optimizers[0])}'
        else:
            ylabel = 'Compute Efficiency'
        ax_gain.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE_GAIN, fontweight='bold')
        ax_gain.set_yscale('log')
        ax_gain.tick_params(axis='y', labelsize=TICK_LABELSIZE_GAIN, which='major', direction='out')
        ax_gain.tick_params(axis='y', labelsize=TICK_LABELSIZE_GAIN, which='minor', direction='out')
        # Format y-axis to avoid scientific notation and remove "× 10^0"
        from matplotlib.ticker import FuncFormatter, LogFormatterExponent
        # Use a custom formatter that formats numbers directly - always 2 decimal places
        def format_func(x, p):
            return f'{x:.2f}'
        formatter = FuncFormatter(format_func)
        # Explicitly set both formatters to override default log formatter
        ax_gain.yaxis.set_major_formatter(formatter)
        ax_gain.yaxis.set_minor_formatter(formatter)
        
        # Add horizontal dashed grey line at compute gain 1.0 (without affecting y-axis limits)
        ax_gain.axhline(y=1.0, color='grey', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
        
        # Hide top, bottom, and left spines of ax_gain (only show right spine)
        ax_gain.spines['top'].set_visible(False)
        ax_gain.spines['bottom'].set_visible(False)
        ax_gain.spines['left'].set_visible(False)
    else:
        # Only show baseline name if there's a single baseline
        if len(baseline_optimizers) == 1:
            ylabel = f'Compute Efficiency vs {_display_name(baseline_optimizers[0])}'
        else:
            ylabel = 'Compute Efficiency'
        ax_gain.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE_GAIN_SECONDARY, fontweight='bold', color='#000000')
        ax_gain.set_yscale('log')
        ax_gain.tick_params(axis='y', labelcolor='#000000', labelsize=TICK_LABELSIZE_GAIN, which='major', direction='out')
        ax_gain.tick_params(axis='y', labelcolor='#000000', labelsize=TICK_LABELSIZE_GAIN, which='minor', direction='out')
        # Format y-axis to avoid scientific notation and remove "× 10^0"
        from matplotlib.ticker import FuncFormatter
        # Use a custom formatter that formats numbers directly - always 2 decimal places
        def format_func(x, p):
            return f'{x:.2f}'
        formatter = FuncFormatter(format_func)
        # Explicitly set both formatters to override default log formatter
        ax_gain.yaxis.set_major_formatter(formatter)
        ax_gain.yaxis.set_minor_formatter(formatter)
        
        # Hide top, bottom, and left spines of ax_gain (only show right spine)
        ax_gain.spines['top'].set_visible(False)
        ax_gain.spines['bottom'].set_visible(False)
        ax_gain.spines['left'].set_visible(False)
    
    if show_title:
        opts_str = ', '.join([_display_name(opt) for opt in optimizer_shorts])
        rules_str = ' vs '.join(scaling_rules)
        if no_loss:
            # Only show baseline name in title if there's a single baseline
            if len(baseline_optimizers) == 1:
                title_suffix = f' (relative to {_display_name(baseline_optimizers[0])})'
            else:
                title_suffix = ''
            
            # Update subtitle based on whether fitted curves are shown
            if no_fit:
                subtitle = 'Affine-by-part only'
            else:
                subtitle = 'Solid lines = affine-by-part, Dashed lines = fitted curves'
            
            ax.set_title(f'Compute Gain Comparison: {rules_str}\nOptimizers: {opts_str}{title_suffix}\n{subtitle}',
                        fontsize=20, fontweight='bold', pad=20)
        else:
            ax.set_title(f'Scaling Laws Comparison: {rules_str}\nOptimizers: {opts_str} (Shared saturation a = {fit_results["a"]:.4f})',
                        fontsize=20, fontweight='bold', pad=20)
    
    if len(legend_handles) > 0:
        # Use ax_gain for legend when no_loss, otherwise use ax
        legend_ax = ax_gain if no_loss else ax
        legend_ax.legend(legend_handles, legend_labels, fontsize=LEGEND_FONTSIZE, loc='lower center', framealpha=0.95, ncol=2, 
                  edgecolor='#333333', fancybox=True, shadow=True, bbox_to_anchor=(0.5, 0.02))
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    if no_loss:
        ax_gain.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Remove spines for cleaner look
    # Hide top spine when heads axis is not shown
    if not show_heads_axis:
        ax.spines['top'].set_visible(False)
        ax.tick_params(top=False)
    else:
        # When heads axis is shown, hide top spine of main axis (ax2 will have its own)
        ax.spines['top'].set_visible(False)
        ax.tick_params(top=False)
    
    # Hide left spine when showing only compute gains
    if no_loss:
        ax.spines['left'].set_visible(False)
    else:
        # Hide right spine when showing both loss and compute gain (ax_gain uses right)
        ax.spines['right'].set_visible(False)
        ax.tick_params(right=False)
    
    # Add second x-axis showing size (heads or depth) on top
    if show_heads_axis:
        ax2 = ax.twiny()
        ax2.set_xscale('log')
        ax2.set_xlim(ax.get_xlim())
        
        # Collect all sizes from data
        all_sizes = set()
        size_to_metric = {}
        for opt_type in optimizer_types:
            for rule in scaling_rules:
                df = data_dict[opt_type][rule]
                if len(df) > 0:
                    for _, row in df.iterrows():
                        size = row['size']
                        metric_val = row[fit_metric]
                        all_sizes.add(size)
                        if size not in size_to_metric:
                            size_to_metric[size] = metric_val
        
        if len(all_sizes) > 0:
            all_sizes_sorted = sorted(all_sizes)
            
            # Show fewer ticks - select every other one or every third one depending on count
            if len(all_sizes_sorted) > 10:
                # Show every third tick
                selected_indices = range(0, len(all_sizes_sorted), 3)
            elif len(all_sizes_sorted) > 6:
                # Show every other tick
                selected_indices = range(0, len(all_sizes_sorted), 2)
            else:
                # Show all ticks
                selected_indices = range(len(all_sizes_sorted))
            
            selected_sizes = [all_sizes_sorted[i] for i in selected_indices]
            ax2.set_xticks([size_to_metric[size] for size in selected_sizes])
            ax2.set_xticklabels([str(size) for size in selected_sizes], fontweight='bold', fontsize=TICK_LABELSIZE_HEADS)
            
            # Set tick parameters for top x-axis
            ax2.tick_params(axis='x', which='major', labelsize=TICK_LABELSIZE_HEADS, width=TICK_WIDTH_MAJOR, length=TICK_LENGTH_MAJOR, pad=10)
            ax2.tick_params(axis='x', which='minor', width=TICK_WIDTH_MINOR, length=TICK_LENGTH_MINOR)
            
            # Hide bottom, left, and right spines of ax2 (only show top spine)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # Determine label based on scaling rules
            if 'BigHead' in scaling_rules and len(scaling_rules) == 1:
                size_label = 'Depth'
            else:
                size_label = 'Heads'
            ax2.set_xlabel(size_label, fontsize=AXIS_LABEL_FONTSIZE_HEADS, fontweight='bold')

    plt.tight_layout()

    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Validate arguments and prepare baseline optimizer if needed
    baseline_optimizer_shorts = []
    baseline_optimizer_types = []
    
    if args.affine_by_part and not args.plot_compute_gain:
        print("ERROR: --affine-by-part requires --plot-compute-gain to be set")
        exit(1)
    
    if args.no_loss and not args.plot_compute_gain:
        print("ERROR: --no-loss requires --plot-compute-gain to be set")
        exit(1)
    
    if args.no_fit and not args.plot_compute_gain:
        print("ERROR: --no-fit requires --plot-compute-gain to be set")
        exit(1)
    
    if args.plot_compute_gain:
        # args.plot_compute_gain is now a list
        baseline_optimizer_shorts = args.plot_compute_gain
        baseline_optimizer_types = [optimizer_map[opt] for opt in baseline_optimizer_shorts]
        
        # Add baseline optimizers to the list if not already present
        for baseline_short, baseline_type in zip(baseline_optimizer_shorts, baseline_optimizer_types):
            if baseline_short not in args.optimizers:
                print(f"Note: Adding baseline optimizer '{baseline_short}' to fitting")
                args.optimizers.append(baseline_short)
                optimizer_types.append(baseline_type)
    
    print("="*70)
    print(f"Scaling Rules Comparison")
    print(f"Scaling Rules: {', '.join(args.scaling_rules)}")
    print(f"Optimizers: {', '.join(args.optimizers)} ({', '.join(optimizer_types)})")
    print(f"Fit Metric: {args.fit_metric}")
    if args.min_compute:
        print(f"Min Compute: {args.min_compute:.4e} PFH")
    if args.head_min:
        print(f"Min Head/Depth for Fitting: {args.head_min}")
    if args.head_max:
        print(f"Max Head/Depth for Fitting: {args.head_max}")
    print(f"Lower Bound on 'a': {args.a_lower_bound}")
    if args.plot_compute_gain:
        method = "piecewise affine interpolation" if args.affine_by_part else "fitted curve"
        print(f"Compute Gain Plot: Enabled (relative to {', '.join(baseline_optimizer_shorts)}, using {method})")
    print("="*70)

    # Load data for all optimizer x scaling_rule combinations
    data_dict = {}  # {optimizer_type: {scaling_rule: df}}

    for optimizer_idx, optimizer_type in enumerate(optimizer_types):
        optimizer_short = args.optimizers[optimizer_idx]
        print(f"\nLoading data for {optimizer_short} ({optimizer_type})...")

        data_dict[optimizer_type] = {}
        # Check if we need to filter by kappa (for dana-star-mk4-kappa-0-85)
        kappa_filter = None
        if optimizer_short == 'dana-star-mk4-kappa-0-85':
            kappa_filter = 0.85
        
        for scaling_rule in args.scaling_rules:
            df = load_scaling_rule_data(
                scaling_rule=scaling_rule,
                project=args.project,
                entity=args.entity,
                optimizer_type=optimizer_type,
                min_compute=args.min_compute,
                head_min=args.head_min,
                head_max=args.head_max,
                kappa_filter=kappa_filter
            )
            data_dict[optimizer_type][scaling_rule] = df
            if len(df) > 0:
                print(f"  {scaling_rule}: {len(df)} data points")
            else:
                print(f"  {scaling_rule}: No data")

    # Prepare data for joint fitting across ALL optimizers and scaling rules
    # Only use points marked with use_for_fit=True
    joint_fit_data = []

    for optimizer_idx, optimizer_type in enumerate(optimizer_types):
        optimizer_short = args.optimizers[optimizer_idx]
        
        # Skip fitting for dana-star-no-tau-kappa-1-0
        if optimizer_short == 'dana-star-no-tau-kappa-1-0':
            continue

        for scaling_rule in args.scaling_rules:
            df = data_dict[optimizer_type][scaling_rule]

            if len(df) > 0:
                # Filter to only points used for fitting
                df_fit = df[df['use_for_fit']] if 'use_for_fit' in df.columns else df
                
                if len(df_fit) > 0:
                    joint_fit_data.append({
                        'compute': df_fit[args.fit_metric].values,
                        'loss': df_fit['val_loss'].values,
                        'name': f'{optimizer_short}_{scaling_rule}'
                    })

    # Check if we have any data
    if len(joint_fit_data) == 0:
        print("\nNo data found for any optimizer/scaling rule combination. Exiting.")
        exit(1)

    print(f"\n{'='*70}")
    print(f"Joint Fitting {len(joint_fit_data)} Curves")
    print("Using broken power law: loss = a + b*C^{-c} + e*C^{-f}")
    print(f"{'='*70}")

    # Perform joint fit
    fit_results = fit_all_saturated_power_laws_joint(
        joint_fit_data,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        a_lower_bound=args.a_lower_bound
    )

    # Print results
    print(f"\n{'='*70}")
    print("Fit Results")
    print(f"{'='*70}")
    print(f"\nShared saturation level a = {fit_results['a']:.6f}")

    for curve_name, curve_params in fit_results['curves'].items():
        print(f"\n{curve_name}:")
        print(f"  b = {curve_params['b']:.6e}")
        print(f"  c = {curve_params['c']:.6f}")
        print(f"  e = {curve_params['e']:.6e}")
        print(f"  f = {curve_params['f']:.6f}")
        print(f"  R² = {curve_params['r_squared']:.6f}")

    # Choose which plot to create based on flags
    show_title = not args.no_title
    if args.plot_compute_gain:
        # Create compute gain plot
        print(f"\nCreating compute gain plot (relative to {', '.join(baseline_optimizer_shorts)})...")
        print(f"  - Plotting affine-by-part compute gains (solid lines)")
        if not args.no_fit:
            print(f"  - Plotting fitted curve compute gains (dashed lines)")
        else:
            print(f"  - Skipping fitted curve compute gains (--no-fit enabled)")
        fig = plot_compute_gain(
            data_dict,
            fit_results,
            args.scaling_rules,
            args.optimizers,
            optimizer_types,
            args.fit_metric,
            baseline_optimizer_shorts,
            use_affine_by_part=args.affine_by_part,
            show_title=show_title,
            no_loss=args.no_loss,
            show_heads_axis=not args.no_heads_axis,
            head_max=args.head_max,
            no_fit=args.no_fit,
            show_vertical_bar=args.vertical_bar,
            kappa_ablation=args.kappa_ablation
        )
    elif args.fit_relative_to_adamw:
        # Create relative plot with AdamW as baseline
        if 'adamw' not in args.optimizers:
            print("\nWarning: --fit-relative-to-adamw requires 'adamw' in optimizers list. Falling back to standard plot.")
            fig = plot_comparison_multi_optimizer(
                data_dict,
                fit_results,
                args.scaling_rules,
                args.optimizers,
                optimizer_types,
                args.fit_metric,
                show_title=show_title,
                show_heads_axis=not args.no_heads_axis,
                head_max=args.head_max,
                show_vertical_bar=args.vertical_bar
            )
        else:
            print(f"\nCreating relative comparison plot (AdamW as baseline)...")
            fig = plot_comparison_relative_to_adamw(
                data_dict,
                fit_results,
                args.scaling_rules,
                args.optimizers,
                optimizer_types,
                args.fit_metric,
                show_title=show_title,
                head_max=args.head_max,
                show_vertical_bar=args.vertical_bar
            )
    else:
        # Create standard comparison plot
        fig = plot_comparison_multi_optimizer(
            data_dict,
            fit_results,
            args.scaling_rules,
            args.optimizers,
            optimizer_types,
            args.fit_metric,
            show_title=show_title,
            show_heads_axis=not args.no_heads_axis,
            head_max=args.head_max,
            show_vertical_bar=args.vertical_bar
        )

    # Save plot
    if args.output:
        output_file = args.output
    else:
        rules_str = '_'.join(args.scaling_rules)
        opts_str = '_'.join([_filename_safe_name(opt) for opt in args.optimizers])
        if args.plot_compute_gain:
            baseline_str = '_vs_'.join([_filename_safe_name(b) for b in baseline_optimizer_shorts])
            if args.no_loss:
                suffix = f'_compute_gain_both_no_loss_vs_{baseline_str}'
            else:
                suffix = f'_compute_gain_both_vs_{baseline_str}'
        elif args.fit_relative_to_adamw and 'adamw' in args.optimizers:
            suffix = '_relative'
        else:
            suffix = ''
        output_file = f'ScalingComparison_{rules_str}_{opts_str}{suffix}.pdf'

    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    print(f"\n{'='*70}")
    print(f"Plot saved to: {os.path.abspath(output_file)}")
    print(f"{'='*70}")
