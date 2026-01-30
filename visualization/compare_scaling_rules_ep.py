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

from opt_colors import OPT_COLORS, OPT_LINESTYLES, OPT_DISPLAY_NAMES

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
rcParams['font.size'] = 20
rcParams['figure.figsize'] = (16, 10)
rcParams['axes.linewidth'] = 2.5
rcParams['axes.edgecolor'] = '#333333'
rcParams['axes.labelsize'] = 22
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 16
rcParams['grid.alpha'] = 0.3
rcParams['grid.linestyle'] = '--'
rcParams['legend.framealpha'] = 0.95
rcParams['legend.edgecolor'] = '#333333'

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description='Compare scaling rules performance')
parser.add_argument('--scaling-rules', type=str, nargs='+', required=True,
                    choices=['BigHead', 'EggHead', 'Enoki', 'Enoki_Scaled', 'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', 'Qwen3_Hoyer'],
                    help='Scaling rules to compare (can specify multiple)')
parser.add_argument('--optimizers', type=str, nargs='+', required=True,
                    choices=['adamw', 'mk4', 'dana', 'ademamix', 'd-muon', 'manau', 'manau-hard', 'adamw-decaying-wd', 'dana-mk4', 'ademamix-decaying-wd', 'dana-star-no-tau', 'dana-star', 'dana-star-no-tau-kappa-0-75', 'dana-star-no-tau-kappa-0-8', 'dana-star-no-tau-kappa-0-85', 'dana-star-no-tau-kappa-0-9', 'dana-mk4-kappa-0-85', 'dana-mk4-kappa-0-75', 'dana-star-mk4-kappa-0-75', 'dana-star-mk4-kappa-0-85', 'dana-star-no-tau-dana-constant', 'dana-star-no-tau-beta2-constant', 'dana-star-no-tau-beta1', 'dana-star-no-tau-dana-constant-beta2-constant', 'dana-star-no-tau-dana-constant-beta1', 'dana-star-no-tau-dana-constant-beta2-constant-beta1', 'dana-star-no-tau-kappa-1-0', 'adana-kappa-0-85'],
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
parser.add_argument('--a-upper-bound', type=float, default=None,
                    help='Upper bound constant for saturation parameter a (default: 0.95 * min_loss)')
parser.add_argument('--equal-weight', action='store_true', default=True,
                    help='Use equal weights for all data points instead of weighting by compute (default: True)')
parser.add_argument('--compute-weight', action='store_true',
                    help='Weight by compute (larger models matter more) - overrides --equal-weight')
parser.add_argument('--fit-relative-to-adamw', action='store_true',
                    help='Plot relative to AdamW baseline (AdamW appears as horizontal line at 0)')
parser.add_argument('--plot-compute-gain', type=str, default=None,
                    choices=['adamw', 'mk4', 'dana', 'ademamix', 'd-muon', 'manau', 'manau-hard', 'adamw-decaying-wd', 'dana-mk4', 'ademamix-decaying-wd', 'dana-star-no-tau', 'dana-star', 'dana-star-no-tau-kappa-0-75', 'dana-star-no-tau-kappa-0-8', 'dana-star-no-tau-kappa-0-85', 'dana-star-no-tau-kappa-0-9', 'dana-mk4-kappa-0-85', 'dana-mk4-kappa-0-75', 'dana-star-mk4-kappa-0-75', 'dana-star-mk4-kappa-0-85', 'dana-star-no-tau-dana-constant', 'dana-star-no-tau-beta2-constant', 'dana-star-no-tau-beta1', 'dana-star-no-tau-dana-constant-beta2-constant', 'dana-star-no-tau-dana-constant-beta1', 'dana-star-no-tau-dana-constant-beta2-constant-beta1', 'adana-kappa-0-85'],
                    help='Plot compute gain relative to specified baseline optimizer (will be included in fitting even if not in --optimizers)')
parser.add_argument('--affine-by-part', action='store_true',
                    help='Use piecewise affine interpolation of baseline data instead of fitted curve for compute gain calculation (requires --plot-compute-gain)')
parser.add_argument('--head-min', type=int, default=None,
                    help='Minimum head/depth size to include in fitting (still plots all points)')
parser.add_argument('--no-title', action='store_true',
                    help='Do not show title on plot')
parser.add_argument('--no-loss', action='store_true',
                    help='Hide loss points and fitted curves, show only compute gains (requires --plot-compute-gain)')
parser.add_argument('--single-power-law', action='store_true',
                    help='Use single power law (a + b*C^{-c}) instead of broken power law (a + b*C^{-c} + e*C^{-f})')
parser.add_argument('--no-heads-axis', action='store_true',
                    help='Hide the heads/depth axis at top of plot')
parser.add_argument('--cache-dir', type=str, default='wandb_cache',
                    help='Directory for caching WandB data (default: wandb_cache)')
parser.add_argument('--no-cache', action='store_true',
                    help='Disable caching (always fetch fresh data from WandB)')
parser.add_argument('--skip-fit', type=str, nargs='+', default=[],
                    help='Optimizers to exclude from curve fitting (data points still shown)')
parser.add_argument('--mark-outlier', type=str, nargs='+', default=[],
                    help='Mark outlier points with red star. Format: optimizer:size (e.g., ademamix:17)')
parser.add_argument('--legend-suffix', type=str, nargs='+', default=[],
                    help='Add suffix to optimizer legend labels. Format: optimizer:suffix (e.g., "mk4:(κ=0.75)")')
parser.add_argument('--compute-formula', type=str, default='default',
                    choices=['default', '6N1', '6N2', 'M'],
                    help='Compute formula to use: default (6*P*20*D), 6N1 (Kaplan: 72*n_layer*d_model^2), '
                         '6N2 (Chinchilla: includes embedding), M (DeepSeek: includes attention). Default: default')
parser.add_argument('--seq-length', type=int, default=2048,
                    help='Sequence length for M formula (default: 2048)')
parser.add_argument('--efficiency-ymin', type=float, default=None,
                    help='Minimum y-value for compute efficiency axis (cuts off whitespace below this value)')
parser.add_argument('--broken-axis', action='store_true',
                    help='Use broken y-axis for compute efficiency, removing gap between --broken-axis-lower and --broken-axis-upper')
parser.add_argument('--broken-axis-lower', type=float, default=0.1,
                    help='Lower bound of gap to remove in broken axis (default: 0.1)')
parser.add_argument('--broken-axis-upper', type=float, default=0.9,
                    help='Upper bound of gap to remove in broken axis (default: 0.9)')
args = parser.parse_args()

# Parse legend suffixes into a dictionary
legend_suffixes = {}
for suffix_spec in args.legend_suffix:
    try:
        opt, suffix = suffix_spec.split(':', 1)
        legend_suffixes[opt] = suffix
    except ValueError:
        print(f"Warning: Invalid legend suffix format '{suffix_spec}'. Expected 'optimizer:suffix'")

# Map optimizer names
optimizer_map = {'adamw': 'adamw', 'mk4': 'dana-star-mk4', 'dana': 'dana', 'ademamix': 'ademamix',
                 'd-muon': 'd-muon', 'manau': 'manau', 'manau-hard': 'manau-hard', 'adamw-decaying-wd': 'adamw-decaying-wd', 'dana-mk4': 'dana-mk4', 'ademamix-decaying-wd': 'ademamix-decaying-wd', 'dana-star-no-tau': 'dana-star-no-tau', 'dana-star': 'dana-star', 'dana-star-no-tau-kappa-0-75': 'dana-star-no-tau-kappa-0-75', 'dana-star-no-tau-kappa-0-8': 'dana-star-no-tau-kappa-0-8', 'dana-star-no-tau-kappa-0-85': 'dana-star-no-tau-kappa-0-85', 'dana-star-no-tau-kappa-0-9': 'dana-star-no-tau-kappa-0-9', 'dana-mk4-kappa-0-85': 'dana-mk4-kappa-0-85', 'dana-star-mk4-kappa-0-75': 'dana-star-mk4-kappa-0-75', 'dana-star-mk4-kappa-0-85': 'dana-star-mk4-kappa-0-85', 'dana-star-no-tau-dana-constant': 'dana-star-no-tau-dana-constant', 'dana-star-no-tau-beta2-constant': 'dana-star-no-tau-beta2-constant', 'dana-star-no-tau-beta1': 'dana-star-no-tau-beta1', 'dana-star-no-tau-dana-constant-beta2-constant': 'dana-star-no-tau-dana-constant-beta2-constant', 'dana-star-no-tau-dana-constant-beta1': 'dana-star-no-tau-dana-constant-beta1', 'dana-star-no-tau-dana-constant-beta2-constant-beta1': 'dana-star-no-tau-dana-constant-beta2-constant-beta1', 'dana-star-no-tau-kappa-1-0': 'dana-star-no-tau-kappa-1-0', 'adana-kappa-0-85': 'adana-kappa-0-85', 'dana-mk4-kappa-0-75': 'dana-mk4-kappa-0-75', 'dana-mk4-kappa-0-85': 'dana-mk4-kappa-0-85'}
optimizer_types = [optimizer_map[opt] for opt in args.optimizers]

# =============================================================================
# PARAMETER CALCULATION FUNCTIONS
# =============================================================================

def compute_params(size, scaling_rule, compute_formula='default', seq_length=2048):
    """
    Compute parameters for a given size and scaling rule.

    Args:
        size: For BigHead, this is depth. For EggHead/Enoki/Eryngii/Qwen3_Scaled/Qwen3_Hoyer, this is heads.
        scaling_rule: One of 'BigHead', 'EggHead', 'Enoki', 'Enoki_Scaled', 'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', 'Qwen3_Hoyer'
        compute_formula: One of 'default', '6N1', '6N2', 'M' (DeepSeek formulas)
            - 'default': 6 * non_emb * total_params * 20 (Chinchilla-style with N=20D)
            - '6N1': 72 * n_layer * d_model^2 * D (Kaplan - non-embedding only)
            - '6N2': (72 * n_layer * d_model^2 + 6 * n_vocab * d_model) * D (includes embedding)
            - 'M': (72 * n_layer * d_model^2 + 12 * n_layer * d_model * l_seq) * D (includes attention)
        seq_length: Sequence length for M formula (default: 2048)

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

    # Compute in FLOPs based on formula choice
    # All formulas use D = total_params * 20 as the number of tokens (Chinchilla scaling)
    n_tokens = total_params * 20.0
    vocab_size = 50304  # Standard vocab size

    if compute_formula == 'default':
        # Original formula: C = 6 * P * N where P = non_emb, N = 20 * D
        compute_flops = 6.0 * non_emb * n_tokens
    elif compute_formula == '6N1':
        # Kaplan et al. (2020): 6N1 = 72 * n_layer * d_model^2
        # This ignores embedding parameters
        flops_per_token = 72.0 * n_layer * n_embd * n_embd
        compute_flops = flops_per_token * n_tokens
    elif compute_formula == '6N2':
        # Hoffmann et al. (2022) / Chinchilla: includes embedding
        # 6N2 = 72 * n_layer * d_model^2 + 6 * n_vocab * d_model
        flops_per_token = 72.0 * n_layer * n_embd * n_embd + 6.0 * vocab_size * n_embd
        compute_flops = flops_per_token * n_tokens
    elif compute_formula == 'M':
        # DeepSeek: includes attention overhead
        # M = 72 * n_layer * d_model^2 + 12 * n_layer * d_model * l_seq
        flops_per_token = 72.0 * n_layer * n_embd * n_embd + 12.0 * n_layer * n_embd * seq_length
        compute_flops = flops_per_token * n_tokens
    else:
        raise ValueError(f"Unknown compute formula: {compute_formula}")

    # Convert to PetaFlop-Hours: 1 PFH = 3600e15 FLOPs
    compute_pfh = compute_flops / (3600e15)

    return {
        'non_emb': int(non_emb),
        'total_params': int(total_params),
        'compute': compute_pfh,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
        'compute_formula': compute_formula
    }

# =============================================================================
# DATA LOADING
# =============================================================================

import pickle
import hashlib

def get_cache_path(cache_dir, scaling_rule, optimizer_type, project, entity, compute_formula='default'):
    """Generate cache file path for a specific query."""
    # Create a unique key for this query
    key = f"{scaling_rule}_{optimizer_type}_{project}_{entity}_{compute_formula}"
    filename = f"cache_{key}.pkl"
    return os.path.join(cache_dir, filename)

def load_scaling_rule_data(scaling_rule, project, entity, optimizer_type, min_compute=None, head_min=None,
                           cache_dir=None, use_cache=True, compute_formula='default', seq_length=2048):
    """Load data for a scaling rule and get best loss for each size.

    Args:
        scaling_rule: One of 'BigHead', 'EggHead', 'Enoki'
        project: WandB project name
        entity: WandB entity name
        optimizer_type: Type of optimizer to filter for
        min_compute: Minimum compute threshold in PFH (optional)
        head_min: Minimum head/depth size to include in fitting (optional)
        cache_dir: Directory for caching (optional)
        use_cache: Whether to use cached data if available (default: True)
        compute_formula: Compute formula to use (default, 6N1, 6N2, M)
        seq_length: Sequence length for M formula
    """
    # Check cache first
    if cache_dir and use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = get_cache_path(cache_dir, scaling_rule, optimizer_type, project, entity, compute_formula)
        if os.path.exists(cache_path):
            print(f"Loading {scaling_rule}/{optimizer_type} from cache...")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            df = cached_data['df']
            print(f"  Loaded {len(df)} {scaling_rule} runs from cache")

            # Apply filters (min_compute and head_min) to cached data
            if len(df) > 0:
                if min_compute is not None:
                    df = df[df['compute'] >= min_compute].copy()
                    print(f"  Filtered to {len(df)} data points with compute >= {min_compute:.4e} PFH")
                if head_min is not None:
                    df['use_for_fit'] = df['size'] >= head_min
                    n_fit = df['use_for_fit'].sum()
                    print(f"  Using {n_fit}/{len(df)} data points for fitting (head/depth >= {head_min})")
            return df

    api = wandb.Api()

    config = SCALING_RULE_CONFIG[scaling_rule]
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

        # Handle ADANA kappa=0.85: stored as dana-mk4 with clipsnr=1000 (effectively no clipping)
        if optimizer_type == 'adana-kappa-0-85':
            run_kappa = run_config.get('kappa', None)
            run_clipsnr = run_config.get('clipsnr', None)
            # Match dana-mk4 with very large clipsnr (>=100) and kappa=0.85
            if opt != 'dana-mk4' or run_kappa != 0.85 or (run_clipsnr is not None and run_clipsnr < 100):
                continue
        # Handle Dana-MK4 kappa=0.75: stored as dana-mk4 with clipsnr around 1-2 and kappa=0.75
        elif optimizer_type == 'dana-mk4-kappa-0-75':
            run_kappa = run_config.get('kappa', None)
            run_clipsnr = run_config.get('clipsnr', None)
            # Match dana-mk4 with moderate clipsnr (<10) and kappa=0.75
            if opt != 'dana-mk4' or run_kappa != 0.75 or (run_clipsnr is not None and run_clipsnr >= 10):
                continue
        # Handle Dana-MK4 kappa=0.85: stored as dana-mk4 with clipsnr around 0.25 and kappa=0.85
        elif optimizer_type == 'dana-mk4-kappa-0-85':
            run_kappa = run_config.get('kappa', None)
            run_clipsnr = run_config.get('clipsnr', None)
            # Match dana-mk4 with moderate clipsnr (<10) and kappa=0.85 (distinct from ADANA which has clipsnr>=100)
            if opt != 'dana-mk4' or run_kappa != 0.85 or (run_clipsnr is not None and run_clipsnr >= 10):
                continue
        # Handle kappa-specific filtering for dana-star-mk4 variants
        # e.g., optimizer_type='dana-star-mk4-kappa-0-75' should match opt='dana-star-mk4' with kappa=0.75
        elif 'dana-star-mk4-kappa-' in optimizer_type:
            # Extract target kappa from optimizer_type name (e.g., '0-75' -> 0.75)
            kappa_str = optimizer_type.split('dana-star-mk4-kappa-')[1]
            target_kappa = float(kappa_str.replace('-', '.'))
            run_kappa = run_config.get('kappa', None)

            # Match base optimizer type and kappa value
            if opt != 'dana-star-mk4' or run_kappa != target_kappa:
                continue
        elif 'dana-star-no-tau-kappa-' in optimizer_type:
            # Handle dana-star-no-tau with kappa variants
            # These might have opt='dana-star-no-tau' or similar with kappa in config
            kappa_str = optimizer_type.split('kappa-')[1].split('-')[0:2]
            target_kappa = float(f"{kappa_str[0]}.{kappa_str[1]}")
            run_kappa = run_config.get('kappa', None)
            base_opt = optimizer_type.rsplit('-kappa-', 1)[0]

            if opt != base_opt and opt != optimizer_type:
                # Also check if it matches exactly (some runs have full name in opt)
                if opt != optimizer_type:
                    continue
            if run_kappa is not None and run_kappa != target_kappa:
                continue
        else:
            # Standard filtering - exact match
            if opt != optimizer_type:
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
        params = compute_params(size, scaling_rule, compute_formula, seq_length)

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

        # Only add use_for_fit column if head_min is specified
        if head_min is not None:
            result_row['use_for_fit'] = (size >= head_min)

        results.append(result_row)

    result_df = pd.DataFrame(results)

    # Save to cache (before applying filters, so cache has all data)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = get_cache_path(cache_dir, scaling_rule, optimizer_type, project, entity, compute_formula)
        # Save unfiltered data to cache
        cache_df = result_df.copy()
        if 'use_for_fit' in cache_df.columns:
            cache_df = cache_df.drop(columns=['use_for_fit'])
        with open(cache_path, 'wb') as f:
            pickle.dump({'df': cache_df}, f)
        print(f"  Cached data to {cache_path}")

    if min_compute is not None and len(result_df) > 0:
        print(f"  Filtered to {len(result_df)} data points with compute >= {min_compute:.4e} PFH")
    if head_min is not None and len(result_df) > 0 and 'use_for_fit' in result_df.columns:
        n_fit = result_df['use_for_fit'].sum()
        print(f"  Using {n_fit}/{len(result_df)} data points for fitting (head/depth >= {head_min})")

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

        # Set weights: equal weights by default, compute-weighted if --compute-weight specified
        if args.compute_weight:
            weights_arr = compute_arr  # Weight by compute (larger models matter more)
        else:
            weights_arr = jnp.ones_like(compute_arr)  # Equal weights (default)

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
        # [log(b_i), c_i, log(e_i), f_i] with initial values b=0.40, c=0.200, e=2.50, f=0.030
        init_params.extend([
            jnp.log(0.40),  # log(b)
            0.200,          # c (positive, will be used as -c in power)
            jnp.log(2.50),  # log(e)
            0.030           # f (positive, will be used as -f in power)
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


def fit_all_single_power_laws_joint(data_list, n_steps=50000, learning_rate=0.1, a_lower_bound=0.0, a_upper_bound=None, equal_weight=False):
    """
    Fit SINGLE POWER LAWS to multiple datasets with a SHARED saturation level 'a'.

    Single power law: loss = a + b*C^{-c}
    where:
        - 'a' (saturation level) is shared across all curves
        - 'b', 'c' are fit per optimizer

    This is the traditional Chinchilla-style scaling law.

    Args:
        data_list: List of dicts, each with 'compute', 'loss', and 'name' keys
        n_steps: Number of optimization steps
        learning_rate: Learning rate for Adagrad optimizer
        a_lower_bound: Non-trainable lower bound constant for parameter a (default: 0.0)
        a_upper_bound: Upper bound constant for parameter a (default: 0.95 * min_loss)
        equal_weight: If True, use equal weights; otherwise weight by compute

    Returns:
        Dict with 'a' (shared saturation) and 'curves' (dict mapping name -> params)
    """
    print(f"\nPreparing {len(data_list)} curves for joint SINGLE POWER LAW fitting...")
    print(f"Model: loss = a + b*C^{{-c}}")
    print(f"  - 'a' shared across all optimizers")
    print(f"  - 'b', 'c' fit per optimizer")

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
        if equal_weight:
            weights_arr = jnp.ones_like(compute_arr)
        else:
            weights_arr = compute_arr  # Weight by compute (larger models matter more)

        jax_data.append({
            'compute': compute_arr,
            'loss': loss_arr,
            'name': name,
            'weights': weights_arr
        })

    # Initialize parameters for single power law
    # Params: [a_raw, log(b_0), c_0, log(b_1), c_1, ...]
    n_curves = len(jax_data)

    # Better initialization: estimate from data
    min_loss_val = float(jnp.min(jnp.array([jnp.min(d['loss']) for d in jax_data])))
    max_loss_val = float(jnp.max(jnp.array([jnp.max(d['loss']) for d in jax_data])))

    # Determine upper bound for a: use provided value or default to 0.95 * min_loss
    if a_upper_bound is not None:
        effective_a_upper = a_upper_bound
    else:
        effective_a_upper = min_loss_val * 0.95

    print(f"  Saturation bounds: a ∈ [{a_lower_bound:.4f}, {effective_a_upper:.4f}]")

    # Initialize a to be small (10% of the allowed range)
    init_a = a_lower_bound + 0.1 * (effective_a_upper - a_lower_bound)
    target_sigmoid = (init_a - a_lower_bound) / (effective_a_upper - a_lower_bound)
    init_a_raw = float(jnp.log(target_sigmoid / (1 - target_sigmoid + 1e-8)))

    init_params = [init_a_raw]  # a_raw

    for i in range(n_curves):
        # Initial estimates
        init_b = 0.5
        init_c = 0.05

        init_params.extend([jnp.log(init_b), init_c])  # [log(b_i), c_i]

    fit_params = jnp.array(init_params, dtype=jnp.float32)

    print(f"\nInitialization:")
    print(f"  a_raw = {init_params[0]:.4f} (maps to a ≈ {init_a:.4f})")
    print(f"  Data loss range: [{min_loss_val:.4f}, {max_loss_val:.4f}]")

    # Store effective upper bound for use in loss function
    a_upper_for_opt = effective_a_upper

    @jit
    def loss_fn(params):
        """Joint loss function for all curves with shared saturation (single power law)."""
        a_raw = params[0]
        a = a_lower_bound + jax.nn.sigmoid(a_raw) * (a_upper_for_opt - a_lower_bound)

        total_loss = 0.0
        total_weight = 0.0

        for i in range(n_curves):
            log_b = params[1 + 2*i]
            c = params[1 + 2*i + 1]

            compute_i = jax_data[i]['compute']
            loss_i = jax_data[i]['loss']
            weights_i = jax_data[i]['weights']

            # Log-space fitting: log(loss - a) = log(b) - c*log(compute)
            log_compute = jnp.log(compute_i)
            log_loss_shifted = jnp.log(loss_i - a + 1e-8)
            pred_log_loss_shifted = log_b - c * log_compute

            # Weighted MSE
            residuals = (log_loss_shifted - pred_log_loss_shifted) ** 2
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

    print(f"\nStarting optimization (single power law)...")
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
            a = float(a_lower_bound + jax.nn.sigmoid(a_raw) * (a_upper_for_opt - a_lower_bound))
            print(f"  Step {step:5d}: loss={best_loss:.6e}, a={a:.4f}")

    # Extract final parameters
    a_raw = best_params[0]
    a = float(a_lower_bound + jax.nn.sigmoid(a_raw) * (a_upper_for_opt - a_lower_bound))

    results = {
        'a': a,
        'curves': {},
        'is_single_power_law': True  # Flag to indicate single power law
    }

    for i in range(n_curves):
        log_b = float(best_params[1 + 2*i])
        c = float(best_params[1 + 2*i + 1])

        b = float(jnp.exp(log_b))

        name = jax_data[i]['name']
        compute_vals = np.array(jax_data[i]['compute'])
        loss_vals = np.array(jax_data[i]['loss'])

        # Compute R-squared using single power law
        predictions = a + b * np.power(compute_vals, -c)
        residuals = loss_vals - predictions
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((loss_vals - np.mean(loss_vals))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        results['curves'][name] = {
            'b': b,
            'c': c,
            'e': 0.0,  # Set to 0 for compatibility with plotting code
            'f': 0.0,  # Set to 0 for compatibility with plotting code
            'r_squared': r_squared
        }

    return results


# =============================================================================
# PLOTTING
# =============================================================================

def _display_name(opt_short):
    """Convert optimizer short name to display name for plots."""
    name = OPT_DISPLAY_NAMES.get(opt_short, opt_short)
    # Add legend suffix if specified
    if opt_short in legend_suffixes:
        name = f"{name} {legend_suffixes[opt_short]}"
    return name

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

def plot_comparison_multi_optimizer(data_dict, fit_results, scaling_rules, optimizer_shorts, optimizer_types, fit_metric, show_title=True, compute_formula='default', seq_length=2048):
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
        compute_formula: Compute formula used (default, 6N1, 6N2, M)
        seq_length: Sequence length for M formula
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
        plot_range = np.logspace(np.log10(metric_min * 0.5), np.log10(metric_max * 2.0), 200)
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

                line, = ax.plot(plot_range, loss_fit, linestyle=linestyle, color=color, linewidth=7.0,
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
        xlabel = 'Compute (PetaFlop-Hours)'
    else:  # non_emb
        xlabel = 'Non-embedding Parameters'

    # Formatting - improved aesthetics with larger axis titles
    ax.set_xlabel(xlabel, fontsize=28, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=28, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Improve tick labels - much larger
    ax.tick_params(axis='x', which='major', labelsize=32, width=2.5, length=12)
    ax.tick_params(axis='x', which='minor', width=2.0, length=8)
    ax.tick_params(axis='y', which='major', labelsize=36, width=2.5, length=12)
    ax.tick_params(axis='y', which='minor', width=2.0, length=8)

    if show_title:
        opts_str = ', '.join([_display_name(opt) for opt in optimizer_shorts])
        rules_str = ' vs '.join(scaling_rules)
        ax.set_title(f'Scaling Laws Comparison: {rules_str}\nOptimizers: {opts_str} (Shared saturation a = {fit_results["a"]:.4f})',
                    fontsize=20, fontweight='bold', pad=20)

    ax.legend(legend_handles, legend_labels, fontsize=16, loc='best', framealpha=0.95, ncol=2, 
              edgecolor='#333333', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Add vertical line at 24 heads
    # Calculate metric value for 24 heads from any available scaling rule
    metric_24_heads = None
    for rule in scaling_rules:
        if rule in ['Enoki', 'Enoki_Scaled', 'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', 'Qwen3_Hoyer']:
            # For head-based scaling rules, compute metric for 24 heads
            params_24 = compute_params(24, rule, compute_formula, seq_length)
            metric_24_heads = params_24[fit_metric]
            break
    
    if metric_24_heads is not None:
        ax.axvline(x=metric_24_heads, color='gray', linestyle='-', linewidth=2.0, alpha=0.5, zorder=1)

    # Add second x-axis showing size (heads or depth) on top
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
        ax2.set_xticklabels([str(size) for size in selected_sizes], fontweight='bold', fontsize=24)
        
        # Set tick parameters for top x-axis
        ax2.tick_params(axis='x', which='major', labelsize=24, width=2.0, length=10)
        ax2.tick_params(axis='x', which='minor', width=1.5, length=6)
        
        # Determine label based on scaling rules (use 'Heads' for head-based, 'Depth' for BigHead)
        if 'BigHead' in scaling_rules and len(scaling_rules) == 1:
            size_label = 'Depth'
        else:
            size_label = 'Heads'
        ax2.set_xlabel(size_label, fontsize=32, fontweight='bold')

    plt.tight_layout()

    return fig

def plot_comparison_relative_to_adamw(data_dict, fit_results, scaling_rules, optimizer_shorts, optimizer_types, fit_metric, show_title=True, compute_formula='default', seq_length=2048):
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
        compute_formula: Compute formula used (default, 6N1, 6N2, M)
        seq_length: Sequence length for M formula
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
        plot_range = np.logspace(np.log10(metric_min * 0.5), np.log10(metric_max * 2.0), 200)
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

                line, = ax.plot(plot_range, normalized_fit, linestyle=linestyle, color=color, linewidth=7.0,
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
        xlabel = 'Compute (PetaFlop-Hours)'
    else:  # non_emb
        xlabel = 'Non-embedding Parameters'

    # Formatting - both axes in log scale, improved aesthetics with larger axis titles
    ax.set_xlabel(xlabel, fontsize=28, fontweight='bold')
    ax.set_ylabel('Validation Loss Ratio (Loss / Loss$_{AdamW}$)', fontsize=28, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Improve tick labels - much larger
    ax.tick_params(axis='x', which='major', labelsize=32, width=2.5, length=12)
    ax.tick_params(axis='x', which='minor', width=2.0, length=8)
    ax.tick_params(axis='y', which='major', labelsize=36, width=2.5, length=12)
    ax.tick_params(axis='y', which='minor', width=2.0, length=8)
    
    # Horizontal line at y=1 represents AdamW baseline (slope 0 in log-log)
    ax.axhline(y=1.0, color='#333333', linestyle='--', linewidth=2.5, alpha=0.8, label='AdamW baseline (ratio=1)', zorder=5)

    if show_title:
        opts_str = ', '.join([_display_name(opt) for opt in optimizer_shorts])
        rules_str = ' vs '.join(scaling_rules)
        ax.set_title(f'Scaling Laws Comparison (Relative to AdamW, Log-Log): {rules_str}\nOptimizers: {opts_str}',
                    fontsize=20, fontweight='bold', pad=20)

    ax.legend(legend_handles, legend_labels, fontsize=16, loc='best', framealpha=0.95, ncol=2,
              edgecolor='#333333', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Add vertical line at 24 heads
    # Calculate metric value for 24 heads from any available scaling rule
    metric_24_heads = None
    for rule in scaling_rules:
        if rule in ['Enoki', 'Enoki_Scaled', 'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', 'Qwen3_Hoyer']:
            # For head-based scaling rules, compute metric for 24 heads
            params_24 = compute_params(24, rule, compute_formula, seq_length)
            metric_24_heads = params_24[fit_metric]
            break
    
    if metric_24_heads is not None:
        ax.axvline(x=metric_24_heads, color='gray', linestyle='-', linewidth=2.0, alpha=0.5, zorder=1)

    # Add second x-axis showing size (heads or depth) on top
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
        ax2.set_xticks([size_to_metric[size] for size in all_sizes_sorted])
        ax2.set_xticklabels([str(size) for size in all_sizes_sorted])
        
        # Determine label based on scaling rules (use 'Depth' for BigHead, 'Heads' for head-based)
        if 'BigHead' in scaling_rules and len(scaling_rules) == 1:
            size_label = 'Depth'
        else:
            size_label = 'Heads'
        ax2.set_xlabel(size_label, fontsize=20)

    plt.tight_layout()

    return fig

def plot_compute_gain(data_dict, fit_results, scaling_rules, optimizer_shorts, optimizer_types, fit_metric, baseline_optimizer, use_affine_by_part=False, show_title=True, no_loss=False, mark_outliers=None, compute_formula='default', seq_length=2048, efficiency_ymin=None, broken_axis=False, broken_axis_lower=0.1, broken_axis_upper=0.9):
    """
    Plot compute gain percentage relative to specified baseline optimizer.
    Same as plot_comparison_multi_optimizer but with added right y-axis for compute gain.

    Now plots BOTH affine-by-part and fitted curve compute gains simultaneously:
    - Affine-by-part: scatter points + connecting lines (solid/dashed based on optimizer)
    - Fitted curve: smooth dashed lines with lower opacity

    Args:
        baseline_optimizer: Short name of the baseline optimizer (e.g., 'adamw-decaying-wd')
        use_affine_by_part: If True, plot affine-by-part compute gain (always plotted now)
        show_title: Whether to show the plot title (default: True)
        no_loss: If True, hide loss points and fitted curves, show only compute gains (default: False)
        mark_outliers: List of outlier specs in format "optimizer:size" to mark with red stars
        compute_formula: Compute formula used (default, 6N1, 6N2, M)
        seq_length: Sequence length for M formula
        efficiency_ymin: Minimum y-value for efficiency axis (cuts off whitespace)
        broken_axis: If True, use broken y-axis removing gap between broken_axis_lower and broken_axis_upper
        broken_axis_lower: Lower bound of gap to remove (default: 0.1)
        broken_axis_upper: Upper bound of gap to remove (default: 0.9)
    """

    # Define transformation functions for broken axis
    def transform_y(y):
        """Transform y-values for broken axis: compress the gap region."""
        if not broken_axis:
            return y
        y = np.asarray(y)
        result = np.where(y <= broken_axis_lower, y,
                         np.where(y >= broken_axis_upper, y - (broken_axis_upper - broken_axis_lower),
                                  broken_axis_lower))  # values in gap map to lower bound
        return result

    def inverse_transform_y(y_trans):
        """Inverse transform for tick labels."""
        if not broken_axis:
            return y_trans
        y_trans = np.asarray(y_trans)
        return np.where(y_trans <= broken_axis_lower, y_trans, y_trans + (broken_axis_upper - broken_axis_lower))
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
        plot_range = np.logspace(np.log10(metric_min * 0.5), np.log10(metric_max * 2.0), 200)
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
    baseline_fits = {}
    baseline_affine_funcs = {}
    
    # Create piecewise affine interpolation from baseline optimizer's actual data
    print(f"\nPreparing baseline optimizer data: {baseline_optimizer}")
    
    # Get baseline optimizer's full name
    baseline_opt_type = optimizer_map[baseline_optimizer]
    
    for rule in scaling_rules:
        # Get baseline optimizer's data for this scaling rule
        if baseline_opt_type in data_dict and rule in data_dict[baseline_opt_type]:
            baseline_df = data_dict[baseline_opt_type][rule]
            
            if len(baseline_df) > 0:
                # Sort by metric (compute or non_emb)
                baseline_df_sorted = baseline_df.sort_values(by=fit_metric)
                baseline_metric = baseline_df_sorted[fit_metric].values
                baseline_loss = baseline_df_sorted['val_loss'].values
                
                print(f"  {rule}: {len(baseline_metric)} affine points from {baseline_metric[0]:.4e} to {baseline_metric[-1]:.4e}")
                
                # Create interpolation function (linear interpolation in log-log space)
                # This creates an affine-by-part function: loss = f(metric)
                # We'll store both forward (metric -> loss) and need inverse (loss -> metric)
                baseline_affine_funcs[rule] = {
                    'metric': baseline_metric,
                    'loss': baseline_loss
                }
        
        # Also get fitted curves
        baseline_curve_name = f'{baseline_optimizer}_{rule}'
        if baseline_curve_name in fit_results['curves']:
            baseline_fits[rule] = {
                'b': fit_results['curves'][baseline_curve_name]['b'],
                'c': fit_results['curves'][baseline_curve_name]['c'],
                'e': fit_results['curves'][baseline_curve_name]['e'],
                'f': fit_results['curves'][baseline_curve_name]['f']
            }
            print(f"  {rule}: fitted curve available")
    
    # Create second y-axis for compute gain
    ax_gain = ax.twinx()
    
    # Plot each optimizer x scaling_rule combination
    for opt_idx, opt_type in enumerate(optimizer_types):
        opt_short = optimizer_shorts[opt_idx]
        color = OPT_COLORS.get(opt_short, 'black')
        
        # Use same color for compute gain as for loss plot
        gain_color = color
        
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
                              zorder=10, alpha=0.5)
                    
                    obs_key = (rule, opt_short)
                    obs_handles[obs_key] = (scatter, f'{_display_name(opt_short)} {_display_rule(rule)}')
                
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
                    
                    line, = ax.plot(plot_range, loss_fit, linestyle=linestyle, color=color, linewidth=7.0,
                           zorder=9, alpha=0.5)
                    
                    fit_key = (rule, opt_short)
                    # Print fit formula to output
                    print(f"  Fit: {_display_name(opt_short)} {_display_rule(rule)}: {a:.3f}+{b:.2e}{metric_symbol}^{{-{c:.3f}}}+{e:.2e}{metric_symbol}^{{-{f:.3f}}} (R²={r2:.3f})")
                    fit_handles[fit_key] = (line, f'{_display_name(opt_short)} {_display_rule(rule)}')
            
            # Plot compute gain on right axis (skip baseline itself)
            if opt_short != baseline_optimizer:
                    # Plot affine-by-part compute gain (if available)
                    if rule in baseline_affine_funcs:
                        # Use piecewise affine interpolation
                        baseline_metric = baseline_affine_funcs[rule]['metric']
                        baseline_loss = baseline_affine_funcs[rule]['loss']

                        # Get optimizer's actual observed data points for this rule
                        opt_type_full = optimizer_map[opt_short]
                        opt_df = data_dict[opt_type_full][rule]

                        # Filter to only use points marked for fitting (respects --head-min)
                        if 'use_for_fit' in opt_df.columns:
                            opt_df = opt_df[opt_df['use_for_fit']]

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
                                scatter_size = 150 if no_loss else 100
                                line_alpha = 1.0 if no_loss else 1.0
                                line_width = 7.0 if no_loss else 6.0
                                # Use optimizer-specific linestyle if available, otherwise use default based on no_loss
                                default_line_style = '-' if no_loss else ':'
                                line_style = OPT_LINESTYLES.get(opt_short, default_line_style)
                                
                                # Transform y-values for broken axis
                                plot_gains = transform_y(compute_gains)
                                scatter_gain = ax_gain.scatter(valid_opt_metrics, plot_gains,
                                              s=scatter_size, marker=marker, c=gain_color, edgecolors='white',
                                              linewidths=1.5, zorder=11, alpha=1.0)

                                # Also plot connecting lines for visualization
                                sorted_indices = np.argsort(valid_opt_metrics)
                                sorted_metrics = np.array(valid_opt_metrics)[sorted_indices]
                                sorted_gains = transform_y(np.array(compute_gains)[sorted_indices])
                                line_gain, = ax_gain.plot(sorted_metrics, sorted_gains, linestyle=line_style, color=gain_color,
                                           linewidth=line_width, alpha=line_alpha, zorder=8)
                                
                                # Store handle for legend when no_loss
                                if no_loss:
                                    gain_key = (rule, opt_short, 'affine')
                                    gain_handles[gain_key] = (line_gain, f'{_display_name(opt_short)} {_display_rule(rule)} (affine)')
                    
                    # Plot fitted curve compute gain (if available)
                    if rule in baseline_fits and curve_name in fit_results['curves']:
                        # Use fitted curve with broken power law
                        baseline_b = baseline_fits[rule]['b']
                        baseline_c = baseline_fits[rule]['c']
                        baseline_e = baseline_fits[rule]['e']
                        baseline_f = baseline_fits[rule]['f']
                        
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
                            line_alpha_fit = 1.0 if no_loss else 1.0
                            line_width_fit = 6.0 if no_loss else 5.0
                            line_style_fit = '--'  # Always dashed for fitted curves

                            # Transform y-values for broken axis
                            plot_gain_ratio_fit = transform_y(compute_gain_ratio_fit)
                            line_gain_fit, = ax_gain.plot(valid_plot_range_fit, plot_gain_ratio_fit, linestyle=line_style_fit, color=gain_color,
                                       linewidth=line_width_fit, alpha=line_alpha_fit, zorder=7)
                            
                            # Store handle for legend when no_loss
                            if no_loss:
                                gain_key = (rule, opt_short, 'fit')
                                gain_handles[gain_key] = (line_gain_fit, f'{_display_name(opt_short)} {_display_rule(rule)} (fit)')
    
    # Add baseline line at 1.0 (ratio = 1 means same efficiency) on gain axis
    # Skip this line when using affine-by-part to focus on other optimizers
    if not use_affine_by_part:
        baseline_color = OPT_COLORS.get(baseline_optimizer, '#00CED1')
        # Transform y=1.0 for broken axis
        baseline_y = transform_y(np.array([1.0]))[0]
        ax_gain.axhline(y=baseline_y, color=baseline_color, linestyle='--', linewidth=1.5, alpha=0.6, zorder=7)
    
    # Add vertical line at 24 heads
    # Calculate metric value for 24 heads from any available scaling rule
    metric_24_heads = None
    for rule in scaling_rules:
        if rule in ['Enoki', 'Enoki_Scaled', 'Eryngii', 'Eryngii_Scaled', 'Qwen3_Scaled', 'Qwen3_Hoyer']:
            # For head-based scaling rules, compute metric for 24 heads
            params_24 = compute_params(24, rule, compute_formula, seq_length)
            metric_24_heads = params_24[fit_metric]
            break
    
    if metric_24_heads is not None:
        ax.axvline(x=metric_24_heads, color='gray', linestyle='-', linewidth=2.0, alpha=0.5, zorder=1)

    # Draw outlier markers (red stars) for specified points
    if mark_outliers:
        for outlier_spec in mark_outliers:
            try:
                opt_short, size_str = outlier_spec.split(':')
                size = float(size_str)

                # Find the data point
                opt_type_full = optimizer_map.get(opt_short, opt_short)
                for rule in scaling_rules:
                    if opt_type_full in data_dict and rule in data_dict[opt_type_full]:
                        df = data_dict[opt_type_full][rule]
                        point = df[df['size'] == size]
                        if len(point) > 0:
                            point = point.iloc[0]
                            # Draw red star on loss panel
                            if not no_loss:
                                ax.scatter([point[fit_metric]], [point['val_loss']],
                                          s=400, marker='*', c='red', edgecolors='black', linewidths=1.5,
                                          zorder=20, alpha=1.0)
                            # Also mark on compute gain panel if applicable
                            if opt_short != baseline_optimizer and rule in baseline_affine_funcs:
                                baseline_metric = baseline_affine_funcs[rule]['metric']
                                baseline_loss = baseline_affine_funcs[rule]['loss']
                                opt_loss = point['val_loss']
                                opt_metric = point[fit_metric]
                                # Interpolate to find baseline compute for same loss
                                if baseline_loss.min() <= opt_loss <= baseline_loss.max():
                                    baseline_compute_equiv = np.interp(opt_loss, baseline_loss[::-1], baseline_metric[::-1])
                                    gain = baseline_compute_equiv / opt_metric
                                    # Transform for broken axis
                                    plot_gain = transform_y(np.array([gain]))[0]
                                    ax_gain.scatter([opt_metric], [plot_gain],
                                                   s=400, marker='*', c='red', edgecolors='black', linewidths=1.5,
                                                   zorder=20, alpha=1.0)
                            print(f"  Marked outlier: {opt_short} size={size} at metric={point[fit_metric]:.2f}, loss={point['val_loss']:.4f}")
            except Exception as e:
                print(f"  Warning: Could not mark outlier '{outlier_spec}': {e}")

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
                if opt_short != baseline_optimizer:
                    # Add affine-by-part gain
                    gain_key_affine = (rule, opt_short, 'affine')
                    if gain_key_affine in gain_handles:
                        handle, label = gain_handles[gain_key_affine]
                        legend_handles.append(handle)
                        legend_labels.append(label)
                    
                    # Add fitted curve gain
                    gain_key_fit = (rule, opt_short, 'fit')
                    if gain_key_fit in gain_handles:
                        handle, label = gain_handles[gain_key_fit]
                        legend_handles.append(handle)
                        legend_labels.append(label)
    
    # Get metric info
    if fit_metric == 'compute':
        xlabel = 'Compute (PetaFlop-Hours)'
    else:
        xlabel = 'Non-embedding Parameters'
    
    # Formatting - improved aesthetics with larger axis titles
    ax.set_xlabel(xlabel, fontsize=28, fontweight='bold')
    if no_loss:
        # Hide left y-axis when showing only compute gains
        ax.set_ylabel('', fontsize=28, fontweight='bold')
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    else:
        # Validation loss axis has transparency (gray color)
        ax.set_ylabel('Validation Loss', fontsize=28, fontweight='bold', color='#666666')
        ax.set_yscale('log')
    
    ax.set_xscale('log')
    
    # Improve tick labels - much larger
    ax.tick_params(axis='x', which='major', labelsize=32, width=2.5, length=12)
    ax.tick_params(axis='x', which='minor', width=2.0, length=8)
    ax.tick_params(axis='y', which='major', labelsize=36, width=2.5, length=12)
    ax.tick_params(axis='y', which='minor', width=2.0, length=8)
    
    # Swap axes: compute efficiency on LEFT, validation loss on RIGHT
    # Move ax (validation loss) to right side
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()

    # Move ax_gain (compute efficiency) to left side
    ax_gain.yaxis.set_label_position('left')
    ax_gain.yaxis.tick_left()

    # Add grid lines
    ax_gain.grid(True, which='major', linestyle='-', alpha=0.3, color='gray')
    ax_gain.grid(True, which='minor', linestyle=':', alpha=0.2, color='gray')
    ax.grid(True, which='major', linestyle='-', alpha=0.3, color='gray', axis='x')
    ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='gray', axis='x')

    if no_loss:
        ax_gain.set_ylabel(f'Compute Efficiency vs {_display_name(baseline_optimizer)}', fontsize=35, fontweight='bold', color='#000000')
        ax_gain.tick_params(axis='y', labelcolor='#000000', labelsize=36)
        ax_gain.set_yscale('log')
        # Format y-axis to avoid scientific notation
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax_gain.yaxis.set_major_formatter(formatter)

        # Add horizontal dashed grey line at compute gain 1.0 (without affecting y-axis limits)
        y_1_trans = transform_y(np.array([1.0]))[0]
        ax_gain.axhline(y=y_1_trans, color='grey', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    else:
        # Compute efficiency axis is fully opaque (black)
        ax_gain.set_ylabel(f'Compute Efficiency vs {_display_name(baseline_optimizer)}', fontsize=28, fontweight='bold', color='#000000')
        ax_gain.tick_params(axis='y', labelcolor='#000000', labelsize=36)
        # Format y-axis to avoid scientific notation
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax_gain.yaxis.set_major_formatter(formatter)

    # Handle broken axis setup
    if broken_axis:
        from matplotlib.ticker import FixedLocator, FixedFormatter

        # Get current y-limits
        current_ymin, current_ymax = ax_gain.get_ylim()

        # Define tick values in original space that we want to show
        tick_values_orig = [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

        # Transform tick positions to the compressed space
        # Values <= broken_axis_lower stay as-is
        # Values >= broken_axis_upper are shifted down by (upper - lower)
        gap_size = broken_axis_upper - broken_axis_lower
        tick_positions = []
        tick_labels = []
        for v in tick_values_orig:
            if v <= broken_axis_lower:
                tick_positions.append(v)
                tick_labels.append(f'{v:.1f}')
            elif v >= broken_axis_upper:
                tick_positions.append(v - gap_size)
                if v == int(v):
                    tick_labels.append(f'{int(v)}.0')
                else:
                    tick_labels.append(f'{v:.1f}')

        print(f"  Broken axis tick positions (transformed): {tick_positions}")
        print(f"  Broken axis tick labels: {tick_labels}")

        # Set explicit tick positions and labels
        ax_gain.yaxis.set_major_locator(FixedLocator(tick_positions))
        ax_gain.yaxis.set_major_formatter(FixedFormatter(tick_labels))

        # Adjust y-limits to show all ticks
        # The max tick in transformed space
        max_tick_trans = max(tick_positions)
        if current_ymax < max_tick_trans * 1.05:
            ax_gain.set_ylim(top=max_tick_trans * 1.05)

        # Draw break indicators (diagonal lines)
        gap_y_trans = broken_axis_lower  # This is where the break is in transformed space

        # Draw break indicators as small diagonal lines at the break point
        d = 0.02  # Size of diagonal lines
        kwargs = dict(transform=ax_gain.get_yaxis_transform(), color='k', clip_on=False, linewidth=2)

        # Draw two diagonal lines to indicate the break
        ax_gain.plot((-d, +d), (gap_y_trans - d*0.3, gap_y_trans + d*0.7), **kwargs)
        ax_gain.plot((-d, +d), (gap_y_trans - d*1.0, gap_y_trans), **kwargs)

    # Set minimum y-value for efficiency axis if specified (cuts off whitespace)
    elif efficiency_ymin is not None:
        current_ymin, current_ymax = ax_gain.get_ylim()
        ax_gain.set_ylim(bottom=efficiency_ymin, top=current_ymax)

    if show_title:
        opts_str = ', '.join([_display_name(opt) for opt in optimizer_shorts])
        rules_str = ' vs '.join(scaling_rules)
        if no_loss:
            ax.set_title(f'Compute Gain Comparison: {rules_str}\nOptimizers: {opts_str} (relative to {_display_name(baseline_optimizer)})\nSolid lines = affine-by-part, Dashed lines = fitted curves',
                        fontsize=20, fontweight='bold', pad=20)
        else:
            ax.set_title(f'Scaling Laws Comparison: {rules_str}\nOptimizers: {opts_str} (Shared saturation a = {fit_results["a"]:.4f})',
                        fontsize=20, fontweight='bold', pad=20)
    
    if len(legend_handles) > 0:
        # Use ax_gain for legend when no_loss, otherwise use ax
        legend_ax = ax_gain if no_loss else ax
        legend_ax.legend(legend_handles, legend_labels, fontsize=16, loc='best', framealpha=0.95, ncol=2, 
                  edgecolor='#333333', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    if no_loss:
        ax_gain.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Add second x-axis showing size (heads or depth) on top
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

        # Filter: show all sizes <= 24, skip 26, then show every other starting from 28
        selected_sizes = []
        skip_next = False
        for size in all_sizes_sorted:
            if size <= 24:
                selected_sizes.append(size)
            elif size == 26:
                # Skip 26
                continue
            else:
                # For sizes >= 28, show every other
                if not skip_next:
                    selected_sizes.append(size)
                skip_next = not skip_next

        ax2.set_xticks([size_to_metric[size] for size in selected_sizes])
        ax2.set_xticklabels([str(int(size)) for size in selected_sizes], fontsize=28, fontweight='bold')
        ax2.tick_params(axis='x', which='major', labelsize=28, width=2.5, length=10, pad=8)

        # Determine label based on scaling rules
        if 'BigHead' in scaling_rules and len(scaling_rules) == 1:
            size_label = 'Depth'
        else:
            size_label = 'Heads'
        ax2.set_xlabel(size_label, fontsize=32, fontweight='bold', labelpad=10)
    
    plt.tight_layout()
    
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Validate arguments and prepare baseline optimizer if needed
    baseline_optimizer_short = None
    baseline_optimizer_type = None
    
    if args.affine_by_part and not args.plot_compute_gain:
        print("ERROR: --affine-by-part requires --plot-compute-gain to be set")
        exit(1)
    
    if args.no_loss and not args.plot_compute_gain:
        print("ERROR: --no-loss requires --plot-compute-gain to be set")
        exit(1)
    
    if args.plot_compute_gain:
        baseline_optimizer_short = args.plot_compute_gain
        baseline_optimizer_type = optimizer_map[baseline_optimizer_short]
        
        # Add baseline optimizer to the list if not already present
        if baseline_optimizer_short not in args.optimizers:
            print(f"Note: Adding baseline optimizer '{baseline_optimizer_short}' to fitting")
            args.optimizers.append(baseline_optimizer_short)
            optimizer_types.append(baseline_optimizer_type)
    
    print("="*70)
    print(f"Scaling Rules Comparison")
    print(f"Scaling Rules: {', '.join(args.scaling_rules)}")
    print(f"Optimizers: {', '.join(args.optimizers)} ({', '.join(optimizer_types)})")
    print(f"Fit Metric: {args.fit_metric}")
    print(f"Compute Formula: {args.compute_formula}")
    if args.compute_formula == 'M':
        print(f"  Sequence Length: {args.seq_length}")
    if args.min_compute:
        print(f"Min Compute: {args.min_compute:.4e} PFH")
    if args.head_min:
        print(f"Min Head/Depth for Fitting: {args.head_min}")
    print(f"Lower Bound on 'a': {args.a_lower_bound}")
    if args.plot_compute_gain:
        method = "piecewise affine interpolation" if args.affine_by_part else "fitted curve"
        print(f"Compute Gain Plot: Enabled (relative to {baseline_optimizer_short}, using {method})")
    print("="*70)

    # Load data for all optimizer x scaling_rule combinations
    data_dict = {}  # {optimizer_type: {scaling_rule: df}}

    for optimizer_idx, optimizer_type in enumerate(optimizer_types):
        optimizer_short = args.optimizers[optimizer_idx]
        print(f"\nLoading data for {optimizer_short} ({optimizer_type})...")

        data_dict[optimizer_type] = {}
        for scaling_rule in args.scaling_rules:
            df = load_scaling_rule_data(
                scaling_rule=scaling_rule,
                project=args.project,
                entity=args.entity,
                optimizer_type=optimizer_type,
                min_compute=args.min_compute,
                head_min=args.head_min,
                cache_dir=args.cache_dir,
                use_cache=not args.no_cache,
                compute_formula=args.compute_formula,
                seq_length=args.seq_length
            )
            data_dict[optimizer_type][scaling_rule] = df
            if len(df) > 0:
                print(f"  {scaling_rule}: {len(df)} data points")
            else:
                print(f"  {scaling_rule}: No data")

    # Prepare data for joint fitting across ALL optimizers and scaling rules
    # Only use points marked with use_for_fit=True
    # Skip optimizers in args.skip_fit
    joint_fit_data = []
    skipped_optimizers = set(args.skip_fit)

    for optimizer_idx, optimizer_type in enumerate(optimizer_types):
        optimizer_short = args.optimizers[optimizer_idx]

        # Skip optimizers that should not be fitted
        if optimizer_short in skipped_optimizers:
            print(f"  Skipping {optimizer_short} from curve fitting (--skip-fit)")
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
    if args.single_power_law:
        print("Using SINGLE power law: loss = a + b*C^{-c}")
    else:
        print("Using broken power law: loss = a + b*C^{-c} + e*C^{-f}")
    print(f"{'='*70}")

    # Perform joint fit - choose between single and broken power law
    if args.single_power_law:
        fit_results = fit_all_single_power_laws_joint(
            joint_fit_data,
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            a_lower_bound=args.a_lower_bound,
            a_upper_bound=args.a_upper_bound,
            equal_weight=args.equal_weight
        )
    else:
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

    is_single = fit_results.get('is_single_power_law', False)
    for curve_name, curve_params in fit_results['curves'].items():
        print(f"\n{curve_name}:")
        print(f"  b = {curve_params['b']:.6e}")
        print(f"  c = {curve_params['c']:.6f}")
        if not is_single:
            print(f"  e = {curve_params['e']:.6e}")
            print(f"  f = {curve_params['f']:.6f}")
        print(f"  R² = {curve_params['r_squared']:.6f}")

    # Choose which plot to create based on flags
    show_title = not args.no_title
    if args.plot_compute_gain:
        # Create compute gain plot (now always plots both affine-by-part and fitted curves)
        print(f"\nCreating compute gain plot (relative to {baseline_optimizer_short})...")
        print(f"  - Plotting affine-by-part compute gains (solid lines)")
        print(f"  - Plotting fitted curve compute gains (dashed lines)")
        fig = plot_compute_gain(
            data_dict,
            fit_results,
            args.scaling_rules,
            args.optimizers,
            optimizer_types,
            args.fit_metric,
            baseline_optimizer_short,
            use_affine_by_part=args.affine_by_part,
            show_title=show_title,
            no_loss=args.no_loss,
            mark_outliers=args.mark_outlier,
            compute_formula=args.compute_formula,
            seq_length=args.seq_length,
            efficiency_ymin=args.efficiency_ymin,
            broken_axis=args.broken_axis,
            broken_axis_lower=args.broken_axis_lower,
            broken_axis_upper=args.broken_axis_upper
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
                compute_formula=args.compute_formula,
                seq_length=args.seq_length
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
                compute_formula=args.compute_formula,
                seq_length=args.seq_length
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
            compute_formula=args.compute_formula,
            seq_length=args.seq_length
        )

    # Save plot
    if args.output:
        output_file = args.output
    else:
        rules_str = '_'.join(args.scaling_rules)
        opts_str = '_'.join([_filename_safe_name(opt) for opt in args.optimizers])
        if args.plot_compute_gain:
            if args.no_loss:
                suffix = '_compute_gain_both_no_loss'
            else:
                suffix = '_compute_gain_both'
        elif args.fit_relative_to_adamw and 'adamw' in args.optimizers:
            suffix = '_relative'
        else:
            suffix = ''
        # Add single power law indicator to filename
        if args.single_power_law:
            suffix += '_single_pl'
        output_file = f'ScalingComparison_{rules_str}_{opts_str}{suffix}.pdf'

    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    print(f"\n{'='*70}")
    print(f"Plot saved to: {os.path.abspath(output_file)}")
    print(f"{'='*70}")
