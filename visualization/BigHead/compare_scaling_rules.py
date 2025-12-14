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
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (16, 10)
rcParams['axes.linewidth'] = 1.5
rcParams['axes.edgecolor'] = '#333333'
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
                    choices=['adamw', 'mk4', 'dana', 'ademamix', 'd-muon', 'manau', 'manau-hard', 'adamw-decaying-wd', 'dana-mk4', 'ademamix-decaying-wd', 'dana-star-no-tau', 'dana-star', 'dana-star-no-tau-kappa-0-8', 'dana-star-no-tau-kappa-0-85', 'dana-star-no-tau-kappa-0-9'],
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
args = parser.parse_args()

# Map optimizer names
optimizer_map = {'adamw': 'adamw', 'mk4': 'dana-star-mk4', 'dana': 'dana', 'ademamix': 'ademamix',
                 'd-muon': 'd-muon', 'manau': 'manau', 'manau-hard': 'manau-hard', 'adamw-decaying-wd': 'adamw-decaying-wd', 'dana-mk4': 'dana-mk4', 'ademamix-decaying-wd': 'ademamix-decaying-wd', 'dana-star-no-tau': 'dana-star-no-tau', 'dana-star': 'dana-star', 'dana-star-no-tau-kappa-0-8': 'dana-star-no-tau-kappa-0-8', 'dana-star-no-tau-kappa-0-85': 'dana-star-no-tau-kappa-0-85', 'dana-star-no-tau-kappa-0-9': 'dana-star-no-tau-kappa-0-9'}
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

def load_scaling_rule_data(scaling_rule, project, entity, optimizer_type, min_compute=None):
    """Load data for a scaling rule and get best loss for each size.

    Args:
        scaling_rule: One of 'BigHead', 'EggHead', 'Enoki'
        project: WandB project name
        entity: WandB entity name
        optimizer_type: Type of optimizer to filter for
        min_compute: Minimum compute threshold in PFH (optional)
    """
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
        params = compute_params(size, scaling_rule)

        # Apply minimum compute filter if specified
        if min_compute is not None and params['compute'] < min_compute:
            continue

        results.append({
            'size': size,
            'val_loss': row['val_loss'],
            'compute': params['compute'],
            'non_emb': params['non_emb'],
            'total_params': params['total_params'],
            'scaling_rule': scaling_rule
        })

    result_df = pd.DataFrame(results)
    if min_compute is not None and len(result_df) > 0:
        print(f"  Filtered to {len(result_df)} data points with compute >= {min_compute:.4e} PFH")

    return result_df

# =============================================================================
# JOINT FITTING FUNCTIONS (JAX + Adagrad)
# =============================================================================

@jit
def saturated_power_law(x, a, b, c):
    """Saturated power law: y = a + b * x^c"""
    return a + b * jnp.power(x, c)

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
    Fit multiple saturated power laws simultaneously with a shared saturation level.

    This is the version from compare_bighead_diloco.py that returns results in dict format.

    Joint fitting: loss_i = a + b_i * compute^c_i
    where 'a' (saturation level) is shared across all curves.

    Args:
        data_list: List of dicts, each with 'compute', 'loss', and 'name' keys
        n_steps: Number of optimization steps
        learning_rate: Learning rate for Adagrad optimizer
        a_lower_bound: Non-trainable lower bound constant for parameter a (default: 0.0)

    Returns:
        Dict with 'a' (shared saturation) and 'curves' (dict mapping name -> params)
    """
    print(f"\nPreparing {len(data_list)} curves for joint fitting...")

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

    # Initialize parameters
    # Params: [a_raw, log(b_0), c_0, log(b_1), c_1, ..., log(b_n), c_n]
    n_curves = len(jax_data)
    init_params = [0.0]  # a_raw (sigmoid will map to middle of range)
    for i in range(n_curves):
        init_params.extend([jnp.log(1000.0), -0.1])  # [log(b_i), c_i]

    fit_params = jnp.array(init_params, dtype=jnp.float32)

    @jit
    def loss_fn(params):
        """Joint loss function for all curves with shared saturation."""
        # Extract shared saturation
        a_raw = params[0]
        min_loss = jnp.min(jnp.array([jnp.min(d['loss']) for d in jax_data]))
        a = a_lower_bound + jax.nn.sigmoid(a_raw) * (min_loss * 0.99 - a_lower_bound)

        total_loss = 0.0
        total_weight = 0.0

        for i in range(n_curves):
            log_b = params[1 + 2*i]
            c = params[1 + 2*i + 1]

            compute_i = jax_data[i]['compute']
            loss_i = jax_data[i]['loss']
            weights_i = jax_data[i]['weights']

            # Log-space fitting
            log_compute = jnp.log(compute_i)
            log_loss_shifted = jnp.log(loss_i - a + 1e-8)
            pred_log_loss_shifted = log_b + c * log_compute

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

    print(f"\nStarting optimization...")
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
        log_b = float(best_params[1 + 2*i])
        b = float(jnp.exp(log_b))
        c = float(best_params[1 + 2*i + 1])

        name = jax_data[i]['name']
        compute_vals = np.array(jax_data[i]['compute'])
        loss_vals = np.array(jax_data[i]['loss'])

        # Compute R-squared
        predictions = a + b * np.power(compute_vals, c)
        residuals = loss_vals - predictions
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((loss_vals - np.mean(loss_vals))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        results['curves'][name] = {
            'b': b,
            'c': c,
            'r_squared': r_squared
        }

    return results

# =============================================================================
# PLOTTING
# =============================================================================

def plot_comparison_multi_optimizer(data_dict, fit_results, scaling_rules, optimizer_shorts, optimizer_types, fit_metric):
    """
    Plot comparison of multiple optimizers and scaling rules on a single plot.

    Args:
        data_dict: Dict {optimizer_type: {scaling_rule: DataFrame}}
        fit_results: Dict with 'a' and 'curves' from joint fitting
        scaling_rules: List of scaling rule names
        optimizer_shorts: List of short optimizer names
        optimizer_types: List of full optimizer type names
        fit_metric: 'compute' or 'non_emb'
    """
    fig, ax = plt.subplots(figsize=(18, 10))

    # Color scheme for optimizers - expanded with all variants
    opt_colors = {
        'adamw': '#1f77b4',           # Deep blue
        'adamw-decaying-wd': '#00CED1', # Dark turquoise
        'mk4': '#d62728',              # Red
        'dana': '#2ca02c',             # Green
        'dana-mk4': '#8c564b',         # Brown
        'dana-star': '#7f7f7f',        # Gray
        'dana-star-no-tau': '#bcbd22', # Olive
        'dana-star-no-tau-kappa-0-8': '#006400',   # Dark green
        'dana-star-no-tau-kappa-0-85': '#228B22',  # Forest green
        'dana-star-no-tau-kappa-0-9': '#32CD32',   # Lime green
        'ademamix': '#9467bd',         # Purple
        'ademamix-decaying-wd': '#e377c2', # Pink
        'd-muon': '#ff7f0e',           # Orange
        'manau': '#8B4513',            # Saddle brown
        'manau-hard': '#DC143C'        # Crimson
    }

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
        color = opt_colors.get(opt_short, 'black')

        for rule in scaling_rules:
            df = data_dict[opt_type][rule]

            if len(df) == 0:
                continue

            marker = rule_markers.get(rule, 'x')
            linestyle = rule_linestyles.get(rule, '-')

            # Plot observed data (no label yet) - improved styling
            scatter = ax.scatter(df[fit_metric], df['val_loss'],
                      s=150, marker=marker, c=color, edgecolors='white', linewidths=2.0,
                      zorder=10, alpha=0.85)

            obs_key = (rule, opt_short)
            obs_handles[obs_key] = (scatter, f'{opt_short} {rule} (observed)')

            # Plot fitted curve if available (no label yet)
            curve_name = f'{opt_short}_{rule}'
            if curve_name in fit_results['curves'] and plot_range is not None:
                a = fit_results['a']
                b = fit_results['curves'][curve_name]['b']
                c = fit_results['curves'][curve_name]['c']
                r2 = fit_results['curves'][curve_name]['r_squared']

                # Saturated power law
                loss_fit = a + b * np.power(plot_range, c)

                line, = ax.plot(plot_range, loss_fit, linestyle=linestyle, color=color, linewidth=3.0,
                       zorder=9, alpha=0.9)

                fit_key = (rule, opt_short)
                fit_handles[fit_key] = (line, f'{opt_short} {rule}: {a:.3f} + {b:.2e} × {metric_symbol}$^{{{c:.4f}}}$ ($R^2$={r2:.3f})')

    # Create custom ordered legend
    # Order: For each scaling rule, show all fit curves, then all observed points
    legend_handles = []
    legend_labels = []

    for rule in scaling_rules:
        # First, all fit curves for this scaling rule
        for opt_short in optimizer_shorts:
            fit_key = (rule, opt_short)
            if fit_key in fit_handles:
                handle, label = fit_handles[fit_key]
                legend_handles.append(handle)
                legend_labels.append(label)

        # Then, all observed data for this scaling rule
        for opt_short in optimizer_shorts:
            obs_key = (rule, opt_short)
            if obs_key in obs_handles:
                handle, label = obs_handles[obs_key]
                legend_handles.append(handle)
                legend_labels.append(label)

    # Get metric info
    if fit_metric == 'compute':
        xlabel = 'Compute (PetaFlop-Hours)'
    else:  # non_emb
        xlabel = 'Non-embedding Parameters'

    # Formatting - improved aesthetics
    ax.set_xlabel(xlabel, fontsize=22, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=22, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1.0, length=4)

    opts_str = ', '.join(optimizer_shorts)
    rules_str = ' vs '.join(scaling_rules)
    ax.set_title(f'Scaling Laws Comparison: {rules_str}\nOptimizers: {opts_str} (Shared saturation a = {fit_results["a"]:.4f})',
                fontsize=20, fontweight='bold', pad=20)

    ax.legend(legend_handles, legend_labels, fontsize=12, loc='best', framealpha=0.95, ncol=2, 
              edgecolor='#333333', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')

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
        
        # Determine label based on scaling rules (use 'Heads' for head-based, 'Depth' for BigHead)
        if 'BigHead' in scaling_rules and len(scaling_rules) == 1:
            size_label = 'Depth'
        else:
            size_label = 'Heads'
        ax2.set_xlabel(size_label, fontsize=20)

    plt.tight_layout()

    return fig

def plot_comparison_relative_to_adamw(data_dict, fit_results, scaling_rules, optimizer_shorts, optimizer_types, fit_metric):
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
    """
    fig, ax = plt.subplots(figsize=(18, 10))

    # Color scheme for optimizers - expanded with all variants
    opt_colors = {
        'adamw': '#1f77b4',           # Deep blue
        'adamw-decaying-wd': '#00CED1', # Dark turquoise
        'mk4': '#d62728',              # Red
        'dana': '#2ca02c',             # Green
        'dana-mk4': '#8c564b',         # Brown
        'dana-star': '#7f7f7f',        # Gray
        'dana-star-no-tau': '#bcbd22', # Olive
        'dana-star-no-tau-kappa-0-8': '#006400',   # Dark green
        'dana-star-no-tau-kappa-0-85': '#228B22',  # Forest green
        'dana-star-no-tau-kappa-0-9': '#32CD32',   # Lime green
        'ademamix': '#9467bd',         # Purple
        'ademamix-decaying-wd': '#e377c2', # Pink
        'd-muon': '#ff7f0e',           # Orange
        'manau': '#8B4513',            # Saddle brown
        'manau-hard': '#DC143C'        # Crimson
    }

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
                'c': fit_results['curves'][curve_name]['c']
            }

    # Plot each optimizer x scaling_rule combination
    for opt_idx, opt_type in enumerate(optimizer_types):
        opt_short = optimizer_shorts[opt_idx]
        color = opt_colors.get(opt_short, 'black')

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

            # Calculate normalized observed losses in log-log space
            # y_norm = log(loss) - log(loss_adamw) = log(loss / loss_adamw)
            metric_vals = df[fit_metric].values
            observed_losses = df['val_loss'].values
            adamw_baseline = adamw_a + adamw_b * np.power(metric_vals, adamw_c)
            
            # Ratio in log space = difference of logs
            normalized_losses = observed_losses / adamw_baseline

            # Plot normalized observed data (log scale for y) - improved styling
            scatter = ax.scatter(metric_vals, normalized_losses,
                      s=150, marker=marker, c=color, edgecolors='white', linewidths=2.0,
                      zorder=10, alpha=0.85)

            obs_key = (rule, opt_short)
            obs_handles[obs_key] = (scatter, f'{opt_short} {rule} (observed)')

            # Plot normalized fitted curve if available
            curve_name = f'{opt_short}_{rule}'
            if curve_name in fit_results['curves'] and plot_range is not None:
                opt_a = fit_results['a']
                opt_b = fit_results['curves'][curve_name]['b']
                opt_c = fit_results['curves'][curve_name]['c']
                r2 = fit_results['curves'][curve_name]['r_squared']

                # Calculate normalized fit in log-log space
                # y_norm = loss / loss_adamw
                opt_fit = opt_a + opt_b * np.power(plot_range, opt_c)
                adamw_baseline_curve = adamw_a + adamw_b * np.power(plot_range, adamw_c)
                normalized_fit = opt_fit / adamw_baseline_curve

                line, = ax.plot(plot_range, normalized_fit, linestyle=linestyle, color=color, linewidth=3.0,
                       zorder=9, alpha=0.9)

                fit_key = (rule, opt_short)
                
                # For AdamW, it will be a constant at 1.0 (log scale makes this a horizontal line)
                if opt_short == 'adamw':
                    fit_handles[fit_key] = (line, f'{opt_short} {rule}: baseline (slope 0)')
                else:
                    # Show the formula as ratio
                    # Calculate effective slope in log-log space: d(log(y_norm))/d(log(x))
                    # For normalized curve: y_norm = (a + b*x^c) / (a + b_adamw*x^c_adamw)
                    # Effective slope ≈ c - c_adamw for large x
                    slope_diff = opt_c - adamw_c
                    fit_handles[fit_key] = (line, f'{opt_short} {rule}: ({opt_a:.3f} + {opt_b:.2e} × {metric_symbol}$^{{{opt_c:.4f}}}$) / adamw, Δc={slope_diff:.4f} ($R^2$={r2:.3f})')

    # Create custom ordered legend
    legend_handles = []
    legend_labels = []

    for rule in scaling_rules:
        # First, all fit curves for this scaling rule
        for opt_short in optimizer_shorts:
            fit_key = (rule, opt_short)
            if fit_key in fit_handles:
                handle, label = fit_handles[fit_key]
                legend_handles.append(handle)
                legend_labels.append(label)

        # Then, all observed data for this scaling rule
        for opt_short in optimizer_shorts:
            obs_key = (rule, opt_short)
            if obs_key in obs_handles:
                handle, label = obs_handles[obs_key]
                legend_handles.append(handle)
                legend_labels.append(label)

    # Get metric info
    if fit_metric == 'compute':
        xlabel = 'Compute (PetaFlop-Hours)'
    else:  # non_emb
        xlabel = 'Non-embedding Parameters'

    # Formatting - both axes in log scale, improved aesthetics
    ax.set_xlabel(xlabel, fontsize=22, fontweight='bold')
    ax.set_ylabel('Validation Loss Ratio (Loss / Loss$_{AdamW}$)', fontsize=22, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1.0, length=4)
    
    # Horizontal line at y=1 represents AdamW baseline (slope 0 in log-log)
    ax.axhline(y=1.0, color='#333333', linestyle='--', linewidth=2.5, alpha=0.8, label='AdamW baseline (ratio=1)', zorder=5)

    opts_str = ', '.join(optimizer_shorts)
    rules_str = ' vs '.join(scaling_rules)
    ax.set_title(f'Scaling Laws Comparison (Relative to AdamW, Log-Log): {rules_str}\nOptimizers: {opts_str}',
                fontsize=20, fontweight='bold', pad=20)

    ax.legend(legend_handles, legend_labels, fontsize=12, loc='best', framealpha=0.95, ncol=2,
              edgecolor='#333333', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')

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

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print(f"Scaling Rules Comparison")
    print(f"Scaling Rules: {', '.join(args.scaling_rules)}")
    print(f"Optimizers: {', '.join(args.optimizers)} ({', '.join(optimizer_types)})")
    print(f"Fit Metric: {args.fit_metric}")
    if args.min_compute:
        print(f"Min Compute: {args.min_compute:.4e} PFH")
    print(f"Lower Bound on 'a': {args.a_lower_bound}")
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
                min_compute=args.min_compute
            )
            data_dict[optimizer_type][scaling_rule] = df
            if len(df) > 0:
                print(f"  {scaling_rule}: {len(df)} data points")
            else:
                print(f"  {scaling_rule}: No data")

    # Prepare data for joint fitting across ALL optimizers and scaling rules
    joint_fit_data = []

    for optimizer_idx, optimizer_type in enumerate(optimizer_types):
        optimizer_short = args.optimizers[optimizer_idx]

        for scaling_rule in args.scaling_rules:
            df = data_dict[optimizer_type][scaling_rule]

            if len(df) > 0:
                joint_fit_data.append({
                    'compute': df[args.fit_metric].values,
                    'loss': df['val_loss'].values,
                    'name': f'{optimizer_short}_{scaling_rule}'
                })

    # Check if we have any data
    if len(joint_fit_data) == 0:
        print("\nNo data found for any optimizer/scaling rule combination. Exiting.")
        exit(1)

    print(f"\n{'='*70}")
    print(f"Joint Fitting {len(joint_fit_data)} Curves")
    print(f"{'='*70}")

    # Perform single joint fit with shared saturation level
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
        print(f"  R² = {curve_params['r_squared']:.6f}")

    # Choose which plot to create based on --fit-relative-to-adamw flag
    if args.fit_relative_to_adamw:
        # Create relative plot with AdamW as baseline
        if 'adamw' not in args.optimizers:
            print("\nWarning: --fit-relative-to-adamw requires 'adamw' in optimizers list. Falling back to standard plot.")
            fig = plot_comparison_multi_optimizer(
                data_dict,
                fit_results,
                args.scaling_rules,
                args.optimizers,
                optimizer_types,
                args.fit_metric
            )
        else:
            print(f"\nCreating relative comparison plot (AdamW as baseline)...")
            fig = plot_comparison_relative_to_adamw(
                data_dict,
                fit_results,
                args.scaling_rules,
                args.optimizers,
                optimizer_types,
                args.fit_metric
            )
    else:
        # Create standard comparison plot
        fig = plot_comparison_multi_optimizer(
            data_dict,
            fit_results,
            args.scaling_rules,
            args.optimizers,
            optimizer_types,
            args.fit_metric
        )

    # Save plot
    if args.output:
        output_file = args.output
    else:
        rules_str = '_'.join(args.scaling_rules)
        opts_str = '_'.join(args.optimizers)
        suffix = '_relative' if args.fit_relative_to_adamw and 'adamw' in args.optimizers else ''
        output_file = f'ScalingComparison_{rules_str}_{opts_str}{suffix}.pdf'

    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    print(f"\n{'='*70}")
    print(f"Plot saved to: {os.path.abspath(output_file)}")
    print(f"{'='*70}")
