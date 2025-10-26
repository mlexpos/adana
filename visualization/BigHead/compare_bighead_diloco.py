#!/usr/bin/env python3
"""
Compare BigHead vs DiLoco Scaling Performance

This script compares the scaling performance between:
1. BigHead architecture (width and depth scale together)
2. DiLoco architecture (standard scaling from model_library)

For each architecture and model size, it takes the best final-val/loss achieved,
plots loss vs compute, and fits saturated power laws: loss = a + b * compute^c

Joint Fitting Approach:
- All curves are fit simultaneously with a SHARED saturation level 'a'
- Uses JAX + Adagrad optimization (inspired by bighead_lr_scaling.py)
- Log-space fitting for numerical stability
- Weighted MSE loss (larger models get more weight)
- Constraint: 0 < a < min(observed losses) via sigmoid transformation

Usage:
    python compare_bighead_diloco.py --optimizers adamw
    python compare_bighead_diloco.py --optimizers adamw mk4
    python compare_bighead_diloco.py --optimizers adamw mk4 --n-steps 100000 --learning-rate 0.2
    python compare_bighead_diloco.py --optimizers adamw --min-compute 0.001  # Filter out data with < 0.001 PFH
    python compare_bighead_diloco.py --optimizers adamw --fit-metric non_emb  # Fit using non-embedding parameters
"""

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style, rc, rcParams
from scipy.optimize import curve_fit
import argparse
import warnings
import sys
import os
import jax
import jax.numpy as jnp
from jax import grad, jit
import optax

# Add parent directory to path to import model_library
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from model_library import MODEL_CONFIGS

warnings.filterwarnings('ignore')

# Matplotlib formatting
style.use('default')
rc('font', family='sans-serif')
rcParams['font.weight'] = 'light'
rcParams['font.size'] = 18
rcParams['figure.figsize'] = (14, 8)

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description='Compare BigHead vs DiLoco scaling performance')
parser.add_argument('--optimizers', type=str, nargs='+', required=True,
                    choices=['adamw', 'mk4', 'dana', 'ademamix', 'd-muon'],
                    help='Optimizer types to compare (can specify multiple, e.g., --optimizers adamw mk4)')
parser.add_argument('--project', type=str, default='danastar',
                    help='WandB project name (default: danastar)')
parser.add_argument('--bighead-group', type=str, default='DanaStar_MK4_BigHead_Sweep',
                    help='WandB group name for BigHead (default: DanaStar_MK4_BigHead_Sweep)')
parser.add_argument('--entity', type=str, default='ep-rmt-ml-opt',
                    help='WandB entity name (default: ep-rmt-ml-opt)')
parser.add_argument('--output', type=str, default=None,
                    help='Output filename for plot (default: auto-generated)')
parser.add_argument('--n-steps', type=int, default=50000,
                    help='Number of optimization steps for joint fitting (default: 50000)')
parser.add_argument('--learning-rate', type=float, default=0.1,
                    help='Learning rate for Adagrad optimizer (default: 0.1)')
parser.add_argument('--min-compute', type=float, default=None,
                    help='Minimum compute threshold in PetaFlop-Hours. Data points with compute below this value will be discarded (default: None)')
parser.add_argument('--fit-metric', type=str, default='compute',
                    choices=['compute', 'non_emb'],
                    help='Metric to use as independent variable for fitting: compute (PFH) or non_emb (non-embedding parameters) (default: compute)')
args = parser.parse_args()

# Map optimizer names
optimizer_map = {'adamw': 'adamw', 'mk4': 'dana-star-mk4', 'dana': 'dana', 'ademamix': 'ademamix', 'd-muon': 'd-muon'}
optimizer_types = [optimizer_map[opt] for opt in args.optimizers]

# =============================================================================
# PARAMETER CALCULATION FUNCTIONS
# =============================================================================

def compute_bighead_params(depth):
    """
    Compute parameters for BigHead architecture.

    BigHead scaling:
    - head_dim = 16 * depth
    - n_embd = 16 * depth^2
    - mlp_hidden = 32 * depth^2
    - n_head = depth
    - n_layer = depth
    """
    head_dim = 16 * depth
    n_embd = 16 * depth * depth
    mlp_hidden = 32 * depth * depth
    n_head = depth
    n_layer = depth

    # Non-embedding params (use float to avoid overflow)
    non_emb = float(depth * (3 * head_dim * n_embd * n_head + n_embd * n_embd +
                             2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd)

    # Total params (use float to avoid overflow)
    vocab_size = 50304
    total_params = float(non_emb + 2 * n_embd * vocab_size)

    # Iterations (from BigHead.sh)
    iterations = int(20 * total_params / 65536)

    # Compute in FLOPs (use float to avoid overflow)
    # compute = 6 * non_emb * total_params * 20
    compute_flops = 6.0 * non_emb * total_params * 20.0

    # Convert to PetaFlop-Hours: 1 PFH = 3600e15 FLOPs
    compute_pfh = compute_flops / (3600e15) 

    return {
        'non_emb': int(non_emb),
        'total_params': int(total_params),
        'iterations': iterations,
        'compute': compute_pfh,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd
    }

def compute_diloco_params(n_head, qkv_dim, n_layer):
    """
    Compute parameters for DiLoco architecture (from model_library).

    DiLoco uses the standard scaling from softmax_lr_scaling.py:
    - n_embd = n_head * qkv_dim
    - mlp_hidden = 4 * n_embd
    - Non-emb = 12 * n_embd^2 * n_layer
    """
    n_embd = n_head * qkv_dim

    # Non-embedding params (use float to avoid overflow)
    non_emb = float(12 * n_embd * n_embd * n_layer)

    # Total params including embeddings (use float to avoid overflow)
    vocab_size = 50340  # Used in DiLoco
    total_params = float(non_emb + n_embd * 2 * vocab_size)

    # Compute in FLOPs (use float to avoid overflow)
    # compute = 6 * non_emb * total_params * 20
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

def load_bighead_data(project, group, entity, optimizer_type, min_compute=None):
    """Load BigHead data and get best loss for each depth.

    Args:
        project: WandB project name
        group: WandB group name
        entity: WandB entity name
        optimizer_type: Type of optimizer to filter for
        min_compute: Minimum compute threshold in PFH (optional)
    """
    api = wandb.Api()

    print(f"Loading BigHead data from {group}...")
    runs = api.runs(f"{entity}/{project}", filters={"group": group})

    data = []
    for run in runs:
        config = run.config
        summary = run.summary

        # Filter by optimizer
        opt = config.get('opt', '')
        if opt != optimizer_type:
            continue

        # Check completion
        actual_iter = summary.get('iter', 0)
        iterations_config = config.get('iterations', 0)
        if actual_iter > 0 and iterations_config > 0 and actual_iter < iterations_config:
            continue

        depth = config.get('n_layer')
        val_loss = summary.get('final-val/loss')

        if depth is None or val_loss is None:
            continue

        data.append({
            'depth': depth,
            'val_loss': val_loss,
            'run_name': run.name
        })

    df = pd.DataFrame(data)
    print(f"  Loaded {len(df)} BigHead runs")

    if len(df) == 0:
        return df

    # Get best loss for each depth
    best_by_depth = df.groupby('depth')['val_loss'].min().reset_index()

    # Add compute information
    results = []
    for _, row in best_by_depth.iterrows():
        depth = row['depth']
        params = compute_bighead_params(depth)

        # Apply minimum compute filter if specified
        if min_compute is not None and params['compute'] < min_compute:
            continue

        results.append({
            'depth': depth,
            'val_loss': row['val_loss'],
            'compute': params['compute'],
            'non_emb': params['non_emb'],
            'total_params': params['total_params']
        })

    result_df = pd.DataFrame(results)
    if min_compute is not None and len(result_df) > 0:
        print(f"  Filtered to {len(result_df)} data points with compute >= {min_compute:.4e} PFH")

    return result_df

def load_diloco_data(project, entity, optimizer_type, min_compute=None):
    """Load DiLoco data from model_library groups and get best loss for each model size.

    Args:
        project: WandB project name
        entity: WandB entity name
        optimizer_type: Type of optimizer to filter for
        min_compute: Minimum compute threshold in PFH (optional)
    """
    api = wandb.Api()

    # Get model configs for this optimizer
    if optimizer_type == 'adamw':
        model_prefix = 'AW'
    elif optimizer_type == 'dana-star-mk4':
        model_prefix = 'MK4_'
    elif optimizer_type == 'dana':
        model_prefix = 'DS'
    elif optimizer_type == 'ademamix':
        print("Warning: ademamix not in model_library, skipping DiLoco")
        return pd.DataFrame()
    elif optimizer_type == 'd-muon':
        print("Warning: d-muon not in model_library, skipping DiLoco")
        return pd.DataFrame()

    diloco_models = {k: v for k, v in MODEL_CONFIGS.items()
                     if k.startswith(model_prefix) and v['optimizer'] ==
                     ('mk4' if optimizer_type == 'dana-star-mk4' else optimizer_type)}

    print(f"\nLoading DiLoco data for {len(diloco_models)} model sizes...")

    results = []
    for model_key, model_info in diloco_models.items():
        group_name = model_info['group']
        print(f"  Loading {model_key} from {group_name}...")

        runs = api.runs(f"{entity}/{project}", filters={"group": group_name})

        best_loss = float('inf')
        for run in runs:
            config = run.config
            summary = run.summary

            # Check completion
            actual_iter = summary.get('iter', 0)
            iterations_config = config.get('iterations', 0)
            if actual_iter > 0 and iterations_config > 0 and actual_iter < iterations_config:
                continue

            val_loss = summary.get('final-val/loss')
            if val_loss is not None and val_loss < best_loss:
                best_loss = val_loss

        if best_loss < float('inf'):
            params = compute_diloco_params(
                model_info['n_head'],
                model_info['qkv_dim'],
                model_info['n_layer']
            )

            # Apply minimum compute filter if specified
            if min_compute is not None and params['compute'] < min_compute:
                print(f"    Skipped (compute {params['compute']:.4e} < {min_compute:.4e} PFH)")
                continue

            results.append({
                'model': model_key,
                'n_layer': model_info['n_layer'],
                'val_loss': best_loss,
                'compute': params['compute'],
                'non_emb': params['non_emb'],
                'total_params': params['total_params']
            })
            print(f"    Best loss: {best_loss:.4f}")

    result_df = pd.DataFrame(results)
    if min_compute is not None and len(result_df) > 0:
        print(f"  Filtered to {len(result_df)} data points with compute >= {min_compute:.4e} PFH")

    return result_df

# =============================================================================
# SATURATED POWER LAW FITTING (JAX-based Joint Fitting)
# =============================================================================

def saturated_power_law(compute, a, b, c):
    """Saturated power law: loss = a + b * compute^c"""
    return a + b * np.power(compute, c)

def fit_all_saturated_power_laws_joint(data_list, n_steps=50000, learning_rate=0.1):
    """
    Fit multiple saturated power laws simultaneously with a shared saturation level.

    Joint fitting: loss_i = a + b_i * compute^c_i
    where 'a' (saturation level) is shared across all curves.

    Args:
        data_list: List of dicts, each containing:
            - 'compute': array of compute values (PetaFlop-Hours)
            - 'loss': array of loss values
            - 'name': string identifier for the curve
        n_steps: Number of optimization steps
        learning_rate: Learning rate for Adagrad optimizer

    Returns:
        dict: {
            'a': shared saturation level (float),
            'curves': {
                curve_name: {'b': float, 'c': float, 'r_squared': float}
            }
        }

    Loss function inspired by bighead_lr_scaling.py:
        - Log-space fitting for numerical stability
        - Weighted by compute values (larger models get more weight)
        - MSE loss in log-space after subtracting saturation
    """
    if len(data_list) == 0:
        return None

    print(f"\n{'='*70}")
    print("Joint Saturated Power Law Fitting (JAX + Adagrad)")
    print(f"{'='*70}")
    print(f"Fitting {len(data_list)} curves with shared saturation level 'a'")
    print(f"Optimization: {n_steps} steps, learning rate = {learning_rate}")

    # Prepare data for each curve
    jax_data = []
    for i, data in enumerate(data_list):
        compute_arr = jnp.array(data['compute'], dtype=jnp.float32)
        loss_arr = jnp.array(data['loss'], dtype=jnp.float32)
        name = data['name']

        print(f"  Curve {i}: {name}")
        print(f"    Data points: {len(compute_arr)}")
        print(f"    Compute range: {float(jnp.min(compute_arr)):.4e} to {float(jnp.max(compute_arr)):.4e} PFH")
        print(f"    Loss range: {float(jnp.min(loss_arr)):.4f} to {float(jnp.max(loss_arr)):.4f}")

        jax_data.append({
            'compute': compute_arr,
            'loss': loss_arr,
            'name': name,
            'weights': compute_arr  # Weight by compute (larger models matter more)
        })

    # Initialize parameters
    # Params: [a_raw, log(b_0), c_0, log(b_1), c_1, ..., log(b_n), c_n]
    # a_raw: raw saturation parameter (a = sigmoid(a_raw) * min_loss * 0.99)
    #        Initialize to 0.0 so sigmoid(0) = 0.5, giving a ≈ 0.5 * min_loss
    # log(b_i): log of scale parameter for curve i (initialize to log(1000))
    # c_i: exponent for curve i (initialize to -0.1)
    n_curves = len(jax_data)
    init_params = [0.0]  # a_raw (sigmoid will map to middle of range)
    for i in range(n_curves):
        init_params.extend([jnp.log(1000.0), -0.1])  # [log(b_i), c_i]

    fit_params = jnp.array(init_params, dtype=jnp.float32)

    @jit
    def loss_fn(params):
        """
        Joint loss function for all curves with shared saturation.

        Log-space fitting: log(loss - a) = log(b) + c * log(compute)
        Weighted MSE loss: sum_i sum_j w_ij * [log(loss_ij - a) - (log(b_i) + c_i * log(compute_ij))]^2

        Constraint: 0 < a < min(observed losses)
        """
        # Extract shared saturation (constrain to be positive and less than min loss)
        a_raw = params[0]
        # Find minimum loss across all curves
        min_loss = jnp.min(jnp.array([jnp.min(d['loss']) for d in jax_data]))
        # Constrain: a = sigmoid(a_raw) * min_loss * 0.99
        # This ensures 0 < a < min_loss (with small safety margin)
        a = jax.nn.sigmoid(a_raw) * min_loss * 0.99

        total_loss = 0.0
        total_weight = 0.0

        for i in range(n_curves):
            # Extract parameters for this curve
            log_b = params[1 + 2*i]
            c = params[1 + 2*i + 1]

            # Get data for this curve
            compute_i = jax_data[i]['compute']
            loss_i = jax_data[i]['loss']
            weights_i = jax_data[i]['weights']

            # Compute predictions in log-space
            # log(loss - a) = log(b) + c * log(compute)
            log_compute = jnp.log(compute_i)
            log_loss_shifted = jnp.log(loss_i - a + 1e-8)  # Add epsilon for numerical stability

            pred_log_loss_shifted = log_b + c * log_compute

            # Weighted MSE loss in log space
            residuals = (log_loss_shifted - pred_log_loss_shifted) ** 2
            curve_loss = jnp.sum(weights_i * residuals)
            curve_weight = jnp.sum(weights_i)

            total_loss += curve_loss
            total_weight += curve_weight

        # Normalize by total weight
        return total_loss / total_weight

    # Set up optimizer (Adagrad as in bighead_lr_scaling.py)
    optimizer = optax.adagrad(learning_rate)
    opt_state = optimizer.init(fit_params)

    # JIT compile gradient
    grad_fn = jit(grad(loss_fn))

    # Optimization loop
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
            # Extract current parameters for display
            a_raw = best_params[0]
            min_loss = float(jnp.min(jnp.array([jnp.min(d['loss']) for d in jax_data])))
            a = float(jax.nn.sigmoid(a_raw) * min_loss * 0.99)
            print(f"  Step {step:5d}: loss={best_loss:.6e}, a={a:.4f}")

    # Extract final parameters
    a_raw = best_params[0]
    min_loss = float(jnp.min(jnp.array([jnp.min(d['loss']) for d in jax_data])))
    a = float(jax.nn.sigmoid(a_raw) * min_loss * 0.99)

    print(f"\nFinal shared saturation level: a = {a:.6f}")

    # Extract parameters for each curve and compute R-squared
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

        print(f"\n{name}:")
        print(f"  b = {b:.6e}")
        print(f"  c = {c:.6f}")
        print(f"  R² = {r_squared:.6f}")

    return results

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(data_dict, optimizer_names, fit_metric='compute'):
    """Create comparison plot of BigHead vs DiLoco scaling for multiple optimizers.

    Args:
        data_dict: Dictionary containing BigHead and DiLoco data for each optimizer
        optimizer_names: List of optimizer names
        fit_metric: Metric to use for fitting ('compute' or 'non_emb')
    """
    fig, ax = plt.subplots(figsize=(16, 9))

    # Color schemes for different optimizers (bighead solid, diloco dashed)
    colors = {
        'adamw': 'tab:blue',
        'mk4': 'tab:red',
        'dana': 'tab:green',
        'ademamix': 'tab:purple',
        'd-muon': 'tab:orange'
    }

    markers = {'bighead': 'D', 'diloco': 'o'}
    linestyles = {'bighead': '-', 'diloco': '--'}

    optimizer_title_map = {
        'adamw': 'AdamW',
        'mk4': 'Dana-Star-MK4',
        'dana': 'Dana-Star',
        'ademamix': 'AdemaMix',
        'd-muon': 'D-Muon'
    }

    # Collect all metric values to determine plot range
    all_metric_values = []
    for opt_name in optimizer_names:
        bighead_df = data_dict[opt_name]['bighead']
        diloco_df = data_dict[opt_name]['diloco']
        if len(bighead_df) > 0:
            all_metric_values.extend(bighead_df[fit_metric].values)
        if len(diloco_df) > 0:
            all_metric_values.extend(diloco_df[fit_metric].values)

    if len(all_metric_values) > 0:
        metric_min = np.min(all_metric_values)
        metric_max = np.max(all_metric_values)
        # Set plot range with some padding
        plot_range = np.logspace(np.log10(metric_min * 0.3),
                                 np.log10(metric_max * 3.0), 200)
    else:
        plot_range = None

    # Prepare data for joint fitting
    joint_fit_data = []
    for opt_name in optimizer_names:
        bighead_df = data_dict[opt_name]['bighead']
        diloco_df = data_dict[opt_name]['diloco']
        opt_title = optimizer_title_map.get(opt_name, opt_name)

        if len(bighead_df) > 0:
            joint_fit_data.append({
                'compute': bighead_df[fit_metric].values,
                'loss': bighead_df['val_loss'].values,
                'name': f'{opt_title}_BigHead'
            })

        if len(diloco_df) > 0:
            joint_fit_data.append({
                'compute': diloco_df[fit_metric].values,
                'loss': diloco_df['val_loss'].values,
                'name': f'{opt_title}_DiLoco'
            })

    # Perform joint fitting with shared saturation level
    if len(joint_fit_data) > 0:
        # Get optimization parameters from args (passed through)
        n_steps = getattr(args, 'n_steps', 50000)
        learning_rate = getattr(args, 'learning_rate', 0.1)
        fit_results = fit_all_saturated_power_laws_joint(joint_fit_data, n_steps=n_steps, learning_rate=learning_rate)
    else:
        fit_results = None

    # Define variable name for legend based on metric
    metric_symbol = 'C' if fit_metric == 'compute' else 'P'

    # Plot data and fitted curves for each optimizer
    for opt_name in optimizer_names:
        bighead_df = data_dict[opt_name]['bighead']
        diloco_df = data_dict[opt_name]['diloco']

        color = colors.get(opt_name, 'black')
        opt_title = optimizer_title_map.get(opt_name, opt_name)

        # Plot BigHead data
        if len(bighead_df) > 0:
            ax.scatter(bighead_df[fit_metric], bighead_df['val_loss'],
                      s=150, marker='D', c=color, edgecolors='black', linewidths=2,
                      label=f'{opt_title} BigHead (observed)', zorder=10)

            # Plot fitted curve if available
            if fit_results is not None and plot_range is not None:
                curve_name = f'{opt_title}_BigHead'
                if curve_name in fit_results['curves']:
                    a = fit_results['a']
                    b = fit_results['curves'][curve_name]['b']
                    c = fit_results['curves'][curve_name]['c']
                    r2 = fit_results['curves'][curve_name]['r_squared']

                    loss_fit = saturated_power_law(plot_range, a, b, c)

                    ax.plot(plot_range, loss_fit, '-', color=color, linewidth=3,
                           label=f'{opt_title} BigHead: {a:.3f} + {b:.2e} × ${metric_symbol}^{{{c:.3f}}}$ ($R^2$={r2:.4f})',
                           zorder=9)

        # Plot DiLoco data
        if len(diloco_df) > 0:
            ax.scatter(diloco_df[fit_metric], diloco_df['val_loss'],
                      s=150, marker='o', c=color, edgecolors='black', linewidths=2,
                      alpha=0.7, label=f'{opt_title} DiLoco (observed)', zorder=10)

            # Plot fitted curve if available
            if fit_results is not None and plot_range is not None:
                curve_name = f'{opt_title}_DiLoco'
                if curve_name in fit_results['curves']:
                    a = fit_results['a']
                    b = fit_results['curves'][curve_name]['b']
                    c = fit_results['curves'][curve_name]['c']
                    r2 = fit_results['curves'][curve_name]['r_squared']

                    loss_fit = saturated_power_law(plot_range, a, b, c)

                    ax.plot(plot_range, loss_fit, '--', color=color, linewidth=3,
                           label=f'{opt_title} DiLoco: {a:.3f} + {b:.2e} × ${metric_symbol}^{{{c:.3f}}}$ ($R^2$={r2:.4f})',
                           zorder=9)

    # Formatting
    if fit_metric == 'compute':
        ax.set_xlabel('Compute (PetaFlop-Hours)', fontsize=20)
    else:  # non_emb
        ax.set_xlabel('Non-Embedding Parameters', fontsize=20)
    ax.set_ylabel('Final Validation Loss', fontsize=20)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set y-axis ticks with 0.1 spacing in log space
    # Get current y-axis limits
    if len(all_metric_values) > 0:
        all_loss_values = []
        for opt_name in optimizer_names:
            bighead_df = data_dict[opt_name]['bighead']
            diloco_df = data_dict[opt_name]['diloco']
            if len(bighead_df) > 0:
                all_loss_values.extend(bighead_df['val_loss'].values)
            if len(diloco_df) > 0:
                all_loss_values.extend(diloco_df['val_loss'].values)

        if len(all_loss_values) > 0:
            min_loss = np.min(all_loss_values)
            max_loss = np.max(all_loss_values)

            # Generate ticks from floor(min_loss*10)/10 to ceil(max_loss*10)/10 with 0.1 spacing
            y_min = np.floor(min_loss * 10) / 10
            y_max = np.ceil(max_loss * 10) / 10
            y_ticks = np.arange(y_min, y_max + 0.05, 0.1)  # +0.05 to include upper bound

            ax.set_yticks(y_ticks)
            ax.set_ylim(y_min - 0.05, y_max + 0.05)

    # Title with multiple optimizers
    if len(optimizer_names) == 1:
        opt_title = optimizer_title_map.get(optimizer_names[0], optimizer_names[0])
        title = f'Scaling Law Comparison: BigHead vs DiLoco ({opt_title})'
    else:
        title = 'Scaling Law Comparison: BigHead vs DiLoco (Multiple Optimizers)'

    ax.set_title(title, fontsize=22, fontweight='bold')
    ax.legend(fontsize=12, loc='best', ncol=1)

    # Enable grid with enhanced visibility
    ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.5)

    # Add text box with architecture and fitting info
    info_lines = [
        'BigHead: width ∝ depth²',
        '(n_embd = 16×depth²)',
        '',
        'DiLoco: fixed width per size',
        '(standard scaling)'
    ]

    # Add shared saturation level if available
    if fit_results is not None:
        info_lines.append('')
        info_lines.append(f'Shared saturation: a = {fit_results["a"]:.4f}')

    info_text = '\n'.join(info_lines)
    ax.text(0.98, 0.98, info_text,
            transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("BigHead vs DiLoco Scaling Comparison")
    print(f"Optimizers: {', '.join(args.optimizers)}")
    if args.min_compute is not None:
        print(f"Minimum compute filter: {args.min_compute:.4e} PFH")
    print("="*70)

    # Load data for all optimizers
    data_dict = {}

    for opt_abbrev, opt_type in zip(args.optimizers, optimizer_types):
        print(f"\n{'='*70}")
        print(f"Loading data for: {opt_abbrev} ({opt_type})")
        print(f"{'='*70}")

        # Load BigHead data
        bighead_df = load_bighead_data(args.project, args.bighead_group, args.entity, opt_type,
                                       min_compute=args.min_compute)

        if len(bighead_df) > 0:
            print(f"\n{opt_abbrev} BigHead results:")
            print(bighead_df[['depth', 'val_loss', 'compute']].to_string(index=False))
        else:
            print(f"\nNo BigHead data found for {opt_abbrev}!")

        # Load DiLoco data
        diloco_df = load_diloco_data(args.project, args.entity, opt_type,
                                     min_compute=args.min_compute)

        if len(diloco_df) > 0:
            print(f"\n{opt_abbrev} DiLoco results:")
            print(diloco_df[['model', 'n_layer', 'val_loss', 'compute']].to_string(index=False))
        else:
            print(f"\nNo DiLoco data found for {opt_abbrev}!")

        # Store in dictionary
        data_dict[opt_abbrev] = {
            'bighead': bighead_df,
            'diloco': diloco_df
        }

    # Check if we have any data
    has_data = any(len(data_dict[opt]['bighead']) > 0 or len(data_dict[opt]['diloco']) > 0
                   for opt in args.optimizers)

    if not has_data:
        print("\nNo data available for any optimizer. Exiting.")
        exit(1)

    # Create comparison plot
    print(f"\n{'='*70}")
    print("Generating comparison plot...")
    print(f"Fit metric: {args.fit_metric}")
    print(f"{'='*70}")

    fig = plot_comparison(data_dict, args.optimizers, fit_metric=args.fit_metric)

    # Save plot
    if args.output:
        output_file = args.output
    else:
        if len(args.optimizers) == 1:
            optimizer_filename_map = {'adamw': 'AdamW', 'mk4': 'DanaStar-MK4',
                                     'dana': 'DanaStar', 'ademamix': 'AdemaMix', 'd-muon': 'D-Muon'}
            optimizer_name = optimizer_filename_map[args.optimizers[0]]
            output_file = f'BigHead_vs_DiLoco_{optimizer_name}_scaling.pdf'
        else:
            opt_names = '_'.join([opt.upper() for opt in args.optimizers])
            output_file = f'BigHead_vs_DiLoco_{opt_names}_scaling.pdf'

    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    print(f"\nPlot saved to: {os.path.abspath(output_file)}")
