#!/usr/bin/env python3
"""
Tokens Saved Analysis - Compare optimizers by computing how many tokens are saved
relative to AdamW baseline.

For each optimizer and model size:
1. Get the final-val/loss for AdamW at that size
2. Look at the optimizer's loss curve history (val/loss vs tokens)
3. Find the first 'tokens' value where the optimizer's loss drops below AdamW's final loss
4. Compute tokens_saved = final_tokens - crossing_tokens

This measures how much earlier an optimizer reaches AdamW's final performance.

Usage:
    python tokens_saved.py --scaling-rules Enoki_Scaled --optimizers mk4 d-muon
    python tokens_saved.py --scaling-rules Enoki_Scaled --optimizers mk4 ademamix --fit-metric non_emb
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
    }
}

# Matplotlib formatting
style.use('default')
rc('font', family='sans-serif')
rcParams['font.weight'] = 'light'
rcParams['font.size'] = 18
rcParams['figure.figsize'] = (14, 8)

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description='Tokens Saved Analysis - Compare optimizers to AdamW baseline')
parser.add_argument('--scaling-rules', type=str, nargs='+', required=True,
                    choices=['BigHead', 'EggHead', 'Enoki', 'Enoki_Scaled', 'Eryngii', 'Eryngii_Scaled'],
                    help='Scaling rules to compare (can specify multiple)')
parser.add_argument('--optimizers', type=str, nargs='+', required=True,
                    choices=['mk4', 'dana', 'ademamix', 'd-muon', 'manau', 'manau-hard', 'adamw-decaying-wd', 'dana-mk4', 'ademamix-decaying-wd', 'dana-star-no-tau', 'dana-star'],
                    help='Optimizer types to analyze (AdamW is always the baseline)')
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
parser.add_argument('--equal-weight', action='store_true',
                    help='Use equal weights for all data points instead of weighting by compute (default: False)')
args = parser.parse_args()

# Map optimizer names
optimizer_map = {
    'mk4': 'dana-star-mk4', 'dana': 'dana', 'ademamix': 'ademamix',
    'd-muon': 'd-muon', 'manau': 'manau', 'manau-hard': 'manau-hard',
    'adamw-decaying-wd': 'adamw-decaying-wd', 'dana-mk4': 'dana-mk4',
    'ademamix-decaying-wd': 'ademamix-decaying-wd', 'dana-star-no-tau': 'dana-star-no-tau',
    'dana-star': 'dana-star'
}
optimizer_types = [optimizer_map[opt] for opt in args.optimizers]

# =============================================================================
# PARAMETER CALCULATION FUNCTIONS
# =============================================================================

def compute_params(size, scaling_rule):
    """
    Compute parameters for a given size and scaling rule.
    """
    if scaling_rule == 'BigHead':
        depth = size
        head_dim = 16 * depth
        n_embd = 16 * depth * depth
        mlp_hidden = 32 * depth * depth
        n_head = depth
        n_layer = depth
        non_emb = float(depth * (3 * head_dim * n_embd * n_head + n_embd * n_embd +
                                 2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd)
        vocab_size = 50304
        total_params = float(non_emb + 2 * n_embd * vocab_size)

    elif scaling_rule == 'EggHead':
        heads = size
        head_dim = 16 * heads
        n_embd = 16 * heads * heads
        mlp_hidden = 32 * heads * heads
        n_head = heads
        n_layer = int(heads * (heads - 1) / 2)
        non_emb = float(n_layer * (3 * head_dim * n_embd * n_head + n_embd * n_embd +
                                    2 * n_embd * mlp_hidden + 8 * n_embd) + 2 * n_embd)
        vocab_size = 50304
        total_params = float(non_emb + 2 * n_embd * vocab_size)

    elif scaling_rule in ('Enoki', 'Enoki_Scaled'):
        heads = size
        head_dim = 64
        n_embd = heads * 64
        mlp_hidden = 4 * n_embd
        n_head = heads
        n_layer = int(3 * heads // 4)
        non_emb = float(12 * n_embd * n_embd * n_layer)
        vocab_size = 50304
        total_params = float(non_emb + 2 * n_embd * vocab_size)

    elif scaling_rule in ('Eryngii', 'Eryngii_Scaled'):
        heads = size
        head_dim = int(round(32 * heads / 3 / 8) * 8)
        n_head = heads
        n_layer = int(heads * heads // 8)
        n_embd = n_head * head_dim
        mlp_hidden = 4 * n_embd
        non_emb = float(12 * n_embd * n_embd * n_layer)
        vocab_size = 50304
        total_params = float(non_emb + 2 * n_embd * vocab_size)

    else:
        raise ValueError(f"Unknown scaling rule: {scaling_rule}")

    compute_flops = 6.0 * non_emb * total_params * 20.0
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

def extract_config_value(config_dict):
    """Extract values from nested config structure."""
    result = {}
    for key, val in config_dict.items():
        if isinstance(val, dict) and 'value' in val:
            result[key] = val['value']
        else:
            result[key] = val
    return result


def load_adamw_final_losses(scaling_rule, project, entity, min_compute=None):
    """Load AdamW final-val/loss for each size.

    Returns:
        dict: {size: {'final_loss': loss, 'final_tokens': tokens, 'compute': compute, 'non_emb': non_emb}}
    """
    api = wandb.Api()
    config = SCALING_RULE_CONFIG[scaling_rule]
    group = config['group']

    print(f"Loading AdamW baseline data from {group}...")
    runs = api.runs(f"{entity}/{project}", filters={"group": group})

    data = {}
    for run in runs:
        run_config = run.config
        if isinstance(run_config, str):
            try:
                run_config = json.loads(run_config)
            except (json.JSONDecodeError, TypeError):
                continue
        if hasattr(run_config, 'as_dict'):
            run_config = run_config.as_dict()
        elif not isinstance(run_config, dict):
            try:
                run_config = dict(run_config)
            except (TypeError, ValueError):
                continue
        run_config = extract_config_value(run_config)

        summary = run.summary
        if hasattr(summary, '_json_dict') and isinstance(summary._json_dict, str):
            try:
                summary = json.loads(summary._json_dict)
            except (json.JSONDecodeError, TypeError):
                continue

        # Filter for AdamW only
        opt = run_config.get('opt', '')
        if opt != 'adamw':
            continue

        # Check completion
        actual_iter = summary.get('iter', 0)
        iterations_config = run_config.get('iterations', 0)
        if actual_iter > 0 and iterations_config > 0 and actual_iter < iterations_config:
            continue

        # Get size parameter
        if scaling_rule == 'BigHead':
            size = run_config.get('n_layer')
        else:
            size = run_config.get('n_head')

        final_loss = summary.get('final-val/loss')
        final_tokens = summary.get('tokens')

        if size is None or final_loss is None:
            continue

        params = compute_params(size, scaling_rule)

        # Apply min compute filter
        if min_compute is not None and params['compute'] < min_compute:
            continue

        # Keep the best (lowest loss) run for each size
        if size not in data or final_loss < data[size]['final_loss']:
            data[size] = {
                'final_loss': final_loss,
                'final_tokens': final_tokens,
                'compute': params['compute'],
                'non_emb': params['non_emb'],
                'run_name': run.name
            }

    print(f"  Loaded AdamW baselines for {len(data)} sizes")
    return data


def load_optimizer_loss_history(scaling_rule, project, entity, optimizer_type, min_compute=None):
    """Load loss curve history for an optimizer.

    Returns:
        dict: {size: [{'run_name': str, 'history': [(tokens, val_loss), ...], 'final_tokens': int, ...}]}
    """
    api = wandb.Api()
    config = SCALING_RULE_CONFIG[scaling_rule]
    group = config['group']

    print(f"Loading {optimizer_type} history from {group}...")
    runs = api.runs(f"{entity}/{project}", filters={"group": group})

    data = {}
    for run in runs:
        run_config = run.config
        if isinstance(run_config, str):
            try:
                run_config = json.loads(run_config)
            except (json.JSONDecodeError, TypeError):
                continue
        if hasattr(run_config, 'as_dict'):
            run_config = run_config.as_dict()
        elif not isinstance(run_config, dict):
            try:
                run_config = dict(run_config)
            except (TypeError, ValueError):
                continue
        run_config = extract_config_value(run_config)

        summary = run.summary
        if hasattr(summary, '_json_dict') and isinstance(summary._json_dict, str):
            try:
                summary = json.loads(summary._json_dict)
            except (json.JSONDecodeError, TypeError):
                continue

        # Filter by optimizer
        opt = run_config.get('opt', '')
        if opt != optimizer_type:
            continue

        # Check completion
        actual_iter = summary.get('iter', 0)
        iterations_config = run_config.get('iterations', 0)
        if actual_iter > 0 and iterations_config > 0 and actual_iter < iterations_config:
            continue

        # Get size parameter
        if scaling_rule == 'BigHead':
            size = run_config.get('n_layer')
        else:
            size = run_config.get('n_head')

        if size is None:
            continue

        params = compute_params(size, scaling_rule)

        # Apply min compute filter
        if min_compute is not None and params['compute'] < min_compute:
            continue

        # Fetch loss curve history
        try:
            history = run.history(keys=['tokens', 'val/loss'], pandas=False)
            loss_curve = []
            for row in history:
                tokens = row.get('tokens')
                val_loss = row.get('val/loss')
                if tokens is not None and val_loss is not None:
                    loss_curve.append((tokens, val_loss))

            if not loss_curve:
                continue

            # Sort by tokens
            loss_curve.sort(key=lambda x: x[0])

            final_tokens = summary.get('tokens')
            final_loss = summary.get('final-val/loss')

            if size not in data:
                data[size] = []

            data[size].append({
                'run_name': run.name,
                'history': loss_curve,
                'final_tokens': final_tokens,
                'final_loss': final_loss,
                'compute': params['compute'],
                'non_emb': params['non_emb']
            })

        except Exception as e:
            print(f"  Warning: Could not fetch history for {run.name}: {e}")
            continue

    total_runs = sum(len(runs_list) for runs_list in data.values())
    print(f"  Loaded {total_runs} runs across {len(data)} sizes")
    return data


def compute_tokens_saved(adamw_baselines, optimizer_history):
    """
    Compute tokens saved for each size.

    For each size:
    1. Get AdamW's final loss
    2. Find the earliest point in the optimizer's loss curve where loss <= adamw_final_loss
    3. tokens_saved = final_tokens - crossing_tokens

    Returns:
        list of dicts with size, tokens_saved, compute, non_emb, etc.
    """
    results = []

    for size, adamw_data in adamw_baselines.items():
        if size not in optimizer_history:
            continue

        adamw_final_loss = adamw_data['final_loss']
        adamw_final_tokens = adamw_data['final_tokens']

        # Find the best run for this size (lowest final loss that beats AdamW)
        best_crossing_tokens = None
        best_final_tokens = None
        best_run_name = None

        for run_data in optimizer_history[size]:
            history = run_data['history']
            final_tokens = run_data['final_tokens']

            # Find first crossing point
            crossing_tokens = None
            for tokens, val_loss in history:
                if val_loss <= adamw_final_loss:
                    crossing_tokens = tokens
                    break

            if crossing_tokens is not None:
                # This run beats AdamW - check if it's the best
                if best_crossing_tokens is None or crossing_tokens < best_crossing_tokens:
                    best_crossing_tokens = crossing_tokens
                    best_final_tokens = final_tokens
                    best_run_name = run_data['run_name']

        if best_crossing_tokens is not None and best_final_tokens is not None:
            tokens_saved = best_final_tokens - best_crossing_tokens
            results.append({
                'size': size,
                'tokens_saved': tokens_saved,
                'crossing_tokens': best_crossing_tokens,
                'final_tokens': best_final_tokens,
                'adamw_final_loss': adamw_final_loss,
                'compute': adamw_data['compute'],
                'non_emb': adamw_data['non_emb'],
                'run_name': best_run_name
            })

    return results

# =============================================================================
# JOINT FITTING FUNCTIONS (Power Law: tokens_saved = b * X^c)
# =============================================================================

def fit_power_laws_joint(datasets, n_steps=50000, lr=0.1):
    """
    Fit power laws to multiple datasets: tokens_saved = b * X^c

    Note: Unlike loss fitting, we don't need a saturation level here.
    Each curve has its own b and c parameters.

    Args:
        datasets: List of dicts with 'x', 'y', 'weights', 'name'
        n_steps: Number of optimization steps
        lr: Learning rate

    Returns:
        Dict with 'curves' mapping name -> {b, c, r_squared}
    """
    n_curves = len(datasets)

    # Prepare data
    x_data = [jnp.array(d['x'], dtype=jnp.float32) for d in datasets]
    y_data = [jnp.array(d['y'], dtype=jnp.float32) for d in datasets]
    weights_data = [jnp.array(d['weights'], dtype=jnp.float32) for d in datasets]

    # Initialize parameters: log(b), c for each curve
    init_params = []
    for i in range(n_curves):
        init_params.extend([jnp.log(1e10), 1.0])  # [log(b_i), c_i]

    fit_params = jnp.array(init_params, dtype=jnp.float32)

    @jit
    def loss_fn(params):
        """Weighted MSE loss in log space"""
        total_loss = 0.0
        total_weight = 0.0

        for i in range(n_curves):
            log_b = params[2*i]
            c = params[2*i + 1]

            # Power law: y = b * x^c
            # Log space: log(y) = log(b) + c * log(x)
            log_x = jnp.log(x_data[i])
            log_y_true = jnp.log(y_data[i] + 1)  # +1 to avoid log(0)
            log_y_pred = log_b + c * log_x

            residuals = (log_y_pred - log_y_true) ** 2
            curve_loss = jnp.sum(weights_data[i] * residuals)
            curve_weight = jnp.sum(weights_data[i])

            total_loss += curve_loss
            total_weight += curve_weight

        return total_loss / total_weight

    # Optimize
    optimizer = optax.adagrad(lr)
    opt_state = optimizer.init(fit_params)
    grad_fn = jit(grad(loss_fn))

    best_loss = float('inf')
    best_params = fit_params

    print(f"\nFitting {n_curves} power law curves...")
    for step in range(n_steps):
        grads = grad_fn(fit_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        fit_params = optax.apply_updates(fit_params, updates)

        current_loss = float(loss_fn(fit_params))
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = fit_params

        if step % 10000 == 0 or step == n_steps - 1:
            print(f"  Step {step:5d}: loss={best_loss:.6e}")

    # Extract results
    results = {'curves': {}}

    for i, dataset in enumerate(datasets):
        log_b = float(best_params[2*i])
        b = float(jnp.exp(log_b))
        c = float(best_params[2*i + 1])
        name = dataset['name']

        # Compute R-squared
        x_vals = np.array(x_data[i])
        y_vals = np.array(y_data[i])
        predictions = b * np.power(x_vals, c)
        residuals = y_vals - predictions
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_vals - np.mean(y_vals))**2)
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

def plot_tokens_saved(data_dict, fit_results, scaling_rules, optimizer_shorts, fit_metric):
    """
    Plot tokens saved vs compute (or non_emb) for each optimizer.
    """
    fig, ax = plt.subplots(figsize=(18, 10))

    # Color scheme for optimizers
    opt_colors = {
        'mk4': 'tab:red',
        'dana': 'tab:green',
        'ademamix': 'tab:purple',
        'd-muon': 'tab:orange',
        'manau': 'tab:brown',
        'manau-hard': 'tab:pink',
        'adamw-decaying-wd': 'tab:gray',
        'dana-mk4': 'tab:olive',
        'ademamix-decaying-wd': 'tab:cyan',
        'dana-star-no-tau': 'darkblue',
        'dana-star': 'darkgreen'
    }

    # Scaling rule markers
    rule_markers = {
        'BigHead': 'D',
        'EggHead': 's',
        'Enoki': 'o',
        'Enoki_Scaled': 'o',
        'Eryngii': '^',
        'Eryngii_Scaled': '^'
    }

    # Scaling rule line styles
    rule_linestyles = {
        'BigHead': '-',
        'EggHead': '--',
        'Enoki': ':',
        'Enoki_Scaled': '-',
        'Eryngii': '-.',
        'Eryngii_Scaled': '-'
    }

    # Collect all metric values for plot range
    all_metric_vals = []
    for opt_short in optimizer_shorts:
        for rule in scaling_rules:
            key = f'{opt_short}_{rule}'
            if key in data_dict and len(data_dict[key]) > 0:
                all_metric_vals.extend([d[fit_metric] for d in data_dict[key]])

    if len(all_metric_vals) > 0:
        metric_min = np.min(all_metric_vals)
        metric_max = np.max(all_metric_vals)
        plot_range = np.logspace(np.log10(metric_min * 0.5), np.log10(metric_max * 2.0), 200)
    else:
        plot_range = None

    fit_handles = {}
    obs_handles = {}

    metric_symbol = 'C' if fit_metric == 'compute' else 'P'

    # Plot each optimizer x scaling_rule combination
    for opt_short in optimizer_shorts:
        color = opt_colors.get(opt_short, 'black')

        for rule in scaling_rules:
            key = f'{opt_short}_{rule}'

            if key not in data_dict or len(data_dict[key]) == 0:
                continue

            data_list = data_dict[key]
            metric_vals = np.array([d[fit_metric] for d in data_list])
            tokens_saved_vals = np.array([d['tokens_saved'] for d in data_list])

            marker = rule_markers.get(rule, 'x')
            linestyle = rule_linestyles.get(rule, '-')

            # Plot observed data
            scatter = ax.scatter(metric_vals, tokens_saved_vals,
                      s=120, marker=marker, c=color, edgecolors='black', linewidths=1.5,
                      zorder=10, alpha=0.8)

            obs_handles[(rule, opt_short)] = (scatter, f'{opt_short} {rule} (observed)')

            # Plot fitted curve if available
            if key in fit_results['curves'] and plot_range is not None:
                b = fit_results['curves'][key]['b']
                c = fit_results['curves'][key]['c']
                r2 = fit_results['curves'][key]['r_squared']

                # Power law: tokens_saved = b * X^c
                tokens_fit = b * np.power(plot_range, c)

                line, = ax.plot(plot_range, tokens_fit, linestyle=linestyle, color=color, linewidth=2.5,
                       zorder=9)

                fit_handles[(rule, opt_short)] = (line, f'{opt_short} {rule}: {b:.2e} × {metric_symbol}$^{{{c:.4f}}}$ ($R^2$={r2:.3f})')

    # Create legend
    legend_handles = []
    legend_labels = []

    for rule in scaling_rules:
        for opt_short in optimizer_shorts:
            fit_key = (rule, opt_short)
            if fit_key in fit_handles:
                handle, label = fit_handles[fit_key]
                legend_handles.append(handle)
                legend_labels.append(label)

        for opt_short in optimizer_shorts:
            obs_key = (rule, opt_short)
            if obs_key in obs_handles:
                handle, label = obs_handles[obs_key]
                legend_handles.append(handle)
                legend_labels.append(label)

    # Labels
    if fit_metric == 'compute':
        xlabel = 'Compute (PetaFlop-Hours)'
    else:
        xlabel = 'Non-embedding Parameters'

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel('Tokens Saved (vs AdamW)', fontsize=20)
    ax.set_xscale('log')
    ax.set_yscale('log')

    opts_str = ', '.join(optimizer_shorts)
    rules_str = ' vs '.join(scaling_rules)
    ax.set_title(f'Tokens Saved Analysis: {rules_str}\nOptimizers: {opts_str} (relative to AdamW)',
                fontsize=18, fontweight='bold')

    ax.legend(legend_handles, legend_labels, fontsize=11, loc='best', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print(f"Tokens Saved Analysis")
    print(f"Scaling Rules: {', '.join(args.scaling_rules)}")
    print(f"Optimizers: {', '.join(args.optimizers)} (vs AdamW baseline)")
    print(f"Fit Metric: {args.fit_metric}")
    if args.min_compute:
        print(f"Min Compute: {args.min_compute:.4e} PFH")
    print("="*70)

    # Load AdamW baselines for each scaling rule
    adamw_baselines = {}  # {scaling_rule: {size: data}}
    for scaling_rule in args.scaling_rules:
        adamw_baselines[scaling_rule] = load_adamw_final_losses(
            scaling_rule=scaling_rule,
            project=args.project,
            entity=args.entity,
            min_compute=args.min_compute
        )

    # Load optimizer histories and compute tokens saved
    data_dict = {}  # {opt_short_rule: [results]}

    for opt_idx, optimizer_type in enumerate(optimizer_types):
        opt_short = args.optimizers[opt_idx]
        print(f"\nProcessing {opt_short} ({optimizer_type})...")

        for scaling_rule in args.scaling_rules:
            # Load history
            opt_history = load_optimizer_loss_history(
                scaling_rule=scaling_rule,
                project=args.project,
                entity=args.entity,
                optimizer_type=optimizer_type,
                min_compute=args.min_compute
            )

            # Compute tokens saved
            results = compute_tokens_saved(adamw_baselines[scaling_rule], opt_history)

            key = f'{opt_short}_{scaling_rule}'
            data_dict[key] = results

            print(f"  {scaling_rule}: {len(results)} data points with tokens_saved")
            for r in results:
                print(f"    Size {r['size']}: tokens_saved={r['tokens_saved']:,.0f}, crossing_tokens={r['crossing_tokens']:,.0f}")

    # Prepare data for fitting
    fit_data = []
    for key, results in data_dict.items():
        if len(results) > 0:
            x_vals = [r[args.fit_metric] for r in results]
            y_vals = [r['tokens_saved'] for r in results]

            if args.equal_weight:
                weights = [1.0] * len(x_vals)
            else:
                weights = x_vals  # Weight by compute

            fit_data.append({
                'x': x_vals,
                'y': y_vals,
                'weights': weights,
                'name': key
            })

    if len(fit_data) == 0:
        print("\nNo data found. Exiting.")
        exit(1)

    # Fit power laws
    print(f"\n{'='*70}")
    print(f"Fitting {len(fit_data)} Power Law Curves")
    print(f"{'='*70}")

    fit_results = fit_power_laws_joint(
        fit_data,
        n_steps=args.n_steps,
        lr=args.learning_rate
    )

    # Print results
    print(f"\n{'='*70}")
    print("Fit Results: tokens_saved = b × X^c")
    print(f"{'='*70}")

    for curve_name, curve_params in fit_results['curves'].items():
        print(f"\n{curve_name}:")
        print(f"  b = {curve_params['b']:.6e}")
        print(f"  c = {curve_params['c']:.6f}")
        print(f"  R² = {curve_params['r_squared']:.6f}")

    # Create plot
    fig = plot_tokens_saved(
        data_dict,
        fit_results,
        args.scaling_rules,
        args.optimizers,
        args.fit_metric
    )

    # Save plot
    if args.output:
        output_file = args.output
    else:
        rules_str = '_'.join(args.scaling_rules)
        opts_str = '_'.join(args.optimizers)
        output_file = f'TokensSaved_{rules_str}_{opts_str}.pdf'

    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    print(f"\n{'='*70}")
    print(f"Plot saved to: {os.path.abspath(output_file)}")
    print(f"{'='*70}")
