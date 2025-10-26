#!/usr/bin/env python3
"""
Compare Scaling Rules Performance (BigHead vs EggHead vs Enoki)

This script compares the scaling performance between different model architectures:
1. BigHead: depth-based scaling (n_layer = depth)
2. EggHead: quadratic depth scaling (n_layer = heads * (heads-1) / 2)
3. Enoki: DiLoco scaling (n_layer = 3 * heads / 4)

For each architecture and model size, it takes the best final-val/loss achieved,
plots loss vs compute (or non-emb params), and fits saturated power laws: loss = a + b * X^c

Joint Fitting Approach:
- All curves are fit simultaneously with a SHARED saturation level 'a'
- Uses JAX + Adagrad optimization
- Log-space fitting for numerical stability
- Weighted MSE loss (larger models get more weight)
- Constraint: 0 < a < min(observed losses) via sigmoid transformation

Usage:
    python compare_scaling_rules.py --scaling-rules BigHead Enoki --optimizer adamw
    python compare_scaling_rules.py --scaling-rules BigHead EggHead Enoki --optimizer mk4
    python compare_scaling_rules.py --scaling-rules BigHead Enoki --optimizer adamw --fit-metric non_emb
"""

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style, rc, rcParams
import argparse
import warnings
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

parser = argparse.ArgumentParser(description='Compare scaling rules performance')
parser.add_argument('--scaling-rules', type=str, nargs='+', required=True,
                    choices=['BigHead', 'EggHead', 'Enoki'],
                    help='Scaling rules to compare (can specify multiple)')
parser.add_argument('--optimizer', type=str, required=True,
                    choices=['adamw', 'mk4', 'dana', 'ademamix', 'd-muon'],
                    help='Optimizer type to analyze')
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
args = parser.parse_args()

# Map optimizer name
optimizer_map = {'adamw': 'adamw', 'mk4': 'dana-star-mk4', 'dana': 'dana', 'ademamix': 'ademamix', 'd-muon': 'd-muon'}
optimizer_type = optimizer_map[args.optimizer]

# =============================================================================
# PARAMETER CALCULATION FUNCTIONS
# =============================================================================

def compute_params(size, scaling_rule):
    """
    Compute parameters for a given size and scaling rule.

    Args:
        size: For BigHead, this is depth. For EggHead/Enoki, this is heads.
        scaling_rule: One of 'BigHead', 'EggHead', 'Enoki'

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

    elif scaling_rule == 'Enoki':
        # Enoki: DiLoco scaling
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
        run_config = run.config
        summary = run.summary

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
        else:  # EggHead or Enoki
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

def joint_fit_saturated_power_laws(datasets, n_steps=50000, lr=0.1):
    """
    Fit saturated power laws to multiple datasets with a SHARED saturation level 'a'.

    Args:
        datasets: List of dicts, each with 'x' and 'y' arrays
        n_steps: Number of optimization steps
        lr: Learning rate for Adagrad

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
        a = min_loss * jax.nn.sigmoid(params['a_raw'])
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
    a_fit = float(min_loss * jax.nn.sigmoid(params['a_raw']))
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

# =============================================================================
# PLOTTING
# =============================================================================

def plot_comparison(all_data, fit_results, fit_metric):
    """
    Plot comparison of scaling rules.

    Args:
        all_data: Dict mapping scaling_rule -> DataFrame
        fit_results: Dict mapping scaling_rule -> fit parameters
        fit_metric: 'compute' or 'non_emb'
    """
    fig, ax = plt.subplots(figsize=(16, 9))

    # Get metric info
    if fit_metric == 'compute':
        metric_col = 'compute'
        xlabel = 'Compute (PetaFlop-Hours)'
    else:  # non_emb
        metric_col = 'non_emb'
        xlabel = 'Non-embedding Parameters'

    # Plot each scaling rule
    for scaling_rule in all_data.keys():
        df = all_data[scaling_rule]
        if len(df) == 0:
            continue

        config = SCALING_RULE_CONFIG[scaling_rule]
        fit_params = fit_results[scaling_rule]

        # Plot data points
        x = df[metric_col].values
        y = df['val_loss'].values

        ax.scatter(x, y, s=100, c=config['color'], marker=config['marker'],
                  label=f'{scaling_rule} (data)', alpha=0.7, edgecolors='black', linewidth=1.5)

        # Plot fit line
        x_range = np.logspace(np.log10(min(x) * 0.5), np.log10(max(x) * 2), 200)
        y_fit = saturated_power_law(x_range, fit_params['a'], fit_params['b'], fit_params['c'])

        ax.plot(x_range, y_fit, linestyle=config['linestyle'], color=config['color'],
               linewidth=2.5, alpha=0.8,
               label=f'{scaling_rule} fit: {fit_params["a"]:.3f} + {fit_params["b"]:.3e} × X^{{{fit_params["c"]:.3f}}}')

    # Formatting
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel('Validation Loss', fontsize=20)
    ax.set_xscale('log')
    ax.set_yscale('log')

    optimizer_title_map = {'adamw': 'AdamW', 'dana-star-mk4': 'Dana-Star-MK4', 'dana': 'Dana-Star',
                          'ademamix': 'AdemaMix', 'd-muon': 'D-Muon'}
    optimizer_title = optimizer_title_map.get(optimizer_type, optimizer_type)

    scaling_rules_str = ' vs '.join(all_data.keys())
    ax.set_title(f'Scaling Laws Comparison: {scaling_rules_str}\n{optimizer_title} Optimizer',
                fontsize=20, fontweight='bold')

    ax.legend(fontsize=14, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print(f"Scaling Rules Comparison")
    print(f"Scaling Rules: {', '.join(args.scaling_rules)}")
    print(f"Optimizer: {args.optimizer} ({optimizer_type})")
    print(f"Fit Metric: {args.fit_metric}")
    if args.min_compute:
        print(f"Min Compute: {args.min_compute:.4e} PFH")
    print("="*70)

    # Load data for each scaling rule
    all_data = {}
    for scaling_rule in args.scaling_rules:
        df = load_scaling_rule_data(
            scaling_rule=scaling_rule,
            project=args.project,
            entity=args.entity,
            optimizer_type=optimizer_type,
            min_compute=args.min_compute
        )
        all_data[scaling_rule] = df

    # Check if we have data
    if all(len(df) == 0 for df in all_data.values()):
        print("\nNo data found for any scaling rule. Exiting.")
        exit(1)

    # Prepare datasets for joint fitting
    datasets = []
    scaling_rule_names = []

    for scaling_rule, df in all_data.items():
        if len(df) == 0:
            continue

        x = df[args.fit_metric].values
        y = df['val_loss'].values

        # Weight by x value (larger models get more weight)
        weights = x / np.sum(x) * len(x)

        datasets.append({
            'x': x,
            'y': y,
            'weights': weights
        })
        scaling_rule_names.append(scaling_rule)

    # Joint fit
    fit_params_list = joint_fit_saturated_power_laws(
        datasets,
        n_steps=args.n_steps,
        lr=args.learning_rate
    )

    # Map results back to scaling rules
    fit_results = {name: params for name, params in zip(scaling_rule_names, fit_params_list)}

    # Print results
    print(f"\n{'='*70}")
    print("Fit Results")
    print(f"{'='*70}")

    shared_a = fit_params_list[0]['a'] if fit_params_list else None
    if shared_a:
        print(f"\nShared saturation level a = {shared_a:.6f}")

    for scaling_rule, params in fit_results.items():
        print(f"\n{scaling_rule}:")
        print(f"  y = {params['a']:.6f} + {params['b']:.6e} × X^{params['c']:.6f}")

    # Plot
    fig = plot_comparison(all_data, fit_results, args.fit_metric)

    # Save plot
    if args.output:
        output_file = args.output
    else:
        rules_str = '_'.join(args.scaling_rules)
        output_file = f'ScalingComparison_{rules_str}_{args.optimizer}.pdf'

    import os
    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    print(f"\nPlot saved to: {os.path.abspath(output_file)}")
