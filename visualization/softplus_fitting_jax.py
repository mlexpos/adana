#!/usr/bin/env python3
"""
DanaStar Softplus Function Fitting with JAX

OVERVIEW:
========
This script fits a softplus-like function to validation loss data from DanaStar experiments
to model the relationship between learning rate, weight decay, and final validation loss.

The goal is to create a consistent fit that captures the optimal learning rate for different
weight decay values (parameterized by omega = weight_decay_ts * learning_rate) and to model
the validation loss landscape around the minimum.

FUNCTIONAL FORM:
===============
For each omega value, we fit: f(log_lr) = a + log(exp(b*(log_lr-c)) + exp(-d*(log_lr-c)))

Where:
- a: vertical offset (baseline loss level)
- b: left slope parameter (steepness for lr < optimal)
- c: center location (optimal log learning rate)
- d: right slope parameter (steepness for lr > optimal)

The minimum occurs at: log_lr_min = c + log(d/b) / (b+d)

WEIGHTING STRATEGY:
==================
Points closer to the theoretical minimum for each omega get higher weights in the loss function.
This focuses the fit on the critical region around the optimal learning rate, reducing the
influence of potentially noisy points far from the optimum.

REGULARIZATION:
==============
Parameters are regularized to vary smoothly across omega values using quartic weighting:
reg_loss = λ * Σ(||θ_i - θ_j||² / |log_ω_i - log_ω_j|⁴)

This encourages similar parameter values for nearby omega values while allowing larger
differences for distant omega values.

USAGE:
======
python softplus_fitting_jax.py --model 35M    # Fit 35M model data
python softplus_fitting_jax.py --model 90M    # Fit 90M model data

OUTPUT:
=======
- PDF plot showing data points, fitted curves, and optimal points
- Console output with fitted parameters and minimum locations for each omega
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')
import argparse

import jax
import jax.numpy as jnp
from jax import grad, jit
import optax

# =============================================================================
# CONFIGURATION AND HYPERPARAMETERS
# =============================================================================

# Initial parameter values for softplus function fitting
A_INIT = None  # Will be set to mean validation loss (computed from data)
B_INIT = 1.0   # Initial left slope parameter
C_INIT = -7.0  # Initial center location (log learning rate)
D_INIT = 1.0   # Initial right slope parameter

# Optimization hyperparameters
LAMBDA_REG = 1e-3      # Regularization strength for parameter smoothness
LEARNING_RATE = 1e-1   # Adagrad learning rate
N_STEPS = 20000        # Number of optimization steps

# Data filtering
MAX_VAL_LOSS = 4.0     # Filter out validation losses above this threshold

# Parse command line arguments
parser = argparse.ArgumentParser(description='JAX softplus fitting for DanaStar experiment data')
parser.add_argument('--model', type=str, default='35M',
                    choices=['35M', '90M'],
                    help='Model size to fit (default: 35M)')
args = parser.parse_args()

# Configuration mapping for different model sizes
model_size = args.model
configs = {
    '35M': {
        'group': 'DanaStar_35M_LR_WD_Sweep',      # WandB experiment group name
        'title': 'DanaStar 35M: JAX Softplus-like Fits',  # Plot title
        'filename': '35m_lrwd_sweep_jax.pdf'      # Output filename
    },
    '90M': {
        'group': 'DanaStar_90M_LR_WD_Sweep',      # WandB experiment group name
        'title': 'DanaStar 90M: JAX Softplus-like Fits',  # Plot title
        'filename': '90m_lrwd_sweep_jax.pdf'      # Output filename
    }
}

config = configs[model_size]
print(f"Using configuration for {model_size}: {config}")

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_wandb_data(project_name, group_name):
    """
    Load experiment data from Weights & Biases.

    Downloads runs from the specified project and group, extracting:
    - lr: learning rate
    - wd_ts: weight decay (timescale)
    - dataset: dataset name (fineweb vs fineweb_100)
    - val_loss: final validation loss

    Computes omega = wd_ts * lr for parameterization.

    Args:
        project_name (str): WandB project name
        group_name (str): WandB experiment group name

    Returns:
        pd.DataFrame: Processed experiment data
    """
    # Initialize WandB API
    api = wandb.Api()

    print(f"Downloading data from project: {project_name}, group: {group_name}")

    # Get all runs in the specified group
    runs = api.runs(f"ep-rmt-ml-opt/{project_name}", filters={"group": group_name})

    data = []
    for run in runs:
        print(f"Processing run: {run.name}")

        # Extract configuration and summary from run
        config = run.config
        summary = run.summary

        # Get required fields
        lr = config.get('lr')
        wd_ts = config.get('wd_ts')
        dataset = config.get('dataset')
        val_loss = summary.get('val/loss')

        # Only include runs with complete data
        if all(x is not None for x in [lr, wd_ts, dataset, val_loss]):
            # Calculate omega = wd_ts * lr (rounded for consistent grouping)
            omega_3digits = round(wd_ts * lr, 3)

            data.append({
                'lr': lr,
                'wd_ts': wd_ts,
                'dataset': dataset,
                'val_loss': val_loss,
                'omega_3digits': omega_3digits,
                'log_omega': np.log(omega_3digits) if omega_3digits > 0 else np.nan,
                'log_lr': np.log(lr)
            })
        else:
            print(f"Skipping run {run.name} - missing data: lr={lr}, wd_ts={wd_ts}, dataset={dataset}, val_loss={val_loss}")

    return pd.DataFrame(data)

def preprocess_data(df):
    """
    Clean and prepare data for fitting.

    - Filters out outliers with excessive validation loss
    - Reports data statistics
    - Prepares JAX arrays for optimization

    Args:
        df (pd.DataFrame): Raw experiment data

    Returns:
        tuple: (X, Y, omega_indices, unique_log_omega_jax) for JAX optimization
    """
    print(f"Downloaded {len(df)} runs with complete data")
    print(f"Datasets found: {df['dataset'].unique()}")

    # Remove outliers with high validation loss
    print(f"Before filtering outliers: {len(df)} runs")
    df = df[df['val_loss'] < MAX_VAL_LOSS]
    print(f"After filtering outliers: {len(df)} runs")

    # Report omega value statistics
    unique_omega_3digits = sorted(df['omega_3digits'].unique())
    unique_log_omega = sorted(df['log_omega'].unique())
    print(f"Unique omega values (3 digits): {unique_omega_3digits}")
    print(f"Unique log_omega values: {[f'{x:.3f}' for x in unique_log_omega]}")
    print(f"Number of unique omega values: {len(unique_omega_3digits)}")

    # Convert to JAX arrays
    X = jnp.array(df['log_lr'].values)                    # Log learning rates
    Y = jnp.array(df['val_loss'].values)                  # Validation losses
    log_omega_vals = jnp.array(df['log_omega'].values)    # Log omega values for each point
    unique_log_omega_jax = jnp.array(unique_log_omega)    # Unique log omega values

    # Create omega indices mapping each data point to its omega group
    n_omega = len(unique_log_omega)
    omega_indices = jnp.zeros(len(X), dtype=jnp.int32)
    for i, log_omega in enumerate(unique_log_omega):
        # Find points belonging to this omega value
        mask = jnp.abs(log_omega_vals - log_omega) < 1e-6
        omega_indices = jnp.where(mask, i, omega_indices)

    print(f"Data prepared for JAX: X shape {X.shape}, Y shape {Y.shape}")
    print(f"Omega indices shape: {omega_indices.shape}")

    return df, X, Y, omega_indices, unique_log_omega_jax

# =============================================================================
# MATHEMATICAL FUNCTIONS
# =============================================================================

def softplus_like_function_jax(log_lr, a, b, c, d):
    """
    Softplus-like function for modeling validation loss vs log learning rate.

    Form: f(x) = a + log(exp(b*(x-c)) + exp(-d*(x-c)))

    This function has a minimum at x = c + log(d/b) / (b+d) when b,d > 0.
    The function approaches:
    - a + b*(x-c) for x << c (left tail, linear with slope b)
    - a - d*(x-c) for x >> c (right tail, linear with slope -d)

    Uses log-sum-exp trick for numerical stability.

    Args:
        log_lr (jnp.ndarray): Log learning rate values
        a (float): Vertical offset parameter
        b (float): Left slope parameter (b > 0)
        c (float): Center location parameter
        d (float): Right slope parameter (d > 0)

    Returns:
        jnp.ndarray: Function values
    """
    x_shifted = log_lr - c  # Center the input around c

    # Use log-sum-exp trick: log(exp(u) + exp(v)) = max(u,v) + log(exp(u-max) + exp(v-max))
    # This prevents overflow when b*(x-c) or -d*(x-c) are large
    max_term = jnp.maximum(b * x_shifted, -d * x_shifted)

    return a + max_term + jnp.log(
        jnp.exp(b * x_shifted - max_term) + jnp.exp(-d * x_shifted - max_term)
    )

def compute_function_minimum(a, b, c, d):
    """
    Compute the analytical minimum of the softplus-like function.

    For f(x) = a + log(exp(b*(x-c)) + exp(-d*(x-c))), the minimum occurs where
    the derivative equals zero: b*exp(b*y) = d*exp(-d*y) where y = x-c.

    Solving: x_min = c + log(d/b) / (b+d)

    Args:
        a, b, c, d (float): Function parameters

    Returns:
        tuple: (x_min, y_min) - location and value of minimum
    """
    if b > 0 and d > 0:
        x_min = c + jnp.log(d/b) / (b + d)
        y_min = softplus_like_function_jax(x_min, a, b, c, d)
    else:
        # Fallback for invalid parameters
        x_min = c
        y_min = a + jnp.log(2)  # Value when b=d=0

    return x_min, y_min

# =============================================================================
# OBJECTIVE FUNCTION WITH WEIGHTED RESIDUALS
# =============================================================================

def objective_function_jax(theta, X, Y, omega_indices, unique_log_omega, lambda_reg=LAMBDA_REG):
    """
    Vectorized JAX objective function with weighted residuals and regularization.

    Computes weighted least squares loss where points closer to the theoretical minimum
    for each omega get higher weights. Also includes regularization to encourage smooth
    parameter variation across omega values.

    Algorithm:
    1. Reshape parameters into (n_omega, 4) matrix
    2. Compute function predictions for all points
    3. Calculate distance-based weights for each point
    4. Compute weighted residual loss
    5. Add regularization penalty for parameter smoothness

    Args:
        theta (jnp.ndarray): Flattened parameter vector [a1,b1,c1,d1,a2,b2,c2,d2,...]
        X (jnp.ndarray): Log learning rate values
        Y (jnp.ndarray): Validation loss values
        omega_indices (jnp.ndarray): Omega group index for each data point
        unique_log_omega (jnp.ndarray): Unique log omega values
        lambda_reg (float): Regularization strength

    Returns:
        float: Total loss (data loss + regularization)
    """
    n_points = len(X)
    n_omega = len(unique_log_omega)

    # Reshape flattened parameter vector into (n_omega, 4) matrix
    # Each row contains [a, b, c, d] parameters for one omega value
    params = theta.reshape(n_omega, 4)

    # Vectorized parameter lookup: get parameters for each data point
    # omega_indices maps each point to its omega group
    point_params = params[omega_indices]  # Shape: (n_points, 4)
    a_vals = point_params[:, 0]  # a parameter for each point
    b_vals = point_params[:, 1]  # b parameter for each point
    c_vals = point_params[:, 2]  # c parameter for each point
    d_vals = point_params[:, 3]  # d parameter for each point

    # Compute function predictions for all points simultaneously
    predictions = softplus_like_function_jax(X, a_vals, b_vals, c_vals, d_vals)

    # WEIGHT COMPUTATION: Points closer to minimum get higher weights
    # ================================================================

    # Compute theoretical minimum location for each point's omega
    min_x_points = jnp.where(
        (b_vals > 0) & (d_vals > 0),  # Valid parameter condition
        c_vals + jnp.log(d_vals/b_vals) / (b_vals + d_vals),  # Analytical minimum
        c_vals  # Fallback if parameters invalid
    )

    # Distance from each point to its omega's minimum
    distances_from_min = jnp.abs(X - min_x_points)

    # Fully vectorized ranking computation without loops
    # Create matrix where entry (i,j) = 1 if points i,j belong to same omega
    same_omega_matrix = (omega_indices[:, None] == omega_indices[None, :]).astype(jnp.float32)

    # Create matrix where entry (i,j) = 1 if point j is closer to min than point i
    distance_comparison = (distances_from_min[:, None] > distances_from_min[None, :]).astype(jnp.float32)

    # Count points in same omega that are closer (gives rank: 0=closest, 1=second closest, etc.)
    same_omega_closer = distance_comparison * same_omega_matrix
    ranks = jnp.sum(same_omega_closer, axis=1)

    # Count total points in each omega
    points_in_omega = jnp.sum(same_omega_matrix, axis=1)

    # Convert ranks to weights: closest point gets highest weight
    # If omega has n points: closest gets weight n, furthest gets weight 1
    weights = points_in_omega - ranks

    # Compute weighted data fitting loss with squared weights for emphasis
    weighted_residuals = (weights**2) * (Y - predictions)**2
    data_loss = jnp.mean(weighted_residuals)

    # REGULARIZATION: Encourage smooth parameter variation across omega values
    # =======================================================================

    # Get all pairs of omega indices (upper triangular to avoid double counting)
    i_indices, j_indices = jnp.triu_indices(n_omega, k=1)

    # Parameter vectors for all pairs
    params_i = params[i_indices]  # Shape: (n_pairs, 4)
    params_j = params[j_indices]  # Shape: (n_pairs, 4)

    # Squared differences between parameter vectors
    param_diffs = jnp.sum((params_i - params_j)**2, axis=1)  # Shape: (n_pairs,)

    # Quartic differences in log omega values (larger separation allows more difference)
    log_omega_i = unique_log_omega[i_indices]
    log_omega_j = unique_log_omega[j_indices]
    log_omega_diffs = (log_omega_i - log_omega_j)**4

    # Regularization: penalize parameter differences scaled by omega separation
    # Nearby omegas should have similar parameters, distant ones can differ more
    reg_terms = param_diffs / (log_omega_diffs + 1e-12)  # Add small constant to avoid division by zero
    reg_loss = jnp.sum(reg_terms) * lambda_reg

    return data_loss + reg_loss

# =============================================================================
# OPTIMIZATION
# =============================================================================

def optimize_parameters(X, Y, omega_indices, unique_log_omega_jax):
    """
    Optimize softplus function parameters using JAX and Optax.

    Uses Adagrad optimizer with JIT compilation for performance.
    Tracks best parameters during optimization.

    Args:
        X, Y, omega_indices, unique_log_omega_jax: Data arrays from preprocessing

    Returns:
        jnp.ndarray: Optimized parameter vector
    """
    n_omega = len(unique_log_omega_jax)

    # Initialize parameters
    global A_INIT
    A_INIT = float(jnp.mean(Y))  # Set baseline to mean validation loss

    # Create initial parameter vector: [a,b,c,d] repeated for each omega
    theta_init_jax = jnp.array([A_INIT, B_INIT, C_INIT, D_INIT] * n_omega)

    print(f"\\nInitial parameters shape: {theta_init_jax.shape}")
    print(f"Initial values: a={A_INIT:.4f}, b={B_INIT:.6f}, c={C_INIT:.4f}, d={D_INIT:.6f}")
    print(f"Optimization settings: lr={LEARNING_RATE}, steps={N_STEPS}, λ_reg={LAMBDA_REG}")

    # Set up Adagrad optimizer
    optimizer = optax.adagrad(LEARNING_RATE)
    opt_state = optimizer.init(theta_init_jax)

    # JIT compile objective function for performance
    objective_function_jax_jit = jit(objective_function_jax)

    # Create loss function with fixed data arguments
    def loss_fn(theta):
        return objective_function_jax_jit(theta, X, Y, omega_indices, unique_log_omega_jax, LAMBDA_REG)

    # JIT compile gradient function
    grad_fn = jit(grad(loss_fn))

    # Compute initial loss
    initial_loss = loss_fn(theta_init_jax)
    print(f"Initial loss: {float(initial_loss):.8f}")

    # Optimization loop
    theta = theta_init_jax
    best_loss = float('inf')
    best_theta = theta

    print(f"\\nStarting optimization...")
    for step in range(N_STEPS):
        # Compute gradients and update parameters
        grads = grad_fn(theta)
        updates, opt_state = optimizer.update(grads, opt_state)
        theta = optax.apply_updates(theta, updates)

        # Track best parameters
        current_loss = float(loss_fn(theta))
        if current_loss < best_loss:
            best_loss = current_loss
            best_theta = theta

        # Print progress periodically
        if step % 1000 == 0 or step == N_STEPS - 1:
            grad_norm = float(jnp.linalg.norm(grads))
            print(f"Step {step:5d}: loss={current_loss:.8f}, grad_norm={grad_norm:.2e}")

    print(f"\\nOptimization completed!")
    print(f"Final loss: {best_loss:.8f}")
    print(f"Improvement: {float(initial_loss) - best_loss:.8f}")

    return best_theta

def display_results(theta_opt, unique_log_omega):
    """
    Display fitted parameters and minimum locations for each omega.

    Args:
        theta_opt (jnp.ndarray): Optimized parameters
        unique_log_omega (list): Unique log omega values
    """
    print("\\nOptimal parameters for each omega:")
    print("=" * 60)

    for i, log_omega in enumerate(unique_log_omega):
        # Extract parameters for this omega
        start_idx = i * 4
        a, b, c, d = theta_opt[start_idx:start_idx+4]

        # Compute minimum location and value
        x_min, y_min = compute_function_minimum(a, b, c, d)

        # Display results
        print(f"log_ω={log_omega:6.3f}, ω={np.exp(log_omega):6.3f}: "
              f"a={float(a):.4f}, b={float(b):.6f}, c={float(c):.4f}, d={float(d):.6f}")
        print(f"  Minimum at log_lr={float(x_min):.4f} "
              f"(lr={float(jnp.exp(x_min)):.6f}), min_loss={float(y_min):.4f}")

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization(df, theta_opt, X, unique_log_omega, config):
    """
    Create and save publication-quality plot of data and fitted curves.

    Shows:
    - Scatter points colored by log(omega) with different markers for datasets
    - Fitted softplus curves for each omega value
    - Star markers indicating theoretical minima
    - Colorbar for omega values

    Args:
        df (pd.DataFrame): Original data for plotting
        theta_opt (jnp.ndarray): Optimized parameters
        X (jnp.ndarray): Log learning rate range for data
        unique_log_omega (list): Unique log omega values
        config (dict): Configuration with title and filename
    """
    plt.figure(figsize=(12, 8))

    # Create clipped plasma colormap (avoid too-bright yellow)
    plasma_cmap = plt.cm.plasma
    plasma_colors = plasma_cmap(np.linspace(0, 0.85, 256))
    clipped_plasma = ListedColormap(plasma_colors)

    # Separate data by dataset for different markers
    fineweb_data = df[df['dataset'] == 'fineweb']
    fineweb_100_data = df[df['dataset'] == 'fineweb_100']

    # Plot scatter points with dataset-specific markers
    if len(fineweb_data) > 0:
        plt.scatter(fineweb_data['lr'], fineweb_data['val_loss'],
                   c=fineweb_data['log_omega'], cmap=clipped_plasma,
                   marker='o', s=50, alpha=0.7, label='fineweb')

    if len(fineweb_100_data) > 0:
        plt.scatter(fineweb_100_data['lr'], fineweb_100_data['val_loss'],
                   c=fineweb_100_data['log_omega'], cmap=clipped_plasma,
                   marker='x', s=50, alpha=0.7, label='fineweb_100')

    # Generate smooth curves for fitted functions
    log_lr_range = jnp.linspace(float(X.min()), float(X.max()), 200)
    lr_range = jnp.exp(log_lr_range)

    # Plot fitted curves and minima for each omega
    for i, log_omega in enumerate(unique_log_omega):
        # Extract parameters for this omega
        start_idx = i * 4
        a, b, c, d = theta_opt[start_idx:start_idx+4]

        # Generate smooth fitted curve
        loss_pred = softplus_like_function_jax(log_lr_range, a, b, c, d)

        # Compute minimum location
        x_min, y_min = compute_function_minimum(a, b, c, d)

        # Color based on omega value
        log_omega_min = df['log_omega'].min()
        log_omega_max = df['log_omega'].max()
        normalized_log_omega = (log_omega - log_omega_min) / (log_omega_max - log_omega_min)
        color = clipped_plasma(normalized_log_omega)

        # Plot fitted curve (only label first few to avoid legend clutter)
        plt.plot(np.array(lr_range), np.array(loss_pred), '-',
                color=color, alpha=0.8, linewidth=2,
                label=f'ω={np.exp(log_omega):.3f} JAX fit' if i < 3 else "")

        # Mark minimum with star
        plt.plot(float(jnp.exp(x_min)), float(y_min), '*',
                color=color, markersize=12, markeredgecolor='black', markeredgewidth=1,
                label=f'ω={np.exp(log_omega):.3f} minimum' if i < 3 else "")

    # Add colorbar for omega values
    if len(df) > 0:
        cbar = plt.colorbar(label='log(ω) where ω = wd_ts × lr')

    # Formatting
    plt.xlabel('Learning Rate (lr)')
    plt.ylabel('Validation Loss (val/loss)')
    plt.title(config['title'])
    plt.legend()
    plt.xscale('log')  # Log scale for learning rate axis
    plt.tight_layout()

    # Save high-quality PDF
    plt.savefig(config['filename'], format='pdf', dpi=300, bbox_inches='tight')
    print(f"\\nPlot saved as {config['filename']}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("DanaStar Softplus Function Fitting with JAX")
    print("=" * 50)

    # Load and preprocess data
    df = load_wandb_data("danastar", config['group'])
    df, X, Y, omega_indices, unique_log_omega_jax = preprocess_data(df)

    # Optimize parameters
    theta_opt = optimize_parameters(X, Y, omega_indices, unique_log_omega_jax)

    # Display results
    display_results(theta_opt, list(unique_log_omega_jax))

    # Create visualization
    create_visualization(df, theta_opt, X, list(unique_log_omega_jax), config)

    print("\\nFitting completed successfully!")