#!/usr/bin/env python3
"""
DanaStar and AdamW Softplus Function Fitting with JAX

OVERVIEW:
========
This script fits a softplus-like function to validation loss data from optimization experiments
to model the relationship between learning rate, weight decay, and final validation loss.

The goal is to create a consistent fit that captures the optimal learning rate for different
weight decay values and to model the validation loss landscape around the minimum.

SUPPORTED OPTIMIZERS:
====================
1. **DanaStar**: Uses wd_ts (weight decay timescale) parameter
   - omega = wd_ts * learning_rate
   - Groups: DanaStar_35M_LR_WD_Sweep, DanaStar_90M_LR_WD_Sweep

2. **AdamW**: Uses weight_decay and iterations parameters
   - omega_T = weight_decay * learning_rate * iterations
   - Groups: AdamW_small_lr_weight_decay_sweeps, AdamW_35M_lr_weight_decay_sweeps,
            AdamW_90M_lr_weight_decay_sweeps, AdamW_180M_lr_weight_decay_sweeps

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
Basic usage:
python softplus_fitting_jax.py --model DS35M     # DanaStar 35M model data
python softplus_fitting_jax.py --model DS90M     # DanaStar 90M model data
python softplus_fitting_jax.py --model AWsmall   # AdamW small model data
python softplus_fitting_jax.py --model AW35M     # AdamW 35M model data
python softplus_fitting_jax.py --model AW90M     # AdamW 90M model data
python softplus_fitting_jax.py --model AW180M    # AdamW 180M model data

With custom filtering and optimization:
python softplus_fitting_jax.py --model DS35M --maxloss 3.5 --minpoints 3 --lambda 1e-4    # Custom loss, points, and regularization
python softplus_fitting_jax.py --model AW35M --maxomega 2.0 --lambda 1e-2                # Limit to omega ≤ 2.0, stronger regularization

OPTIONS:
========
--model: Model configuration (DS35M, DS90M, AWsmall, AW35M, AW90M, AW180M)
--maxloss: Maximum validation loss threshold for filtering outliers (default: 4.0)
           Data points with validation loss above this value are excluded
--minpoints: Minimum number of data points required per omega group (default: 2)
             Omega groups with fewer points are excluded from fitting
--lambda: Regularization parameter for smooth parameter variation across omega (default: 1e-3)
          Higher values enforce smoother parameter changes, lower values allow more variation
--maxomega: Maximum omega value to include in fitting (default: no limit)
            Omega groups above this threshold are excluded from analysis

OUTPUT:
=======
- PDF plot showing data points, fitted curves, and optimal points
- Console output with fitted parameters and minimum locations for each omega
- For DanaStar: omega = wd_ts × lr
- For AdamW: omega_T = weight_decay × lr × iterations
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
LEARNING_RATE = 1e-1   # Adagrad learning rate
N_STEPS = 20000        # Number of optimization steps

# Note: Regularization parameter and data filtering thresholds are now set via command line arguments

# Parse command line arguments
parser = argparse.ArgumentParser(description='JAX softplus fitting for DanaStar and AdamW experiment data')
parser.add_argument('--model', type=str, default='DS35M',
                    choices=['DS35M', 'DS90M', 'AWsmall', 'AW35M', 'AW90M', 'AW180M'],
                    help='Model configuration to fit (default: DS35M)')
parser.add_argument('--maxloss', type=float, default=4.0,
                    help='Maximum validation loss threshold - data points above this are filtered out (default: 4.0)')
parser.add_argument('--minpoints', type=int, default=2,
                    help='Minimum number of data points required per omega group (default: 2)')
parser.add_argument('--lambda', type=float, default=1e-3, dest='lambda_reg',
                    help='Regularization parameter for smooth parameter variation (default: 1e-3)')
parser.add_argument('--maxomega', type=float, default=None,
                    help='Maximum omega value to include - omega groups above this are filtered out (default: no limit)')
args = parser.parse_args()

# Configuration mapping for different optimizers and model sizes
model_config = args.model
configs = {
    # DanaStar configurations
    'DS35M': {
        'group': 'DanaStar_35M_LR_WD_Sweep',                          # WandB experiment group name
        'title': 'DanaStar 35M: JAX Softplus-like Fits',             # Plot title
        'filename': 'danastar_35m_lrwd_sweep_jax.pdf',               # Output filename
        'optimizer': 'danastar',                                      # Optimizer type
        'omega_label': 'log(ω) where ω = wd_ts × lr'                # Colorbar label
    },
    'DS90M': {
        'group': 'DanaStar_90M_LR_WD_Sweep',                          # WandB experiment group name
        'title': 'DanaStar 90M: JAX Softplus-like Fits',             # Plot title
        'filename': 'danastar_90m_lrwd_sweep_jax.pdf',               # Output filename
        'optimizer': 'danastar',                                      # Optimizer type
        'omega_label': 'log(ω) where ω = wd_ts × lr'                # Colorbar label
    },
    # AdamW configurations
    'AWsmall': {
        'group': 'AdamW_small_lr_weight_decay_sweeps',                # WandB experiment group name
        'title': 'AdamW Small: JAX Softplus-like Fits',              # Plot title
        'filename': 'adamw_small_lrwd_sweep_jax.pdf',                # Output filename
        'optimizer': 'adamw',                                         # Optimizer type
        'omega_label': 'log(ω_T) where ω_T = weight_decay × lr × T'  # Colorbar label
    },
    'AW35M': {
        'group': 'AdamW_35M_lr_weight_decay_sweeps',                  # WandB experiment group name
        'title': 'AdamW 35M: JAX Softplus-like Fits',                # Plot title
        'filename': 'adamw_35m_lrwd_sweep_jax.pdf',                  # Output filename
        'optimizer': 'adamw',                                         # Optimizer type
        'omega_label': 'log(ω_T) where ω_T = weight_decay × lr × T'  # Colorbar label
    },
    'AW90M': {
        'group': 'AdamW_90M_lr_weight_decay_sweeps',                  # WandB experiment group name
        'title': 'AdamW 90M: JAX Softplus-like Fits',                # Plot title
        'filename': 'adamw_90m_lrwd_sweep_jax.pdf',                  # Output filename
        'optimizer': 'adamw',                                         # Optimizer type
        'omega_label': 'log(ω_T) where ω_T = weight_decay × lr × T'  # Colorbar label
    },
    'AW180M': {
        'group': 'AdamW_180M_lr_weight_decay_sweeps',                 # WandB experiment group name
        'title': 'AdamW 180M: JAX Softplus-like Fits',               # Plot title
        'filename': 'adamw_180m_lrwd_sweep_jax.pdf',                 # Output filename
        'optimizer': 'adamw',                                         # Optimizer type
        'omega_label': 'log(ω_T) where ω_T = weight_decay × lr × T'  # Colorbar label
    }
}

config = configs[model_config]
print(f"Using configuration for {model_config}: {config}")

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def group_nearby_omegas(omega_values, tolerance_percent=1.0):
    """
    Group omega values that are within a specified percentage of each other.

    This function clusters omega values that are very close (within tolerance_percent)
    into groups and assigns each group a representative value. This prevents artificial
    separation of essentially identical experimental conditions due to minor numerical
    differences.

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

def load_wandb_data(project_name, group_name, optimizer_type):
    """
    Load experiment data from Weights & Biases for different optimizers.

    Downloads runs from the specified project and group, extracting different parameters
    based on the optimizer type:

    DanaStar:
    - lr: learning rate
    - wd_ts: weight decay (timescale)
    - dataset: dataset name (fineweb vs fineweb_100)
    - val_loss: final validation loss
    - omega = wd_ts * lr

    AdamW:
    - lr: learning rate
    - weight_decay: weight decay parameter
    - iterations: number of training iterations (T)
    - dataset: dataset name (fineweb vs fineweb_100)
    - val_loss: final validation loss
    - omega_T = weight_decay * lr * iterations

    Args:
        project_name (str): WandB project name
        group_name (str): WandB experiment group name
        optimizer_type (str): 'danastar' or 'adamw'

    Returns:
        pd.DataFrame: Processed experiment data
    """
    # Initialize WandB API
    api = wandb.Api()

    print(f"Downloading data from project: {project_name}, group: {group_name}")
    print(f"Optimizer type: {optimizer_type}")

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

        # Get common fields
        lr = config.get('lr')
        dataset = config.get('dataset')
        val_loss = summary.get('val/loss')

        if optimizer_type == 'danastar':
            # DanaStar-specific fields
            wd_ts = config.get('wd_ts')
            required_fields = [lr, wd_ts, dataset, val_loss]

            if all(x is not None for x in required_fields):
                # Calculate raw omega = wd_ts * lr (not rounded yet)
                raw_omega = wd_ts * lr

                raw_data.append({
                    'lr': lr,
                    'wd_ts': wd_ts,
                    'dataset': dataset,
                    'val_loss': val_loss,
                    'raw_omega': raw_omega,
                    'log_lr': np.log(lr),
                    'optimizer': 'danastar'
                })
                raw_omega_values.append(raw_omega)
            else:
                print(f"Skipping run {run.name} - missing DanaStar data: lr={lr}, wd_ts={wd_ts}, dataset={dataset}, val_loss={val_loss}")

        elif optimizer_type == 'adamw':
            # AdamW-specific fields
            weight_decay = config.get('weight_decay')
            iterations = config.get('iterations')
            required_fields = [lr, weight_decay, iterations, dataset, val_loss]

            if all(x is not None for x in required_fields):
                # Calculate raw omega_T = weight_decay * lr * iterations (not rounded yet)
                raw_omega_T = weight_decay * lr * iterations

                raw_data.append({
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'iterations': iterations,
                    'dataset': dataset,
                    'val_loss': val_loss,
                    'raw_omega': raw_omega_T,  # Use same field name for consistency
                    'log_lr': np.log(lr),
                    'optimizer': 'adamw'
                })
                raw_omega_values.append(raw_omega_T)
            else:
                print(f"Skipping run {run.name} - missing AdamW data: lr={lr}, weight_decay={weight_decay}, iterations={iterations}, dataset={dataset}, val_loss={val_loss}")

        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

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
        final_entry['log_omega'] = np.log(grouped_omega) if grouped_omega > 0 else np.nan
        del final_entry['raw_omega']  # Remove temporary field

        data.append(final_entry)

    return pd.DataFrame(data)

def preprocess_data(df, max_loss_threshold, min_points_per_omega, max_omega_threshold=None):
    """
    Clean and prepare data for fitting from either DanaStar or AdamW experiments.

    CRITICAL: Multi-stage filtering to ensure quality fitting:
    1. Filter out high validation losses
    2. Filter out omega groups above maximum omega threshold
    3. Filter out omega groups with too few data points
    4. Only keep omega values that have sufficient data for reliable fitting

    Args:
        df (pd.DataFrame): Raw experiment data with unified omega representation
        max_loss_threshold (float): Maximum validation loss threshold for filtering
        min_points_per_omega (int): Minimum number of data points required per omega group
        max_omega_threshold (float, optional): Maximum omega value to include (None = no limit)

    Returns:
        tuple: (df, X, Y, omega_indices, unique_log_omega_jax) for JAX optimization
    """
    print(f"Downloaded {len(df)} runs with complete data")
    print(f"Datasets found: {df['dataset'].unique()}")

    # STEP 1: Filter out outliers with high validation loss FIRST
    # This is critical - we must filter before determining omega values
    print(f"Before filtering outliers: {len(df)} runs")
    print(f"Filtering out runs with validation loss > {max_loss_threshold}")
    df_filtered = df[df['val_loss'] < max_loss_threshold].copy()
    print(f"After filtering outliers: {len(df_filtered)} runs")

    # STEP 2: Filter out omega groups above maximum omega threshold (if specified)
    if max_omega_threshold is not None:
        print(f"Filtering out runs with omega > {max_omega_threshold}")
        initial_count = len(df_filtered)
        df_filtered = df_filtered[df_filtered['omega_3digits'] <= max_omega_threshold].copy()
        print(f"After filtering high omega values: {len(df_filtered)} runs (removed {initial_count - len(df_filtered)})")

    # STEP 3: Determine omega values from filtered data
    initial_unique_omega_3digits = sorted(df_filtered['omega_3digits'].unique())
    initial_unique_log_omega = sorted(df_filtered['log_omega'].unique())

    print(f"Omega values from fully filtered data:")
    print(f"Initial unique omega values (3 digits): {initial_unique_omega_3digits}")
    print(f"Number of initial omega values: {len(initial_unique_omega_3digits)}")

    # STEP 4: Filter out omega groups with too few data points
    print(f"\\nFiltering omega groups with < {min_points_per_omega} data points:")
    valid_omegas = []
    valid_log_omegas = []

    for log_omega in initial_unique_log_omega:
        omega_count = np.sum(np.abs(df_filtered['log_omega'].values - log_omega) < 1e-6)
        omega_value = np.exp(log_omega)

        if omega_count >= min_points_per_omega:
            valid_omegas.append(omega_value)
            valid_log_omegas.append(log_omega)
            print(f"  ✓ Omega {omega_value:.3f} (log={log_omega:.3f}): {omega_count} data points - KEEPING")
        else:
            print(f"  ✗ Omega {omega_value:.3f} (log={log_omega:.3f}): {omega_count} data points - REMOVING (< {min_points_per_omega})")

    # STEP 5: Keep only data points from valid omega groups
    if len(valid_log_omegas) == 0:
        raise ValueError(f"No omega groups have >= {min_points_per_omega} data points. Try lowering --minpoints, --maxloss, or --maxomega thresholds.")

    # Filter dataframe to keep only points from valid omega groups
    valid_mask = np.zeros(len(df_filtered), dtype=bool)
    for log_omega in valid_log_omegas:
        omega_mask = np.abs(df_filtered['log_omega'].values - log_omega) < 1e-6
        valid_mask |= omega_mask

    df_final = df_filtered[valid_mask].copy()

    # Update omega lists to only include valid ones
    unique_omega_3digits = sorted([np.exp(log_omega) for log_omega in valid_log_omegas])
    unique_log_omega = sorted(valid_log_omegas)

    print(f"\\nFinal omega groups for fitting:")
    print(f"Unique omega values (3 digits): {unique_omega_3digits}")
    print(f"Unique log_omega values: {[f'{x:.3f}' for x in unique_log_omega]}")
    print(f"Number of omega groups: {len(unique_omega_3digits)}")
    print(f"Total data points for fitting: {len(df_final)}")

    # Verify final counts
    for i, log_omega in enumerate(unique_log_omega):
        omega_count = np.sum(np.abs(df_final['log_omega'].values - log_omega) < 1e-6)
        print(f"  Final Omega {np.exp(log_omega):.3f}: {omega_count} data points")

    # STEP 6: Convert to JAX arrays (only from final filtered data)
    X = jnp.array(df_final['log_lr'].values)                    # Log learning rates
    Y = jnp.array(df_final['val_loss'].values)                  # Validation losses
    log_omega_vals = jnp.array(df_final['log_omega'].values)    # Log omega values for each point
    unique_log_omega_jax = jnp.array(unique_log_omega)          # Unique log omega values

    # STEP 7: Create omega indices mapping each data point to its omega group
    n_omega = len(unique_log_omega)
    omega_indices = jnp.zeros(len(X), dtype=jnp.int32)
    for i, log_omega in enumerate(unique_log_omega):
        # Find points belonging to this omega value
        mask = jnp.abs(log_omega_vals - log_omega) < 1e-6
        omega_indices = jnp.where(mask, i, omega_indices)

    print(f"\\nData prepared for JAX: X shape {X.shape}, Y shape {Y.shape}")
    print(f"Omega indices shape: {omega_indices.shape}")

    return df_final, X, Y, omega_indices, unique_log_omega_jax

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

def objective_function_jax(theta, X, Y, omega_indices, unique_log_omega, lambda_reg):
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

def optimize_parameters(X, Y, omega_indices, unique_log_omega_jax, lambda_reg):
    """
    Optimize softplus function parameters using JAX and Optax.

    Uses Adagrad optimizer with JIT compilation for performance.
    Tracks best parameters during optimization.

    Args:
        X, Y, omega_indices, unique_log_omega_jax: Data arrays from preprocessing
        lambda_reg (float): Regularization parameter for smooth parameter variation

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
    print(f"Optimization settings: lr={LEARNING_RATE}, steps={N_STEPS}, λ_reg={lambda_reg}")

    # Set up Adagrad optimizer
    optimizer = optax.adagrad(LEARNING_RATE)
    opt_state = optimizer.init(theta_init_jax)

    # JIT compile objective function for performance
    objective_function_jax_jit = jit(objective_function_jax)

    # Create loss function with fixed data arguments
    def loss_fn(theta):
        return objective_function_jax_jit(theta, X, Y, omega_indices, unique_log_omega_jax, lambda_reg)

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

    # First pass: calculate fitted minimums and find observed minimums for each omega
    fitted_minima = []
    observed_minima = []

    for i, log_omega in enumerate(unique_log_omega):
        # Extract parameters for this omega
        start_idx = i * 4
        a, b, c, d = theta_opt[start_idx:start_idx+4]

        # Compute fitted minimum
        x_min, y_min = compute_function_minimum(a, b, c, d)
        fitted_minima.append((log_omega, float(y_min), np.exp(log_omega)))

        # Find observed minimum for this omega group
        omega_mask = np.abs(df['log_omega'].values - log_omega) < 1e-6
        omega_data = df[omega_mask]
        if len(omega_data) > 0:
            observed_min_loss = omega_data['val_loss'].min()
            observed_minima.append((log_omega, observed_min_loss, np.exp(log_omega)))

    # Sort by loss values and get top 3
    fitted_minima.sort(key=lambda x: x[1])  # Sort by fitted min loss
    observed_minima.sort(key=lambda x: x[1])  # Sort by observed min loss

    best_fitted = fitted_minima[:3]
    best_observed = observed_minima[:3]

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

        # Plot fitted curve (no labels - we'll add custom legend)
        plt.plot(np.array(lr_range), np.array(loss_pred), '-',
                color=color, alpha=0.8, linewidth=2)

        # Mark minimum with star (no labels - we'll add custom legend)
        plt.plot(float(jnp.exp(x_min)), float(y_min), '*',
                color=color, markersize=12, markeredgecolor='black', markeredgewidth=1)

    # Add colorbar for omega values
    if len(df) > 0:
        cbar = plt.colorbar(label=config['omega_label'])

    # Create custom legend
    legend_elements = []

    # Add dataset markers if they exist
    if len(fineweb_data) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor='gray', markersize=8, alpha=0.7,
                                        label='fineweb', linestyle='None'))

    if len(fineweb_100_data) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='x', color='w',
                                        markerfacecolor='gray', markersize=8, alpha=0.7,
                                        label='fineweb_100', linestyle='None'))

    # Add best omegas section
    legend_elements.append(plt.Line2D([0], [0], color='none', label='Best omegas:'))

    # Add best fitted minimums
    for i, (log_omega, fitted_min, omega) in enumerate(best_fitted):
        # Calculate color for this omega
        log_omega_min = df['log_omega'].min()
        log_omega_max = df['log_omega'].max()
        normalized_log_omega = (log_omega - log_omega_min) / (log_omega_max - log_omega_min)
        color = clipped_plasma(normalized_log_omega)

        legend_elements.append(plt.Line2D([0], [0], marker='*', color='w',
                                        markerfacecolor=color, markeredgecolor='black',
                                        markersize=10, markeredgewidth=1,
                                        label=f'  Fitted: ω={omega:.3f} (min={fitted_min:.3f})',
                                        linestyle='None'))

    # Add best observed minimums
    for i, (log_omega, observed_min, omega) in enumerate(best_observed):
        # Calculate color for this omega
        log_omega_min = df['log_omega'].min()
        log_omega_max = df['log_omega'].max()
        normalized_log_omega = (log_omega - log_omega_min) / (log_omega_max - log_omega_min)
        color = clipped_plasma(normalized_log_omega)

        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=color, markersize=8,
                                        label=f'  Observed: ω={omega:.3f} (min={observed_min:.3f})',
                                        linestyle='None'))

    # Formatting
    plt.xlabel('Learning Rate (lr)')
    plt.ylabel('Validation Loss (val/loss)')
    plt.title(config['title'])
    plt.legend(handles=legend_elements, loc='upper right')
    plt.xscale('log')  # Log scale for learning rate axis
    plt.tight_layout()

    # Save high-quality PDF
    plt.savefig(config['filename'], format='pdf', dpi=300, bbox_inches='tight')
    print(f"\\nPlot saved as {config['filename']}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("DanaStar and AdamW Softplus Function Fitting with JAX")
    print("=" * 60)

    # Load and preprocess data
    df = load_wandb_data("danastar", config['group'], config['optimizer'])
    df, X, Y, omega_indices, unique_log_omega_jax = preprocess_data(df, args.maxloss, args.minpoints, args.maxomega)

    # Optimize parameters
    theta_opt = optimize_parameters(X, Y, omega_indices, unique_log_omega_jax, args.lambda_reg)

    # Display results
    display_results(theta_opt, list(unique_log_omega_jax))

    # Create visualization
    create_visualization(df, theta_opt, X, list(unique_log_omega_jax), config)

    print("\\nFitting completed successfully!")