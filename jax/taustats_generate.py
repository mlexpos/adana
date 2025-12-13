#!/usr/bin/env python
"""
Tau Statistics Data Generation: Train dana-star-mk4 optimizer and collect tau order statistics.

This script trains a MoE PLRF model with dana-star-mk4 optimizer and collects
tau order statistics at times which are powers of 2 times 1000 (1000, 2000, 4000, 8000, ...).
The tau statistics data allows plotting cumulative distribution functions on log-log axes.
"""
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import pickle
from tqdm import tqdm
from typing import NamedTuple, Callable, Dict, List, Union, Optional, Tuple
import numpy as np
import argparse
import os
import json
from datetime import datetime

from optimizers import powerlaw_schedule, get_dana_star_mk4, TaneaOptimizerState
from moe_plrf import MixtureOfExpertsPLRF


class LabelNoiseMixtureOfExpertsPLRF(MixtureOfExpertsPLRF):
    """Mixture of Experts PLRF with added student-t label noise."""

    def __init__(self,
                 alpha: float,
                 beta: float,
                 v: int,
                 d: int,
                 m: int,
                 zeta: float,
                 student_t_dof: float,
                 sigma: float,
                 key: random.PRNGKey):
        """Initialize the MoE PLRF model with label noise parameters.

        Args:
            alpha: Power law exponent for eigenvalue decay
            beta: Power law exponent for target coefficient decay
            v: Hidden dimension (number of random features)
            d: Embedded dimension (parameter dimension)
            m: Number of experts
            zeta: Power law exponent for expert selection (p(i) ∝ i^(-zeta))
            student_t_dof: Degrees of freedom for student-t distribution
            sigma: Scaling factor for the noise
            key: JAX random key
        """
        super().__init__(alpha, beta, v, d, m, zeta, key)
        self.student_t_dof = student_t_dof
        self.sigma = sigma

    def generate_batch(self, key: random.PRNGKey, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate a batch of (X, y) training data with added student-t noise.

        Args:
            key: JAX random key
            batch_size: Number of samples to generate

        Returns:
            X: Input features of shape (batch_size, d)
            y: Target values of shape (batch_size,) with added noise
        """
        # Split key for data generation and noise
        key_data, key_noise = random.split(key)

        # Generate random features x ~ N(0, 1)
        x = random.normal(key_data, (batch_size, self.v))

        # Transform to get inputs and targets
        X = jnp.matmul(x, self.checkW)  # (batch_size, d)
        y_clean = jnp.matmul(x, self.checkb)  # (batch_size,)

        # Add student-t noise
        noise = random.t(key_noise, df=self.student_t_dof, shape=(batch_size,))
        y_noisy = y_clean + self.sigma * noise

        return X, y_noisy


def compute_tau_order_statistics(tau_vector):
    """Compute order statistics for tau vector in a jittable way.

    Args:
        tau_vector: A 1D array of non-negative tau values

    Returns:
        Tuple of (largest_order_stats, smallest_order_stats) where:
        - largest_order_stats: [largest, (1.1)^1-th largest, (1.1)^2-th largest, ...]
        - smallest_order_stats: [smallest, (1.1)^1-th smallest, (1.1)^2-th smallest, ...]
        where we take the (1.1)^k-th for k = 0, 1, 2, ..., up to n
    """
    n = len(tau_vector)
    if n == 0:
        return np.array([]), np.array([])

    # Sort in ascending order
    sorted_tau_asc = jnp.sort(tau_vector)

    # Compute powers of 1.1 up to n, similar to evaluation times
    max_k = jnp.ceil(jnp.log(n) / jnp.log(1.1)).astype(jnp.int32)
    indices = jnp.int32(1.1 ** jnp.arange(max_k + 1)) - 1  # 0-indexed: [0, 0, 1, 2, 3, 4, ...]

    # Remove duplicates and clamp to valid range
    indices = jnp.unique(indices)
    indices = jnp.minimum(indices, n - 1)

    # Get smallest order statistics
    smallest_order_stats = sorted_tau_asc[indices]

    # Get largest order statistics using reversed indices
    # For largest: indices from the end of the sorted array
    reversed_indices = (n - 1) - indices
    largest_order_stats = sorted_tau_asc[reversed_indices]

    return largest_order_stats, smallest_order_stats


def extract_tau_statistics(opt_state):
    """Extract tau statistics from optimizer state.

    Args:
        opt_state: Optimizer state (may be from optax.chain)

    Returns:
        Dictionary with tau statistics including both largest and smallest order statistics
    """
    # Handle optax.chain optimizer - extract the dana-star-mk4 state
    dana_state = opt_state
    if hasattr(opt_state, '__len__') and len(opt_state) > 1:
        # optax.chain creates a tuple: (dana_star_mk4_state, scale_state)
        dana_state = opt_state[0]

    if not isinstance(dana_state, TaneaOptimizerState):
        return {}

    def compute_tau_order_stats_wrapper(x):
        if x is None:
            return None
        else:
            # x should be shape (d, m) for MoE PLRF model
            # We want to compute order statistics PER EXPERT (per column)
            if x.ndim == 2:
                # Process each expert (column) separately
                d, m = x.shape
                per_expert_stats = []
                for expert_idx in range(m):
                    tau_col = x[:, expert_idx]  # Extract column for this expert
                    u, v = compute_tau_order_statistics(tau_col)
                    per_expert_stats.append((np.array(u), np.array(v)))
                return per_expert_stats
            else:
                # Fallback: treat as a single vector
                u, v = compute_tau_order_statistics(jnp.ravel(x))
                return [(np.array(u), np.array(v))]

    # Flatten tau tree into order statistics
    tau_stats = jax.tree.map(compute_tau_order_stats_wrapper, dana_state.tau)

    return tau_stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tau Statistics Generation: Collect tau order statistics from dana-star-mk4 optimizer")

    # Model parameters
    parser.add_argument("--alpha", type=float, default=1.0, help="Power law exponent for eigenvalue decay")
    parser.add_argument("--beta", type=float, default=-0.3, help="Beta value (power law exponent for target coefficient decay)")
    parser.add_argument("--v", type=int, default=2000, help="Hidden dimension (number of random features)")
    parser.add_argument("--d", type=int, default=500, help="Embedded dimension (parameter dimension)")
    parser.add_argument("--m", type=int, default=1000, help="Number of experts")
    parser.add_argument("--zeta", type=float, default=1.0, help="Power-law exponent for expert selection")

    # Training parameters
    parser.add_argument("--steps", type=int, default=1000000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=100, help="Training batch size")
    parser.add_argument("--g2_scale", type=float, default=0.5, help="Base learning rate scale")
    parser.add_argument("--g3_over_g2", type=float, default=1.0, help="G3 to G2 ratio for momentum")
    parser.add_argument("--tanea_lr_scalar", type=float, default=1.0, help="Learning rate scalar")

    # Optimizer parameters
    parser.add_argument("--kappa", type=float, default=1.0, help="Kappa value for dana-star-mk4")
    parser.add_argument("--clipsnr", type=float, default=2.0, help="Clipsnr parameter for Dana-Star-MK4")
    parser.add_argument("--delta", type=float, default=8.0, help="Delta parameter for EMA coefficient")

    # Label noise parameters
    parser.add_argument("--student_t_dof", type=float, default=3.0, help="Degrees of freedom for student-t distribution")
    parser.add_argument("--sigma", type=float, default=0.0, help="Scaling factor for the noise")

    # Output parameters
    parser.add_argument("--data_dir", type=str, default="jax/taustats_data", help="Directory to store tau statistics data")
    parser.add_argument("--output_prefix", type=str, default="taustats", help="Prefix for output files")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


def get_traceK(alpha, v):
    """Compute trace of the kernel matrix."""
    x_grid = jnp.arange(1, v+1).reshape(1, v)
    population_eigs = x_grid ** -alpha
    population_trace = jnp.sum(population_eigs**2)
    return population_trace


def create_optimizer(args, traceK):
    """Create dana-star-mk4 optimizer.

    Args:
        args: Command line arguments
        traceK: Trace of the kernel matrix

    Returns:
        Optimizer object
    """
    # Compute learning rate scaling
    learning_rate_g2 = args.g2_scale * jnp.minimum(1.0, jnp.float32(args.batch_size) / traceK)
    g2_constant = args.tanea_lr_scalar * learning_rate_g2
    g3_constant = args.g3_over_g2 * learning_rate_g2 * args.tanea_lr_scalar

    optimizer = get_dana_star_mk4(
        g2_constant,
        g3_constant,
        1.0,  # learning_rate (already scaled in g2/g3)
        epsilon=1e-8,
        kappa=float(args.kappa),
        clipsnr=args.clipsnr,
        delta=args.delta
    )

    return optimizer


def train_with_tau_stats(model, optimizer, key, num_steps, batch_size):
    """Train with dana-star-mk4 optimizer and collect tau statistics.

    Args:
        model: The MoE PLRF model
        optimizer: The dana-star-mk4 optimizer
        key: JAX random key
        num_steps: Number of training steps
        batch_size: Batch size

    Returns:
        Dict with training results and tau statistics
    """
    # Initialize parameters
    init_params = jnp.zeros((model.d, model.m))
    opt_state = optimizer.init(init_params)
    params = init_params

    # Determine tau statistics collection times: powers of 2 times 1000
    # tau_stats_times = [1000, 2000, 4000, 8000, 16000, 32000, ...]
    max_power = int(np.floor(np.log2(num_steps / 1000)))
    tau_stats_times = [0] + [1000 * (2 ** k) for k in range(max_power + 1) if 1000 * (2 ** k) <= num_steps]
    tau_stats_times = sorted(set(tau_stats_times))  # Remove duplicates and sort

    print(f"Tau statistics collection times: {tau_stats_times}")

    # Determine loss evaluation times (logarithmic spacing)
    eval_times = jnp.unique(jnp.concatenate([
        jnp.array([0]),
        jnp.int32(1.1 ** jnp.arange(1, jnp.ceil(jnp.log(num_steps) / jnp.log(1.1)))),
        jnp.array([num_steps])
    ]))

    # Batch loss function
    @jax.jit
    def batch_loss_moe(params_single, X, y, expert_indices):
        """Compute mean squared error loss with expert routing."""
        R = model.create_routing_matrix(expert_indices, batch_size)
        all_predictions = jnp.matmul(X, params_single)  # (batch_size, m)
        predictions = jnp.sum(all_predictions * R.T, axis=1)  # (batch_size,)
        return jnp.mean(optax.l2_loss(predictions, y))

    # Gradient computation
    @jax.jit
    def compute_gradient(params, X, y, expert_indices):
        """Compute gradients."""
        return jax.grad(batch_loss_moe)(params, X, y, expert_indices)

    # Training step
    @jax.jit
    def training_step(params, opt_state, X, y, expert_indices):
        """Single training step."""
        grads = compute_gradient(params, X, y, expert_indices)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # Initialize results tracking
    results = {
        'timestamps': [0],
        'losses': [model.population_risk(init_params)],
        'tau_statistics': {
            'timestamps': [],
            'tau_stats': []
        }
    }

    # Collect initial tau statistics
    initial_tau_stats = extract_tau_statistics(opt_state)
    if initial_tau_stats:
        results['tau_statistics']['timestamps'].append(0)
        results['tau_statistics']['tau_stats'].append(initial_tau_stats)

    eval_idx = 1
    next_eval = eval_times[eval_idx] if eval_idx < len(eval_times) else num_steps + 1

    tau_stats_idx = 1
    next_tau_stats = tau_stats_times[tau_stats_idx] if tau_stats_idx < len(tau_stats_times) else num_steps + 1

    # Training loop
    for step in tqdm(range(num_steps), desc="Training"):
        # Generate batch
        key, key_data, key_expert = random.split(key, 3)
        X, y = model.generate_batch(key_data, batch_size)
        expert_indices = model.sample_expert_batch(key_expert, batch_size)

        # Training step
        params, opt_state = training_step(params, opt_state, X, y, expert_indices)

        # Evaluate loss if needed
        if step + 1 == next_eval:
            pop_risk = model.population_risk(params)
            results['timestamps'].append(step + 1)
            results['losses'].append(pop_risk)

            eval_idx += 1
            if eval_idx < len(eval_times):
                next_eval = eval_times[eval_idx]
            else:
                next_eval = num_steps + 1

        # Collect tau statistics if needed
        if step + 1 == next_tau_stats:
            tau_stats = extract_tau_statistics(opt_state)
            if tau_stats:
                results['tau_statistics']['timestamps'].append(step + 1)
                results['tau_statistics']['tau_stats'].append(tau_stats)

            tau_stats_idx += 1
            if tau_stats_idx < len(tau_stats_times):
                next_tau_stats = tau_stats_times[tau_stats_idx]
            else:
                next_tau_stats = num_steps + 1

    # Convert lists to arrays
    results['timestamps'] = jnp.array(results['timestamps'])
    results['losses'] = jnp.array(results['losses'])

    return results


def save_results(results, args, traceK):
    """Save tau statistics results to disk with all metadata.

    Args:
        results: Dict with training results and tau statistics
        args: Command line arguments
        traceK: Trace of kernel matrix
    """
    # Create output directory
    os.makedirs(args.data_dir, exist_ok=True)

    # Create filename based on configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.output_prefix}_alpha{args.alpha}_m{args.m}_zeta{args.zeta}_beta{args.beta}_sigma{args.sigma}_kappa{args.kappa}_steps{args.steps}_{timestamp}"

    # Prepare metadata
    metadata = {
        'timestamp': timestamp,
        'optimizer': 'dana-star-mk4',
        'model_params': {
            'alpha': args.alpha,
            'beta': args.beta,
            'v': args.v,
            'd': args.d,
            'm': args.m,
            'zeta': args.zeta,
            'student_t_dof': args.student_t_dof,
            'sigma': args.sigma,
        },
        'optimizer_params': {
            'kappa': args.kappa,
            'clipsnr': args.clipsnr,
            'delta': args.delta,
            'g2_scale': args.g2_scale,
            'g3_over_g2': args.g3_over_g2,
            'tanea_lr_scalar': args.tanea_lr_scalar,
        },
        'training_params': {
            'steps': args.steps,
            'batch_size': args.batch_size,
        },
        'traceK': float(traceK),
        'random_seed': args.random_seed,
    }

    # Convert JAX arrays to numpy for serialization
    results_numpy = {
        'timestamps': np.array(results['timestamps']),
        'losses': np.array(results['losses']),
        'tau_statistics': {
            'timestamps': results['tau_statistics']['timestamps'],
            'tau_stats': results['tau_statistics']['tau_stats']
        }
    }

    # Save data
    data = {
        'metadata': metadata,
        'results': results_numpy
    }

    # Save as pickle
    pickle_path = os.path.join(args.data_dir, f"{filename}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved tau statistics data to: {pickle_path}")

    # Also save metadata as JSON for easy inspection
    json_path = os.path.join(args.data_dir, f"{filename}_metadata.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {json_path}")


def main():
    """Main function."""
    args = parse_args()

    # Set random seed
    key = random.PRNGKey(args.random_seed)

    # Compute traceK for hyperparameter scaling
    traceK = get_traceK(args.alpha, args.v)

    # Create model
    key, model_key = random.split(key)
    model = LabelNoiseMixtureOfExpertsPLRF(
        alpha=args.alpha,
        beta=args.beta,
        v=args.v,
        d=args.d,
        m=args.m,
        zeta=args.zeta,
        student_t_dof=args.student_t_dof,
        sigma=args.sigma,
        key=model_key
    )

    print(f"Model created:")
    print(f"  Expert probabilities: {model.expert_probs}")
    print(f"  Optimal risk: {model.population_risk(model.optimal_params_per_expert()):.6f}")
    print(f"  traceK: {traceK:.6f}")

    # Create optimizer
    optimizer = create_optimizer(args, traceK)
    print(f"\nCreated optimizer: dana-star-mk4 (kappa={args.kappa})")

    # Train with tau statistics collection
    print(f"\nTraining for {args.steps} steps...")
    key, train_key = random.split(key)
    results = train_with_tau_stats(
        model,
        optimizer,
        train_key,
        args.steps,
        args.batch_size
    )

    # Save results
    print("\nSaving tau statistics data...")
    save_results(results, args, traceK)

    print("\nDone!")


if __name__ == "__main__":
    main()
