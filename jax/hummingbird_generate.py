#!/usr/bin/env python
"""
Hummingbird Data Generation: Train optimizers and save results.

This script trains a single MoE PLRF model with multiple optimizers simultaneously
(sharing the same gradients) to efficiently compare performance across different
kappa values. Results are saved in a format that records all relevant command line
details and kappa information.
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

from optimizers import powerlaw_schedule, get_dana_star_mk4, get_dana_mk4, get_dana_star
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hummingbird Data Generation: Generate optimizer training data")

    # Model parameters
    parser.add_argument("--alpha", type=float, default=1.0, help="Power law exponent for eigenvalue decay")
    parser.add_argument("--beta", type=float, default=0.0, help="Beta value (power law exponent for target coefficient decay)")
    parser.add_argument("--v", type=int, default=2000, help="Hidden dimension (number of random features)")
    parser.add_argument("--d", type=int, default=500, help="Embedded dimension (parameter dimension)")
    parser.add_argument("--m", type=int, default=100, help="Number of experts")
    parser.add_argument("--zeta", type=float, default=0.5, help="Power-law exponent for expert selection")

    # Training parameters
    parser.add_argument("--steps", type=int, default=125000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=100, help="Training batch size")
    parser.add_argument("--g2_scale", type=float, default=0.5, help="Base learning rate scale")
    parser.add_argument("--g3_over_g2", type=float, default=1.0, help="G3 to G2 ratio for momentum")
    parser.add_argument("--tanea_lr_scalar", type=float, default=1.0, help="Learning rate scalar")

    # Optimizer selection
    parser.add_argument("--optimizer", type=str, default="dana-star-mk4",
                       choices=["ademamix", "dana-mk4", "dana-star-mk4", "dana-star"],
                       help="Optimizer to use for kappa sweep")

    # Kappa sweep parameters
    parser.add_argument("--kappa_min", type=float, default=0.0, help="Minimum kappa value")
    parser.add_argument("--kappa_max", type=float, default=1.0, help="Maximum kappa value")
    parser.add_argument("--kappa_step", type=float, default=0.1, help="Kappa step size")
    parser.add_argument("--clipsnr", type=float, default=2.0, help="Clipsnr parameter for Dana-Star-MK4")
    parser.add_argument("--delta", type=float, default=8.0, help="Delta parameter for EMA coefficient")

    # AdEMAMix specific parameters
    parser.add_argument("--ademamix_beta1", type=float, default=0.9, help="AdEMAMix beta1 (fast EMA)")
    parser.add_argument("--ademamix_beta2", type=float, default=0.999, help="AdEMAMix beta2 (second moment)")
    parser.add_argument("--ademamix_beta3", type=float, default=0.9999, help="AdEMAMix beta3 (slow EMA)")
    parser.add_argument("--ademamix_alpha", type=float, default=2.0, help="AdEMAMix alpha (slow EMA weight)")

    # Adam baseline parameters
    parser.add_argument("--adam_lr", type=float, default=0.2, help="Adam learning rate scale")
    parser.add_argument("--adam_beta1", type=float, default=0.99, help="Adam beta1 parameter")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2 parameter")

    # Label noise parameters
    parser.add_argument("--student_t_dof", type=float, default=3.0, help="Degrees of freedom for student-t distribution")
    parser.add_argument("--sigma", type=float, default=0.1, help="Scaling factor for the noise")

    # Output parameters
    parser.add_argument("--data_dir", type=str, default="hummingbird_data", help="Directory to store training data")
    parser.add_argument("--output_prefix", type=str, default="hummingbird", help="Prefix for output files")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


def get_traceK(alpha, v):
    """Compute trace of the kernel matrix."""
    x_grid = jnp.arange(1, v+1).reshape(1, v)
    population_eigs = x_grid ** -alpha
    population_trace = jnp.sum(population_eigs**2)
    return population_trace


def make_alpha_warmup_schedule(alpha_final, num_steps):
    """Create linear warmup schedule for alpha (matching PyTorch AdEMAMix).

    Linearly warms up from 0 to alpha_final over num_steps.

    Args:
        alpha_final: Final value of alpha after warmup
        num_steps: Number of steps for warmup

    Returns:
        Callable schedule function that takes step and returns alpha value
    """
    def schedule(step):
        progress = jnp.minimum(step / num_steps, 1.0)
        return progress * alpha_final
    return schedule


def make_beta3_warmup_schedule(beta1, beta3_final, num_steps):
    """Create half-life warmup schedule for beta3 (matching PyTorch AdEMAMix).

    Warms up from beta1 to beta3_final using a half-life based interpolation.
    This matches the PyTorch implementation's linear_hl_warmup_scheduler.

    Args:
        beta1: Starting beta value (typically 0.9)
        beta3_final: Final beta3 value after warmup (typically 0.9999)
        num_steps: Number of steps for warmup

    Returns:
        Callable schedule function that takes step and returns beta3 value
    """
    def beta_to_half_life(beta, eps=1e-8):
        """Convert beta to half-life: hl = log(0.5) / log(beta) - 1"""
        return jnp.log(0.5) / jnp.log(beta + eps) - 1.0

    def half_life_to_beta(hl):
        """Convert half-life to beta: beta = 0.5^(1/(hl+1))"""
        return jnp.power(0.5, 1.0 / (hl + 1.0))

    def schedule(step):
        progress = jnp.minimum(step / num_steps, 1.0)
        hl_start = beta_to_half_life(beta1)
        hl_end = beta_to_half_life(beta3_final)
        hl = (1.0 - progress) * hl_start + progress * hl_end
        return half_life_to_beta(hl)

    return schedule


def create_optimizers(args, traceK, num_steps):
    """Create all optimizers for the hummingbird plot.

    Args:
        args: Command line arguments
        traceK: Trace of the kernel matrix
        num_steps: Total number of training steps (for schedules)

    Returns:
        Dict mapping optimizer names to optimizer objects, and array of kappa values
    """
    optimizers = {}

    # Compute learning rate scaling
    learning_rate_g2 = args.g2_scale * jnp.minimum(1.0, jnp.float32(args.batch_size) / traceK)
    g2_constant = args.tanea_lr_scalar * learning_rate_g2
    g3_constant = args.g3_over_g2 * learning_rate_g2 * args.tanea_lr_scalar

    # Create kappa sweep for selected optimizer
    kappa_values = np.arange(args.kappa_min, args.kappa_max + args.kappa_step/2, args.kappa_step)

    if args.optimizer == "dana-star-mk4":
        for kappa in kappa_values:
            opt_name = f"dana-star-mk4_kappa{kappa:.1f}"
            optimizers[opt_name] = get_dana_star_mk4(
                g2_constant,
                g3_constant,
                1.0,  # learning_rate (already scaled in g2/g3)
                epsilon=1e-8,
                kappa=float(kappa),
                clipsnr=args.clipsnr,
                delta=args.delta
            )
    elif args.optimizer == "dana-mk4":
        for kappa in kappa_values:
            opt_name = f"dana-mk4_kappa{kappa:.1f}"
            optimizers[opt_name] = get_dana_mk4(
                g2_constant,
                g3_constant,
                1.0,  # learning_rate (already scaled in g2/g3)
                epsilon=1e-8,
                kappa=float(kappa),
                clipsnr=args.clipsnr,
                delta=args.delta
            )
    elif args.optimizer == "dana-star":
        for kappa in kappa_values:
            opt_name = f"dana-star_kappa{kappa:.1f}"
            optimizers[opt_name] = get_dana_star(
                g2_constant,
                g3_constant,
                float(kappa),  # kappa_exponent parameter
                1.0,  # learning_rate (already scaled in g2/g3)
                epsilon=1e-8
            )
    elif args.optimizer == "ademamix":
        # Import ademamix optimizer
        from optimizers import get_ademamix

        # Scale learning rate similar to other optimizers
        ademamix_lr = args.g2_scale * jnp.minimum(1.0, jnp.float32(args.batch_size) / traceK)

        for kappa in kappa_values:
            opt_name = f"ademamix_kappa{kappa:.1f}"

            # Calculate alpha and beta3 based on kappa and delta (matching PyTorch implementation)
            # In PyTorch: alpha = iterations^(1 - kappa), beta3 = 1 - delta / iterations
            alpha_final = float(num_steps ** (1.0 - kappa))
            beta3_final = 1.0 - args.delta / num_steps

            # Create warmup schedules (matching PyTorch AdEMAMix warmup behavior)
            alpha_schedule = make_alpha_warmup_schedule(alpha_final, num_steps)
            beta3_schedule = make_beta3_warmup_schedule(args.ademamix_beta1, beta3_final, num_steps)

            optimizers[opt_name] = get_ademamix(
                learning_rate=float(ademamix_lr),
                beta1=args.ademamix_beta1,
                beta2=args.ademamix_beta2,
                beta3=beta3_schedule,  # Warmup from beta1 to beta3_final
                alpha=alpha_schedule,  # Warmup from 0 to alpha_final
                gamma_3_factor=1.0,  # Fixed at 1.0 as in PyTorch config
                epsilon=1e-8
            )

    # Add Adam baseline
    adam_lr = args.adam_lr * jnp.minimum(1.0, jnp.float32(args.batch_size) / traceK)
    optimizers['adam'] = optax.adam(adam_lr, b1=args.adam_beta1, b2=args.adam_beta2)

    return optimizers, kappa_values


def train_multi_optimizer(model, optimizers, key, num_steps, batch_size, eval_freq=None):
    """Train with multiple optimizers simultaneously using the same gradients.

    Uses Optax multi_transform to apply different optimizers to different parameter slices,
    computing gradients only once per step.

    Args:
        model: The MoE PLRF model
        optimizers: Dict of optimizer name -> optimizer object
        key: JAX random key
        num_steps: Number of training steps
        batch_size: Batch size
        eval_freq: Evaluation frequency (if None, uses logarithmic spacing)

    Returns:
        Dict mapping optimizer name to training results
    """
    # Initialize parameters - create a dict tree where each optimizer has its own params
    init_params_single = jnp.zeros((model.d, model.m))

    # Create a parameter tree: {opt_name: params}
    params = {name: init_params_single.copy() for name in optimizers.keys()}

    # Create multi-transform: apply each optimizer to its corresponding parameter slice
    multi_opt = optax.multi_transform(
        transforms=optimizers,
        param_labels={name: name for name in optimizers.keys()}
    )

    # Initialize the combined optimizer state
    opt_state = multi_opt.init(params)

    # Determine evaluation times
    if eval_freq is None:
        eval_times = jnp.unique(jnp.concatenate([
            jnp.array([0]),
            jnp.int32(1.1 ** jnp.arange(1, jnp.ceil(jnp.log(num_steps) / jnp.log(1.1)))),
            jnp.array([num_steps])
        ]))
    else:
        eval_times = jnp.arange(0, num_steps + 1, eval_freq)

    # Batch loss function for a single optimizer's parameters
    @jax.jit
    def batch_loss_moe(params_single, X, y, expert_indices):
        """Compute mean squared error loss with expert routing for single optimizer."""
        R = model.create_routing_matrix(expert_indices, batch_size)
        all_predictions = jnp.matmul(X, params_single)  # (batch_size, m)
        predictions = jnp.sum(all_predictions * R.T, axis=1)  # (batch_size,)
        return jnp.mean(optax.l2_loss(predictions, y))

    # Combined loss function that sums losses across all optimizers
    @jax.jit
    def combined_loss(params_dict, X, y, expert_indices):
        """Compute total loss summed across all optimizer parameter sets."""
        return jax.tree.reduce(
            lambda acc, p: acc + batch_loss_moe(p, X, y, expert_indices),
            params_dict,
            initializer=0.0
        )

    # Gradient computation for the entire parameter tree
    @jax.jit
    def compute_gradients(params_dict, X, y, expert_indices):
        """Compute gradients for all optimizers at once."""
        return jax.grad(combined_loss)(params_dict, X, y, expert_indices)

    # Training step using multi_transform
    @jax.jit
    def training_step(params, opt_state, X, y, expert_indices):
        """Single training step for all optimizers."""
        # Compute gradients once for all optimizers
        grads = compute_gradients(params, X, y, expert_indices)

        # Apply multi-transform update
        updates, opt_state = multi_opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state

    # Initialize results tracking
    results = {
        name: {
            'timestamps': [0],
            'losses': [model.population_risk(init_params_single)]
        }
        for name in optimizers.keys()
    }

    eval_idx = 1
    next_eval = eval_times[eval_idx] if eval_idx < len(eval_times) else num_steps + 1

    # Training loop
    for step in tqdm(range(num_steps), desc="Training"):
        # Generate batch (shared across all optimizers)
        key, key_data, key_expert = random.split(key, 3)
        X, y = model.generate_batch(key_data, batch_size)
        expert_indices = model.sample_expert_batch(key_expert, batch_size)

        # Single update step for all optimizers
        params, opt_state = training_step(params, opt_state, X, y, expert_indices)

        # Evaluate if needed
        if step + 1 == next_eval:
            for name in optimizers.keys():
                pop_risk = model.population_risk(params[name])
                results[name]['timestamps'].append(step + 1)
                results[name]['losses'].append(pop_risk)

            eval_idx += 1
            if eval_idx < len(eval_times):
                next_eval = eval_times[eval_idx]
            else:
                next_eval = num_steps + 1

    # Convert lists to arrays
    for name in results.keys():
        results[name]['timestamps'] = jnp.array(results[name]['timestamps'])
        results[name]['losses'] = jnp.array(results[name]['losses'])

    return results


def save_results(results, kappa_values, args, traceK):
    """Save training results to disk with all metadata.

    Args:
        results: Dict mapping optimizer name to training results
        kappa_values: Array of kappa values used
        args: Command line arguments
        traceK: Trace of kernel matrix
    """
    # Create output directory
    os.makedirs(args.data_dir, exist_ok=True)

    # Create filename based on configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.output_prefix}_{args.optimizer}_alpha{args.alpha}_m{args.m}_zeta{args.zeta}_beta{args.beta}_sigma{args.sigma}_steps{args.steps}_{timestamp}"

    # Prepare metadata
    metadata = {
        'timestamp': timestamp,
        'optimizer': args.optimizer,
        'kappa_values': kappa_values.tolist(),
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
        'training_params': {
            'steps': args.steps,
            'batch_size': args.batch_size,
            'g2_scale': args.g2_scale,
            'g3_over_g2': args.g3_over_g2,
            'tanea_lr_scalar': args.tanea_lr_scalar,
            'clipsnr': args.clipsnr,
            'delta': args.delta,
        },
        'adam_params': {
            'lr': args.adam_lr,
            'beta1': args.adam_beta1,
            'beta2': args.adam_beta2,
        },
        'traceK': float(traceK),
        'random_seed': args.random_seed,
    }

    # Add optimizer-specific parameters
    if args.optimizer == 'ademamix':
        metadata['ademamix_params'] = {
            'beta1': args.ademamix_beta1,
            'beta2': args.ademamix_beta2,
            'beta3': args.ademamix_beta3,
            'alpha': args.ademamix_alpha,
        }

    # Convert JAX arrays to numpy for serialization
    results_numpy = {}
    for name, res in results.items():
        results_numpy[name] = {
            'timestamps': np.array(res['timestamps']),
            'losses': np.array(res['losses'])
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
    print(f"Saved training data to: {pickle_path}")

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

    # Create optimizers
    optimizers, kappa_values = create_optimizers(args, traceK, args.steps)
    print(f"\nCreated {len(optimizers)} optimizers:")
    for name in optimizers.keys():
        print(f"  - {name}")

    # Train with all optimizers simultaneously
    print(f"\nTraining for {args.steps} steps...")
    key, train_key = random.split(key)
    results = train_multi_optimizer(
        model,
        optimizers,
        train_key,
        args.steps,
        args.batch_size
    )

    # Save results
    print("\nSaving training data...")
    save_results(results, kappa_values, args, traceK)

    print("\nDone!")


if __name__ == "__main__":
    main()
