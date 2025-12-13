#!/usr/bin/env python
"""
Hummingbird Plot: Visualizing optimizer behavior across kappa values.

This script trains a single MoE PLRF model with multiple optimizers simultaneously
(sharing the same gradients) to efficiently compare performance across different
kappa values for Dana-Star-MK4.
"""
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from typing import NamedTuple, Callable, Dict, List, Union, Optional, Tuple
import numpy as np
import argparse
import os

from optimizers import powerlaw_schedule, get_dana_star_mk4, get_dana_mk4, get_dana_star

# Import MoE PLRF implementations
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
    parser = argparse.ArgumentParser(description="Hummingbird Plot: Visualize optimizer behavior across kappa values")

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

    # Adam baseline parameters
    parser.add_argument("--adam_lr", type=float, default=0.2, help="Adam learning rate scale")
    parser.add_argument("--adam_beta1", type=float, default=0.99, help="Adam beta1 parameter")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2 parameter")

    # Label noise parameters
    parser.add_argument("--student_t_dof", type=float, default=3.0, help="Degrees of freedom for student-t distribution")
    parser.add_argument("--sigma", type=float, default=0.1, help="Scaling factor for the noise")

    # Output parameters
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store results")
    parser.add_argument("--output_prefix", type=str, default="hummingbird", help="Prefix for output files")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


def get_traceK(alpha, v):
    """Compute trace of the kernel matrix."""
    x_grid = jnp.arange(1, v+1).reshape(1, v)
    population_eigs = x_grid ** -alpha
    population_trace = jnp.sum(population_eigs**2)
    return population_trace


def create_optimizers(args, traceK):
    """Create all optimizers for the hummingbird plot.

    Returns:
        Dict mapping optimizer names to optimizer objects
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
        raise NotImplementedError("ademamix optimizer not yet implemented")

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


def plot_hummingbird(results, kappa_values, args):
    """Create the hummingbird plot.

    Args:
        results: Dict mapping optimizer name to training results
        kappa_values: Array of kappa values used
        args: Command line arguments
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Get viridis colormap
    viridis = cm.get_cmap('viridis', len(kappa_values))

    # Plot kappa sweep curves
    for i, kappa in enumerate(kappa_values):
        opt_name = f"{args.optimizer}_kappa{kappa:.1f}"
        if opt_name in results:
            timestamps = results[opt_name]['timestamps']
            losses = results[opt_name]['losses']

            ax.loglog(timestamps, losses,
                     color=viridis(i),
                     alpha=0.8,
                     linewidth=2,
                     label=f'κ={kappa:.1f}')

    # Plot Adam baseline in black
    if 'adam' in results:
        timestamps = results['adam']['timestamps']
        losses = results['adam']['losses']
        ax.loglog(timestamps, losses,
                 color='black',
                 alpha=0.9,
                 linewidth=3,
                 linestyle='--',
                 label='Adam (baseline)')

    ax.set_xlabel('Training Iteration', fontsize=14)
    ax.set_ylabel('Population Risk', fontsize=14)

    title = f'Hummingbird Plot: {args.optimizer}\n'
    title += f'α={args.alpha}, m={args.m}, ζ={args.zeta}, d={args.d}, β={args.beta}\n'
    title += f'batch={args.batch_size}, steps={args.steps}, clipsnr={args.clipsnr}, δ={args.delta}'
    ax.set_title(title, fontsize=14)

    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()

    # Save plot
    filename = f"{args.output_prefix}_{args.optimizer}_alpha{args.alpha}_m{args.m}_zeta{args.zeta}_beta{args.beta}_sigma{args.sigma}_steps{args.steps}.pdf"
    filepath = os.path.join(args.results_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nHummingbird plot saved to: {filepath}")
    plt.close()


def main():
    """Main function."""
    args = parse_args()

    # Set random seed
    key = random.PRNGKey(args.random_seed)

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

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

    # Create optimizers
    optimizers, kappa_values = create_optimizers(args, traceK)
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

    # Create hummingbird plot
    print("\nGenerating hummingbird plot...")
    plot_hummingbird(results, kappa_values, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
