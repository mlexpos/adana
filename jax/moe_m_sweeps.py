#!/usr/bin/env python
"""
MoE PLRF Training with sweeps over the value of m, comparing dana_star vs dana_star_mk4.

This script trains Mixture of Experts PLRF models using different optimizers
(Tanea, TarMSProp-SGD, Adam) and collects tau statistics from TaneaOptimizer.
It generates one main plot:
1. Learning curves comparison across optimizers and different m values
"""
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import NamedTuple, Callable, Dict, List, Union, Optional, Tuple
import numpy as np
import argparse
import os

from optimizers import powerlaw_schedule, GalaxyOptimizerState, get_adam_star, get_adam_nesterov_star, get_dana_star, get_dana_star_mk4, get_long_adam, get_long_adam_nesterov

# Import MoE PLRF implementations from installed package
from moe_plrf import (
    TwoExpertPLRF,
    MoEPLRFTrainer,
    MixtureOfExpertsPLRF
)
# from power_law_rf.moe_plrf.moe_plrf_ode import (
#     ode_moe_dana_log_implicit,
#     MoEODEInputs
# )
# from power_law_rf.ode import DanaHparams

class TaneaHparams(NamedTuple):
    g2: Callable[[Union[float, jnp.ndarray]], float]  # learning rate function
    g3: Callable[[Union[float, jnp.ndarray]], float]  # learning rate function
    delta: Callable[[Union[float, jnp.ndarray]], float]  # momentum function

# import power_law_rf.deterministic_equivalent as theory

def parse_args():
    """Parse command line arguments for the label noise experiment."""
    parser = argparse.ArgumentParser(description="Label Noise Experiment for Momentum Strategy Comparison")
    
    # Model parameters
    parser.add_argument("--alpha", type=float, default=1.0, help="Power law exponent for eigenvalue decay")
    parser.add_argument("--beta", type=float, default=0.0, help="Beta value (power law exponent for target coefficient decay)")
    parser.add_argument("--v", type=int, default=2000, help="Hidden dimension (number of random features)")
    parser.add_argument("--d", type=int, default=500, help="Embedded dimension (parameter dimension)")
    
    # MoE parameters
    parser.add_argument("--m_range", type=str, default="50,100,200", help="Comma-separated list of m values (number of experts)")
    parser.add_argument("--zeta", type=float, default=0.5, help="Single zeta value (power-law exponent for expert selection: p(i) ∝ i^(-zeta))")
    
    # Training parameters
    parser.add_argument("--steps", type=int, default=125000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=100, help="Training batch size")
    parser.add_argument("--g2_scale", type=float, default=0.2, help="Base learning rate scale")
    parser.add_argument("--g3_over_g2", type=float, default=0.01, help="G3 to G2 ratio for momentum")
    parser.add_argument("--tanea_lr_scalar", type=float, default=1, help="Tanea learning rate scalar") #default=1e-2
    parser.add_argument("--tanea_global_exponent", type=float, default=0.0, help="Tanea global time exponent")
    parser.add_argument("--tanea_kappa", type=float, default=None, help="Tanea kappa parameter to override powerlaw_schedule exponent")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2 parameter (also used for RMSprop decay in RMSprop+Dana)")
    parser.add_argument("--adam_beta1", type=float, default=0.99, help="Adam beta1 parameter (also used for RMSprop decay in RMSprop+Dana)")
    parser.add_argument("--adam_lr", type=float, default=0.2, help="Adam learning rate scale")
    parser.add_argument("--adam_star_lr", type=float, default=0.2, help="Adam-star learning rate scale")
    parser.add_argument("--long_adam_lr", type=float, default=0.2, help="Long Adam learning rate scale")

    
  
    # Label noise parameters
    parser.add_argument("--student_t_dof", type=float, default=3.0, help="Degrees of freedom for student-t distribution")
    parser.add_argument("--sigma", type=float, default=0.1, help="Scaling factor for the noise")
    parser.add_argument("--enable_adam", action="store_true", default=True, help="Enable Adam optimizer")
    parser.add_argument("--disable_adam", action="store_true", help="Disable Adam optimizer")
    
    parser.add_argument("--enable_long_adam", action="store_true", default=True, help="Enable Long Adam optimizer")
    parser.add_argument("--disable_long_adam", action="store_true", help="Disable Long Adam optimizer")
    parser.add_argument("--enable_long_adam_nesterov", action="store_true", default=True, help="Enable Long Adam Nesterov optimizer")
    parser.add_argument("--disable_long_adam_nesterov", action="store_true", help="Disable Long Adam Nesterov optimizer")

    parser.add_argument("--enable_adam_star", action="store_true", default=True, help="Enable Adam Star optimizer")
    parser.add_argument("--disable_adam_star", action="store_true", help="Disable Adam Star optimizer")
    parser.add_argument("--enable_adam_nesterov_star", action="store_true", default=True, help="Enable Adam Nesterov Star optimizer")
    parser.add_argument("--disable_adam_nesterov_star", action="store_true", help="Disable Adam Nesterov Star optimizer")
    
    parser.add_argument("--enable_dana_star", action="store_true", default=True, help="Enable DANA Star optimizer")
    parser.add_argument("--disable_dana_star", action="store_true", help="Disable DANA Star optimizer")
    
    parser.add_argument("--enable_dana_star_mk4", action="store_true", default=True, help="Enable DANA Star MK4 optimizer")
    parser.add_argument("--disable_dana_star_mk4", action="store_true", help="Disable DANA Star MK4 optimizer")

    
    # Output parameters
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store results")
    parser.add_argument("--output_prefix", type=str, default="label_noise", help="Prefix for output files")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser.parse_args()


# Set random seed (will be overridden by command line args)
key = random.PRNGKey(42)


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
        return jnp.array([]), jnp.array([])
    
    # Sort in descending order for largest stats
    sorted_tau_desc = jnp.sort(tau_vector)[::-1]
    
    # Compute powers of 1.1 up to n, similar to evaluation times
    max_k = jnp.ceil(jnp.log(n) / jnp.log(1.1)).astype(jnp.int32)
    indices = jnp.int32(1.1 ** jnp.arange(max_k + 1)) - 1  # 0-indexed: [0, 0, 1, 2, 3, 4, ...]
    
    # Remove duplicates and clamp to valid range
    indices = jnp.unique(indices)
    indices = jnp.minimum(indices, n - 1)
    
    # Get largest order statistics (same as before)
    largest_order_stats = sorted_tau_desc[indices]
    
    # Get smallest order statistics using reversed indices
    # For smallest: indices from the end of the sorted array
    reversed_indices = (n - 1) - indices
    smallest_order_stats = sorted_tau_desc[reversed_indices]
    
    return largest_order_stats, smallest_order_stats

def extract_tau_statistics(opt_state):
    """Extract tau statistics from TaneaOptimizerState.
    
    Args:
        opt_state: TaneaOptimizerState containing tau tree
        
    Returns:
        Dictionary with tau statistics including both largest and smallest order statistics
    """
    # if not isinstance(opt_state, GalaxyOptimizerState):
    #     return {}
    opt_state = opt_state[0]
    if not isinstance(opt_state, GalaxyOptimizerState):
        return {}
    
    # Flatten tau tree into a single vector

    tau_leaves = jax.tree_util.tree_leaves(opt_state.tau)
    tau_vector = jnp.concatenate([jnp.ravel(leaf) for leaf in tau_leaves])
    
    # Compute order statistics (now returns both largest and smallest)
    order_stats, reversed_order_stats = compute_tau_order_statistics(tau_vector)
    
    return {
        'tau_order_statistics': order_stats,
        'tau_reversed_order_statistics': reversed_order_stats,
        'tau_mean': jnp.mean(tau_vector),
        'tau_std': jnp.std(tau_vector),
        'tau_min': jnp.min(tau_vector),
        'tau_max': jnp.max(tau_vector)
    }


class TauTrackingLabelNoiseMoEPLRFTrainer(MoEPLRFTrainer):
    """Custom MoEPLRFTrainer that tracks tau statistics during training with label noise.
    
    Extends the base MoEPLRFTrainer to collect tau statistics from TaneaOptimizer
    at evaluation steps during training.
    """
    
    def train(self,
              key: random.PRNGKey,
              num_steps: int,
              batch_size: int,
              init_params: Optional[jnp.ndarray] = None,
              eval_freq: Optional[int] = None,
              track_per_expert_loss: bool = False,
              track_update_history: bool = False,
              track_tau_stats: bool = True) -> Dict:
        """Train the MoE model and return training metrics including tau statistics.
        
        Args:
            Same as parent class, plus:
            track_tau_stats: If True, collect tau statistics during eval steps
            
        Returns:
            Same as parent class, plus:
                - tau_statistics: Dictionary with tau stats at each eval step
        """
        if not track_tau_stats:
            return super().train(
                key=key,
                num_steps=num_steps,
                batch_size=batch_size,
                init_params=init_params,
                eval_freq=eval_freq,
                track_per_expert_loss=track_per_expert_loss,
                track_update_history=track_update_history
            )
        
        # Initialize parameters for all experts
        if init_params is None:
            init_params = jnp.zeros((self.model.d, self.model.m))

        params = init_params
        opt_state = self.optimizer.init(params)

        # Determine evaluation times
        if eval_freq is None:
            eval_times = jnp.unique(jnp.concatenate([
                jnp.array([0]),
                jnp.int32(1.1 ** jnp.arange(1, jnp.ceil(jnp.log(num_steps) / jnp.log(1.1)))),
                jnp.array([num_steps])
            ]))
        else:
            eval_times = jnp.arange(0, num_steps + 1, eval_freq)

        # Batch loss function for MoE
        @jax.jit
        def batch_loss_moe(params, X, y, expert_indices):
            """Compute mean squared error loss with expert routing."""
            # Create routing matrix
            R = self.model.create_routing_matrix(expert_indices, batch_size)

            # Compute predictions for all experts
            all_predictions = jnp.matmul(X, params)  # (batch_size, m)

            # Select predictions based on routing
            predictions = jnp.sum(all_predictions * R.T, axis=1)  # (batch_size,)

            # Compute loss
            return jnp.mean(optax.l2_loss(predictions, y))

        # Gradient computation for MoE
        @jax.jit
        def compute_moe_gradients(params, X, y, expert_indices):
            """Compute gradients for MoE model."""
            def loss_fn(params):
                return batch_loss_moe(params, X, y, expert_indices)
            
            return jax.grad(loss_fn)(params)

        # Training step
        @jax.jit
        def train_step(params, opt_state, key):
            """Single training step for MoE."""
            # Split key for data generation
            key_data, key_expert = random.split(key)
            
            # Generate batch with label noise
            X, y = self.model.generate_batch(key_data, batch_size)
            
            # Sample expert indices for this batch
            expert_indices = self.model.sample_expert_batch(key_expert, batch_size)
            
            # Compute gradients
            grads = compute_moe_gradients(params, X, y, expert_indices)
            
            # Update parameters
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            return params, opt_state

        # Training loop with progress bar
        losses = [self.model.population_risk(init_params)]
        timestamps = [0]
        
        # Initialize tau statistics tracking
        tau_statistics = {
            'timestamps': [0],
            'tau_order_statistics': [extract_tau_statistics(opt_state).get('tau_order_statistics', jnp.array([]))],
            'tau_reversed_order_statistics': [extract_tau_statistics(opt_state).get('tau_reversed_order_statistics', jnp.array([]))],
            'tau_mean': [extract_tau_statistics(opt_state).get('tau_mean', 0.0)],
            'tau_std': [extract_tau_statistics(opt_state).get('tau_std', 0.0)],
            'tau_min': [extract_tau_statistics(opt_state).get('tau_min', 0.0)],
            'tau_max': [extract_tau_statistics(opt_state).get('tau_max', 0.0)]
        }
        
        # Initialize per-expert loss tracking if requested
        per_expert_losses = None
        if track_per_expert_loss:
            # Use vectorized per-expert risk computation and store as arrays
            initial_per_expert_risks = self.model.per_expert_population_risk(init_params)
            per_expert_losses = [initial_per_expert_risks]

        eval_idx = 1
        next_eval = eval_times[eval_idx] if eval_idx < len(eval_times) else num_steps + 1

        for step in tqdm(range(num_steps)):
            # Split key for this step
            key, subkey = random.split(key)

            # Perform training step
            params, opt_state = train_step(params, opt_state, subkey)

            # Evaluate if needed
            if step + 1 == next_eval:
                pop_risk = self.model.population_risk(params)
                losses.append(pop_risk)
                timestamps.append(step + 1)
                
                # Extract tau statistics
                tau_stats = extract_tau_statistics(opt_state)
                tau_statistics['timestamps'].append(step + 1)
                tau_statistics['tau_order_statistics'].append(tau_stats.get('tau_order_statistics', jnp.array([])))
                tau_statistics['tau_reversed_order_statistics'].append(tau_stats.get('tau_reversed_order_statistics', jnp.array([])))
                tau_statistics['tau_mean'].append(tau_stats.get('tau_mean', 0.0))
                tau_statistics['tau_std'].append(tau_stats.get('tau_std', 0.0))
                tau_statistics['tau_min'].append(tau_stats.get('tau_min', 0.0))
                tau_statistics['tau_max'].append(tau_stats.get('tau_max', 0.0))
                
                # Track per-expert losses if requested
                if track_per_expert_loss and per_expert_losses is not None:
                    # Use vectorized per-expert risk computation
                    current_per_expert_risks = self.model.per_expert_population_risk(params)
                    per_expert_losses.append(current_per_expert_risks)

                eval_idx += 1
                if eval_idx < len(eval_times):
                    next_eval = eval_times[eval_idx]
                else:
                    next_eval = num_steps + 1

        # Prepare results
        results = {
            'timestamps': jnp.array(timestamps),
            'losses': jnp.array(losses),
            'tau_statistics': tau_statistics
        }
        
        # Add per-expert losses if tracked
        if track_per_expert_loss and per_expert_losses is not None:
            # Convert list of arrays to a single array of shape (n_eval_times, m)
            results['per_expert_losses'] = jnp.array(per_expert_losses)
        
        return results


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
        # Generate student-t random variables
        noise = random.t(key_noise, df=self.student_t_dof, shape=(batch_size,))
        # Scale by sigma
        y_noisy = y_clean + self.sigma * noise

        return X, y_noisy


def get_traceK(alpha, v):
  x_grid = jnp.arange(1, v+1).reshape(1, v)
  population_eigs = x_grid ** -alpha
  population_trace = jnp.sum(population_eigs**2)
  return population_trace

def get_tanea_hparams(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent, tanea_kappa=None):
    """Get Tanea hyperparameters."""
    kappa_b = jnp.log(batch_size) / jnp.log(d)  # exponent for batch wrt d
    learning_rate = g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)
    
    # Use tanea_kappa if provided, otherwise use default exponent
    g3_exponent = -1.0*tanea_kappa if tanea_kappa is not None else -tanea_global_exponent-(1.0 - kappa_b) / (2 * alpha)
    
    tanea_params = TaneaHparams(
        g2=powerlaw_schedule(tanea_lr_scalar*learning_rate, 0.0, -tanea_global_exponent, 1.0),
        g3=powerlaw_schedule(tanea_lr_scalar*learning_rate*g3_over_g2, 0.0, g3_exponent, 1.0),
        delta=powerlaw_schedule(1.0, 0.0, -1.0, 4.0+2*(alpha+beta)/(2*alpha))
    )
    return tanea_params


def get_tanea_kappa1_hparams(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent, tanea_kappa=None):
    """Get Tanea kappa1 hyperparameters (same as tanea for now)."""
    return get_tanea_hparams(alpha, beta, d, batch_size, g2_scale, 1.0, traceK, tanea_lr_scalar, tanea_global_exponent, 1.0)

def get_tarmsprop_sgd_hparams(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent):
  kappa_b = jnp.log(batch_size) / jnp.log(d)  # exponent for batch wrt d
  learning_rate = g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)
  tanea_params = TaneaHparams(
    g2=powerlaw_schedule(tanea_lr_scalar*learning_rate, 0.0, -tanea_global_exponent, 1.0),
    g3=powerlaw_schedule(0.0, 0.0, -tanea_global_exponent-(1.0 - kappa_b) / (2 * alpha), 1.0),
    delta=powerlaw_schedule(1.0, 0.0, -1.0, 4.0+2*(alpha+beta)/(2*alpha))
  )
  return tanea_params

def get_adam_lr(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent, adam_lr):
   return jnp.minimum(1.0, jnp.float32(batch_size) / traceK) * adam_lr

def get_Long_adam_optimizer(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent, long_adam_lr):
    """Get Long-Adam optimizer with hyperparameters based on Tanea/Adam settings."""
    #adam_lr = get_adam_lr(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent)
    long_adam_opt = get_long_adam(long_adam_lr * jnp.minimum(1.0, jnp.float32(batch_size) / traceK))
        
    return long_adam_opt

def get_long_adam_nesterov_optimizer(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent):
    """Get Long-Adam-Nesterov optimizer with hyperparameters based on Tanea/Adam settings."""
    # Get Adam LR for Dana g2
    adam_lr_val = jnp.minimum(1.0, jnp.float32(batch_size) / traceK) * 0.2  # Use default adam_lr of 0.2
    long_adam_nesterov_opt = get_long_adam_nesterov(adam_lr_val)
   
    return long_adam_nesterov_opt

def get_adam_star_lr(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent, adam_star_lr):
    """Get Adam Star learning rate same as Adam"""
    return adam_star_lr * jnp.minimum(1.0, jnp.float32(batch_size) / traceK) #get_adam_lr(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent)

def get_dana_star_optimizer(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent, tanea_kappa=None):
    """Get Dana Star optimizer"""
    kappa_b = jnp.log(batch_size) / jnp.log(d)  # exponent for batch wrt d
    learning_rate_g2 = g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)
    g2_constant = tanea_lr_scalar *learning_rate_g2
    g3_constant = g3_over_g2 * learning_rate_g2 * tanea_lr_scalar
    kappa_exponent = 1.0*tanea_kappa if tanea_kappa is not None else tanea_global_exponent+(1.0 - kappa_b) / (2 * alpha)
    optimizer = get_dana_star(g2_constant, g3_constant, kappa_exponent, 1.0) # This is on the g2 schedule
    return optimizer

def get_dana_star_mk4_optimizer(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent, tanea_kappa=None):
    """Get Dana Star MK4 optimizer"""
    kappa_b = jnp.log(batch_size) / jnp.log(d)  # exponent for batch wrt d
    learning_rate_g2 = g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)
    g2_constant = tanea_lr_scalar *learning_rate_g2
    g3_constant = g3_over_g2 * learning_rate_g2 * tanea_lr_scalar
    optimizer = get_dana_star_mk4(g2_constant, 0.25, 1.0) # This is on the g2 schedule
    return optimizer

def get_adam_nesterov_star_lr(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent):
    """Get Adam Nesterov Star learning rate same as Adam"""
    return jnp.minimum(1.0, jnp.float32(batch_size) / traceK) * 0.2  # Use default adam_lr of 0.2


def main():
    """Main function to run the label noise experiment."""
    args = parse_args()
    
    # Override global random seed
    global key
    key = random.PRNGKey(args.random_seed)
    
    # Parse m_range list from string, beta and zeta are single values
    beta = args.beta
    zeta = args.zeta
    m_list = [int(m.strip()) for m in args.m_range.split(',')]
    
    # Process optimizer flags (disable flags override enable flags)
    enable_adam = args.enable_adam and not args.disable_adam
    enable_long_adam = args.enable_long_adam and not args.disable_long_adam
    enable_long_adam_nesterov = args.enable_long_adam_nesterov and not args.disable_long_adam_nesterov
    enable_adam_star = args.enable_adam_star and not args.disable_adam_star
    enable_adam_nesterov_star = args.enable_adam_nesterov_star and not args.disable_adam_nesterov_star
    enable_dana_star = args.enable_dana_star and not args.disable_dana_star
    enable_dana_star_mk4 = args.enable_dana_star_mk4 and not args.disable_dana_star_mk4

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Compute traceK for hyperparameter scaling
    traceK = get_traceK(args.alpha, args.v)

    # Main training experiment loop
    moe_results = []

    for m in m_list:
        print(f"\nRunning MoE experiments for m = {m}, ζ = {zeta}, β = {beta}")

        # Create MoE model with label noise
        key, model_key = random.split(key)
        model = LabelNoiseMixtureOfExpertsPLRF(
            alpha=args.alpha,
            beta=beta,
            v=args.v,
            d=args.d,
            m=m,
            zeta=zeta,
            student_t_dof=args.student_t_dof,
            sigma=args.sigma,
            key=model_key
        )

        print(f"  Expert probabilities: {model.expert_probs}")
        print(f"  Optimal risk: {model.population_risk(model.optimal_params_per_expert()):.6f}")

        # Create hyperparameters
        optimizers_dict = {}
        
        if enable_adam:
            optimizers_dict['adam'] = optax.adam(get_adam_lr(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.adam_lr), b1=args.adam_beta1, b2=args.adam_beta2)

        if enable_long_adam:
            optimizers_dict['tanea_long_adam'] = get_Long_adam_optimizer(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.long_adam_lr)

        if enable_long_adam_nesterov:
            optimizers_dict['tanea_long_adam_nesterov'] = get_long_adam_nesterov_optimizer(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent)

        if enable_adam_star:
            optimizers_dict['tanea_adam_star'] = get_adam_star(get_adam_star_lr(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.adam_star_lr))

        if enable_adam_nesterov_star:
            optimizers_dict['tanea_adam_nesterov_star'] = get_adam_nesterov_star(get_adam_nesterov_star_lr(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent))

        if enable_dana_star:
            optimizers_dict['tanea_dana_star'] = get_dana_star_optimizer(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.tanea_kappa)

        if enable_dana_star_mk4:
            optimizers_dict['tanea_dana_star_mk4'] = get_dana_star_mk4_optimizer(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.tanea_kappa)

        # Run training experiments for enabled optimizers
        results_dict = {'beta': beta, 'zeta': zeta, 'm': m, 'model': model}

        for opt_name, optimizer in optimizers_dict.items():
            print(f"  Running {opt_name} experiment...")

            # Use TauTrackingLabelNoiseMoEPLRFTrainer for Tanea-family optimizers
            #if opt_name.startswith('tanea'):
            if 'star' in opt_name:
                trainer = TauTrackingLabelNoiseMoEPLRFTrainer(model, optimizer)
                key, train_key = random.split(key)
                results = trainer.train(
                    train_key,
                    num_steps=args.steps,
                    batch_size=args.batch_size,
                    track_per_expert_loss=False,
                    track_tau_stats=False
                )
            else:
                # Use regular MoEPLRFTrainer for non-Tanea optimizers (Adam, RMSprop+Dana)
                trainer = MoEPLRFTrainer(model, optimizer)
                key, train_key = random.split(key)
                results = trainer.train(
                    train_key,
                    num_steps=args.steps,
                    batch_size=args.batch_size,
                    track_per_expert_loss=False
                )

            results_dict[opt_name] = results

        moe_results.append(results_dict)

    # Skip visualization if --no_plots flag is set
    if not args.no_plots:
        # Create single plot showing training loss curves for all m values and optimizers
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Define colors for algorithms (matching moe_learning_rate_sweeps.py tab colors)
        algorithm_colors = {
            'adam': 'tab:red', #'#1f77b4',              # tab:blue
            'tanea_long_adam': 'tab:orange', #'#2ca02c',   # tab:green
            'tanea_long_adam_nesterov': 'indigo', #'2ca02c',  # tab:green (same as long_adam)
            'tanea_adam_star': 'tab:green', #'#ff7f0e',   # tab:orange
            'tanea_adam_nesterov_star': 'navy', #'#d62728',  # tab:red
            'tanea_dana_star': 'tab:blue', #9467bd'    # tab:purple
            'tanea_dana_star_mk4': 'tab:purple' # New color for mk4
        }

        # Define linestyles for m values
        m_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

        for result in moe_results:
            beta = result['beta']
            zeta = result['zeta']
            m = result['m']
            m_idx = m_list.index(m)

            # Get all available optimizers in this result
            available_optimizers = [k for k in result.keys() if k not in ['beta', 'zeta', 'm', 'model']]

            for optimizer_name in available_optimizers:
                if optimizer_name in result and 'timestamps' in result[optimizer_name] and 'losses' in result[optimizer_name]:
                    timestamps = result[optimizer_name]['timestamps']
                    losses = result[optimizer_name]['losses']

                    if len(timestamps) > 0 and len(losses) > 0:
                        # Create display name
                        if optimizer_name == 'adam':
                            display_name = f'Adam (m={m})'
                        elif optimizer_name == 'tanea_long_adam':
                            display_name = f'Long-Adam (m={m})'
                        elif optimizer_name == 'tanea_long_adam_nesterov':
                            display_name = f'Long-Adam-Nesterov (m={m})'
                        elif optimizer_name == 'tanea_adam_star':
                            display_name = f'Adam-Star (m={m})'
                        elif optimizer_name == 'tanea_adam_nesterov_star':
                            display_name = f'Adam-Nesterov-Star (m={m})'
                        elif optimizer_name == 'tanea_dana_star':
                            display_name = f'Dana-Star (m={m})'
                        elif optimizer_name == 'tanea_dana_star_mk4':
                            display_name = f'Dana-Star-MK4 (m={m})'
                        else:
                            display_name = f'{optimizer_name} (m={m})'

                        ax.loglog(timestamps, losses,
                                linestyle=m_styles[m_idx % len(m_styles)],
                                color=algorithm_colors.get(optimizer_name, '#000000'),
                                alpha=0.8, linewidth=3,
                                label=display_name)

        ax.set_xlabel('Training Iteration', fontsize=14)
        ax.set_ylabel('Population Risk', fontsize=14)
        ax.set_title(f'Learning Curves (Dana-Star vs Dana-Star-MK4)\nα={args.alpha}, ζ={zeta}, d={args.d}, β={beta}, Adam learning rate = {args.adam_lr},\n β_1 = {args.adam_beta1}, β_2 = {args.adam_beta2}, κ = {args.tanea_kappa}, batch={args.batch_size}, steps={args.steps}', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                 handlelength=6, handletextpad=0.8, columnspacing=2)

        plt.tight_layout()

        # Save learning curves PDF
        m_str = "_".join([f"m{m}" for m in m_list])
        curves_filename = f"{args.output_prefix}_mk4_comparison_alpha{args.alpha}_zeta{zeta}_D{args.d}_{m_str}_beta{beta}_steps{args.steps}.pdf"
        curves_filepath = os.path.join(args.results_dir, curves_filename)
        plt.savefig(curves_filepath, dpi=300, bbox_inches='tight')
        print(f"Training loss curves saved to: {curves_filepath}")
        plt.close()


    else:
        print("Skipping plots due to --no_plots flag")


if __name__ == "__main__":
    main()