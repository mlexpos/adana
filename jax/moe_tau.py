#!/usr/bin/env python
"""
MoE PLRF Training with Tau Statistics Collection and Visualization.

This script trains Mixture of Experts PLRF models using different optimizers
(Tanea, TarMSProp-SGD, Adam) and collects tau statistics from TaneaOptimizer.
It generates two main plots:
1. Tau order statistics evolution over training
2. Learning curves comparison across optimizers
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

from optimizers import powerlaw_schedule, get_dana_star

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
    parser.add_argument("--beta", type=str, default="-0.3,0.0,0.8", help="Comma-separated list of beta values (power law exponent for target coefficient decay)")
    parser.add_argument("--v", type=int, default=2000, help="Hidden dimension (number of random features)")
    parser.add_argument("--d", type=int, default=500, help="Embedded dimension (parameter dimension)")
    
    # MoE parameters
    parser.add_argument("--m", type=int, default=100, help="Number of experts")
    parser.add_argument("--zeta", type=float, default=0.5, help="Power-law exponent for expert selection (p(i) ∝ i^(-zeta))")
    
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
    parser.add_argument("--adam_lr", type=float, default=1e-2, help="Tanea learning rate scalar") #default=1e-2
    parser.add_argument("--adam_star_lr", type=float, default=1e-2, help="Tanea learning rate scalar") #default=1e-2
    parser.add_argument("--long_adam_lr", type=float, default=1e-2, help="Tanea learning rate scalar") #default=1e-2
    
    
  
    # Label noise parameters
    parser.add_argument("--student_t_dof", type=float, default=3.0, help="Degrees of freedom for student-t distribution")
    parser.add_argument("--sigma", type=float, default=0.1, help="Scaling factor for the noise")
    
    # Optimizer enable/disable flags
    # parser.add_argument("--enable_tanea", action="store_true", default=True, help="Enable Tanea (effective-clip) optimizer")
    # parser.add_argument("--disable_tanea", action="store_true", help="Disable Tanea (effective-clip) optimizer")
    # parser.add_argument("--enable_tanea_theory", action="store_true", help="Enable Tanea (theory) optimizer")
    # parser.add_argument("--enable_tanea_always_on", action="store_true", default=True, help="Enable Tanea (always-on) optimizer")
    # parser.add_argument("--disable_tanea_always_on", action="store_true", help="Disable Tanea (always-on) optimizer")
    # parser.add_argument("--enable_tanea_strong_clip", action="store_true", help="Enable Tanea (strong-clip) optimizer")
    # parser.add_argument("--disable_tanea_strong_clip", action="store_true", help="Disable Tanea (strong-clip) optimizer")
    # parser.add_argument("--enable_tanea_first_moment", action="store_true", help="Enable Tanea (first-moment) optimizer")
    # parser.add_argument("--disable_tanea_first_moment", action="store_true", default=True, help="Disable Tanea (first-moment) optimizer")
    # parser.add_argument("--enable_tanea_mk2", action="store_true", help="Enable Tanea (mk2) optimizer")
    # parser.add_argument("--disable_tanea_mk2", action="store_true", default=True, help="Disable Tanea (mk2) optimizer")
    # parser.add_argument("--enable_tanea_always_on_mk2", action="store_true", help="Enable Tanea (always-on-mk2) optimizer")
    # parser.add_argument("--disable_tanea_always_on_mk2", action="store_true", help="Disable Tanea (always-on-mk2) optimizer")
    # parser.add_argument("--enable_tanea_mk3", action="store_true", default=True, help="Enable Tanea (mk3) optimizer")
    # parser.add_argument("--disable_tanea_mk3", action="store_true", help="Disable Tanea (mk3) optimizer")
    # parser.add_argument("--enable_tanea_kappa1", action="store_true", help="Enable Tanea (kappa1) optimizer")
    # parser.add_argument("--disable_tanea_kappa1", action="store_true", help="Disable Tanea (kappa1) optimizer")
    # parser.add_argument("--enable_tanea_g3zero", action="store_true", default=True, help="Enable Tanea G3=0 (formerly TarMSProp-SGD) optimizer")
    # parser.add_argument("--disable_tanea_g3zero", action="store_true", help="Disable Tanea G3=0 optimizer")
    # parser.add_argument("--enable_rmsprop_dana", action="store_true", default=True, help="Enable RMSprop+Dana optimizer")
    # parser.add_argument("--disable_rmsprop_dana", action="store_true", help="Disable RMSprop+Dana optimizer")
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
    adam_lr = get_adam_lr(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent)
    long_adam_nesterov_opt = get_long_adam_nesterov(adam_lr)
   
    return long_adam_nesterov_opt

def get_adam_star_lr(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent, adam_star_lr):
    """Get Adam Star learning rate same as Adam"""
    return adam_star_lr * jnp.minimum(1.0, jnp.float32(batch_size) / traceK) #get_adam_lr(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent)


def get_dana_star_optimizer(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent, tanea_kappa=None):
    """Get Adam Star learning rate same as Adam"""
    kappa_b = jnp.log(batch_size) / jnp.log(d)  # exponent for batch wrt d
    learning_rate_g2 = g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)
    g2_constant = tanea_lr_scalar *learning_rate_g2
    g3_constant = g3_over_g2 * learning_rate_g2 * tanea_lr_scalar
    kappa_exponent = 1.0*tanea_kappa if tanea_kappa is not None else tanea_global_exponent+(1.0 - kappa_b) / (2 * alpha)
    optimizer = get_dana_star(g2_constant, g3_constant, kappa_exponent, 1.0) # This is on the g2 schedule
    return optimizer

def get_adam_nesterov_star_lr(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent):
    """Get Adam Nesterov Star learning rate same as Adam"""
    return get_adam_lr(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent)


# def get_rmsprop_dana_optimizer(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent, adam_beta2):
#     """Get RMSprop+Dana optimizer with hyperparameters based on Tanea/Adam settings."""
#     # Get Adam LR for Dana g2
#     adam_lr = get_adam_lr(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent)
    
#     # # Get Tanea hyperparameters for kappa (delta parameter)
#     # tanea_hparams = get_tanea_hparams(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent)
    
#     # RMSprop decay (same as Adam beta2)
#     rms_decay = adam_beta2
    
#     # Dana parameters
#     dana_g2 = adam_lr
#     dana_g3 = g3_over_g2 * adam_lr

#     kappa_b = jnp.log(batch_size) / jnp.log(d)  # exponent for batch wrt d
    
#     # Create Dana optimizer schedules
#     g1 = powerlaw_schedule(1.0, 0.0, 0.0, 1)
#     g2 = powerlaw_schedule(dana_g2, 0.0, 0.0, 1)
#     g3 = powerlaw_schedule(dana_g3, 0.0, -1.0*(1.0 - kappa_b) / (2 * alpha), 1)
#     Delta = powerlaw_schedule(1.0, 0.0, -1.0, 4.0+2*(alpha+beta)/(2*alpha))
    
#     # Create Dana optimizer
#     dana_opt = dana_optimizer(g1=g1, g2=g2, g3=g3, Delta=Delta)
    
#     # Chain RMSProp and Dana optimizers
#     optimizer = optax.chain(
#         optax.scale_by_rms(decay=rms_decay, eps=1e-8, bias_correction=True),
#         dana_opt
#     )
    
#     return optimizer


def main():
    """Main function to run the label noise experiment."""
    args = parse_args()
    
    # Override global random seed
    global key
    key = random.PRNGKey(args.random_seed)
    
    # Parse beta list from string
    beta_list = [float(b.strip()) for b in args.beta.split(',')]
    
    # Process optimizer flags (disable flags override enable flags)
    # enable_tanea = args.enable_tanea and not args.disable_tanea
    # enable_tanea_theory = args.enable_tanea_theory
    # enable_tanea_always_on = args.enable_tanea_always_on and not args.disable_tanea_always_on
    # enable_tanea_strong_clip = args.enable_tanea_strong_clip
    # enable_tanea_first_moment = args.enable_tanea_first_moment and not args.disable_tanea_first_moment
    # enable_tanea_mk2 = args.enable_tanea_mk2 and not args.disable_tanea_mk2
    # enable_tanea_always_on_mk2 = args.enable_tanea_always_on_mk2 and not args.disable_tanea_always_on_mk2
    # enable_tanea_mk3 = args.enable_tanea_mk3 and not args.disable_tanea_mk3
    # enable_tanea_kappa1 = args.enable_tanea_kappa1 and not args.disable_tanea_kappa1
    # enable_tanea_g3zero = args.enable_tanea_g3zero and not args.disable_tanea_g3zero
    # enable_rmsprop_dana = args.enable_rmsprop_dana and not args.disable_rmsprop_dana
    enable_adam = args.enable_adam and not args.disable_adam
    enable_long_adam = args.enable_long_adam and not args.disable_long_adam
    enable_long_adam_nesterov = args.enable_long_adam_nesterov and not args.disable_long_adam_nesterov
    enable_adam_star = args.enable_adam_star and not args.disable_adam_star
    enable_adam_nesterov_star = args.enable_adam_nesterov_star and not args.disable_adam_nesterov_star
    enable_dana_star = args.enable_dana_star and not args.disable_dana_star

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Compute traceK for hyperparameter scaling
    traceK = get_traceK(args.alpha, args.v)

    # print("="*60)
    # print("Label Noise Experiment")
    # print("="*60)
    # print(f"Model parameters: α={args.alpha}, β={beta_list}, V={args.v}, D={args.d}")
    # print(f"MoE parameters: M={args.m}, ζ={args.zeta}")
    # print(f"Training parameters: STEPS={args.steps}, BATCH_SIZE={args.batch_size}")
    # print(f"Label noise parameters: Student-t DOF={args.student_t_dof}, σ={args.sigma}")
    # print(f"Enabled optimizers: Tanea={enable_tanea}, TaneaTheory={enable_tanea_theory}, TaneaAlwaysOn={enable_tanea_always_on}")
    # print(f"                   TaneaStrongClip={enable_tanea_strong_clip}, TaneaFirstMoment={enable_tanea_first_moment}, TaneaMk2={enable_tanea_mk2}")
    # print(f"                   TaneaAlwaysOnMk2={enable_tanea_always_on_mk2}, TaneaMk3={enable_tanea_mk3}, TaneaKappa1={enable_tanea_kappa1}")
    # print(f"                   TaneaG3Zero={enable_tanea_g3zero}, RMSpropDana={enable_rmsprop_dana}, Adam={enable_adam}")
    # print(f"Results directory: {args.results_dir}")
    # print("="*60)

    # Main training experiment loop
    moe_results = []

    for beta in beta_list:
        print(f"\nRunning MoE experiments for β = {beta}")

        # Create MoE model with label noise
        key, model_key = random.split(key)
        model = LabelNoiseMixtureOfExpertsPLRF(
            alpha=args.alpha,
            beta=beta,
            v=args.v,
            d=args.d,
            m=args.m,
            zeta=args.zeta,
            student_t_dof=args.student_t_dof,
            sigma=args.sigma,
            key=model_key
        )

        print(f"  Expert probabilities: {model.expert_probs}")
        print(f"  Optimal risk: {model.population_risk(model.optimal_params_per_expert()):.6f}")

        # Create hyperparameters
        optimizers_dict = {}
        
        # if enable_tanea:
        #     tanea_hparams = get_tanea_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.tanea_kappa)
        #     optimizers_dict['tanea'] = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta)
        
        # if enable_tanea_theory:
        #     tanea_hparams = get_tanea_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.tanea_kappa)
        #     optimizers_dict['tanea_theory'] = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, momentum_flavor="theory")
        
        # if enable_tanea_always_on:
        #     tanea_hparams = get_tanea_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.tanea_kappa)
        #     optimizers_dict['tanea_always_on'] = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, momentum_flavor="always-on")
        
        # if enable_tanea_strong_clip:
        #     tanea_hparams = get_tanea_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.tanea_kappa)
        #     optimizers_dict['tanea_strong_clip'] = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, momentum_flavor="strong-clip")
        
        # if enable_tanea_first_moment:
        #     tanea_hparams = get_tanea_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.tanea_kappa)
        #     optimizers_dict['tanea_first_moment'] = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, tau_flavor="first-moment")
        
        # if enable_tanea_mk2:
        #     tanea_hparams = get_tanea_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.tanea_kappa)
        #     optimizers_dict['tanea_mk2'] = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, momentum_flavor="mk2")
        
        # if enable_tanea_always_on_mk2:
        #     tanea_hparams = get_tanea_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.tanea_kappa)
        #     optimizers_dict['tanea_always_on_mk2'] = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, momentum_flavor="always-on-mk2")
        
        # if enable_tanea_mk3:
        #     tanea_hparams = get_tanea_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.tanea_kappa)
        #     optimizers_dict['tanea_mk3'] = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, momentum_flavor="mk3")
        
        # if enable_tanea_kappa1:
        #     tanea_kappa1_hparams = get_tanea_kappa1_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.tanea_kappa)
        #     optimizers_dict['tanea_kappa1'] = tanea_optimizer(tanea_kappa1_hparams.g2, tanea_kappa1_hparams.g3, tanea_kappa1_hparams.delta)
        
        # if enable_tanea_g3zero:
        #     tanea_g3zero_hparams = get_tarmsprop_sgd_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, traceK, args.tanea_lr_scalar, args.tanea_global_exponent)
        #     optimizers_dict['tanea_g3zero'] = tanea_optimizer(tanea_g3zero_hparams.g2, tanea_g3zero_hparams.g3, tanea_g3zero_hparams.delta)
        
        # if enable_rmsprop_dana:
        #     optimizers_dict['rmsprop_dana'] = get_rmsprop_dana_optimizer(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.adam_beta2)
        
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

        # Run training experiments for enabled optimizers
        results_dict = {'beta': beta, 'model': model}
        
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
                    track_per_expert_loss=True,
                    track_tau_stats=True
                )
            else:
                # Use regular MoEPLRFTrainer for non-Tanea optimizers (Adam, RMSprop+Dana)
                trainer = MoEPLRFTrainer(model, optimizer)
                key, train_key = random.split(key)
                results = trainer.train(
                    train_key,
                    num_steps=args.steps,
                    batch_size=args.batch_size,  
                    track_per_expert_loss=True
                )
            
            results_dict[opt_name] = results

        moe_results.append(results_dict)

    # Skip visualization if --no_plots flag is set
    if not args.no_plots:
        # Create separate PDF files for different plot types
        n_beta = len(moe_results)
        
        # 1. LEARNING CURVES PDF
        fig_curves, axes_curves = plt.subplots(1, n_beta, figsize=(6 * n_beta, 5))
        if n_beta == 1:
            axes_curves = [axes_curves]

        for i, result in enumerate(moe_results):
            beta = result['beta']
            ax = axes_curves[i]
            
            # Plot each optimizer's learning curve
            optimizer_colors = {
            'tanea': 'red', 
            'tanea_theory': 'orange', 
            'tanea_always_on': 'purple', 
            'tanea_strong_clip': 'brown', 
            'tanea_first_moment': 'pink', 
            'tanea_mk2': 'magenta',
            'tanea_always_on_mk2': 'violet',
            'tanea_mk3': 'darkmagenta',
            'tanea_kappa1': 'darkgreen',
            'tanea_g3zero': 'blue',
            'rmsprop_dana': 'darkred',
            'adam': 'tab:red', #'green',
            'tanea_long_adam': 'tab:orange', #'yellow', 
            'tanea_long_adam_nesterov': 'indigo',
            'tanea_adam_star': 'tab:green', #'salmon',
            'tanea_adam_nesterov_star': 'navy',
            'tanea_dana_star': 'tab:blue', #'lightseagreen', 
            }
            
            # Get all available optimizers in this result
            available_optimizers = [k for k in result.keys() if k not in ['beta', 'model']]
            
            for optimizer_name in available_optimizers:
                if optimizer_name in result and 'timestamps' in result[optimizer_name] and 'losses' in result[optimizer_name]:
                    timestamps = result[optimizer_name]['timestamps']
                    losses = result[optimizer_name]['losses']
                    
                    if len(timestamps) > 0 and len(losses) > 0:
                        # Create display name
                        display_name = optimizer_name.upper().replace('_', ' ')
                        if optimizer_name == 'tanea_g3zero':
                            display_name = 'TANEA G3=0'
                        elif optimizer_name == 'tanea_always_on_mk2':
                            display_name = 'TANEA (ALWAYS-ON-MK2)'
                        elif optimizer_name == 'tanea_mk3':
                            display_name = 'TANEA (MK3)'
                        elif optimizer_name == 'tanea_kappa1':
                            display_name = 'TANEA (KAPPA1)'
                        elif optimizer_name == 'rmsprop_dana':
                            display_name = f'RMSPROP+DANA (β₂={args.adam_beta2})'
                        elif optimizer_name == 'adam':
                            display_name = f'Adam (β₁={args.adam_beta1}, β₂={args.adam_beta2})'
                        elif optimizer_name == 'tanea_long_adam':
                            display_name = 'Long-Adam'
                        elif optimizer_name == 'tanea_long_adam_nesterov':
                            display_name = 'Long-Adam-Nesterov'
                        elif optimizer_name == 'tanea_adam_star':
                            display_name = 'Adam-Star'
                        elif optimizer_name == 'tanea_adam_nesterov_star':
                            display_name = 'Adam-Nesterov-Star'
                        elif optimizer_name == 'tanea_dana_star':
                            display_name = 'Dana-Star'
                        
                        ax.loglog(timestamps, losses, 'o-', 
                                color=optimizer_colors.get(optimizer_name, 'black'), 
                                alpha=0.8, markersize=4, linewidth=2, 
                                label=display_name)
            
            ax.set_xlabel('Training Iteration')
            ax.set_ylabel('Population Risk')
            ax.set_title(f'Learning Curves with Label Noise\nβ={beta}, α={args.alpha}, m={args.m}, d={args.d}, ζ={args.zeta}, batch={args.batch_size}\nStudent-t DOF={args.student_t_dof}, σ={args.sigma}, steps={args.steps}')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        
        # Save learning curves PDF
        beta_str = "_".join([f"beta{beta}" for beta in beta_list])
        curves_filename = f"{args.output_prefix}_learning_curves_alpha{args.alpha}_M{args.m}_D{args.d}_zeta{args.zeta}_dof{args.student_t_dof}_sigma{args.sigma}_{beta_str}_steps{args.steps}.pdf"
        curves_filepath = os.path.join(args.results_dir, curves_filename)
        plt.savefig(curves_filepath, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to: {curves_filepath}")
        plt.close()  # Important: close the figure

        # 2. PER-EXPERT COMPARISON PDFS (one for each optimizer comparison)
        # Count enabled Tanea optimizers for per-expert plots
        enabled_tanea_opts = []
        # if enable_tanea:
        #     enabled_tanea_opts.append(('tanea', 'Tanea (Effective-Clip)'))
        # if enable_tanea_theory:
        #     enabled_tanea_opts.append(('tanea_theory', 'Tanea (Theory)'))
        # if enable_tanea_always_on:
        #     enabled_tanea_opts.append(('tanea_always_on', 'Tanea (Always-On)'))
        # if enable_tanea_strong_clip:
        #     enabled_tanea_opts.append(('tanea_strong_clip', 'Tanea (Strong-Clip)'))
        # if enable_tanea_first_moment:
        #     enabled_tanea_opts.append(('tanea_first_moment', 'Tanea (First-Moment)'))
        # if enable_tanea_mk2:
        #     enabled_tanea_opts.append(('tanea_mk2', 'Tanea (MK2)'))
        # if enable_tanea_always_on_mk2:
        #     enabled_tanea_opts.append(('tanea_always_on_mk2', 'Tanea (Always-On-MK2)'))
        # if enable_tanea_mk3:
        #     enabled_tanea_opts.append(('tanea_mk3', 'Tanea (MK3)'))
        # if enable_tanea_kappa1:
        #     enabled_tanea_opts.append(('tanea_kappa1', 'Tanea (Kappa1)'))
        # if enable_rmsprop_dana:
        #     enabled_tanea_opts.append(('rmsprop_dana', 'RMSprop+Dana'))
        if enable_long_adam:
            enabled_tanea_opts.append(('tanea_long_adam', 'Long-Adam'))
        if enable_long_adam_nesterov:
            enabled_tanea_opts.append(('tanea_long_adam_nesterov', 'Long-Adam-Nesterov'))
        if enable_adam_star:
            enabled_tanea_opts.append(('tanea_adam_star', 'Adam-Star'))
        if enable_adam_nesterov_star:
            enabled_tanea_opts.append(('tanea_adam_nesterov_star', 'Adam-Nesterov-Star'))
        if enable_dana_star:
            enabled_tanea_opts.append(('tanea_dana_star', 'Dana-Star'))

        # Create separate PDF for each optimizer comparison
        for tanea_opt, tanea_label in enabled_tanea_opts:
            fig_expert, axes_expert = plt.subplots(1, n_beta, figsize=(6 * n_beta, 5))
            if n_beta == 1:
                axes_expert = [axes_expert]
            
            for i, result in enumerate(moe_results):
                beta = result['beta']
                model = result['model']
                ax_expert = axes_expert[i]
                
                # Get Adam per-expert losses for comparison
                adam_per_expert_array = result.get('adam', {}).get('per_expert_losses', None)
                adam_timestamps = result.get('adam', {}).get('timestamps', [])
                
                # Get Adam-Star per-expert losses for grey dots
                adam_star_per_expert_array = result.get('tanea_adam_star', {}).get('per_expert_losses', None)
                
                # Get expert selection probabilities for color mapping
                expert_probs = model.expert_probs
                log_expert_probs = np.log(expert_probs)
                
                # Normalize log probabilities to [0, 1] for colormap
                if len(log_expert_probs) > 1:
                    log_prob_min = np.min(log_expert_probs)
                    log_prob_max = np.max(log_expert_probs)
                    log_prob_normalized = (log_expert_probs - log_prob_min) / (log_prob_max - log_prob_min)
                else:
                    log_prob_normalized = np.array([0.5])
                
                if tanea_opt in result and 'per_expert_losses' in result[tanea_opt]:
                    tanea_per_expert_array = result[tanea_opt]['per_expert_losses']
                    tanea_timestamps = result[tanea_opt]['timestamps']
                    
                    # First plot Adam-Star vs Adam in grey for all experts (background)
                    if adam_per_expert_array is not None and adam_star_per_expert_array is not None:
                        adam_final_losses = adam_per_expert_array[-1, :]
                        adam_star_final_losses = adam_star_per_expert_array[-1, :]
                        
                        ax_expert.scatter(adam_final_losses, adam_star_final_losses, 
                                        color='grey', alpha=0.4, s=30, 
                                        marker='s', edgecolors='none')
                    
                    # Plot Adam vs current Tanea optimizer per-expert losses with plasma colors
                    points_plotted = 0
                    if adam_per_expert_array is not None and tanea_per_expert_array is not None:
                        adam_final_losses = adam_per_expert_array[-1, :]
                        tanea_final_losses = tanea_per_expert_array[-1, :]
                        
                        colors = plt.cm.plasma(log_prob_normalized)
                        
                        ax_expert.scatter(adam_final_losses, tanea_final_losses, 
                                        c=colors, alpha=0.8, s=60, 
                                        edgecolors='black', linewidth=0.5)
                        points_plotted = len(adam_final_losses)
                    
                    # Add diagonal line for reference
                    if adam_per_expert_array is not None and tanea_per_expert_array is not None:
                        adam_final_losses = adam_per_expert_array[-1, :]
                        tanea_final_losses = tanea_per_expert_array[-1, :]
                        
                        all_finals = np.concatenate([adam_final_losses, tanea_final_losses])
                        
                        if adam_star_per_expert_array is not None:
                            adam_star_final_losses = adam_star_per_expert_array[-1, :]
                            all_finals = np.concatenate([all_finals, adam_star_final_losses])
                        
                        if len(all_finals) > 0:
                            min_loss = np.min(all_finals)
                            max_loss = np.max(all_finals)
                            
                            ax_expert.plot([min_loss, max_loss], [min_loss, max_loss], 
                                        'k--', alpha=0.5, linewidth=1, label='Equal Performance')
                    
                    # Add colorbar for the first plot
                    if i == 0 and points_plotted > 0:
                        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
                        sm.set_array([])
                        cbar = plt.colorbar(sm, ax=ax_expert, fraction=0.05, pad=0.04)
                        cbar.set_label('Log Expert Selection Probability\n(normalized)', fontsize=8)
                        
                        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
                        actual_log_probs = log_prob_min + np.array([0, 0.25, 0.5, 0.75, 1.0]) * (log_prob_max - log_prob_min)
                        cbar.set_ticklabels([f'{val:.1f}' for val in actual_log_probs])
                    
                    # Add legend
                    legend_elements = [
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                                markersize=8, alpha=0.8, markeredgecolor='black', linewidth=0,
                                label=f'{tanea_label}'),
                        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='grey', 
                                markersize=6, alpha=0.4, linewidth=0,
                                label='Adam-Star'),
                        plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.5,
                                label='Equal Performance')
                    ]
                    ax_expert.legend(handles=legend_elements, fontsize=8, loc='upper left')
                    
                    ax_expert.set_xlabel('Adam Final Loss')
                    ax_expert.set_ylabel(f'{tanea_label} Final Loss')
                    ax_expert.set_title(f'{tanea_label} vs Adam Per-Expert Losses\nβ={beta}, α={args.alpha}, ζ={args.zeta}, m={args.m}, d={args.d}, batch={args.batch_size}, steps={args.steps}')
                    ax_expert.grid(True, alpha=0.3)
                    ax_expert.set_xscale('log')
                    ax_expert.set_yscale('log')
                else:
                    # Show empty plot message
                    ax_expert.text(0.5, 0.5, f'No per-expert data\nfor {tanea_label}', 
                                transform=ax_expert.transAxes, ha='center', va='center',
                                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    ax_expert.set_xlabel('Adam Final Loss')
                    ax_expert.set_ylabel(f'{tanea_label} Final Loss')
                    ax_expert.set_title(f'{tanea_label} vs Adam Per-Expert Losses\nβ={beta}, α={args.alpha}, ζ={args.zeta}, m={args.m}, d={args.d}, batch={args.batch_size}, steps={args.steps}')

            plt.tight_layout()
            
            # Save per-expert comparison PDF
            safe_tanea_name = tanea_opt.replace(' ', '_').replace('(', '').replace(')', '')
            expert_filename = f"{args.output_prefix}_per_expert_{safe_tanea_name}_alpha{args.alpha}_M{args.m}_D{args.d}_zeta{args.zeta}_dof{args.student_t_dof}_sigma{args.sigma}_{beta_str}_steps{args.steps}.pdf"
            expert_filepath = os.path.join(args.results_dir, expert_filename)
            plt.savefig(expert_filepath, dpi=300, bbox_inches='tight')
            print(f"Per-expert comparison for {tanea_label} saved to: {expert_filepath}")
            plt.close()  # Important: close the figure

        # 3. TANEA ALGORITHMS VS ADAM PER-EXPERT COMPARISON PDFS
        # Create per-expert comparison plots for all Tanea algorithms vs Adam
        tanea_opts_for_comparison = []
        # if enable_tanea:
        #     tanea_opts_for_comparison.append(('tanea', 'Tanea (Effective-Clip)'))
        # if enable_tanea_theory:
        #     tanea_opts_for_comparison.append(('tanea_theory', 'Tanea (Theory)'))
        # if enable_tanea_always_on:
        #     tanea_opts_for_comparison.append(('tanea_always_on', 'Tanea (Always-On)'))
        # if enable_tanea_strong_clip:
        #     tanea_opts_for_comparison.append(('tanea_strong_clip', 'Tanea (Strong-Clip)'))
        # if enable_tanea_first_moment:
        #     tanea_opts_for_comparison.append(('tanea_first_moment', 'Tanea (First-Moment)'))
        # if enable_tanea_mk2:
        #     tanea_opts_for_comparison.append(('tanea_mk2', 'Tanea (MK2)'))
        # if enable_tanea_always_on_mk2:
        #     tanea_opts_for_comparison.append(('tanea_always_on_mk2', 'Tanea (Always-On-MK2)'))
        # if enable_tanea_mk3:
        #     tanea_opts_for_comparison.append(('tanea_mk3', 'Tanea (MK3)'))
        # if enable_tanea_kappa1:
        #     tanea_opts_for_comparison.append(('tanea_kappa1', 'Tanea (Kappa1)'))
        # if enable_tanea_g3zero:
        #     tanea_opts_for_comparison.append(('tanea_g3zero', 'Tanea G3=0'))
        if enable_long_adam:
            tanea_opts_for_comparison.append(('tanea_long_adam', 'Long-Adam'))
        if enable_long_adam_nesterov:
            tanea_opts_for_comparison.append(('tanea_long_adam_nesterov', 'Long-Adam-Nesterov'))
        if enable_adam_star:
            tanea_opts_for_comparison.append(('tanea_adam_star', 'Adam-Star'))
        if enable_adam_nesterov_star:
            tanea_opts_for_comparison.append(('tanea_adam_nesterov_star', 'Adam-Nesterov-Star'))
        if enable_dana_star:
            tanea_opts_for_comparison.append(('tanea_dana_star', 'Dana-Star'))
        
        # Create separate PDF for each Tanea optimizer comparison with Adam
        for tanea_opt, tanea_label in tanea_opts_for_comparison:
            fig_tanea_vs_adam, axes_tanea_vs_adam = plt.subplots(1, n_beta, figsize=(6 * n_beta, 5))
            if n_beta == 1:
                axes_tanea_vs_adam = [axes_tanea_vs_adam]
            
            for i, result in enumerate(moe_results):
                beta = result['beta']
                model = result['model']
                ax_tanea_vs_adam = axes_tanea_vs_adam[i]
                
                # Get Adam per-expert losses for comparison
                adam_per_expert_array = result.get('adam', {}).get('per_expert_losses', None)
                adam_timestamps = result.get('adam', {}).get('timestamps', [])
                
                # Get expert selection probabilities for color mapping
                expert_probs = model.expert_probs
                log_expert_probs = np.log(expert_probs)
                
                # Normalize log probabilities to [0, 1] for colormap
                if len(log_expert_probs) > 1:
                    log_prob_min = np.min(log_expert_probs)
                    log_prob_max = np.max(log_expert_probs)
                    log_prob_normalized = (log_expert_probs - log_prob_min) / (log_prob_max - log_prob_min)
                else:
                    log_prob_normalized = np.array([0.5])
                
                if tanea_opt in result and 'per_expert_losses' in result[tanea_opt]:
                    tanea_per_expert_array = result[tanea_opt]['per_expert_losses']
                    tanea_timestamps = result[tanea_opt]['timestamps']
                    
                    # Plot Adam vs current Tanea optimizer per-expert losses with plasma colors
                    points_plotted = 0
                    if adam_per_expert_array is not None and tanea_per_expert_array is not None:
                        adam_final_losses = adam_per_expert_array[-1, :]
                        tanea_final_losses = tanea_per_expert_array[-1, :]
                        
                        colors = plt.cm.plasma(log_prob_normalized)
                        
                        ax_tanea_vs_adam.scatter(adam_final_losses, tanea_final_losses, 
                                            c=colors, alpha=0.8, s=60, 
                                            edgecolors='black', linewidth=0.5)
                        points_plotted = len(adam_final_losses)
                    
                    # Add diagonal line for reference
                    if adam_per_expert_array is not None and tanea_per_expert_array is not None:
                        adam_final_losses = adam_per_expert_array[-1, :]
                        tanea_final_losses = tanea_per_expert_array[-1, :]
                        
                        all_finals = np.concatenate([adam_final_losses, tanea_final_losses])
                        
                        if len(all_finals) > 0:
                            min_loss = np.min(all_finals)
                            max_loss = np.max(all_finals)
                            
                            ax_tanea_vs_adam.plot([min_loss, max_loss], [min_loss, max_loss], 
                                            'k--', alpha=0.5, linewidth=1, label='Equal Performance')
                    
                    # Add colorbar for the first plot
                    if i == 0 and points_plotted > 0:
                        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
                        sm.set_array([])
                        cbar = plt.colorbar(sm, ax=ax_tanea_vs_adam, fraction=0.05, pad=0.04)
                        cbar.set_label('Log Expert Selection Probability\n(normalized)', fontsize=8)
                        
                        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
                        actual_log_probs = log_prob_min + np.array([0, 0.25, 0.5, 0.75, 1.0]) * (log_prob_max - log_prob_min)
                        cbar.set_ticklabels([f'{val:.1f}' for val in actual_log_probs])
                    
                    # Add legend
                    legend_elements = [
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                                markersize=8, alpha=0.8, markeredgecolor='black', linewidth=0,
                                label=f'{tanea_label}'),
                        plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.5,
                                label='Equal Performance')
                    ]
                    ax_tanea_vs_adam.legend(handles=legend_elements, fontsize=8, loc='upper left')
                    
                    ax_tanea_vs_adam.set_xlabel('Adam Final Loss')
                    ax_tanea_vs_adam.set_ylabel(f'{tanea_label} Final Loss')
                    ax_tanea_vs_adam.set_title(f'{tanea_label} vs Adam Per-Expert Losses\nβ={beta}, α={args.alpha}, ζ={args.zeta}, m={args.m}, d={args.d}, batch={args.batch_size}, steps={args.steps}')
                    ax_tanea_vs_adam.grid(True, alpha=0.3)
                    ax_tanea_vs_adam.set_xscale('log')
                    ax_tanea_vs_adam.set_yscale('log')
                else:
                    # Show empty plot message
                    ax_tanea_vs_adam.text(0.5, 0.5, f'No per-expert data\nfor {tanea_label}', 
                                transform=ax_tanea_vs_adam.transAxes, ha='center', va='center',
                                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    ax_tanea_vs_adam.set_xlabel('Adam Final Loss')
                    ax_tanea_vs_adam.set_ylabel(f'{tanea_label} Final Loss')
                    ax_tanea_vs_adam.set_title(f'{tanea_label} vs Adam Per-Expert Losses\nβ={beta}, α={args.alpha}, ζ={args.zeta}, m={args.m}, d={args.d}, batch={args.batch_size}, steps={args.steps}')

            plt.tight_layout()
            
            # Save Tanea vs Adam per-expert comparison PDF
            safe_tanea_name = tanea_opt.replace(' ', '_').replace('(', '').replace(')', '')
            tanea_vs_adam_filename = f"{args.output_prefix}_tanea_vs_adam_{safe_tanea_name}_alpha{args.alpha}_M{args.m}_D{args.d}_zeta{args.zeta}_dof{args.student_t_dof}_sigma{args.sigma}_{beta_str}_steps{args.steps}.pdf"
            tanea_vs_adam_filepath = os.path.join(args.results_dir, tanea_vs_adam_filename)
            plt.savefig(tanea_vs_adam_filepath, dpi=300, bbox_inches='tight')
            print(f"Tanea vs Adam per-expert comparison for {tanea_label} saved to: {tanea_vs_adam_filepath}")
            plt.close()  # Important: close the figure

        # 4. TAU ORDER STATISTICS VISUALIZATION PDFS
        # Create tau order statistics plots for Tanea optimizers (matching original structure)
        tanea_opts_with_tau = []
        # if enable_tanea:
        #     tanea_opts_with_tau.append(('tanea', 'Tanea (Effective-Clip)'))
        # if enable_tanea_theory:
        #     tanea_opts_with_tau.append(('tanea_theory', 'Tanea (Theory)'))
        # if enable_tanea_always_on:
        #     tanea_opts_with_tau.append(('tanea_always_on', 'Tanea (Always-On)'))
        # if enable_tanea_strong_clip:
        #     tanea_opts_with_tau.append(('tanea_strong_clip', 'Tanea (Strong-Clip)'))
        # if enable_tanea_first_moment:
        #     tanea_opts_with_tau.append(('tanea_first_moment', 'Tanea (First-Moment)'))
        # if enable_tanea_mk2:
        #     tanea_opts_with_tau.append(('tanea_mk2', 'Tanea (MK2)'))
        # if enable_tanea_always_on_mk2:
        #     tanea_opts_with_tau.append(('tanea_always_on_mk2', 'Tanea (Always-On-MK2)'))
        # if enable_tanea_mk3:
        #     tanea_opts_with_tau.append(('tanea_mk3', 'Tanea (MK3)'))
        # if enable_tanea_kappa1:
        #     tanea_opts_with_tau.append(('tanea_kappa1', 'Tanea (Kappa1)'))
        # if enable_tanea_g3zero:
        #     tanea_opts_with_tau.append(('tanea_g3zero', 'Tanea G3=0'))
        if enable_adam_star:
            tanea_opts_with_tau.append(('tanea_adam_star', 'Adam-Star'))
        if enable_adam_nesterov_star:
            tanea_opts_with_tau.append(('tanea_adam_nesterov_star', 'Adam-Nesterov-Star'))
        if enable_dana_star:
            tanea_opts_with_tau.append(('tanea_dana_star', 'Dana-Star'))
        
        # Create individual tau order statistics plots for each Tanea optimizer and beta combination
        for tanea_opt, tanea_label in tanea_opts_with_tau:
            for i, result in enumerate(moe_results):
                beta = result['beta']
                
                # Create a single plot for this optimizer-beta combination
                fig_tau, ax = plt.subplots(1, 1, figsize=(6, 5))
                
                if tanea_opt in result and 'tau_statistics' in result[tanea_opt]:
                    tau_stats = result[tanea_opt]['tau_statistics']
                    tau_times = tau_stats['timestamps']
                    tau_order_stats = tau_stats['tau_order_statistics']
                    
                    # Check if reversed order statistics are available
                    tau_reversed_order_stats = tau_stats.get('tau_reversed_order_statistics', None)
                    
                    if len(tau_order_stats) > 0:
                        # Create color map for time evolution
                        n_timestamps = len(tau_times)
                        colors = plt.cm.plasma(np.linspace(0, 0.8, n_timestamps))
                        
                        # Find overall max and min for y-axis range (include both regular and reversed stats)
                        all_order_stats = []
                        for order_stats in tau_order_stats:
                            if len(order_stats) > 0:
                                all_order_stats.extend(order_stats)
                        
                        # Also include reversed order statistics if available
                        if tau_reversed_order_stats:
                            for order_stats in tau_reversed_order_stats:
                                if len(order_stats) > 0:
                                    all_order_stats.extend(order_stats)
                        
                        if all_order_stats:
                            max_tau = max(all_order_stats)
                            min_tau_plot = max_tau * 1e-5  # 5 orders of magnitude lower
                            
                            # Plot largest order statistics for each timestamp
                            for t_idx, (timestamp, order_stats) in enumerate(zip(tau_times, tau_order_stats)):
                                if len(order_stats) > 0:
                                    # k values: 0, 1, 2, ..., max_k where (1.1)^k corresponds to order stat index
                                    k_values = np.arange(len(order_stats))
                                    
                                    # Filter order stats to only show those within our range
                                    valid_mask = order_stats >= min_tau_plot
                                    if np.any(valid_mask):
                                        filtered_k = 1.1**(k_values[valid_mask])
                                        filtered_stats = order_stats[valid_mask]
                                        
                                        ax.scatter(filtered_k, filtered_stats, 
                                                 color=colors[t_idx], alpha=0.7, s=20)
                            
                            # Plot smallest order statistics if available (same color scheme)
                            if tau_reversed_order_stats:
                                for t_idx, (timestamp, reversed_order_stats) in enumerate(zip(tau_times, tau_reversed_order_stats)):
                                    if len(reversed_order_stats) > 0:
                                        # k values: 0, 1, 2, ..., max_k where (1.1)^k corresponds to order stat index
                                        k_values = np.arange(len(reversed_order_stats))
                                        
                                        # Filter order stats to only show those within our range
                                        valid_mask = reversed_order_stats >= min_tau_plot
                                        if np.any(valid_mask):
                                            filtered_k = 1.1**(k_values[valid_mask])
                                            filtered_stats = reversed_order_stats[valid_mask]
                                            
                                            ax.scatter(filtered_k, filtered_stats, 
                                                     color=colors[t_idx], alpha=0.7, s=20)
                            
                            # Set y-axis limits
                            ax.set_ylim(min_tau_plot, max_tau * 1.1)
                        
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                        ax.set_xlabel('k (order statistic index)')
                        ax.set_ylabel('τ_k')
                        ax.set_title(f'{tanea_label} Tau Order Statistics Evolution\nβ={beta}, α={args.alpha}, m={args.m}, d={args.d}, ζ={args.zeta}, steps={args.steps}')
                        ax.grid(True, alpha=0.3)
                        
                        # Add colorbar to show time evolution
                        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=0.8))
                        sm.set_array([])
                        
                        # Map timestamp indices to [0, 0.8] range
                        if n_timestamps > 1:
                            time_values = np.linspace(0, 0.8, n_timestamps)
                            actual_times = np.array(tau_times)
                            cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
                            
                            # Set colorbar ticks to show actual iteration numbers
                            tick_positions = np.linspace(0, 0.8, min(5, n_timestamps))
                            tick_labels = []
                            for pos in tick_positions:
                                # Find closest timestamp index
                                idx = int(pos / 0.8 * (n_timestamps - 1))
                                tick_labels.append(f'{actual_times[idx]:.0f}')
                            
                            cbar.set_ticks(tick_positions)
                            cbar.set_ticklabels(tick_labels)
                            cbar.set_label('Training Iteration')
                    else:
                        # Show empty plot message
                        ax.text(0.5, 0.5, f'No tau data\nfor {tanea_label}', 
                                transform=ax.transAxes, ha='center', va='center',
                                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                        ax.set_xlabel('k (order statistic index)')
                        ax.set_ylabel('τ_k')
                        ax.set_title(f'{tanea_label} Tau Order Statistics Evolution\nβ={beta}, α={args.alpha}, m={args.m}, d={args.d}, ζ={args.zeta}, steps={args.steps}')
                else:
                    # Show empty plot message
                    ax.text(0.5, 0.5, f'No tau data\nfor {tanea_label}', 
                            transform=ax.transAxes, ha='center', va='center',
                            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    ax.set_xlabel('k (order statistic index)')
                    ax.set_ylabel('τ_k')
                    ax.set_title(f'{tanea_label} Tau Order Statistics Evolution\nβ={beta}, α={args.alpha}, m={args.m}, d={args.d}, ζ={args.zeta}, steps={args.steps}')
                
                plt.tight_layout()
                
                # Save individual tau order statistics PDF for this optimizer-beta combination
                safe_tanea_name = tanea_opt.replace(' ', '_').replace('(', '').replace(')', '')
                tau_filename = f"{args.output_prefix}_tau_order_stats_{safe_tanea_name}_beta{beta}_alpha{args.alpha}_M{args.m}_D{args.d}_zeta{args.zeta}_dof{args.student_t_dof}_sigma{args.sigma}_steps{args.steps}.pdf"
                tau_filepath = os.path.join(args.results_dir, tau_filename)
                plt.savefig(tau_filepath, dpi=300, bbox_inches='tight')
                print(f"Tau order statistics for {tanea_label} (β={beta}) saved to: {tau_filepath}")
                plt.close()  # Important: close the figure

        print(f"\nAll plots saved as separate PDF files in: {args.results_dir}")
    else:
        print("Skipping plots due to --no_plots flag")


if __name__ == "__main__":
    main()
