#!/usr/bin/env python
"""
Debug script to step through mk4 optimizer and inspect buffer evolution.
"""
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import numpy as np
from typing import Dict, Any

from optimizers import get_dana_star, TaneaOptimizerState
from moe_m_sweeps_mk4 import get_dana_star_mk4_optimizer, get_dana_star_optimizer, LabelNoiseMixtureOfExpertsPLRF

def print_buffer_stats(name: str, buffer: jnp.ndarray, step: int):
    """Print statistics about an optimizer buffer."""
    print(f"  {name} (step {step}):")
    print(f"    shape: {buffer.shape}")
    print(f"    mean: {jnp.mean(buffer):.6f}")
    print(f"    std: {jnp.std(buffer):.6f}")
    print(f"    min: {jnp.min(buffer):.6f}")
    print(f"    max: {jnp.max(buffer):.6f}")
    print(f"    norm: {jnp.linalg.norm(buffer):.6f}")

def debug_optimizer_step(optimizer, params, opt_state, grads, step_num):
    """Debug a single optimizer step and print detailed information."""
    print(f"\n=== STEP {step_num} ===")
    
    # Print gradient information
    print("Gradients:")
    print_buffer_stats("grads", grads, step_num)
    
    # Print optimizer state before update
    if hasattr(opt_state, '__len__') and len(opt_state) > 0:
        tanea_state = opt_state[0]
        if hasattr(tanea_state, 'm'):
            print("Optimizer State (before update):")
            print_buffer_stats("m (momentum)", tanea_state.m, step_num)
            print_buffer_stats("v (second moment)", tanea_state.v, step_num)
            print_buffer_stats("tau", tanea_state.tau, step_num)
            print(f"    count: {tanea_state.count}")
    
    # Perform the update
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    # Print update information
    print("Updates:")
    print_buffer_stats("updates", updates, step_num)
    
    # Print optimizer state after update
    if hasattr(new_opt_state, '__len__') and len(new_opt_state) > 0:
        new_tanea_state = new_opt_state[0]
        if hasattr(new_tanea_state, 'm'):
            print("Optimizer State (after update):")
            print_buffer_stats("m (momentum)", new_tanea_state.m, step_num)
            print_buffer_stats("v (second moment)", new_tanea_state.v, step_num)
            print_buffer_stats("tau", new_tanea_state.tau, step_num)
            print(f"    count: {new_tanea_state.count}")
    
    # Print parameter change
    param_change = new_params - params
    print("Parameter Changes:")
    print_buffer_stats("param_change", param_change, step_num)
    
    print("Parameters:")
    print_buffer_stats("new_params", new_params, step_num)
    
    return new_params, new_opt_state

def get_traceK(alpha, v):
    x_grid = jnp.arange(1, v+1).reshape(1, v)
    population_eigs = x_grid ** -alpha
    population_trace = jnp.sum(population_eigs**2)
    return population_trace

def main():
    # Use same parameters as moe_m_sweeps_mk4.py
    alpha = 1.0
    beta = 0.0
    d = 50
    v = 2000
    m = 50
    zeta = 0.5
    batch_size = 100
    g2_scale = 0.2
    g3_over_g2 = 0.01
    tanea_lr_scalar = 1.0
    tanea_global_exponent = 0.0
    tanea_kappa = None
    student_t_dof = 3.0
    sigma = 0.1
    
    # Set random seed
    key = random.PRNGKey(42)
    
    # Compute traceK for hyperparameter scaling
    traceK = get_traceK(alpha, v)
    print(f"traceK: {traceK}")
    
    # Create MoE model
    key, model_key = random.split(key)
    model = LabelNoiseMixtureOfExpertsPLRF(
        alpha=alpha,
        beta=beta,
        v=v,
        d=d,
        m=m,
        zeta=zeta,
        student_t_dof=student_t_dof,
        sigma=sigma,
        key=model_key
    )
    
    print(f"Optimal risk: {model.population_risk(model.optimal_params_per_expert()):.6f}")
    
    # Create optimizers
    dana_star_opt = get_dana_star_optimizer(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent, tanea_kappa)
    dana_star_mk4_opt = get_dana_star_mk4_optimizer(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent, tanea_kappa)
    
    # Initialize parameters
    init_params = jnp.zeros((d, m))
    dana_star_state = dana_star_opt.init(init_params)
    dana_star_mk4_state = dana_star_mk4_opt.init(init_params)
    
    print(f"Initial population risk: {model.population_risk(init_params):.6f}")
    
    # Create a function to compute gradients
    @jax.jit
    def compute_gradients(params, key):
        # Split key for data generation
        key_data, key_expert = random.split(key)
        
        # Generate batch
        X, y = model.generate_batch(key_data, batch_size)
        
        # Sample expert indices
        expert_indices = model.sample_expert_batch(key_expert, batch_size)
        
        # Compute gradients
        def loss_fn(params):
            # Create routing matrix
            R = model.create_routing_matrix(expert_indices, batch_size)
            
            # Compute predictions for all experts
            all_predictions = jnp.matmul(X, params)  # (batch_size, m)
            
            # Select predictions based on routing
            predictions = jnp.sum(all_predictions * R.T, axis=1)  # (batch_size,)
            
            # Compute loss
            return jnp.mean(optax.l2_loss(predictions, y))
        
        return jax.grad(loss_fn)(params)
    
    # Run debugging for several steps
    dana_star_params = init_params.copy()
    dana_star_mk4_params = init_params.copy()
    
    num_debug_steps = 10
    
    print("\n" + "="*80)
    print("DEBUGGING DANA STAR (ORIGINAL)")
    print("="*80)
    
    for step in range(num_debug_steps):
        key, step_key = random.split(key)
        grads = compute_gradients(dana_star_params, step_key)
        
        dana_star_params, dana_star_state = debug_optimizer_step(
            dana_star_opt, dana_star_params, dana_star_state, grads, step
        )
        
        risk = model.population_risk(dana_star_params)
        print(f"Population risk after step {step}: {risk:.6f}")
    
    print("\n" + "="*80)
    print("DEBUGGING DANA STAR MK4")
    print("="*80)
    
    # Reset key for fair comparison
    key = random.PRNGKey(42)
    
    for step in range(num_debug_steps):
        key, step_key = random.split(key)
        grads = compute_gradients(dana_star_mk4_params, step_key)
        
        dana_star_mk4_params, dana_star_mk4_state = debug_optimizer_step(
            dana_star_mk4_opt, dana_star_mk4_params, dana_star_mk4_state, grads, step
        )
        
        risk = model.population_risk(dana_star_mk4_params)
        print(f"Population risk after step {step}: {risk:.6f}")
    
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    final_dana_star_risk = model.population_risk(dana_star_params)
    final_dana_star_mk4_risk = model.population_risk(dana_star_mk4_params)
    
    print(f"Final Dana Star risk: {final_dana_star_risk:.6f}")
    print(f"Final Dana Star MK4 risk: {final_dana_star_mk4_risk:.6f}")
    print(f"Initial risk: {model.population_risk(init_params):.6f}")
    print(f"Optimal risk: {model.population_risk(model.optimal_params_per_expert()):.6f}")

if __name__ == "__main__":
    main()