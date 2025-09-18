"""
Large Class Logistic Regression on Power Law Gaussian Mixture Model

This script simulates and analyzes the performance of logistic regression on a 
power-law Gaussian Mixture Model (GMM) with many classes. It explores how the 
model's performance scales with dimension and computational resources.

Key features:
- Simulates data from a power-law GMM where class frequencies follow p(i) ∝ i^(-gamma)
- Trains logistic regression models with varying input dimensions (d)
- Tracks loss curves as a function of computational work (FLOPs)
- Generates "chinchilla plots" showing the scaling relationship between:
  1. Model dimension (d) vs final loss (power law relationship)
  2. Computational budget (FLOPs) vs loss for different model sizes

The script implements theoretical predictions from statistical learning theory
about how model performance scales with dimension and computation in the 
high-dimensional regime.

Author: Elliot Paquette
Date: 2025-05-15
"""


import jax
import jax.numpy as jnp
import jax.random as random
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt




def get_power_law_probs(m, gamma):
    """Calculate class probabilities following a power law distribution.
    
    Args:
        m: Number of classes
        gamma: Power law parameter
        
    Returns:
        Array of normalized class probabilities
    """
    probs = jnp.arange(1, m + 1) ** (-gamma)
    return probs / jnp.sum(probs)

def generate_data(key, v, m, gamma, batch_size, W, mu, Sigma_eps):
    """Generate a batch of data for the large class logistic regression problem.
    
    Args:
        key: JAX PRNGKey
        v: Abstract space dimension
        m: Number of classes
        gamma: Power law parameter for class frequencies
        batch_size: Number of samples to generate
        W: Fixed random feature map (d x v)
        mu: Class means (v x m)
        Sigma_eps: Covariance matrix for noise (v x v)
        
    Returns:
        tuple: (embeddings, labels) where embeddings are the projected features
        and labels are the true class probabilities (one-hot in this case)
    """
    key1, key2 = random.split(key)
    
    class_probs = get_power_law_probs(m, gamma)
    class_indices = random.categorical(key1, jnp.log(class_probs), shape=(batch_size,)) # Note that jax.random.catergorical expects log probabilities
    
    # Generate points from the selected class means with noise
    # x ~ N(μ_i, Σ^ε) where μ_i ~ N(0, Σ^μ)
    # Generate standard normal samples and scale by sqrt of Sigma_eps
    z = random.normal(key2, shape=(batch_size, v))
    # Assuming Sigma_eps is diagonal, multiply by sqrt of diagonal elements
    sqrt_Sigma_eps = jnp.sqrt(jnp.diag(Sigma_eps))
    x = z * sqrt_Sigma_eps
    x = x + mu[:, class_indices].T
    
    # Project to feature space
    x_proj = x @ W.T
    
    # Create one-hot labels
    labels = jnp.zeros((batch_size, m))
    labels = labels.at[jnp.arange(batch_size), class_indices].set(1.0)
    
    return x_proj, labels

def cross_entropy_loss(params, x, p_true):
    """Compute the cross entropy loss between true and predicted probabilities.
    
    Args:
        params: Tuple of (theta, b) where theta is the weight matrix and b is the bias vector
        x: Input features
        p_true: True class probabilities
        
    Returns:
        float: Cross entropy loss
    """
    theta, b = params
    logits = x @ theta + b
    p_pred = jax.nn.softmax(logits, axis=1)
    return -jnp.mean(jnp.sum(p_true * jnp.log(p_pred + 1e-8), axis=1))

def train_largeclass_logistic(
    key,
    d,
    v,
    m,
    beta,
    alpha,
    gamma,
    batch_size,
    steps,
    optimizer,
    init_params,
    loss_oracle,
    W,
    mu,
    Sigma_eps,
    tqdm_bar=True,
    train_loss=True
):
    """Train a large class logistic regression model using SGD.
    
    Args:
        key: JAX PRNGKey
        d: Feature dimension
        v: Abstract space dimension
        m: Number of classes
        beta: Power law parameter for class means covariance
        alpha: Power law parameter for noise covariance
        gamma: Power law parameter for class frequencies
        batch_size: Batch size for training
        steps: Total number of training steps
        optimizer: Optimizer
        init_params: Initial parameters (theta, b)
        loss_oracle: Function to compute population loss
        W: Fixed random feature map
        mu: Class means (v x m)
        Sigma_eps: Covariance matrix for noise (v x v)
        tqdm_bar: Whether to show progress bar
        train_loss: Whether to return training loss

    Returns:
        tuple: (timestamps, losses) where timestamps are iteration numbers
        and losses are the corresponding population loss values
    """
    params = init_params
    opt_state = optimizer.init(params)
    state = (params, opt_state)

    def train_step(state, key):
        params, opt_state = state
        x, p_true = generate_data(key, v, m, gamma, batch_size, W, mu, Sigma_eps)
        loss_fn = lambda p: cross_entropy_loss(p, x, p_true)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    # Generate exponentially spaced timestamps for loss recording
    loss_times = jnp.unique(jnp.concatenate([
        jnp.array([0]),
        jnp.int32(1.1**jnp.arange(1, jnp.ceil(jnp.log(steps)/jnp.log(1.1)))),
        jnp.array([steps])
    ]))
    
    loss_time_steps = loss_times[1:] - loss_times[:-1]
    losses = [loss_oracle(init_params, key)]
    
    if tqdm_bar:
        for increment in tqdm(loss_time_steps):
            key, subkey = random.split(key)
            keyz = random.split(subkey, increment)
            state, tl = jax.lax.scan(train_step, state, keyz)
            if train_loss:
                losses.append(jnp.mean(tl))
            else:
                pop_loss = loss_oracle(state[0], key)
                losses.append(pop_loss)
    else:
        for increment in loss_time_steps:
            key, subkey = random.split(key)
            keyz = random.split(subkey, increment)
            state, tl = jax.lax.scan(train_step, state, keyz)
            if train_loss:
                losses.append(jnp.mean(tl))
            else:
                pop_loss = loss_oracle(state[0], key)
                losses.append(pop_loss)
            
    return loss_times, losses

def main():
    # Set parameters
    d_values = [25,50,100,200,400,800]
    beta = 1.5
    alpha = 1.0
    gamma = 1.0
    delta = 0.0
    betaprime = 2
    batch_size = 1000
    steps = 10000000
    learning_rate = 0.01
    
    # Initialize random key
    key = random.PRNGKey(0)
    
    # Create figure for plotting
    plt.figure(figsize=(10, 6))
    
    # Dictionary to store results for pickle
    results = {
        'params': {
            'beta': beta,
            'betaprime': betaprime,
            'alpha': alpha,
            'gamma': gamma,
            'delta': delta,
            'batch_size': batch_size,
            'steps': steps,
            'learning_rate': learning_rate
        },
        'data': {}
    }
    
    # Store final losses for d vs loss plot
    final_losses = []
    
    for d in d_values:
        v = 10 * d
        m = 1000  # Fixed number of classes
        #m = v  # Number of classes equals v
        #mtest = 100

        # Generate fixed random feature map
        key, subkey = random.split(key)
        W = random.normal(subkey, (d, v)) / jnp.sqrt(d)
        
        # Generate class means and covariances
        key, subkey1, subkey2 = random.split(key, 3)
        
        # Generate class means μ_i ~ N(0, Σ^μ)
        Sigma_mu_diag = jnp.arange(1, v + 1) ** (-2 * beta)
        # Generate standard normal samples and scale by sqrt of diagonal covariance
        mu_standard = random.normal(subkey1, shape=(m, v))
        mu = (mu_standard * jnp.sqrt(Sigma_mu_diag)).T
        # Rescale the j-th mu vector by j^{-delta}
        j_indices = jnp.arange(1, m + 1)
        scaling_factors = j_indices ** (-delta)
        mu = mu * scaling_factors * betaprime
        # Equivalent to sampling from N(0, Σ^μ) where Σ^μ is diagonal
        
        # Generate noise covariance Σ^ε
        Sigma_eps = jnp.diag(jnp.arange(1, v + 1) ** (-2 * alpha))
        
        # Initialize parameters
        key, subkey = random.split(key)
        #theta_init = random.normal(subkey, (d, m)) / jnp.sqrt(d)
        theta_init = jnp.zeros((d, m))
        b_init = jnp.zeros(m)
        #probs = get_power_law_probs(m, gamma)
        #b_init = jnp.log(probs)
        #init_entropy = jnp.sum(probs * jnp.log(probs + 1e-8))
        init_params = (theta_init, b_init)
        
        # Define loss oracle
        def loss_oracle(params, key):
            x, p_true = generate_data(key, v, m, gamma, 
                                    batch_size=10000, W=W, mu=mu,
                                    Sigma_eps=Sigma_eps)
            return cross_entropy_loss(params, x, p_true)
        
        # Train model
        key, subkey = random.split(key)
        optimizer = optax.adam(learning_rate)

        times, losses = train_largeclass_logistic(
            subkey, d, v, m, beta, alpha, gamma,
            batch_size, steps, optimizer,
            init_params, loss_oracle, W, mu, Sigma_eps,
            train_loss=False
        )
        
        # Scale times by d*batch_size for flops
        flops = (times*1.0) * (d*1.0) * batch_size
        
        # Calculate average of last 10 losses for this d value
        avg_final_loss = jnp.mean(jnp.array(losses[-10:]))
        final_losses.append(avg_final_loss)
        
        # Store results for this d value
        results['data'][d] = {
            'times': times,
            'losses': losses,
            'flops': flops,
            'avg_final_loss': avg_final_loss
        }
        
        # Plot results
        plt.loglog(flops, losses, label=f'd={d}')
    
    plt.xlabel('Flops')
    plt.ylabel('Loss')
    plt.title('Large Class Logistic Regression Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'largeclass_logistic_losses_alpha{alpha}_beta{beta}_gamma{gamma}_delta{delta}.pdf')
    plt.close()
    
    # Create d vs final loss plot
    plt.figure(figsize=(10, 6))
    d_values_array = jnp.array(d_values)
    final_losses_array = jnp.array(final_losses)
    
    # Log-log plot
    plt.loglog(d_values_array, final_losses_array, 'o-', label='Final Loss')
    
    # Linear fit in log-log space
    log_d = jnp.log(d_values_array)
    log_loss = jnp.log(final_losses_array)
    slope, intercept = jnp.polyfit(log_d, log_loss, 1)
    
    # Plot the fit line
    fit_line = jnp.exp(intercept) * d_values_array**slope
    plt.loglog(d_values_array, fit_line, 'r--', label=f'Fit: slope = {slope:.3f}')
    
    plt.xlabel('Dimension (d)')
    plt.ylabel('Final Loss (avg of last 10)')
    plt.title(f'Dimension vs Final Loss (slope = {slope:.3f})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'largeclass_logistic_d_vs_loss_alpha{alpha}_beta{beta}_gamma{gamma}_delta{delta}_betaprime{betaprime}.pdf')
    plt.close()
    
    # Save results to pickle file
    import pickle
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pickle_filename = f'largeclass_logistic_results_alpha{alpha}_beta{beta}_gamma{gamma}_delta{delta}_betaprime{betaprime}_{timestamp}.pkl'
    
    with open(pickle_filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {pickle_filename}")
    print(f"Dimension vs Loss slope: {slope:.3f}")

if __name__ == "__main__":
    main() 