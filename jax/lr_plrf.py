"""
Logistic Regression Power-Law Random Features (LR-PLRF)

This module implements a logistic regression variant of the MoE PLRF problem.
The number of experts equals the number of classes, with:
- alpha: data covariance exponent
- beta: class mean decay exponent
- zeta: power law decay exponent for class frequency

Based on the semantics from largeclass_logistic.py and structure from moe_plrf.py
"""

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from typing import Tuple, Optional, Union, Dict, List
import tqdm
from functools import partial


class LogisticRegressionPLRF:
    """Logistic Regression Power-Law Random Features model.

    This extends the PLRF concept to logistic regression where:
    - Number of experts = number of classes (m)
    - alpha: power law exponent for data covariance decay
    - beta: power law exponent for class mean decay
    - zeta: power law exponent for class frequency decay
    """

    def __init__(self,
                 alpha: float,
                 beta: float,
                 zeta: float,
                 snr : float,
                 v: int,
                 d: int,
                 m: int,
                 key: random.PRNGKey):
        """Initialize the LR-PLRF model.

        Args:
            alpha: Power law exponent for data covariance decay
            beta: Power law exponent for class mean decay
            zeta: Power law exponent for class frequency decay
            snr: Signal-to-noise ratio
            v: Abstract space dimension
            d: Feature dimension (embedded dimension)
            m: Number of classes (and experts)
            key: JAX random key
        """
        self.alpha = alpha
        self.beta = beta
        self.zeta = zeta
        self.snr = snr
        self.v = v
        self.d = d
        self.m = m

        # Split keys for different random components
        key_W, key_mu, key_sigma = random.split(key, 3)

        # Fixed random feature map W: (d, v)
        self.W = random.normal(key_W, (d, v)) / jnp.sqrt(d)

        # Class frequencies follow power law: p(i) ∝ i^(-zeta)
        class_indices = jnp.arange(1, m + 1)
        unnormalized_probs = class_indices ** (-zeta)
        self.class_probs = unnormalized_probs / jnp.sum(unnormalized_probs)

        # Class means μ_i ~ N(0, Σ^μ) where Σ^μ has power-law decay
        # Σ^μ is diagonal with entries j^(-2β)
        Sigma_mu_diag = self.snr * jnp.arange(1, v + 1) ** (-2 * beta)
        mu_standard = random.normal(key_mu, shape=(m, v))
        self.mu = (mu_standard * jnp.sqrt(Sigma_mu_diag)).T  # (v, m)

        # Data covariance Σ^ε is diagonal with entries j^(-2α)
        self.Sigma_eps_diag = jnp.arange(1, v + 1) ** (-2 * alpha)

    def generate_batch(self, key: random.PRNGKey, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate a batch of (X, y) training data.

        Args:
            key: JAX random key
            batch_size: Number of samples to generate

        Returns:
            X: Input features of shape (batch_size, d)
            y: One-hot class labels of shape (batch_size, m)
        """
        key1, key2 = random.split(key)

        # Sample class indices according to power-law distribution
        class_indices = random.categorical(key1, jnp.log(self.class_probs), shape=(batch_size,))

        # Generate points from selected class means with noise
        # x ~ N(μ_i, Σ^ε) where i is the selected class
        z = random.normal(key2, shape=(batch_size, self.v))
        # Scale by sqrt of diagonal covariance
        sqrt_Sigma_eps = jnp.sqrt(self.Sigma_eps_diag)
        x = z * sqrt_Sigma_eps
        # Add class means
        x = x + self.mu[:, class_indices].T

        # Project to feature space: x_proj = x @ W.T
        X = x @ self.W.T  # (batch_size, d)

        # Create one-hot labels
        y = jax.nn.one_hot(class_indices, self.m)  # (batch_size, m)

        return X, y

    def population_risk(self, params: jnp.ndarray) -> float:
        """Compute population cross-entropy risk.

        Args:
            params: Parameter tuple (theta, b) where theta is (d, m) and b is (m,)

        Returns:
            Population cross-entropy loss
        """
        theta, b = params

        # Monte Carlo estimate of population risk
        key = random.PRNGKey(42)  # Fixed seed for reproducible evaluation
        X, y_true = self.generate_batch(key, 10000)

        # Compute logits and probabilities
        logits = X @ theta + b  # (batch_size, m)
        y_pred = jax.nn.softmax(logits, axis=1)

        # Cross-entropy loss
        return -jnp.mean(jnp.sum(y_true * jnp.log(y_pred + 1e-8), axis=1))

    def cross_entropy_loss(self, params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute cross-entropy loss on given batch.

        Args:
            params: Parameter tuple (theta, b)
            X: Input features (batch_size, d)
            y: One-hot labels (batch_size, m)

        Returns:
            Cross-entropy loss
        """
        theta, b = params
        logits = X @ theta + b
        y_pred = jax.nn.softmax(logits, axis=1)
        return -jnp.mean(jnp.sum(y * jnp.log(y_pred + 1e-8), axis=1))


class LR_PLRFTrainer:
    """Trainer for Logistic Regression PLRF models."""

    def __init__(self, model: LogisticRegressionPLRF, optimizer: optax.GradientTransformation):
        """Initialize the LR-PLRF trainer.

        Args:
            model: LogisticRegressionPLRF instance
            optimizer: Optax optimizer
        """
        self.model = model
        self.optimizer = optimizer

    def train(self,
              key: random.PRNGKey,
              num_steps: int,
              batch_size: int,
              init_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
              eval_freq: Optional[int] = None,
              track_tau_stats: bool = False) -> Dict:
        """Train the LR-PLRF model and return training metrics.

        Args:
            key: JAX random key
            num_steps: Number of optimization steps
            batch_size: Batch size for SGD
            init_params: Initial parameters (theta, b)
            eval_freq: Frequency of evaluation
            track_tau_stats: If True, attempt to extract tau statistics (for compatibility)

        Returns:
            Dictionary containing:
                - timestamps: Evaluation timestamps
                - losses: Population losses
                - tau_statistics: Empty dict (for compatibility with tau tracking)
        """
        # Initialize parameters
        if init_params is None:
            key_init, key = random.split(key)
            theta_init = jnp.zeros((self.model.d, self.model.m))
            b_init = jnp.log(self.model.class_probs)  # Initialize bias with log class probs
            init_params = (theta_init, b_init)

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

        # Training step
        @jax.jit
        def train_step(params, opt_state, key):
            """Single SGD step for logistic regression."""
            # Generate batch
            X, y = self.model.generate_batch(key, batch_size)

            # Compute loss and gradients
            loss_fn = lambda p: self.model.cross_entropy_loss(p, X, y)
            loss, grads = jax.value_and_grad(loss_fn)(params)

            # Update parameters
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            return params, opt_state, loss

        # Training loop
        losses = [self.model.population_risk(init_params)]
        timestamps = [0]

        # Initialize empty tau statistics for compatibility
        tau_statistics = {
            'timestamps': [0],
            'tau_order_statistics': [jnp.array([])],
            'tau_reversed_order_statistics': [jnp.array([])],
            'tau_mean': [0.0],
            'tau_std': [0.0],
            'tau_min': [0.0],
            'tau_max': [0.0]
        }

        eval_idx = 1
        next_eval = eval_times[eval_idx] if eval_idx < len(eval_times) else num_steps + 1

        for step in tqdm.tqdm(range(num_steps)):
            # Split key
            key, subkey = random.split(key)

            # Perform training step
            params, opt_state, batch_loss_val = train_step(params, opt_state, subkey)

            # Evaluate if needed
            if step + 1 == next_eval:
                pop_risk = self.model.population_risk(params)
                losses.append(pop_risk)
                timestamps.append(step + 1)

                # Add empty tau statistics for compatibility
                if track_tau_stats:
                    tau_statistics['timestamps'].append(step + 1)
                    tau_statistics['tau_order_statistics'].append(jnp.array([]))
                    tau_statistics['tau_reversed_order_statistics'].append(jnp.array([]))
                    tau_statistics['tau_mean'].append(0.0)
                    tau_statistics['tau_std'].append(0.0)
                    tau_statistics['tau_min'].append(0.0)
                    tau_statistics['tau_max'].append(0.0)

                eval_idx += 1
                if eval_idx < len(eval_times):
                    next_eval = eval_times[eval_idx]
                else:
                    next_eval = num_steps + 1

        # Prepare results
        results = {
            'timestamps': jnp.array(timestamps),
            'losses': jnp.array(losses),
            'final_params': params
        }

        if track_tau_stats:
            results['tau_statistics'] = tau_statistics

        return results