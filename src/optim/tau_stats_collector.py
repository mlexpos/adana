"""
Tau Statistics Collector for dana-star-mk4 Optimizer

This module collects tau order statistics from the dana-star-mk4 optimizer during
transformer training. Statistics are collected at exponentially spaced eval steps
(10, 20, 40, 80, ...) and stored to disk for later analysis.

The collection mimics the JAX implementation in jax/taustats_generate.py but is
adapted for PyTorch and transformer models.
"""

import os
import pickle
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class TauStatsCollector:
    """Collects and stores tau statistics from dana-star-mk4 optimizer."""

    def __init__(
        self,
        exp_dir: Path,
        cfg: Any,
        param_names: List[str],
    ):
        """
        Initialize tau statistics collector.

        Args:
            exp_dir: Experiment directory for storing tau stats
            cfg: Configuration object with model and optimizer params
            param_names: List of parameter names being optimized
        """
        self.exp_dir = Path(exp_dir)
        self.cfg = cfg
        self.param_names = param_names

        # Create subdirectory for tau stats
        self.tau_stats_dir = self.exp_dir / "tau_stats"
        self.tau_stats_dir.mkdir(parents=True, exist_ok=True)

        # Determine collection schedule: eval steps at 10, 20, 40, 80, 160, ...
        # Since eval_interval determines when evals occur, we collect at specific eval numbers
        self.eval_step_counter = 0
        self.collection_eval_steps = self._compute_collection_schedule()
        self.next_collection_idx = 0

        print(f"[TauStats] Initialized collector")
        print(f"[TauStats] Will collect at eval steps: {self.collection_eval_steps[:10]}...")
        print(f"[TauStats] Storing in: {self.tau_stats_dir}")

        # Save metadata once at initialization
        self._save_metadata()

    def _compute_collection_schedule(self) -> List[int]:
        """
        Compute collection schedule: eval steps 10, 20, 40, 80, ...

        Returns:
            List of eval step numbers to collect at (2^k * 10)
        """
        # Maximum eval steps we might reach (generous upper bound)
        max_eval_steps = self.cfg.iterations // self.cfg.eval_interval + 10

        schedule = []
        k = 0
        while True:
            eval_step = 10 * (2 ** k)
            if eval_step > max_eval_steps:
                break
            schedule.append(eval_step)
            k += 1

        return schedule

    def should_collect(self, curr_iter: int, is_eval_step: bool) -> bool:
        """
        Determine if we should collect tau statistics at this iteration.

        Args:
            curr_iter: Current training iteration
            is_eval_step: Whether this is an eval step

        Returns:
            True if we should collect tau statistics
        """
        if not is_eval_step:
            return False

        # Increment eval step counter
        self.eval_step_counter += 1

        # Check if this eval step is in our collection schedule
        if self.next_collection_idx >= len(self.collection_eval_steps):
            return False

        next_target = self.collection_eval_steps[self.next_collection_idx]
        if self.eval_step_counter == next_target:
            self.next_collection_idx += 1
            return True

        return False

    def collect(self, opt: torch.optim.Optimizer, curr_iter: int):
        """
        Collect tau statistics from optimizer at current iteration.

        Args:
            opt: Optimizer (should be dana-star-mk4)
            curr_iter: Current training iteration
        """
        print(f"[TauStats] Collecting at iteration {curr_iter} (eval step {self.eval_step_counter})")

        # Extract tau statistics from optimizer
        tau_stats = self._extract_tau_statistics(opt)

        if not tau_stats:
            print(f"[TauStats] Warning: No tau statistics found in optimizer state")
            return

        # Save to disk
        self._save_tau_stats(tau_stats, curr_iter)

        print(f"[TauStats] Collected {len(tau_stats)} parameter tau statistics")

    def _extract_tau_statistics(self, opt: torch.optim.Optimizer) -> Dict[str, Dict]:
        """
        Extract tau statistics from dana-star-mk4 optimizer state.

        Args:
            opt: Optimizer with tau in state

        Returns:
            Dictionary mapping parameter names to tau statistics
        """
        tau_stats = {}

        # Iterate through optimizer state
        for group in opt.param_groups:
            for param_idx, param in enumerate(group['params']):
                if param not in opt.state:
                    continue

                state = opt.state[param]
                if 'tau' not in state:
                    continue

                # Get tau tensor and flatten it
                tau = state['tau']
                tau_flat = tau.detach().cpu().flatten().numpy()

                # Compute order statistics
                largest, smallest = compute_tau_order_statistics(tau_flat)

                # Get parameter name (try to get from param_names if available)
                param_name = self._get_param_name(param, group, param_idx)

                tau_stats[param_name] = {
                    'shape': tuple(tau.shape),
                    'largest_order_stats': largest,
                    'smallest_order_stats': smallest,
                    'num_elements': len(tau_flat),
                }

        return tau_stats

    def _get_param_name(self, param: torch.Tensor, group: Dict, param_idx: int) -> str:
        """
        Get descriptive name for parameter.

        Args:
            param: Parameter tensor
            group: Optimizer parameter group
            param_idx: Index of parameter in group

        Returns:
            Parameter name string
        """
        # Try to get from param_names in group
        if 'param_names' in group and param_idx < len(group['param_names']):
            return group['param_names'][param_idx]

        # Fallback to generic name
        return f"param_group_{id(group)}_idx_{param_idx}"

    def _save_tau_stats(self, tau_stats: Dict, curr_iter: int):
        """
        Save tau statistics to disk.

        Args:
            tau_stats: Dictionary of tau statistics
            curr_iter: Current iteration
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tau_stats_iter_{curr_iter:08d}_{timestamp}.pkl"
        filepath = self.tau_stats_dir / filename

        data = {
            'metadata': {
                'timestamp': timestamp,
                'iteration': curr_iter,
                'eval_step_number': self.eval_step_counter,
            },
            'tau_statistics': tau_stats,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"[TauStats] Saved to: {filepath}")

    def _save_metadata(self):
        """Save configuration metadata to JSON file."""
        # Determine architecture name
        architecture = self.cfg.model
        if hasattr(self.cfg, 'n_head'):
            architecture += f"_heads{self.cfg.n_head}"

        metadata = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'optimizer': self.cfg.opt,
            'architecture': architecture,
            'model_params': self._extract_model_params(),
            'optimizer_params': self._extract_optimizer_params(),
            'training_params': self._extract_training_params(),
            'loss_params': self._extract_loss_params(),
            'initialization_params': self._extract_init_params(),
            'scheduler_params': self._extract_scheduler_params(),
            'collection_schedule': self.collection_eval_steps,
        }

        filepath = self.tau_stats_dir / "tau_stats_metadata.json"
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[TauStats] Saved metadata to: {filepath}")

    def _extract_model_params(self) -> Dict:
        """Extract model parameters from config."""
        params = {}
        model_attrs = ['model', 'n_head', 'n_layer', 'n_embd', 'head_dim',
                      'qkv_dim', 'mlp_hidden_dim', 'mlp_hidden_mult', 'vocab_size',
                      'dropout', 'bias', 'weight_tying', 'parallel_block',
                      'normalization_layer_type', 'elementwise_attn_output_gate']
        for attr in model_attrs:
            if hasattr(self.cfg, attr):
                val = getattr(self.cfg, attr)
                # Convert to native Python types for JSON serialization
                if isinstance(val, (np.integer, np.floating)):
                    val = val.item()
                params[attr] = val

        # Add QK normalization status (inverse of no_qknorm flag)
        if hasattr(self.cfg, 'no_qknorm'):
            params['qk_normalization'] = not self.cfg.no_qknorm

        return params

    def _extract_optimizer_params(self) -> Dict:
        """Extract optimizer parameters from config."""
        params = {}
        opt_attrs = ['lr', 'delta', 'kappa', 'clipsnr', 'weight_decay',
                    'mk4A', 'mk4B', 'omega', 'beta1', 'beta2',
                    'wd_decaying', 'wd_ts', 'grad_clip']
        for attr in opt_attrs:
            if hasattr(self.cfg, attr):
                val = getattr(self.cfg, attr)
                # Convert to native Python types for JSON serialization
                if isinstance(val, (np.integer, np.floating)):
                    val = val.item()
                params[attr] = val
        return params

    def _extract_training_params(self) -> Dict:
        """Extract training parameters from config."""
        params = {}
        training_attrs = ['iterations', 'iterations_to_run', 'batch_size', 'acc_steps',
                         'sequence_length', 'eval_interval', 'log_interval',
                         'warmup_steps', 'seed', 'data_seed']
        for attr in training_attrs:
            if hasattr(self.cfg, attr):
                val = getattr(self.cfg, attr)
                # Convert to native Python types for JSON serialization
                if isinstance(val, (np.integer, np.floating)):
                    val = val.item()
                params[attr] = val
        return params

    def _extract_loss_params(self) -> Dict:
        """Extract loss-related parameters from config."""
        params = {}
        loss_attrs = ['z_loss_coeff', 'hoyer_loss_coeff']
        for attr in loss_attrs:
            if hasattr(self.cfg, attr):
                val = getattr(self.cfg, attr)
                # Convert to native Python types for JSON serialization
                if isinstance(val, (np.integer, np.floating)):
                    val = val.item()
                params[attr] = val
        return params

    def _extract_init_params(self) -> Dict:
        """Extract initialization parameters from config."""
        params = {}
        init_attrs = ['init_scheme', 'init_std', 'residual_stream_scalar']
        for attr in init_attrs:
            if hasattr(self.cfg, attr):
                val = getattr(self.cfg, attr)
                # Convert to native Python types for JSON serialization
                if isinstance(val, (np.integer, np.floating)):
                    val = val.item()
                params[attr] = val
        return params

    def _extract_scheduler_params(self) -> Dict:
        """Extract learning rate scheduler parameters from config."""
        params = {}
        scheduler_attrs = ['scheduler', 'div_factor', 'final_div_factor',
                          'cos_inf_steps', 'wsd_final_lr_scale', 'wsd_fract_decay',
                          'decay_type']
        for attr in scheduler_attrs:
            if hasattr(self.cfg, attr):
                val = getattr(self.cfg, attr)
                # Convert to native Python types for JSON serialization
                if isinstance(val, (np.integer, np.floating)):
                    val = val.item()
                params[attr] = val
        return params


def compute_tau_order_statistics(tau_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute order statistics for tau vector.

    This matches the implementation in jax/taustats_generate.py.
    Returns sparse order statistics at positions 1.1^k for efficient storage
    and log-log plotting.

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
    sorted_tau_asc = np.sort(tau_vector)

    # Compute powers of 1.1 up to n, similar to evaluation times
    max_k = int(np.ceil(np.log(n) / np.log(1.1)))
    indices = np.int32(1.1 ** np.arange(max_k + 1)) - 1  # 0-indexed: [0, 0, 1, 2, 3, 4, ...]

    # Remove duplicates and clamp to valid range
    indices = np.unique(indices)
    indices = np.minimum(indices, n - 1)

    # Get smallest order statistics
    smallest_order_stats = sorted_tau_asc[indices]

    # Get largest order statistics using reversed indices
    # For largest: indices from the end of the sorted array
    reversed_indices = (n - 1) - indices
    largest_order_stats = sorted_tau_asc[reversed_indices]

    return largest_order_stats, smallest_order_stats
