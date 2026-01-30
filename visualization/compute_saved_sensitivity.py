#!/usr/bin/env python3
"""
Compute Saved Sensitivity Analysis

This script propagates uncertainty in LR predictions through to compute savings estimates.
It combines:
1. Bootstrap results from LR scaling law fits (from bighead_lr_scaling.py --bootstrap)
2. Curvature analysis results (from lr_curvature_analysis.py)
3. Loss scaling law parameters (can be manually specified or loaded)

The analysis quantifies:
- How LR uncertainty translates to loss uncertainty
- How loss differences translate to compute savings over AdamW
- Confidence intervals on compute savings claims

Usage:
    python compute_saved_sensitivity.py \
        --bootstrap-json adamw_bootstrap.json dana_mk4_bootstrap.json \
        --curvature-json curvature_results.json \
        --optimizer-names adamw dana-mk4 \
        --baseline adamw \
        --output-json compute_sensitivity.json

Or with manually specified scaling law parameters:
    python compute_saved_sensitivity.py \
        --scaling-law-a 2.5 \
        --scaling-law-b 0.5 \
        --scaling-law-c 0.12 \
        --adamw-loss-at-scale 3.45 \
        --opt-loss-at-scale 3.40 \
        --output-json compute_sensitivity.json
"""

import numpy as np
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description='Compute saved sensitivity analysis')

# Input files
parser.add_argument('--bootstrap-json', type=str, nargs='+', default=None,
                    help='Bootstrap results JSON files from bighead_lr_scaling.py')
parser.add_argument('--optimizer-names', type=str, nargs='+', default=None,
                    help='Names for each bootstrap file (same order as --bootstrap-json)')
parser.add_argument('--curvature-json', type=str, default=None,
                    help='Curvature analysis JSON file from lr_curvature_analysis.py')
parser.add_argument('--baseline', type=str, default='adamw',
                    help='Baseline optimizer for compute saved (default: adamw)')

# Manual scaling law parameters (alternative to loading from files)
parser.add_argument('--scaling-law-a', type=float, default=None,
                    help='Loss scaling law saturation parameter a')
parser.add_argument('--scaling-law-b', type=float, default=None,
                    help='Loss scaling law coefficient b')
parser.add_argument('--scaling-law-c', type=float, default=None,
                    help='Loss scaling law exponent c')

# Manual loss values at target scale
parser.add_argument('--target-scale', type=float, default=None,
                    help='Target scale (non-emb params) for analysis')
parser.add_argument('--adamw-loss-at-scale', type=float, default=None,
                    help='AdamW loss at target scale')
parser.add_argument('--opt-loss-at-scale', type=float, default=None,
                    help='Optimizer loss at target scale')
parser.add_argument('--opt-loss-std', type=float, default=None,
                    help='Optimizer loss standard deviation (for uncertainty)')

# Output
parser.add_argument('--output-json', type=str, default=None,
                    help='Output JSON file for results')
parser.add_argument('--confidence-level', type=float, default=0.95,
                    help='Confidence level for intervals (default: 0.95)')

args = parser.parse_args()

# =============================================================================
# LOSS SCALING LAW FUNCTIONS
# =============================================================================

def loss_scaling_law(compute: float, a: float, b: float, c: float) -> float:
    """
    Compute loss from scaling law: L = a + b * C^(-c)

    Args:
        compute: Compute in PetaFlop-Hours (or non-emb params)
        a: Saturation level
        b: Coefficient
        c: Exponent (positive)

    Returns:
        Loss value
    """
    return a + b * (compute ** (-c))


def compute_from_loss(loss: float, a: float, b: float, c: float) -> float:
    """
    Invert scaling law to get compute from loss: C = ((L - a) / b)^(-1/c)

    Args:
        loss: Loss value
        a: Saturation level
        b: Coefficient
        c: Exponent (positive)

    Returns:
        Compute value
    """
    if loss <= a:
        return float('inf')  # Loss below saturation means infinite compute needed
    return ((loss - a) / b) ** (-1/c)


def compute_saved(baseline_loss: float, opt_loss: float, a: float, b: float, c: float) -> float:
    """
    Compute savings: C(baseline) - C(opt)

    Returns the absolute compute saved (in same units as input).
    """
    c_baseline = compute_from_loss(baseline_loss, a, b, c)
    c_opt = compute_from_loss(opt_loss, a, b, c)
    return c_baseline - c_opt


def compute_saved_fraction(baseline_loss: float, opt_loss: float, a: float, b: float, c: float) -> float:
    """
    Compute savings as fraction: (C(baseline) - C(opt)) / C(baseline)

    Returns the fractional compute saved (0 to 1).
    """
    c_baseline = compute_from_loss(baseline_loss, a, b, c)
    c_opt = compute_from_loss(opt_loss, a, b, c)
    return (c_baseline - c_opt) / c_baseline


# =============================================================================
# CURVATURE-BASED LOSS UNCERTAINTY
# =============================================================================

def loss_increase_from_lr_error(lr_error_fraction: float, kappa: float) -> float:
    """
    Compute expected loss increase from LR error.

    Δloss = κ × (log(1 + ε))²

    Args:
        lr_error_fraction: Relative LR error (e.g., 0.2 for 20% error)
        kappa: Local curvature at the optimum

    Returns:
        Expected loss increase
    """
    return kappa * (np.log(1 + lr_error_fraction))**2


def lr_error_from_ci(lr_mean: float, lr_ci_lower: float, lr_ci_upper: float) -> float:
    """
    Estimate effective LR error from confidence interval.

    Returns the average relative error from the CI bounds.
    """
    err_lower = abs(lr_mean - lr_ci_lower) / lr_mean
    err_upper = abs(lr_ci_upper - lr_mean) / lr_mean
    return (err_lower + err_upper) / 2

# =============================================================================
# COMPUTE SAVED WITH UNCERTAINTY
# =============================================================================

def compute_saved_with_uncertainty(
    baseline_loss: float,
    opt_loss: float,
    opt_loss_std: float,
    a: float, b: float, c: float,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute savings with confidence intervals.

    Uses Gaussian approximation for loss uncertainty to propagate
    to compute saved.

    Args:
        baseline_loss: Baseline optimizer loss
        opt_loss: Optimizer loss (mean)
        opt_loss_std: Standard deviation of optimizer loss
        a, b, c: Scaling law parameters
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary with mean, lower, upper compute saved fractions
    """
    from scipy import stats

    z = stats.norm.ppf((1 + confidence_level) / 2)

    # Loss CI
    loss_lower = opt_loss - z * opt_loss_std
    loss_upper = opt_loss + z * opt_loss_std

    # Compute saved for each scenario
    # Note: lower loss -> more compute saved (optimistic)
    # Higher loss -> less compute saved (pessimistic)
    saved_mean = compute_saved_fraction(baseline_loss, opt_loss, a, b, c)
    saved_upper = compute_saved_fraction(baseline_loss, loss_lower, a, b, c)  # optimistic
    saved_lower = compute_saved_fraction(baseline_loss, loss_upper, a, b, c)  # pessimistic

    return {
        'mean': saved_mean,
        'ci_lower': saved_lower,
        'ci_upper': saved_upper,
        'loss_mean': opt_loss,
        'loss_ci_lower': loss_lower,
        'loss_ci_upper': loss_upper
    }

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def load_bootstrap_results(filepath: str) -> Dict[str, Any]:
    """Load bootstrap results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_curvature_results(filepath: str) -> Dict[str, Any]:
    """Load curvature analysis results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def propagate_lr_to_loss_uncertainty(
    bootstrap_results: Dict[str, Any],
    curvature_results: Dict[str, Any],
    target_params: float
) -> Tuple[float, float]:
    """
    Propagate LR uncertainty to loss uncertainty using curvature.

    Args:
        bootstrap_results: Bootstrap results for an optimizer
        curvature_results: Curvature scaling law results
        target_params: Target parameter count

    Returns:
        Tuple of (loss_std, lr_error_fraction)
    """
    # Get LR prediction CI at target scale
    lr_preds = bootstrap_results.get('lr_predictions', {})
    target_key = str(int(target_params))

    if target_key not in lr_preds:
        # Find closest key
        available_keys = [int(k) for k in lr_preds.keys()]
        if not available_keys:
            return None, None
        closest = min(available_keys, key=lambda x: abs(x - target_params))
        target_key = str(closest)

    lr_pred = lr_preds[target_key]
    lr_mean = lr_pred['lr_mean']
    lr_ci_lower = lr_pred['lr_ci_lower']
    lr_ci_upper = lr_pred['lr_ci_upper']

    # Compute effective LR error
    lr_error = lr_error_from_ci(lr_mean, lr_ci_lower, lr_ci_upper)

    # Get curvature at target scale
    kappa_0 = curvature_results['curvature_scaling_law']['kappa_0']
    alpha = curvature_results['curvature_scaling_law']['alpha']
    kappa = kappa_0 * (target_params ** alpha)

    # Compute loss uncertainty
    loss_increase = loss_increase_from_lr_error(lr_error, kappa)

    # Use this as ~1 std of loss (approximate)
    loss_std = loss_increase

    return loss_std, lr_error


if __name__ == '__main__':
    print("="*70)
    print("Compute Saved Sensitivity Analysis")
    print("="*70)

    results = {
        'confidence_level': args.confidence_level,
        'analysis': []
    }

    # Mode 1: Bootstrap + Curvature analysis
    if args.bootstrap_json and args.curvature_json:
        print("\nLoading bootstrap and curvature results...")

        curvature = load_curvature_results(args.curvature_json)
        print(f"  Curvature scaling law: κ = {curvature['curvature_scaling_law']['kappa_0']:.4e} × P^{curvature['curvature_scaling_law']['alpha']:.4f}")

        optimizer_results = {}
        for i, bootstrap_file in enumerate(args.bootstrap_json):
            opt_name = args.optimizer_names[i] if args.optimizer_names else f"opt_{i}"
            bootstrap = load_bootstrap_results(bootstrap_file)
            optimizer_results[opt_name] = bootstrap

            meta = bootstrap.get('metadata', {})
            print(f"  Loaded {opt_name}: {meta.get('scaling_rule')}, ω={meta.get('target_omega')}")

        # Get extrapolation sizes from curvature results
        if 'loss_sensitivity' in curvature:
            extrapolation_sizes = curvature['loss_sensitivity']['extrapolation_sizes']
        else:
            # Default Enoki sizes
            extrapolation_sizes = [8, 12, 16, 20, 24, 28, 32, 36, 40]

        # Need scaling law parameters - these would come from compare_scaling_rules.py
        # For now, use placeholder values that can be overridden
        print("\n  Note: Scaling law parameters (a, b, c) should be provided for compute saved analysis.")
        print("  Using default values - override with --scaling-law-a/b/c for accurate results.")

        a = args.scaling_law_a if args.scaling_law_a else 2.5
        b = args.scaling_law_b if args.scaling_law_b else 0.5
        c = args.scaling_law_c if args.scaling_law_c else 0.12

        print(f"\n  Scaling law: L = {a} + {b} × C^(-{c})")

        results['scaling_law'] = {'a': a, 'b': b, 'c': c}

        # Analyze LR uncertainty propagation
        print(f"\n{'='*70}")
        print("LR Uncertainty → Loss Uncertainty")
        print(f"{'='*70}")

        for opt_name, bootstrap in optimizer_results.items():
            print(f"\n{opt_name.upper()}:")

            lr_preds = bootstrap.get('lr_predictions', {})
            if not lr_preds:
                print("  No LR predictions available")
                continue

            for size_key in sorted(lr_preds.keys(), key=lambda x: int(x)):
                pred = lr_preds[size_key]
                params = pred['params']

                # Get curvature at this scale
                kappa_0 = curvature['curvature_scaling_law']['kappa_0']
                alpha = curvature['curvature_scaling_law']['alpha']
                kappa = kappa_0 * (params ** alpha)

                # LR error from CI
                lr_error = lr_error_from_ci(
                    pred['lr_mean'], pred['lr_ci_lower'], pred['lr_ci_upper']
                )

                # Loss increase
                loss_std = loss_increase_from_lr_error(lr_error, kappa)

                print(f"  P={params/1e6:.1f}M: LR={pred['lr_mean']:.2e} [{pred['lr_ci_lower']:.2e}, {pred['lr_ci_upper']:.2e}]")
                print(f"           LR error={lr_error*100:.1f}%, κ={kappa:.4f}, Δloss≈{loss_std:.4f}")

    # Mode 2: Manual parameters
    elif args.scaling_law_a and args.adamw_loss_at_scale and args.opt_loss_at_scale:
        print("\nUsing manually specified parameters...")

        a = args.scaling_law_a
        b = args.scaling_law_b if args.scaling_law_b else 0.5
        c = args.scaling_law_c if args.scaling_law_c else 0.12

        baseline_loss = args.adamw_loss_at_scale
        opt_loss = args.opt_loss_at_scale
        opt_loss_std = args.opt_loss_std if args.opt_loss_std else 0.01

        print(f"  Scaling law: L = {a} + {b} × C^(-{c})")
        print(f"  Baseline loss: {baseline_loss}")
        print(f"  Optimizer loss: {opt_loss} ± {opt_loss_std}")

        results['scaling_law'] = {'a': a, 'b': b, 'c': c}
        results['baseline_loss'] = baseline_loss
        results['opt_loss'] = opt_loss
        results['opt_loss_std'] = opt_loss_std

        print(f"\n{'='*70}")
        print("Compute Saved Analysis")
        print(f"{'='*70}")

        saved_results = compute_saved_with_uncertainty(
            baseline_loss, opt_loss, opt_loss_std,
            a, b, c, args.confidence_level
        )

        print(f"\nCompute saved vs baseline:")
        print(f"  Mean: {saved_results['mean']*100:.1f}%")
        print(f"  {args.confidence_level*100:.0f}% CI: [{saved_results['ci_lower']*100:.1f}%, {saved_results['ci_upper']*100:.1f}%]")

        results['compute_saved'] = saved_results

    else:
        print("\nNo input provided. Use one of:")
        print("  --bootstrap-json + --curvature-json + --optimizer-names")
        print("  --scaling-law-a + --adamw-loss-at-scale + --opt-loss-at-scale")
        exit(1)

    # Save results
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    print("\nDone.")
