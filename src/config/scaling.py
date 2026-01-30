"""
Scaling rules for auto-computing model dimensions and LR from --heads.

Usage:
    apply_scaling_rule(args) modifies args in-place if --scaling_rule is set.
"""

import os
import yaml
from pathlib import Path


def _load_scaling_rules():
    yaml_path = Path(__file__).parent / "scaling_rules.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def compute_dimensions(rule_name, heads):
    """Compute model dimensions from heads count using scaling rules.

    Returns dict with: n_head, n_layer, n_embd, mlp_hidden_dim, qkv_dim, non_emb_params
    """
    rules = _load_scaling_rules()
    if rule_name not in rules:
        raise ValueError(f"Unknown scaling rule: '{rule_name}'. Available: {list(rules.keys())}")

    rule = rules[rule_name]
    head_dim = rule["head_dim"]
    n_head = heads
    n_layer = eval(rule["n_layer_formula"], {"heads": heads})
    n_embd = eval(rule["n_embd_formula"], {"heads": heads})
    mlp_hidden = eval(rule["mlp_hidden_formula"], {"n_embd": n_embd})
    non_emb = eval(rule["non_emb_formula"], {"n_embd": n_embd, "n_layer": n_layer})
    total_params = eval(rule["total_params_formula"], {"non_emb": non_emb, "n_embd": n_embd})

    return {
        "n_head": n_head,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "mlp_hidden_dim": mlp_hidden,
        "qkv_dim": head_dim,
        "non_emb_params": non_emb,
        "total_params": total_params,
        "head_dim": head_dim,
    }


def compute_lr(rule_name, optimizer_name, non_emb_params, kappa=None, batch_size=None):
    """Compute LR from saturated power law: LR = a * (b + non_emb_params)^d

    For DANA variants, the formula depends on kappa. If the formula entry is a
    nested dict (keyed by kappa strings like "0.85"), the kappa parameter is used
    to look up the correct coefficients. Falls back to closest available kappa
    if exact match not found.

    If batch_size is 512 (global batch), looks up lr_formula_512 first, falling
    back to lr_formula (batch-32) if not found or if coefficients are null.
    """
    rules = _load_scaling_rules()
    rule = rules[rule_name]

    # Select formula key based on batch size
    formula_key = "lr_formula"
    if batch_size == 512:
        formula_key = "lr_formula_512" if "lr_formula_512" in rule else "lr_formula"

    lr_formulas = rule.get(formula_key, {})

    # Fall back to lr_formula if optimizer not in batch-specific formulas
    if optimizer_name not in lr_formulas and formula_key != "lr_formula":
        lr_formulas = rule.get("lr_formula", {})

    if optimizer_name not in lr_formulas:
        return None

    entry = lr_formulas[optimizer_name]

    # Check if this is a kappa-dependent (nested) entry
    if "a" in entry:
        # Flat entry: {a, b, d}
        coeffs = entry
    else:
        # Nested entry keyed by kappa: {"0.75": {a,b,d}, "0.85": {a,b,d}, ...}
        if kappa is None:
            kappa = 0.85  # default
        kappa_str = f"{float(kappa):.2f}"
        # Remove trailing zero for keys like "0.80" -> try both "0.8" and "0.80"
        if kappa_str in entry:
            coeffs = entry[kappa_str]
        else:
            # Try without trailing zero
            kappa_str_short = str(float(kappa))
            if kappa_str_short in entry:
                coeffs = entry[kappa_str_short]
            else:
                # Fall back to closest available kappa
                available = sorted(entry.keys(), key=lambda k: abs(float(k) - float(kappa)))
                closest = available[0]
                coeffs = entry[closest]

    a, b, d = coeffs["a"], coeffs["b"], coeffs["d"]

    # If coefficients are null (TBD placeholders), fall back to batch-32 formulas
    if a is None or b is None or d is None:
        if formula_key != "lr_formula":
            return compute_lr(rule_name, optimizer_name, non_emb_params, kappa=kappa, batch_size=None)
        return None

    lr = a * (b + non_emb_params) ** d
    return lr


def apply_scaling_rule(args):
    """Apply scaling rule to args, auto-computing dimensions and LR.

    Modifies args in-place. Only acts if args.scaling_rule != "none".
    """
    if getattr(args, "scaling_rule", "none") == "none":
        return

    if args.heads is None:
        raise ValueError("--heads is required when --scaling_rule is set")

    rule_name = args.scaling_rule
    dims = compute_dimensions(rule_name, args.heads)

    # Load rule config for model name
    rules = _load_scaling_rules()
    rule = rules[rule_name]

    # Set model architecture
    args.model = rule["model"]
    args.n_head = dims["n_head"]
    args.n_layer = dims["n_layer"]
    args.n_embd = dims["n_embd"]
    args.mlp_hidden_dim = dims["mlp_hidden_dim"]
    args.qkv_dim = dims["qkv_dim"]
    args.weight_tying = rule.get("weight_tying", False)

    # Compute LR if not explicitly set (check if user passed --lr)
    # We use a sentinel: if lr is still at the default 1e-3, try to auto-compute
    kappa = getattr(args, "kappa", None)
    lr = compute_lr(rule_name, args.opt, dims["non_emb_params"], kappa=kappa)
    if lr is not None:
        args.lr = lr

    # Compute Chinchilla-optimal iterations: tokens = 20 * total_params
    # iterations = tokens / (batch_size * acc_steps * sequence_length * world_size)
    # We set iterations here but it can be overridden on the CLI
    chinchilla_tokens = 20 * dims["total_params"]
    tokens_per_step = args.batch_size * args.acc_steps * args.sequence_length
    args.iterations = int(chinchilla_tokens / tokens_per_step)

    print(f"[Scaling Rule: {rule_name}] heads={args.heads}")
    print(f"  n_head={args.n_head}, n_layer={args.n_layer}, n_embd={args.n_embd}")
    print(f"  mlp_hidden_dim={args.mlp_hidden_dim}, qkv_dim={args.qkv_dim}")
    print(f"  non_emb_params={dims['non_emb_params']:,.0f} ({dims['non_emb_params']/1e6:.1f}M)")
    print(f"  total_params={dims['total_params']:,.0f} ({dims['total_params']/1e6:.1f}M)")
    print(f"  lr={args.lr:.6e}, iterations={args.iterations}")
