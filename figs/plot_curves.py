#!/usr/bin/env python3
"""
Simple script to plot metric curves from Weights & Biases runs.

Examples:
  # Basic usage with individual filters
  python wandb/plot_curves.py \
    --entity my-entity \
    --project my-project \
    --metric val/accuracy \
    --tags best latest \
    --max-runs 20 \
    --x _step \
    --out ./wandb_plot.png
  
  # Complex filter expressions (new feature)
  python wandb/plot_curves.py \
    --entity my-entity \
    --project my-project \
    --metric train/loss \
    --filter-expr "(opt=adamw and lr=1e-3) or (opt=mars and dataset=c4)"
  
  # More complex expressions with multiple conditions
  python wandb/plot_curves.py \
    --entity my-entity \
    --project my-project \
    --metric val/accuracy \
    --filter-expr "((opt=adamw and lr=1e-3 and beta1=0.9) or (opt=mars and momentum=0.95)) and state=finished"

Note: When using --filter-expr, individual filter arguments (--opt, --lr, etc.) are ignored.

Requires: wandb, matplotlib
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import yaml
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union


def load_config(config_file: str = None) -> dict:
    """Load configuration from YAML file with environment variable and CLI overrides."""
    # Default config file path
    if config_file is None:
        script_dir = Path(__file__).parent
        config_file = script_dir / "plot_config.yaml"
    
    # Load config file
    config = {}
    if Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
    
    # Override with environment variables
    config['entity'] = config.get('entity') or os.getenv('WANDB_ENTITY')
    config['project'] = config.get('project') or os.getenv('WANDB_PROJECT')
    
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot curves from W&B API using config file")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (default: plot_config.yaml)")
    parser.add_argument("--entity", type=str, help="W&B entity (overrides config)")
    parser.add_argument("--project", type=str, help="W&B project (overrides config)")
    parser.add_argument("--metric", type=str, help="Metric to plot (overrides config)")
    parser.add_argument("--filter-expr", type=str, help="Filter expression (overrides config)")
    parser.add_argument("--out", type=str, help="Output file (overrides config)")
    parser.add_argument("--title", type=str, help="Plot title (overrides config)")
    parser.add_argument("--xlim", type=float, nargs=2, help="X-axis limits (overrides config)")
    parser.add_argument("--ylim", type=float, nargs=2, help="Y-axis limits (overrides config)")
    
    args = parser.parse_args()
    
    # Load config and merge with CLI args
    config = load_config(args.config)
    
    # CLI args override config file
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value
    
    # Convert back to namespace for compatibility
    return argparse.Namespace(**config)


def import_deps():
    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time error path
        print("wandb is required. Install with: pip install wandb", file=sys.stderr)
        raise

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
        raise

    return wandb, plt


def parse_value(value_str: str) -> Union[str, int, float, bool]:
    """Parse a string value to its appropriate type."""
    value_str = value_str.strip()
    
    # Boolean values
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'
    
    # Try integer
    try:
        if '.' not in value_str and 'e' not in value_str.lower():
            return int(value_str)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Return as string
    return value_str


def tokenize_expression(expr: str) -> List[str]:
    """Tokenize a filter expression into operators, parentheses, and conditions."""
    # Replace operators with spaces around them for easier splitting
    expr = re.sub(r'(\(|\))', r' \1 ', expr)
    expr = re.sub(r'\b(and|or)\b', r' \1 ', expr, flags=re.IGNORECASE)
    
    # Split and filter out empty strings
    tokens = [token.strip() for token in expr.split() if token.strip()]
    return tokens


def parse_condition(condition: str) -> Tuple[str, Union[str, int, float, bool], bool]:
    """Parse a single condition like 'opt=adamw', 'lr=1e-3', or 'tags!=hidden'."""
    if '!=' in condition:
        key, value = condition.split('!=', 1)
        negated = True
    elif '=' in condition:
        key, value = condition.split('=', 1)
        negated = False
    else:
        raise ValueError(f"Invalid condition format: {condition}. Expected format: key=value or key!=value")
    
    key = key.strip()
    value = parse_value(value)
    
    return key, value, negated


def evaluate_expression_on_run(expr_tokens: List[str], run) -> bool:
    """Evaluate a tokenized expression against a specific run."""
    # Convert infix to postfix using Shunting Yard algorithm
    output_queue = []
    operator_stack = []
    
    precedence = {'or': 1, 'and': 2}
    
    for token in expr_tokens:
        token_lower = token.lower()
        
        if token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())
            if operator_stack and operator_stack[-1] == '(':
                operator_stack.pop()  # Remove the '('
        elif token_lower in ('and', 'or'):
            while (operator_stack and operator_stack[-1] != '(' and
                   operator_stack[-1].lower() in precedence and
                   precedence[operator_stack[-1].lower()] >= precedence[token_lower]):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token_lower)
        else:
            # This is a condition
            output_queue.append(token)
    
    while operator_stack:
        output_queue.append(operator_stack.pop())
    
    # Evaluate postfix expression
    stack = []
    
    for token in output_queue:
        if token in ('and', 'or'):
            if len(stack) < 2:
                raise ValueError(f"Invalid expression: not enough operands for '{token}'")
            
            right = stack.pop()
            left = stack.pop()
            
            if token == 'and':
                result = left and right
            else:  # or
                result = left or right
            
            stack.append(result)
        else:
            # Evaluate condition
            try:
                key, expected_value, negated = parse_condition(token)
                
                # Handle special keys
                if key == 'tags':
                    # For tags, check if the expected tag is in the run's tags
                    run_tags = getattr(run, 'tags', []) or []
                    actual_value = expected_value in run_tags
                elif key == 'state':
                    actual_value = getattr(run, 'state', None) == expected_value
                else:
                    # Config key
                    actual_value = run.config.get(key)
                    if actual_value is None:
                        actual_value = False
                    else:
                        # Type-aware comparison
                        if isinstance(expected_value, bool):
                            actual_value = bool(actual_value)
                        elif isinstance(expected_value, (int, float)):
                            try:
                                actual_value = type(expected_value)(actual_value)
                            except (ValueError, TypeError):
                                actual_value = False
                        else:
                            actual_value = str(actual_value)
                        
                        actual_value = actual_value == expected_value
                
                # Apply negation if needed
                if negated:
                    actual_value = not actual_value
                
                stack.append(actual_value)
            except Exception as e:
                raise ValueError(f"Error evaluating condition '{token}': {e}")
    
    if len(stack) != 1:
        raise ValueError("Invalid expression: evaluation did not result in a single boolean value")
    
    return stack[0]


def build_filters(tags: Optional[List[str]], state: Optional[str], cfg_filters: Dict[str, Optional[object]], filter_expr: Optional[str] = None) -> Union[Dict, str]:
    # If filter expression is provided, return it for client-side filtering
    if filter_expr:
        return filter_expr
    
    # Otherwise, build traditional server-side filters
    filters: Dict = {}
    if tags:
        # Match runs that contain ANY of the provided tags
        filters["tags"] = {"$in": tags}
    if state:
        filters["state"] = state
    # Config-based equality filters
    for key, value in cfg_filters.items():
        if value is None:
            continue
        # Handle multiple values for any parameter
        if isinstance(value, list) and len(value) > 0:
            filters[f"config.{key}"] = {"$in": value}
        else:
            filters[f"config.{key}"] = value
    return filters


def iter_history(run, x_key: str, y_key: str) -> Iterable[Tuple[float, float]]:
    # Prefer scan_history for streaming to avoid loading entire history in memory
    history = run.scan_history(keys=[x_key, y_key])
    for row in history:
        x_val = row.get(x_key)
        y_val = row.get(y_key)
        if x_val is None or y_val is None:
            continue
        yield float(x_val), float(y_val)


# Consistent color mapping for optimizers
OPTIMIZER_COLORS = {
    "adamw": "#000000",        # Black
    "dana": "#ff7f0e",         # Orange
    "mars": "#2ca02c",         # Green
    "muon": "#d62728",         # Red
    "soap": "#9467bd",         # Purple
    "lion": "#8c564b",         # Brown
    "prodigy": "#e377c2",      # Pink
    "sophiag": "#7f7f7f",      # Gray
    "adopt": "#bcbd22",        # Olive
    "ademamix": "#17becf",     # Cyan
    "sf-adamw": "#ff9896",     # Light Red
    "signum": "#f7b6d3",       # Light Pink
    "d-muon": "#98df8a",       # Light Green
}

def get_dana_star_color_by_kappa(kappa: float) -> str:
    """Get dana-star color based on kappa value in [0.5, 1] range.
    
    Creates a smooth gradient from blue (kappa=0.5) to gray (kappa=1.0).
    Lower kappa = more blue, higher kappa = more gray.
    Values outside [0.5, 1] get a default color.
    """
    # Only apply gradient for kappa in [0.5, 1] range
    if kappa < 0.5 or kappa > 1.0:
        return "#CCCCCC"  # Light gray for out-of-range kappa values
    
    # Normalize kappa to [0, 1] within the [0.5, 1] range
    # But invert it so 0.5 maps to 1 and 1.0 maps to 0
    normalized_kappa = 1.0 - (kappa - 0.5) / 0.5
    
    # Color gradient from gray to blue
    # At kappa=1.0 (normalized=0): Gray (#808080)
    # At kappa=0.5 (normalized=1): Bright blue (#1E88E5)
    
    # RGB values for interpolation
    gray_r, gray_g, gray_b = 0x80, 0x80, 0x80     # Gray
    blue_r, blue_g, blue_b = 0x1E, 0x88, 0xE5     # Bright blue
    
    # Linear interpolation
    r = int(gray_r + (blue_r - gray_r) * normalized_kappa)
    g = int(gray_g + (blue_g - gray_g) * normalized_kappa)
    b = int(gray_b + (blue_b - gray_b) * normalized_kappa)
    
    # Convert to hex
    return f"#{r:02x}{g:02x}{b:02x}"

def get_line_style_by_param(run, param_name: str, param_values: List) -> str:
    """Get line style based on parameter value."""
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (1, 1))]  # Many different styles
    param_val = run.config.get(param_name)
    if param_val in param_values:
        idx = param_values.index(param_val) % len(line_styles)
        return line_styles[idx]
    return '-'  # default solid line

def get_style_by_scheduler_and_decay(run) -> tuple:
    """Get line style and marker based on scheduler and its specific hyperparameters."""
    scheduler = run.config.get("scheduler", "none")
    
    if scheduler == "wsd":
        # For WSD, use decay fraction to determine style
        wsd_fract_decay = run.config.get("wsd_fract_decay", 0.1)
        decay_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (1, 1))]
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        
        if wsd_fract_decay in decay_values:
            idx = decay_values.index(wsd_fract_decay)
        else:
            # Find closest value
            idx = min(range(len(decay_values)), key=lambda i: abs(decay_values[i] - wsd_fract_decay))
        
        return line_styles[idx % len(line_styles)], markers[idx % len(markers)]
    
    elif scheduler == "cos":
        # For cosine, could use other hyperparams or just default
        return '-', 'o'  # solid line, circle
    
    elif scheduler == "linear":
        return '-.', 's'  # dash-dot, square
    
    else:  # none
        return ':', '^'  # dotted, triangle

def get_marker_by_param(run, param_name: str, param_values: List) -> str:
    """Get marker style based on parameter value."""
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']  # many distinct markers
    param_val = run.config.get(param_name)
    if param_val in param_values:
        idx = param_values.index(param_val) % len(markers)
        return markers[idx]
    return 'o'  # default circle marker if value not in expected list

def get_optimizer_color(run) -> str:
    """Get consistent color for optimizer, with special handling for dana-star kappa."""
    opt = run.config.get("opt", "unknown")
    
    if opt == "dana-star":
        kappa = run.config.get("kappa", 1.0)  # Default kappa if not specified
        return get_dana_star_color_by_kappa(kappa)
    
    return OPTIMIZER_COLORS.get(opt, "#CCCCCC")  # Default to light gray


def analyze_run_parameter_differences(runs) -> List[str]:
    """Analyze and print parameters that vary across the filtered runs.
    
    Returns:
        List of parameter names that vary across runs (for use in legend).
    """
    print(f"\nAnalyzing parameter differences across {len(runs)} filtered runs...")
    all_config_keys = set()
    all_tag_values = set()
    config_values = {}
    
    # First pass: collect all possible config keys from all runs
    for run in runs:
        all_config_keys.update(run.config.keys())
    
    # Second pass: for each key, collect values from all runs (including null for missing keys)
    for key in all_config_keys:
        config_values[key] = set()
        for run in runs:
            if key in run.config:
                config_values[key].add(str(run.config[key]))
            else:
                config_values[key].add('null')  # Mark missing parameters as null
    
    for run in runs:
        # Collect tags
        tags = getattr(run, 'tags', []) or []
        all_tag_values.update(tags)
        
        # Add special fields
        if 'state' not in config_values:
            config_values['state'] = set()
        config_values['state'].add(getattr(run, 'state', 'unknown'))
    
    # Find parameters that vary across runs
    varying_params = []
    
    # Check config parameters
    for key, values in config_values.items():
        if len(values) > 1:  # Parameter has different values across runs
            varying_params.append({
                'parameter': key,
                'type': 'config' if key != 'state' else 'metadata',
                'values': sorted(list(values)),
                'count': len(values)
            })
    
    # Check if tags vary (some runs have different tag combinations)
    if len(all_tag_values) > 0:
        tag_combinations = set()
        for run in runs:
            tags = tuple(sorted(getattr(run, 'tags', []) or []))
            tag_combinations.add(tags)
        
        if len(tag_combinations) > 1:
            varying_params.append({
                'parameter': 'tags',
                'type': 'metadata', 
                'values': [list(combo) for combo in sorted(tag_combinations)],
                'count': len(tag_combinations)
            })
    
    # Sort by number of different values (most varying first)
    varying_params.sort(key=lambda x: x['count'], reverse=True)
    
    print(f"\nParameters that vary across the {len(runs)} filtered runs:")
    print("=" * 60)
    
    if not varying_params:
        print("All runs have identical parameters (no variation found).")
    else:
        for param in varying_params:
            print(f"\n• {param['parameter']} ({param['type']}):")
            print(f"  {param['count']} different values: {param['values']}")
    
    print("=" * 60)
    
    # Return list of varying parameter names for legend use
    varying_param_names = [param['parameter'] for param in varying_params if param['type'] == 'config']
    return varying_param_names


def build_legend_label(run, legend_keys: List[str]) -> str:
    """Build a legend label from selected config keys and run metadata."""
    parts = []
    opt = run.config.get("opt", "unknown")
    scheduler = run.config.get("scheduler", "none")
    
    for key in legend_keys:
        if key == "name":
            parts.append(run.name or "unnamed")
        elif key == "id":
            parts.append(run.id[:8])  # Short ID
        elif key == "opt":
            # For dana-star, automatically include kappa without marker symbol
            if opt == "dana-star":
                kappa = run.config.get("kappa", 1.0)
                parts.append(f"dana-star (κ={kappa:.2g})")
            else:
                # For other optimizers, show learning rate without marker symbol
                lr = run.config.get("lr", 1e-3)
                parts.append(f"{opt} (lr={lr:.0e})")
        elif key == "scheduler":
            # Scheduler info with specific hyperparameters (line style/marker determined by function)
            if scheduler == "wsd":
                wsd_fract_decay = run.config.get("wsd_fract_decay", 0.1)
                parts.append(f"wsd (decay={wsd_fract_decay:.2g})")
            elif scheduler == "none":
                parts.append("no-sched")
            else:
                parts.append(f"{scheduler}")
        elif key == "kappa" and opt != "dana-star":
            # Skip kappa for non-dana-star optimizers
            continue
        elif key == "wd_decaying":
            # Handle wd_decaying: show null for optimizers that don't support it
            config_val = run.config.get(key)
            if config_val is not None:
                val_str = str(config_val).lower()
                parts.append(f"wd_decay={val_str}")
            else:
                parts.append("wd_decay=null")
        else:
            # Try to get from config
            config_val = run.config.get(key)
            if config_val is not None:
                # Format boolean/numeric values nicely
                if isinstance(config_val, bool):
                    val_str = str(config_val).lower()
                elif isinstance(config_val, float):
                    val_str = f"{config_val:.3g}"
                else:
                    val_str = str(config_val)
                parts.append(f"{key}={val_str}")
            else:
                # Show null for missing parameters
                parts.append(f"{key}=null")
    
    return ", ".join(parts) if parts else (run.name or run.id[:8])


def main() -> None:
    args = parse_args()
    wandb, plt = import_deps()

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}"
    
    # Use filter_expr directly since config-based approach focuses on expressions
    filters = getattr(args, 'filter_expr', None)
    if not filters:
        print("No filter expression specified in config. Please set 'filter_expr' in your config file.")
        sys.exit(1)

    try:
        # Complex expression - get all runs and filter client-side
        runs = api.runs(project_path, order="-created_at")
        expr_tokens = tokenize_expression(filters)
        
        # Filter runs using the expression
        filtered_runs = []
        for run in runs:
            try:
                if evaluate_expression_on_run(expr_tokens, run):
                    filtered_runs.append(run)
            except Exception as exc:
                print(f"Error evaluating expression for run {run.id}: {exc}", file=sys.stderr)
                continue
        runs = filtered_runs
        print(f"Found {len(runs)} matching runs")
    except Exception as exc:
        print(f"Failed to load runs for {project_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not runs:
        print("No runs found with given filters.")
        print("Try removing some filters or check your project/entity names.")
        return
    
    # Analyze parameter differences across filtered runs
    varying_legend_keys = analyze_run_parameter_differences(runs)

    # Trim to max runs
    max_runs = getattr(args, 'max_runs', 50)
    runs = list(runs)[: max(0, int(max_runs))]

    if not runs:
        print("No runs to plot after applying max-runs limit.")
        return

    plt.figure(figsize=(10, 6))

    # Create unique style mapping for each unique combination of varying parameters
    unique_combinations = {}
    combination_styles = {}
    style_options = {
        'linestyles': ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (1, 1))],
        'markers': ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    }
    
    # First pass: identify unique combinations
    for run in runs:
        legend_keys = varying_legend_keys if varying_legend_keys else getattr(args, 'legend_keys', ['opt', 'scheduler'])
        
        # Create a tuple of varying parameter values for this run
        combination_values = []
        for key in legend_keys:
            if key in run.config:
                combination_values.append(str(run.config[key]))
            else:
                combination_values.append('null')
        
        combination_tuple = tuple(combination_values)
        if combination_tuple not in unique_combinations:
            unique_combinations[combination_tuple] = len(unique_combinations)
    
    # Assign styles to each unique combination
    for combo_tuple, idx in unique_combinations.items():
        linestyle_idx = idx % len(style_options['linestyles'])
        marker_idx = idx % len(style_options['markers'])
        combination_styles[combo_tuple] = {
            'linestyle': style_options['linestyles'][linestyle_idx],
            'marker': style_options['markers'][marker_idx]
        }

    # Collect all runs with their data and labels for sorting
    plot_data = []
    for i, run in enumerate(runs):
        try:
            x_key = getattr(args, 'x', 'iter')
            metric = getattr(args, 'metric', 'val/loss')
            
            points = list(iter_history(run, x_key, metric))
        except Exception as exc:
            print(f"Skipping run {run.id} due to history error: {exc}", file=sys.stderr)
            continue

        if not points:
            continue

        # Sort by x in case history returns out of order
        points.sort(key=lambda p: p[0])
        
        # Filter out x=0 points for log scale
        if getattr(args, 'xlog', False):
            points = [(x, y) for x, y in points if x > 0]
        
        if not points:
            continue
            
        xs, ys = zip(*points)

        # Use varying parameters for legend, fallback to config if none found
        legend_keys = varying_legend_keys if varying_legend_keys else getattr(args, 'legend_keys', ['opt', 'scheduler'])
        label = build_legend_label(run, legend_keys)
        color = get_optimizer_color(run)
        opt = run.config.get("opt", "unknown")
        
        # Get style based on unique combination of varying parameters
        combination_values = []
        for key in legend_keys:
            if key in run.config:
                combination_values.append(str(run.config[key]))
            else:
                combination_values.append('null')
        combination_tuple = tuple(combination_values)
        
        style_info = combination_styles[combination_tuple]
        linestyle = style_info['linestyle']
        marker = style_info['marker']
        
        plot_data.append({
            'xs': xs,
            'ys': ys,
            'label': label,
            'color': color,
            'linestyle': linestyle,
            'marker': marker,
            'opt': opt,
            'run': run
        })

    if not plot_data:
        print("No curves plotted (no runs had the requested metric).")
        return

    # Sort plot data by optimizer name, with dana-star sorted by kappa within its group
    plot_data.sort(key=lambda data: (
        data['opt'], 
        data['run'].config.get("kappa", 1.0) if data['opt'] == "dana-star" else 0
    ))

    # Plot all curves in sorted order
    for data in plot_data:
        plt.plot(data['xs'], data['ys'], 
                label=data['label'], 
                color=data['color'], 
                linestyle=data['linestyle'],
                marker=data['marker'],
                linewidth=2.0,
                alpha=0.85,
                markersize=6,
                markevery=max(1, len(data['xs']) // 20),
                markerfacecolor=data['color'],
                markeredgecolor='white',
                markeredgewidth=0.5)

    title = getattr(args, 'title', None) or f"{getattr(args, 'project', 'Project')} — {getattr(args, 'metric', 'Metric')}"
    plt.title(title)
    plt.xlabel(getattr(args, 'x', 'iter'))
    plt.ylabel(getattr(args, 'metric', 'val/loss'))
    
    # Enhanced grid for better visibility
    plt.grid(True, which="major", linestyle="-", alpha=0.3)
    plt.grid(True, which="minor", linestyle=":", alpha=0.2)
    
    legend_loc = getattr(args, 'legend_loc', 'best')
    plt.legend(loc=legend_loc, fontsize="small")
    
    # Set axis limits BEFORE setting log scale
    xlim = getattr(args, 'xlim', None)
    ylim = getattr(args, 'ylim', None)
    
    # Convert string values to float if needed (from YAML parsing)
    if xlim and isinstance(xlim, list):
        xlim = [float(x) for x in xlim]
    if ylim and isinstance(ylim, list):
        ylim = [float(y) for y in ylim]
    
    # Smart xlim: ensure limits actually contain data
    if xlim and plot_data:
        # Get actual data range
        all_xs = []
        for data in plot_data:
            all_xs.extend(data['xs'])
        actual_min, actual_max = min(all_xs), max(all_xs)
        
        # Adjust xlim to ensure it contains actual data
        adjusted_min = min(xlim[0], actual_min)
        adjusted_max = max(xlim[1], actual_max)
        
        # Only apply xlim if it overlaps with actual data
        if xlim[1] >= actual_min and xlim[0] <= actual_max:
            plt.xlim(max(xlim[0], actual_min), min(xlim[1], actual_max))
        else:
            print(f"Warning: xlim [{xlim[0]:.2e}, {xlim[1]:.2e}] doesn't overlap with data [{actual_min:.2e}, {actual_max:.2e}]. Using auto-scale.")
    
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    
    # Set logarithmic scales AFTER setting limits
    if getattr(args, 'xlog', False):
        plt.xscale('log')
        # Simple, clean x-axis formatting for tokens
        from matplotlib.ticker import FuncFormatter
        def billions_formatter(x, pos):
            if x >= 1e9:
                return f'{x/1e9:.1f}B'
            elif x >= 1e6:
                return f'{x/1e6:.0f}M'
            else:
                return f'{x:.0f}'
        plt.gca().xaxis.set_major_formatter(FuncFormatter(billions_formatter))
        plt.gca().tick_params(axis='x', which='major', labelsize=10)
    if getattr(args, 'ylog', False):
        plt.yscale('log')
    
    plt.tight_layout()

    out_file = getattr(args, 'out', None)
    if out_file:
        # Automatically prefix with figs/ if not an absolute path
        if not os.path.isabs(out_file):
            out_file = os.path.join("figs", out_file)
        
        # Ensure output directory exists
        out_dir = os.path.dirname(os.path.abspath(out_file))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_file, dpi=200)
        print(f"Saved figure to {out_file}")
    else:
        plt.show()


if __name__ == "__main__":
    main()


