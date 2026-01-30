"""
Optimizer color scheme for visualization scripts.

This module provides a consistent color scheme for optimizer types
used across scaling comparison and other visualization scripts.
"""

# Color scheme for optimizers - expanded with all variants
OPT_COLORS = {
    'adamw': '#1f77b4',           # Deep blue
    'adamw-decaying-wd': '#00CED1', # Dark turquoise
    'mk4': '#d62728',              # Red
    'dana': '#2ca02c',             # Green
    'dana-mk4': '#8c564b',         # Brown
    'dana-mk4-kappa-0-75': '#A0522D', # Sienna (DANA-MK4 with kappa=0.75)
    'dana-mk4-kappa-0-85': '#CD853F', # Peru (DANA-MK4 with kappa=0.85)
    'dana-star': '#7f7f7f',        # Gray
    'dana-star-mk4': '#d62728',    # Red (same as mk4)
    'dana-star-mk4-kappa-0-75': '#DC143C',   # Crimson (DANA-Star-MK4 with kappa=0.75)
    'dana-star-mk4-kappa-0-85': '#FF6347',   # Tomato red (DANA-Star-MK4 with kappa=0.85)
    'adana': '#bcbd22',            # Olive/Yellow-green (alias for dana-star-no-tau)
    'adana-kappa-0-85': '#228B22', # Forest green (ADANA with kappa=0.85)
    'dana-star-no-tau': '#bcbd22', # Olive/Yellow-green
    'dana-star-no-tau-kappa-1-0': '#00FF00',   # Light green
    'dana-star-no-tau-kappa-0-8': '#006400',   # Dark green
    'dana-star-no-tau-kappa-0-85': '#228B22',  # Forest green
    'dana-star-no-tau-kappa-0-9': '#32CD32',   # Lime green
    'dana-star-no-tau-dana-constant': '#FF6347',   # Tomato red
    'dana-star-no-tau-beta2-constant': '#1E90FF',   # Dodger blue
    'dana-star-no-tau-beta1': '#FF1493',   # Deep pink
    'dana-star-no-tau-dana-constant-beta2-constant': '#8A2BE2',   # Blue violet
    'dana-star-no-tau-dana-constant-beta1': '#FF8C00',   # Dark orange
    'dana-star-no-tau-dana-constant-beta2-constant-beta1': '#20B2AA',   # Light sea green
    'ademamix': '#9467bd',         # Purple
    'ademamix-decaying-wd': '#e377c2', # Pink
    'd-muon': '#ff7f0e',           # Orange
    'muon': '#ff7f0e',             # Orange (same as d-muon)
    'manau': '#8B4513',            # Saddle brown
    'manau-hard': '#DC143C',       # Crimson
    'laprop': '#17becf',           # Cyan
    'logadam': '#FFD700',          # Gold
    'logadam-nesterov': '#FFA500', # Orange
}

# Kappa-based color scheme - continuous color scale from kappa 0.75 to 1.0
# Using viridis colormap for smooth color progression
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Define kappa values and their corresponding colors from viridis colormap
_kappa_values = [0.75, 0.8, 0.85, 0.9, 1.0]
_norm = mcolors.Normalize(vmin=0.75, vmax=1.0)
_cmap = cm.get_cmap('viridis')

OPT_COLORS_KAPPA = {
    'dana-star-no-tau': _cmap(_norm(0.75)),  # Default kappa (implicit 0.75)
    'dana-star-no-tau-kappa-0-8': _cmap(_norm(0.8)),
    'dana-star-no-tau-kappa-0-85': _cmap(_norm(0.85)),
    'dana-star-no-tau-kappa-0-9': _cmap(_norm(0.9)),
    'dana-star-no-tau-kappa-1-0': _cmap(_norm(1.0)),
    'dana-star-mk4-kappa-0-85': _cmap(_norm(0.85)),  # Same kappa as dana-star-no-tau-kappa-0-85
    'adana': _cmap(_norm(0.75)),  # Alias
}

# Convert colors to hex format for consistency
OPT_COLORS_KAPPA = {k: mcolors.to_hex(v) for k, v in OPT_COLORS_KAPPA.items()}

# Line styles for optimizers - mk4 variants get dashed lines
OPT_LINESTYLES = {
    'dana-mk4': '--',              # Dashed for mk4
    'dana-star-mk4': '--',         # Dashed for mk4
    'dana-mk4-kappa-0-85': '--',   # Dashed for mk4
    'dana-star-mk4-kappa-0-85': '--',   # Dashed for mk4
    'mk4': '--',                   # Dashed for mk4
    # All other optimizers default to solid line '-'
}

# Display names for optimizers - nice formatting for legends
OPT_DISPLAY_NAMES = {
    'adamw': 'AdamW',
    'adamw-decaying-wd': 'AdamW-DecayWD',
    'mk4': 'DANA-Star-MK4',
    'dana': 'DANA',
    'dana-mk4': 'DANA-MK4',
    'dana-mk4-kappa-0-75': 'DANA-MK4 κ=0.75',
    'dana-mk4-kappa-0-85': 'DANA-MK4 κ=0.85',
    'dana-star': 'DANA-Star',
    'dana-star-mk4': 'DANA-Star-MK4',
    'adana': 'ADANA',
    'adana-kappa-0-85': 'ADANA κ=0.85',
    'dana-star-no-tau': 'ADANA κ=0.75',
    'dana-star-no-tau-kappa-1-0': 'ADANA κ=1.0',
    'dana-star-no-tau-kappa-0-8': 'ADANA κ=0.8',
    'dana-star-no-tau-kappa-0-85': 'ADANA κ=0.85',
    'dana-star-no-tau-kappa-0-9': 'ADANA κ=0.9',
    'dana-star-mk4-kappa-0-75': 'DANA-Star-MK4 κ=0.75',
    'dana-star-mk4-kappa-0-85': 'DANA-Star-MK4 κ=0.85',
    'dana-star-no-tau-dana-constant': 'ADANA-DanaConst',
    'dana-star-no-tau-beta2-constant': 'ADANA-β₂Const',
    'dana-star-no-tau-beta1': 'ADANA-β₁',
    'dana-star-no-tau-dana-constant-beta2-constant': 'ADANA-DanaConst-β₂Const',
    'dana-star-no-tau-dana-constant-beta1': 'ADANA-DanaConst-β₁',
    'dana-star-no-tau-dana-constant-beta2-constant-beta1': 'ADANA-DanaConst-β₂Const-β₁',
    'ademamix': 'Ademamix',
    'ademamix-decaying-wd': 'Ademamix-DecayWD',
    'd-muon': 'Muon',
    'manau': 'Manau',
    'manau-hard': 'Manau-Hard',
    'laprop': 'LaProp',
    'logadam': 'LogAdam',
    'logadam-nesterov': 'LogAdam-Nesterov',
}

# Display names for kappa ablation mode - show only kappa values
OPT_DISPLAY_NAMES_KAPPA = {
    'dana-star-no-tau': 'κ = 0.75',  # Default kappa (implicit 0.75)
    'dana-star-no-tau-kappa-0-8': 'κ = 0.8',
    'dana-star-no-tau-kappa-0-85': 'κ = 0.85',
    'dana-star-no-tau-kappa-0-9': 'κ = 0.9',
    'dana-star-no-tau-kappa-1-0': 'κ = 1.0',
    'dana-star-mk4-kappa-0-85': 'κ = 0.85',  # Same kappa as dana-star-no-tau-kappa-0-85
    'adana': 'κ = 0.75',  # Alias
}
