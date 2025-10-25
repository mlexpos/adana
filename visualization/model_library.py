"""
Model Library: Shared configuration for DanaStar and AdamW experiments

This module contains the centralized configuration mapping for all model sizes
and optimizer types used in the experiments.
"""

# Model configurations with WandB groups and metadata
MODEL_CONFIGS = {
    # DanaStar configurations
    'DSsmall': {
        'group': 'DanaStar_Small_LR_WD_Sweep',
        'title': 'DanaStar Small: JAX Softplus-like Fits',
        'filename': 'danastar_small_lrwd_sweep_jax.pdf',
        'optimizer': 'danastar',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 4,  # Small model: 4 layers
        'n_head': 6,
        'qkv_dim': 64,
        'n_params': 10_000_000  # Approximate parameter count
    },
    'DS35M': {
        'group': 'DanaStar_35M_LR_WD_Sweep',
        'title': 'DanaStar 35M: JAX Softplus-like Fits',
        'filename': 'danastar_35m_lrwd_sweep_jax.pdf',
        'optimizer': 'danastar',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 6,  # 35M model: 6 layers
        'n_head': 8,
        'qkv_dim': 64,
        'n_params': 35_000_000
    },
    'DS90M': {
        'group': 'DanaStar_90M_LR_WD_Sweep',
        'title': 'DanaStar 90M: JAX Softplus-like Fits',
        'filename': 'danastar_90m_lrwd_sweep_jax.pdf',
        'optimizer': 'danastar',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 9,  # 90M model: 9 layers
        'n_head': 12,
        'qkv_dim': 64,
        'n_params': 90_000_000
    },
    'DS120M': {
        'group': 'DanaStar_120M_lr_weight_decay_sweeps',
        'title': 'DanaStar 120M: JAX Softplus-like Fits',
        'filename': 'danastar_120m_lrwd_sweep_jax.pdf',
        'optimizer': 'danastar',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 10,  # 120M model: 10 layers
        'n_head': 14,
        'qkv_dim': 64,
        'n_params': 120_000_000
    },
    'DS180M': {
        'group': 'DanaStar_180M_lr_weight_decay_sweeps',
        'title': 'DanaStar 180M: JAX Softplus-like Fits',
        'filename': 'danastar_180m_lrwd_sweep_jax.pdf',
        'optimizer': 'danastar',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 12,  # 180M model: 12 layers
        'n_head': 16,
        'qkv_dim': 64,
        'n_params': 180_000_000
    },
    'DS550M': {
        'group': 'DanaStar_550M_lr_weight_decay_sweeps',
        'title': 'DanaStar 550M: JAX Softplus-like Fits',
        'filename': 'danastar_550m_lrwd_sweep_jax.pdf',
        'optimizer': 'danastar',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 18,  # 550M model: 18 layers
        'n_head': 24,
        'qkv_dim': 64,
        'n_params': 550_000_000
    },
    'DS13B': {
        'group': 'DanaStar_13B_lr_weight_decay_sweeps',
        'title': 'DanaStar 13B: JAX Softplus-like Fits',
        'filename': 'danastar_13b_lrwd_sweep_jax.pdf',
        'optimizer': 'danastar',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 24,  # 13B model: 24 layers
        'n_head': 32,
        'qkv_dim': 64,
        'n_params': 13_000_000_000
    },

    # AdamW configurations
    'AWsmall': {
        'group': 'AdamW_small_lr_weight_decay_sweeps',
        'title': 'AdamW Small: JAX Softplus-like Fits',
        'filename': 'adamw_small_lrwd_sweep_jax.pdf',
        'optimizer': 'adamw',
        'omega_label': 'log(ω_T) where ω_T = weight_decay × lr × T',
        'n_layer': 4,  # Small model: 4 layers
        'n_head': 6,
        'qkv_dim': 64,
        'n_params': 10_000_000
    },
    'AW35M': {
        'group': 'AdamW_35M_lr_weight_decay_sweeps',
        'title': 'AdamW 35M: JAX Softplus-like Fits',
        'filename': 'adamw_35m_lrwd_sweep_jax.pdf',
        'optimizer': 'adamw',
        'omega_label': 'log(ω_T) where ω_T = weight_decay × lr × T',
        'n_layer': 6,  # 35M model: 6 layers
        'n_head': 8,
        'qkv_dim': 64,
        'n_params': 35_000_000
    },
    'AW90M': {
        'group': 'AdamW_90M_lr_weight_decay_sweeps',
        'title': 'AdamW 90M: JAX Softplus-like Fits',
        'filename': 'adamw_90m_lrwd_sweep_jax.pdf',
        'optimizer': 'adamw',
        'omega_label': 'log(ω_T) where ω_T = weight_decay × lr × T',
        'n_layer': 9,  # 90M model: 9 layers
        'n_head': 12,
        'qkv_dim': 64,
        'n_params': 90_000_000
    },
    'AW180M': {
        'group': 'AdamW_180M_lr_weight_decay_sweeps',
        'title': 'AdamW 180M: JAX Softplus-like Fits',
        'filename': 'adamw_180m_lrwd_sweep_jax.pdf',
        'optimizer': 'adamw',
        'omega_label': 'log(ω_T) where ω_T = weight_decay × lr × T',
        'n_layer': 12,  # 180M model: 12 layers
        'n_head': 16,
        'qkv_dim': 64,
        'n_params': 180_000_000
    },
    'AW330M': {
        'group': 'AdamW_330M_lr_weight_decay_sweeps',
        'title': 'AdamW 330M: JAX Softplus-like Fits',
        'filename': 'adamw_330m_lrwd_sweep_jax.pdf',
        'optimizer': 'adamw',
        'omega_label': 'log(ω_T) where ω_T = weight_decay × lr × T',
        'n_layer': 15,  # 330M model: 15 layers
        'n_head': 20,
        'qkv_dim': 64,
        'n_params': 330_000_000
    },
    'AW550M': {
        'group': 'AdamW_550M_lr_weight_decay_sweeps',
        'title': 'AdamW 550M: JAX Softplus-like Fits',
        'filename': 'adamw_550m_lrwd_sweep_jax.pdf',
        'optimizer': 'adamw',
        'omega_label': 'log(ω_T) where ω_T = weight_decay × lr × T',
        'n_layer': 18,  # 550M model: 18 layers
        'n_head': 24,
        'qkv_dim': 64,
        'n_params': 550_000_000
    },
    'AW13B': {
        'group': 'AdamW_13B_lr_weight_decay_sweeps',
        'title': 'AdamW 13B: JAX Softplus-like Fits',
        'filename': 'adamw_13b_lrwd_sweep_jax.pdf',
        'optimizer': 'adamw',
        'omega_label': 'log(ω_T) where ω_T = weight_decay × lr × T',
        'n_layer': 24,  # 13B model: 24 layers
        'n_head': 32,
        'qkv_dim': 64,
        'n_params': 13_000_000_000
    },

    # Adam-Star configurations
    'AS35M': {
        'group': 'AdamStar_35M_lr_weight_decay_sweeps',
        'title': 'Adam-Star 35M: JAX Softplus-like Fits',
        'filename': 'adamstar_35m_lrwd_sweep_jax.pdf',
        'optimizer': 'adamstar',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 6,  # 35M model: 6 layers
        'n_head': 8,
        'qkv_dim': 64,
        'n_params': 35_000_000
    },
    'AS90M': {
        'group': 'AdamStar_90M_lr_weight_decay_sweeps',
        'title': 'Adam-Star 90M: JAX Softplus-like Fits',
        'filename': 'adamstar_90m_lrwd_sweep_jax.pdf',
        'optimizer': 'adamstar',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 9,  # 90M model: 9 layers
        'n_head': 12,
        'qkv_dim': 64,
        'n_params': 90_000_000
    },
    'AS180M': {
        'group': 'AdamStar_180M_lr_weight_decay_sweeps',
        'title': 'Adam-Star 180M: JAX Softplus-like Fits',
        'filename': 'adamstar_180m_lrwd_sweep_jax.pdf',
        'optimizer': 'adamstar',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 12,  # 180M model: 12 layers
        'n_head': 16,
        'qkv_dim': 64,
        'n_params': 180_000_000
    },

    # DanaStar MK4 configurations
    'MK4_35M': {
        'group': 'DanaStar_MK4_35M_Sweep_formula12',
        'title': 'DanaStar MK4 35M: JAX Softplus-like Fits',
        'filename': 'danastar_mk4_35m_lrwd_sweep_jax.pdf',
        'optimizer': 'mk4',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 6,  # 35M model: 6 layers
        'n_head': 8,
        'qkv_dim': 64,
        'n_params': 35_000_000
    },
    'MK4_90M': {
        'group': 'DanaStar_MK4_90M_Sweep_formula12',
        'title': 'DanaStar MK4 90M: JAX Softplus-like Fits',
        'filename': 'danastar_mk4_90m_lrwd_sweep_jax.pdf',
        'optimizer': 'mk4',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 9,  # 90M model: 9 layers
        'n_head': 12,
        'qkv_dim': 64,
        'n_params': 90_000_000
    },
    'MK4_180M': {
        'group': 'DanaStar_MK4_180M_Sweep_formula12',
        'title': 'DanaStar MK4 180M: JAX Softplus-like Fits',
        'filename': 'danastar_mk4_180m_lrwd_sweep_jax.pdf',
        'optimizer': 'mk4',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 12,  # 180M model: 12 layers
        'n_head': 16,
        'qkv_dim': 64,
        'n_params': 180_000_000
    },
    'MK4_330M': {
        'group': 'DanaStar_MK4_330M_Sweep_formula12',
        'title': 'DanaStar MK4 330M: JAX Softplus-like Fits',
        'filename': 'danastar_mk4_330m_lrwd_sweep_jax.pdf',
        'optimizer': 'mk4',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 15,  # 330M model: 15 layers
        'n_head': 20,
        'qkv_dim': 64,
        'n_params': 330_000_000
    },
    'MK4_550M': {
        'group': 'DanaStar_MK4_550M_Sweep_formula12',
        'title': 'DanaStar MK4 550M: JAX Softplus-like Fits',
        'filename': 'danastar_mk4_550m_lrwd_sweep_jax.pdf',
        'optimizer': 'mk4',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 18,  # 550M model: 18 layers
        'n_head': 24,
        'qkv_dim': 64,
        'n_params': 550_000_000
    },
    'MK4_13B': {
        'group': 'DanaStar_MK4_13B_Sweep_formula12',
        'title': 'DanaStar MK4 13B: JAX Softplus-like Fits',
        'filename': 'danastar_mk4_13b_lrwd_sweep_jax.pdf',
        'optimizer': 'mk4',
        'omega_label': 'log(ω) where ω = wd_ts × lr × weight_decay',
        'n_layer': 24,  # 13B model: 24 layers
        'n_head': 32,
        'qkv_dim': 64,
        'n_params': 13_000_000_000
    }
}

# Helper functions to query the model library
def get_models_by_optimizer(optimizer_type):
    """Get all model configs for a given optimizer type (danastar or adamw)"""
    return {k: v for k, v in MODEL_CONFIGS.items() if v['optimizer'] == optimizer_type}

def get_model_keys_by_optimizer(optimizer_type):
    """Get model keys for a given optimizer type"""
    return [k for k, v in MODEL_CONFIGS.items() if v['optimizer'] == optimizer_type]
