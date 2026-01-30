# DSTAR (Dana Star) - Project Guide

## Project Overview

Research codebase for the paper **"Logarithmic-time Schedules for Scaling Language Models with Momentum"** (ICML 2026 submission). Repository: https://github.com/LosVolterrosHermanos/danastar

The paper is located at `../68a78fcfedd649b2fea47ddd/` (LaTeX source with `paper.tex` as the main file).

## Paper Summary

### Core Idea
Standard AdamW uses fixed momentum parameters (beta1, beta2) and weight decay lambda. This paper shows that *logarithmic-time scheduling* of these parameters -- where the optimizer's gradient memory horizon grows with training time -- delivers substantial performance gains for LLM pretraining.

### The ADana Optimizer (Adaptive Damped Nesterov Acceleration)
ADana modifies AdamW with three key changes:
1. **Log-time momentum:** `beta1(t) = beta2(t) = 1 - delta/(delta + t)` (gradients from the entire training history contribute, not just recent ones)
2. **Decaying weight decay:** `lambda(t) = omega/t` (stronger regularization early, relaxed later)
3. **Damped Nesterov update:** `alpha(t) = (1+t)^(1-kappa)` where kappa in (0,1) balances stability vs acceleration

The update rule: `theta_{t+1} = theta_t - gamma(t) * (gamma* * (g_{t+1} + alpha(t)*m_{t+1}) / (sqrt(v_{t+1}) + eps) + lambda(t)*theta_t)`

### Key Hyperparameters
- **kappa** (spectral dimension): Controls damping. Optimal ~0.85, scale-invariant (doesn't need retuning across model sizes). Range [0.75, 0.9] works well.
- **delta**: EMA coefficient, typically 8. Not very sensitive.
- **omega**: Weight decay constant, typically 4.0. Constant across scales.
- **clipSNR**: For MK4 variants, typically 2.0.

### Algorithm Variants
| Variant | Key Feature | Code File |
|---------|-------------|-----------|
| **ADana** | Base log-time scheduling with damped Nesterov | `src/optim/dana_star.py` (confusingly named) |
| **Dana-MK4** | + SNR clipping for kappa robustness across layers | `src/optim/dana_star_mk4.py` |
| **Dana-Star** | + tau probability estimator for sparse gradient handling | `src/optim/dana_star.py` |
| **Dana-Star-MK4** | + both SNR clipping and tau estimator | `src/optim/dana_star_mk4.py` |

### Main Experimental Results

**Architectures tested:**
- **Enoki**: GPT-3-like with RoPE, QK-LayerNorm, pre-LN, residual scaling, head dim=64. Scales: 186M to 2.62B params.
- **Qwen3**: SwiGLU activation, RMSNorm, elementwise attention gating, head dim=128.

**Key findings:**
1. **ADana achieves ~40% compute savings** vs tuned AdamW, with gains that *increase* with scale
2. **Dana-Star-MK4** (kappa=0.85) significantly outperforms AdamW; gap decreases with scale vs ADana
3. **Muon and Ademamix** show competitive performance at small scales but diminishing gains at larger scales
4. **Log-time weight decay alone** gives ~10% compute savings even for standard AdamW
5. **kappa is scale-invariant** -- no retuning needed across model sizes
6. Constant beta1 provides no benefit over SGD (proven in specific settings); log-time beta1 is needed
7. Naive log-time Nesterov (without damping) is unstable; the alpha(t) damping schedule is essential
8. Constant beta2 with log-time beta1 can cause divergence at scale; log-time beta2 is needed

**Training protocol:** Chinchilla-optimal scaling (N = 20D tokens), FineWeb dataset, cosine LR decay, batch size 32x2048 tokens.

## Repository Structure

```
src/
  main.py              # Training entry point (torchrun)
  config/base.py       # CLI argument parsing and hyperparameter config
  data/                # Dataset loaders (Fineweb, C4, arXiv, OpenWebText2, etc.)
    utils.py           # DataReader, MultiFileDataReader, batch loading
  distributed/         # Multi-GPU/multi-node training backends (NCCL)
  logger/              # Training dynamics logging (norms, gradients, angular updates)
  models/              # Model architectures
    base.py            # GPT-style transformer (Enoki) with Flash Attention
    llama.py           # LLaMA with RoPE
    qwen3.py           # Qwen3 (SwiGLU, RMSNorm, elementwise gating)
    moe.py             # Mixture of Experts
    diloco.py          # DiLoCo distributed training
  optim/               # Optimizer implementations
    dana_star.py       # Dana-Star (ADana with tau estimator)
    dana_star_mk4.py   # Dana-Star MK4 (+ SNR clipping)
    auto_dana.py       # Auto-DANA
    sign_dana.py       # Sign-DANA
    snoo_dana.py       # SNOO-DANA
    muon.py            # Muon optimizer
    schedule.py        # LR schedules (cos, wsd, powerlaw, cos_inf, linear)
    utils.py           # Optimizer utilities and checkpoint management
    tau_stats_collector.py  # Collects tau statistics for analysis
    # + baseline optimizers: adamw, sgd, lion, soap, ademamix, prodigy, sophia, etc.
scripts/               # Cluster job submission scripts (organized by model size / cluster)
jax/                   # JAX-based theoretical analysis (PLRF, MOE-PLRF, scaling laws)
visualization/         # WandB result parsing and plot generation
inference.py           # Checkpoint loading and text generation
```

## Key Commands

### Training
```bash
# Single GPU
torchrun --nproc_per_node=1 ./src/main.py --model llama --opt adamw --lr 1e-3 ...

# Multi-GPU
torchrun --nproc_per_node=4 ./src/main.py --model llama --opt dana-star --lr 1e-3 ...
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Tech Stack

- **Language:** Python (PyTorch for training, JAX for theoretical analysis)
- **Experiment tracking:** Weights & Biases (wandb)
- **Tokenizer:** tiktoken (GPT-2 BPE, vocab size 50,304)
- **Data:** Hugging Face datasets (primarily Fineweb 10BT/100BT)
- **Distributed:** torchrun + NCCL

## Supported Optimizers

adamw, sgd, muon, soap, ademamix, lion, prodigy, sophia, adopt, mars, adafactor, lamb, dana-star, dana, dana-star-mk4, auto-dana, sign_dana, snoo-dana, snoo

## Supported Models

base (GPT/Enoki), llama, qwen3, qwen3next, moe, moe_expert_parallel, diloco

## Supported LR Schedules

cos, wsd, powerlaw, cos_inf, linear, none

## Code Conventions

- Config is via argparse in `src/config/base.py`; all hyperparameters are CLI flags
- Optimizers follow a common interface pattern; see existing implementations in `src/optim/`
- Models are registered by name string and selected via `--model` flag
- WandB is optional (enabled with `--wandb` flag)
- Checkpoints saved to `exps/` directory (gitignored)
- Uses independent (decoupled) weight decay, not PyTorch's default coupled weight decay
- LR scaling rules are fit as saturated power laws: `gamma(P) = a * (b + P)^d`

## Remote Test Environment

- **Host:** `math-slurm` (H100 NVL 96GB, via SSH through `jump` bastion)
- **Remote repo path:** `~/danastar`
- **Module load required:** `module load miniconda/miniconda-winter2025` (provides Python 3.12)
- **PyTorch:** 2.9.0.dev+cu126 (user-installed in `~/.local`, shadows the module's older 2.4.1 -- this is intentional, use the newer version)
- **User packages in `~/.local`:** wandb, jax, tiktoken, optax, einops, and other project deps (~92 packages)
- **Data available:** `~/danastar/src/data/datasets/fineweb-10BT/` (has train.bin + val.bin, works for testing). `fineweb-100BT/` is empty (no parquet files processed).
- **SSH ControlMaster:** configured, persistent for 4h after initial auth
- **Sync command:** `rsync --exclude '.git' --exclude 'exps' --exclude 'wandb' --exclude 'fineweb-*' --exclude 'logs' --exclude '__pycache__' --exclude '*.pyc' -avz /Users/paquette.30/Documents/DSTAR/danastar/ math-slurm:~/danastar/`

## Current Branch

`refactor/cleanup` - Large-scale refactoring and code cleanup
