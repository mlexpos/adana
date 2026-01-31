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
- DANA variants and decaying-WD optimizers use independent weight decay (paper convention): WD is multiplied by the LR schedule γ(t) but NOT by the peak LR γ*. The optimizer computes `schedule_factor = group['lr'] / self.lr` for this. AdamW and AdEMAMix (non-decaying) use the standard PyTorch coupled convention where WD is multiplied by the full lr.
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

## TP Benchmarking Results (24-head Enoki, 550M params, seq=2048, 2x A100-80GB)

| Config | Global Batch | iter_dt | Tokens/s | Peak Mem |
|--------|-------------|---------|----------|----------|
| DDP | 64 (16×4 acc) | 938ms | ~140K | 48.33 GiB |
| FSDP | 64 (16×4 acc) | 961ms | ~136K | 44.87 GiB |
| FSDP+TP2 (no ckpt) | 32 (32×1) | 666ms | ~98K | 57.73 GiB |
| FSDP+TP2+ckpt | 64 (64×1) | 1610ms | ~81K | 32.28 GiB |
| FSDP+TP2+ckpt | 128 (128×1) | 3160ms | ~83K | 61.44 GiB |

**Key findings:** TP2 without ckpt at batch 64 OOMs (~1 GiB short at 76 GiB). Activation checkpointing adds ~2.4x overhead. TP communication + no fused Adam costs ~40% vs DDP. TP's main value is enabling larger batches that wouldn't fit otherwise, not raw throughput on 2 GPUs.

### 40-head Enoki (2.1B params, seq=2048, 2x A100-80GB, FSDP+TP2, no checkpointing)

| Batch | Acc Steps | Global Batch | iter_dt | Peak Mem |
|-------|-----------|-------------|---------|----------|
| 4 | 1 | 4 | 422ms | 26.9 GiB |
| 8 | 4 | 32 | 2.08s | 46.2 GiB |
| 16 | 2 | 32 | OOM | ~76 GiB |

With activation checkpointing, batch 4 runs at 580ms (27% slower) and batch 32 at 2.44s. Checkpointing enables larger batches but hurts throughput. The no-ckpt limit is batch 8-ish for this model size.

## DanaStar Test Cluster

- **Host:** `danastar` (2x H100-80GB HBM3, direct SSH)
- **Remote repo path:** `~/Dana_hummingbird/danastar`
- **Python venv:** `source /home/ubuntu/Danastar/danastarenv/bin/activate` (Python 3.10, PyTorch 2.8.0+cu128)
- **Data available:** `~/Dana_hummingbird/danastar/datasets/fineweb-100BT/` (symlinked to `/home/ubuntu/Danastar/datasets`; val.bin + train shards)
- **Dataset flag:** `--datasets_dir ./datasets --dataset fineweb_100` (the loader auto-detects pre-tokenized .bin files)
- **Sync command:** `rsync --exclude '.git' --exclude 'exps' --exclude 'wandb' --exclude 'fineweb-*' --exclude 'logs' --exclude '__pycache__' --exclude '*.pyc' --exclude 'datasets' -avz /Users/elliotpaquette/Documents/DSTAR/DSTAR2/ danastar:~/Dana_hummingbird/danastar/`
- **Example launch (2 GPUs, FSDP):**
  ```bash
  ssh danastar 'cd ~/Dana_hummingbird/danastar && source /home/ubuntu/Danastar/danastarenv/bin/activate && \
    torchrun --standalone --nproc_per_node=2 src/main.py \
    --model enoki --n_head 40 --n_layer 30 --n_embd 2560 \
    --datasets_dir ./datasets --dataset fineweb_100 \
    --opt adamw --lr 1e-4 \
    --batch_size 16 --acc_steps 1 --iterations 10 --warmup_steps 3 \
    --eval_interval 9999 --log_interval 1 --scheduler cos \
    --distributed_backend fsdp --sequence_length 2048'
  ```
- **Known issues:**
  - No `--no_compile` flag on refactor branch; torch.compile runs by default
  - No `--no_wandb` flag; wandb is opt-in via `--wandb` flag
  - Default warmup_steps is 3000; must set `--warmup_steps` explicitly for short runs
  - Cosine scheduler with `--warmup_steps 1` and very few iterations causes ZeroDivisionError; use `--warmup_steps 3` minimum
  - WandB with `WANDB_MODE=offline` can cause apparent hangs due to sync overhead; omit `--wandb` for test runs
  - When using `--wandb` on a machine with no internet, set `WANDB_MODE=offline` in the environment

### Activation Checkpointing Benchmark (40-head Enoki, 2.1B params, seq=2048, 2x H100-80GB, FSDP+TP2)

All tests use `--tp_size 2` (tensor parallelism across both GPUs), `--distributed_backend fsdp`, global batch 256, 15 iterations.

| Config | batch_size | acc_steps | Peak Mem (GPU 0) | iter_dt (steady) | Status |
|--------|-----------|-----------|-----------------|-------------------|--------|
| No checkpointing | 8 | 32 | 46.17 GiB | ~16.3s | OK |
| `--activation_checkpointing --checkpoint_every_n 2` | 16 | 16 | 52.60 GiB | ~17.5s | OK |
| `--activation_checkpointing --checkpoint_every_n 2` | 32 | 8 | ~76.9 GiB | -- | OOM |
| `--activation_checkpointing --checkpoint_every_n 5` | 32 | 8 | ~76.9 GiB | -- | OOM |
| `--activation_checkpointing --checkpoint_every_n 2` | 64 | 4 | ~75.9 GiB | -- | OOM |
| `--activation_checkpointing --checkpoint_every_n 2` | 128 | 2 | ~75.8 GiB | -- | OOM |
| `--activation_checkpointing --checkpoint_every_n 2` | 256 | 1 | ~74.1 GiB | -- | OOM |
| No checkpointing | 16 | 16 | ~75.9 GiB | -- | OOM (no TP2 also OOM) |

**Key findings:** With FSDP+TP2, the no-checkpointing limit is batch_size=8. Activation checkpointing (every 2nd block) allows batch_size=16 (52.6 GiB, ~7% slower). Batch sizes 32+ all OOM at ~76 GiB regardless of checkpointing frequency (every-2 and every-5 give identical memory at batch 32). The compiled chunked cross-entropy kernel (`_compiled_chunk_ce_zloss`) and torch.compile overhead dominate memory at these sizes, not transformer block activations. The near-constant memory across batch 32-256 confirms the bottleneck is model/optimizer/compile state, not activation storage. Disabling torch.compile may be needed to unlock larger batch sizes.

## Current Branch

`refactor/cleanup` - Large-scale refactoring and code cleanup
