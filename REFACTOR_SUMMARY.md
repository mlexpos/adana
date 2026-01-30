# DSTAR Refactoring Summary

This document describes all changes made during the `refactor/cleanup` branch refactoring, organized by phase.

---

## Phase 2: Optimizer Cleanup

**Goal:** Consolidate ~10 DANA optimizer variants into a clean set of 2 files based on the fast MK4 compiled kernel.

### New Optimizer Layout

| CLI Name | Tau Buffer | SNR Clip | Class | File |
|----------|-----------|----------|-------|------|
| `adana` | No | No | `ADana` | `src/optim/adana.py` |
| `dana-mk4` | No | Yes | `ADana(clipsnr=X)` | `src/optim/adana.py` |
| `dana-star` | Yes | No | `DANA_STAR_MK4(clipsnr=None)` | `src/optim/dana_star_mk4.py` |
| `dana-star-mk4` | Yes | Yes | `DANA_STAR_MK4(clipsnr=X)` | `src/optim/dana_star_mk4.py` |

All four variants share the same compiled update kernel. The difference is:
- **Without tau** (`adana.py`): No tau probability estimator buffer. Uses `norm_term = 1/(sqrt(v)+eps)` directly.
- **With tau** (`dana_star_mk4.py`): Allocates tau buffer for sparse gradient handling.
- **clipsnr=None**: Skips SNR clamping in the update step.
- **clipsnr=float**: Applies `clamp(alpha_factor, max=clipsnr)` for kappa robustness.

### Optimizer Registry

The 180-line if-elif chain in `main.py` was replaced with a `build_optimizer()` function using `OPTIMIZER_REGISTRY` dict (`src/main.py:62-236`). Each optimizer maps to a lambda constructor.

### Files Created
- `src/optim/adana.py` — ADana optimizer (compiled, no tau, optional SNR clip)
- `src/optim/adamw_decaying_wd.py` — Standalone AdamWDecayingWD (extracted from ablation.py)
- `src/optim/ademamix_decaying_wd.py` — Standalone AdEMAMix_DecayingWD (extracted from ablation.py)

### Files Deleted
- `src/optim/dana_star.py` — Replaced by adana.py + dana_star_mk4.py
- `src/optim/auto_dana.py` — Experimental, not in paper
- `src/optim/sign_dana.py` — Experimental, not in paper
- `src/optim/snoo_dana.py` — Experimental, not in paper
- `src/optim/ablation.py` — Components moved to standalone files

### Files Modified
- `src/optim/dana_star_mk4.py` — Added `clipsnr=None` mode
- `src/main.py` — Registry-based optimizer construction
- `src/config/base.py` — Cleaned optimizer choices list
- `src/optim/utils.py` — Updated optimizer name checks in `log_optimizer_schedules`
- `src/optim/base.py` — Updated tau_stats_collector condition

---

## Phase 3: Model Renaming + Scaling Rules

**Goal:** Rename diloco -> enoki, add Python-accessible scaling rules for auto-computing model dimensions and LR.

### Model Rename

- Created `src/models/enoki.py` — Copy of diloco.py with class renames: `DiLoCo` -> `Enoki`, `DiLoCoBlock` -> `EnokiBlock`, etc.
- Updated `src/models/utils.py` — Added `"enoki"` model mapping; kept `"diloco"` as backward-compat alias.
- `src/config/base.py` — Added `"enoki"` to model choices.

### Scaling Rules

Two new files provide auto-computation of model dimensions and learning rate from a `--heads` argument:

**`src/config/scaling_rules.yaml`:**
```yaml
enoki:
  head_dim: 64
  n_layer_formula: "3 * heads // 4"    # 3 heads -> 2 layers, 12 heads -> 9 layers
  n_embd_formula: "heads * head_dim"
  mlp_hidden_formula: "4 * n_embd"
  lr_formulas:
    adana: {a: 3.72e3, b: 3.70e3, d: -0.894}   # LR = a * (b + P)^d

qwen3:
  head_dim: 128
  n_layer_formula: "2 * heads"
  n_embd_formula: "heads * head_dim"
  mlp_hidden_formula: "3 * n_embd"
```

**`src/config/scaling.py`:**
- `compute_dimensions(arch, heads)` — Returns dict with n_head, n_layer, n_embd, mlp_hidden_dim, head_dim, non_emb_params, total_params
- `compute_lr(arch, opt, non_emb_params)` — Returns LR from saturated power law formula
- `apply_scaling_rule(args)` — Modifies args in-place when `--scaling_rule` is set

### New CLI Arguments
```
--scaling_rule {enoki,qwen3,none}   # Auto-compute dims from --heads
--heads N                            # Number of attention heads
```

---

## Phase 1: Scripts Consolidation

**Goal:** Replace ~100+ cluster/model-size specific scripts with 4 universal files.

### New Script Layout

```
scripts/
  config.sh              # Cluster detection, WANDB config, module loads, venv activation
  launch.sh              # Universal training launcher
  restart_enoki.sh       # SLURM restart wrapper for Enoki architecture
  restart_qwen3.sh       # SLURM restart wrapper for Qwen3 architecture
```

### `config.sh`
- Auto-detects cluster from hostname (narval, tamia, fir, math-slurm, local)
- Loads appropriate modules and activates venv per cluster
- Sets WANDB, data paths, caches, PyTorch memory config

### `launch.sh`
Universal launcher accepting:
```bash
bash scripts/launch.sh --arch enoki --opt adana --heads 3
bash scripts/launch.sh --arch qwen3 --opt dana-star-mk4 --heads 6 --wandb_group my_sweep
bash scripts/launch.sh --arch enoki --opt adamw --heads 12 --nproc 4 --distributed_backend fsdp
```

Key features:
- Computes model dimensions via Python scaling rules (`src/config/scaling.py`)
- Computes Chinchilla-optimal iterations: `tokens = 20 * non_emb_params`
- Computes weight decay from omega: `wd = omega / (lr * wd_ts)`
- Optimizer-specific flag blocks (case statement)
- Architecture-specific flags (weight tying, normalization, gating)
- Launches via `torchrun`

### `restart_enoki.sh` / `restart_qwen3.sh`
- SLURM restart wrappers that call `launch.sh --arch {enoki,qwen3}`
- Auto-requeue with `--iterations_to_run` for long training runs
- Preserve SLURM resource allocation (account, time, nodes, GPUs, memory)

### Files Deleted
- All of `scripts/narval/`, `scripts/tamia/`, `scripts/fir/`, `scripts/BigHead/`
- All model-size directories (`scripts/124m/`, `scripts/210m/`, `scripts/720m/`)
- All diloco, moe, scripts_dfer, rorqual directories
- ~20 loose scripts

---

## Phase 4: FSDP Implementation

**Goal:** Add FSDP2 as a distributed backend option for multi-GPU training with per-parameter sharding.

### New Backend: `src/distributed/fsdp.py`

Implements `DistributedBackend` interface using PyTorch's composable `fully_shard()` API (FSDP2):

- **Sharding strategy:** FULL_SHARD (parameters, gradients, and optimizer states sharded across ranks)
- **Mixed precision:** fp32 params, bf16 reductions
- **Wrapping:** Per-transformer-block sharding, then root-level sharding
- **Gradient accumulation:** `set_requires_gradient_sync(False)` on non-final microsteps
- **Parameter names:** Preserved (no prefix mangling like DDP's `module.` prefix)

### Checkpoint Support

FSDP checkpoints use `torch.distributed.checkpoint` (DCP) for sharded state dicts:
- **Save:** `dcp.save()` writes each rank's shard; `meta.pt` stores scheduler + iteration (rank 0 only)
- **Load:** `dcp.load()` reads each rank's shard; `set_model_state_dict` / `set_optimizer_state_dict` apply them
- All ranks must participate in save/load (collective operation)

### Usage
```bash
# DDP (default, unchanged)
torchrun --nproc_per_node=4 ./src/main.py --distributed_backend nccl ...

# FSDP
torchrun --nproc_per_node=4 ./src/main.py --distributed_backend fsdp ...

# Via launch.sh
bash scripts/launch.sh --arch enoki --opt adana --heads 6 --nproc 4 --distributed_backend fsdp
```

### Files Created
- `src/distributed/fsdp.py` — FSDPDistributedBackend class

### Files Modified
- `src/distributed/__init__.py` — Added `"fsdp"` to backend map
- `src/distributed/backend.py` — Added `all_ranks_checkpoint()` method
- `src/optim/utils.py` — FSDP-aware `save_checkpoint` / `load_checkpoint` with DCP
- `src/optim/base.py` — Checkpoint save from all ranks when FSDP; grad clipping uses `get_raw_model()`
- `scripts/launch.sh` — Added `--distributed_backend` argument

---

## Phase 5: Checkpoint Reliability

**Goal:** Write checkpoints to fast local NVMe first, then atomically rsync to project directory.

### How It Works

When `$SLURM_TMPDIR` is set (standard on SLURM clusters):

1. **Write locally:** Checkpoint writes to `$SLURM_TMPDIR/ckpts/<name>/` (fast local NVMe)
2. **Background rsync:** A daemon thread rsyncs to the project directory:
   - `rsync -a --delete` to `<project>/.ckpt_tmp_$SLURM_JOB_ID/`
   - On success: atomic `mv` to final location (old checkpoint renamed to `.ckpt_old_*`, then removed)
   - On failure: temp dir left intact, existing checkpoint not corrupted
3. **Serialized:** Each new rsync waits for the previous one to complete
4. **Drain on exit:** `wait_for_rsync()` called at end of training loop

When `$SLURM_TMPDIR` is not set (local development), behavior is unchanged — checkpoints write directly to the project directory.

### Files Modified
- `src/optim/utils.py` — `_rsync_checkpoint_to_project()`, `_background_rsync()`, `wait_for_rsync()`, updated `save_checkpoint`, `save_worker_state`, `load_worker_state`
- `src/optim/base.py` — Calls `wait_for_rsync()` before returning from `train()`

---

## Bugs Found and Fixed During Testing

Three bugs were discovered during multi-GPU FSDP testing on the danastar cluster (2x H100 80GB):

### Bug 1: FSDP Eval Deadlock
**Symptom:** Training hung at 100% CPU during evaluation with FSDP + 2 GPUs.
**Root Cause:** `eval_and_log()` returned early on non-master ranks (`if not is_master_process: return`). With FSDP, all ranks must participate in forward passes because parameter all-gather is a collective operation. Rank 0 tried to forward, rank 1 had already exited → deadlock.
**Fix:** Made all ranks participate in eval forward passes when FSDP is active. Only master rank logs results. (`src/optim/base.py`)

### Bug 2: FSDP + ADana Optimizer State Logging Crash
**Symptom:** `RuntimeError: tensor data not allocated` when logging optimizer schedules with FSDP + ADana.
**Root Cause:** `log_optimizer_schedules()` called `torch.tensor(alpha_values).mean().item()` where `alpha_values` contained FSDP DTensors (distributed, not locally materialized). Creating a new tensor from DTensors fails.
**Fix:** Added `_to_float()` helper with try/except for safe DTensor-to-float conversion. Replaced `torch.tensor().mean().item()` with plain Python `sum()/len()` arithmetic. (`src/optim/utils.py`)

### Bug 3: FSDP Auto-Resume Not Detecting Checkpoints
**Symptom:** `--auto_resume` didn't find existing FSDP checkpoints, starting training from scratch.
**Root Cause:** Auto-resume checked for `main.pt` (DDP checkpoint format), but FSDP saves `meta.pt` + `sharded/` directory.
**Fix:** Added `_checkpoint_exists()` helper in `main.py` that checks for both `main.pt` (DDP) and `meta.pt` (FSDP).

---

## Quick Reference

### Training Commands

```bash
# Single GPU, ADana optimizer, 3-head Enoki model
bash scripts/launch.sh --arch enoki --opt adana --heads 3

# 4-GPU DDP, Dana-Star-MK4, 6-head Enoki
bash scripts/launch.sh --arch enoki --opt dana-star-mk4 --heads 6 --nproc 4

# 4-GPU FSDP, AdamW, 12-head Qwen3
bash scripts/launch.sh --arch qwen3 --opt adamw --heads 12 --nproc 4 --distributed_backend fsdp

# Override LR and iterations
bash scripts/launch.sh --arch enoki --opt adana --heads 3 --lr 0.01 --iterations 500

# Disable wandb for quick testing
bash scripts/launch.sh --arch enoki --opt adamw --heads 3 --no_wandb
```

### SLURM Submission

```bash
# Submit Enoki training job
sbatch --time=4:00:00 --nodes=1 --gpus-per-node=h100:4 --mem=320GB \
  scripts/restart_enoki.sh --opt dana-star-mk4 --heads 6 --nproc 4

# With auto-restart for long runs
sbatch --time=4:00:00 --nodes=1 --gpus-per-node=h100:4 --mem=320GB \
  scripts/restart_enoki.sh --opt adana --heads 12 --nproc 4 --iterations_to_run 5000
```

### Direct torchrun (bypassing launch.sh)

```bash
torchrun --standalone --nproc_per_node=1 ./src/main.py \
  --config_format base --model enoki --opt adana \
  --n_embd 192 --n_head 3 --n_layer 2 --qkv_dim 64 --mlp_hidden_dim 768 \
  --batch_size 32 --sequence_length 2048 --dataset fineweb \
  --iterations 100 --lr 0.01 --weight_decay 100 \
  --scheduler cos --warmup_steps 10 --grad_clip 2.5 \
  --init-scheme ScaledGPT --dropout 0.0 --seed 0 \
  --weight_tying False --delta 8.0 --kappa 0.85 --wd_decaying --wd_ts 1.0
```

---

## Cluster Setup Guide

### Setting Up `scripts/config.sh`

`config.sh` auto-detects the cluster from hostname and configures modules, venvs, and paths. To add a new cluster:

1. Add a hostname pattern to the `case` statement in `config.sh`:
   ```bash
   mycluster*)
       DSTAR_CLUSTER="mycluster"
       ;;
   ```
2. Add a corresponding block to load modules and activate the venv:
   ```bash
   mycluster)
       module load python/3.12 2>/dev/null || true
       source "$HOME/danastarenv/bin/activate" 2>/dev/null || true
       export DATASETS_DIR="$HOME/scratch/datasets"
       ;;
   ```
3. Alternatively, skip auto-detection by setting `DSTAR_CLUSTER=mycluster` before sourcing.

### Required Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WANDB_API_KEY` | Weights & Biases API key | (none, wandb disabled without it) |
| `WANDB_ENTITY` | WandB team/user entity | (none) |
| `WANDB_PROJECT` | WandB project name | `danastar` |
| `DATASETS_DIR` | Path to dataset directory | `./src/data/datasets/` |
| `RESULTS_BASE_FOLDER` | Path for checkpoints/results | `./exps` |
| `TIKTOKEN_CACHE_DIR` | Tiktoken tokenizer cache | `$HOME/tiktoken_cache` |

### Data Directory Layout

The dataset directory should contain tokenized binary files:

```
$DATASETS_DIR/
  fineweb-10BT/
    train.bin          # Tokenized training data (tiktoken GPT-2 BPE, uint16)
    val.bin            # Tokenized validation data
  fineweb-100BT/       # Larger dataset (same format)
    train.bin
    val.bin
```

To prepare data, run `python3 process_fineweb_100bt.py` (downloads from HuggingFace and tokenizes).

### FSDP vs DDP

| Feature | DDP (`--distributed_backend nccl`) | FSDP (`--distributed_backend fsdp`) |
|---------|-----|------|
| Memory | Full model on each GPU | Parameters sharded across GPUs |
| Use when | Model fits in single GPU memory | Model too large for single GPU |
| Checkpoint format | Single `main.pt` file | Sharded via `torch.distributed.checkpoint` |
| All-rank operations | Only forward/backward | Forward, backward, eval, and checkpoint |

```bash
# DDP (default)
bash scripts/launch.sh --arch enoki --opt adana --heads 6 --nproc 4

# FSDP
bash scripts/launch.sh --arch enoki --opt adana --heads 12 --nproc 4 --distributed_backend fsdp
```

### Auto-Resume

When `--auto_resume` is enabled (default in `launch.sh`), training automatically resumes from the latest checkpoint if one exists. The checkpoint path is derived from the run configuration (architecture, optimizer, model size, wandb group).

- **DDP checkpoints:** Detected by presence of `main.pt`
- **FSDP checkpoints:** Detected by presence of `meta.pt`

To disable: pass `--no_auto_resume` to `launch.sh`.

### Checkpoint Reliability (SLURM)

On SLURM clusters with `$SLURM_TMPDIR`, checkpoints are:
1. Written to fast local NVMe (`$SLURM_TMPDIR/ckpts/`)
2. Background-rsynced to the project directory atomically
3. Old checkpoint is only removed after new one is fully synced

This prevents checkpoint corruption from preemption or network issues.

### Example SLURM Submission

```bash
# Single-node, 4x H100, ADana optimizer, 6-head Enoki (~210M params)
sbatch --account=def-myacct --time=12:00:00 \
  --nodes=1 --gpus-per-node=h100:4 --mem=320GB --cpus-per-task=16 \
  scripts/restart_enoki.sh --opt adana --heads 6 --nproc 4 \
  --wandb_group my_sweep

# Long training with auto-restart (each job runs 5000 iterations, then requeues)
sbatch --account=def-myacct --time=4:00:00 \
  --nodes=1 --gpus-per-node=h100:4 --mem=320GB --cpus-per-task=16 \
  scripts/restart_enoki.sh --opt dana-star-mk4 --heads 12 --nproc 4 \
  --iterations_to_run 5000

# Qwen3 architecture
sbatch --account=def-myacct --time=12:00:00 \
  --nodes=1 --gpus-per-node=h100:4 --mem=320GB --cpus-per-task=16 \
  scripts/restart_qwen3.sh --opt adamw --heads 6 --nproc 4
```

The `restart_*.sh` scripts handle SLURM requeuing: when `--iterations_to_run` is set, the job automatically resubmits itself upon completion until total `--iterations` is reached.
