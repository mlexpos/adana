# Quickstart Guide

Training code for [ADana](README.md). See the [README](README.md) for an overview of the algorithm.

## Quickstart

### 1. Install dependencies

```bash
python -m venv adana-env
source adana-env/bin/activate
pip install -r requirements.txt

# Optional: for JAX theoretical analysis and visualization
pip install -r requirements-viz.txt
```

### 2. Download and tokenize data

Download and tokenize 10B tokens from FineWeb:

```bash
python src/data/fineweb.py
```

For the full 100B-token dataset:

```bash
python src/data/download_fineweb_100bt.py --local-dir /path/to/fineweb/
```

### 3. Configure paths

```bash
cp scripts/config.local.example scripts/config.local
```

Edit `scripts/config.local` with your paths and (optionally) WandB credentials:

```bash
DATASETS_DIR="/path/to/datasets"           # must contain fineweb-10BT/ or fineweb-100BT/
RESULTS_BASE_FOLDER="./exps"               # where checkpoints are saved
# WANDB_API_KEY="your-key"                 # optional
# WANDB_ENTITY="your-team"                 # optional
# WANDB_PROJECT="adana"                    # optional
```

### 4. Launch training

```bash
# Single GPU, 3-head Enoki (~2M non-emb params) with ADana
bash scripts/launch.sh --arch enoki --opt adana --heads 3

# Larger model, 12 heads (~88M non-emb params)
bash scripts/launch.sh --arch enoki --opt adana --heads 12

# Multi-GPU with FSDP
bash scripts/launch.sh --arch enoki --opt adana --heads 12 --nproc 4 \
  --distributed_backend fsdp --batch_size 8 --acc_steps 16

# Qwen3 architecture with Dana-Star-MK4
bash scripts/launch.sh --arch qwen3 --opt dana-star-mk4 --heads 6
```

The launcher automatically computes model dimensions, learning rate, weight decay, and iteration count from the number of heads using scaling rules in `src/config/scaling_rules.yaml`.

## Available optimizers

| Flag | Optimizer | Description |
|------|-----------|-------------|
| `adana` | ADana | Log-time momentum + decaying WD + damped Nesterov |
| `dana-star-mk4` | Dana-Star-MK4 | ADana + tau estimator + SNR clipping |
| `adamw` | AdamW | Standard baseline |
| `muon` | Muon | Momentum + orthogonalization |
| `ademamix` | AdEMAMix | Triple EMA optimizer |
| `ademamix-decaying-wd` | AdEMAMix (decay WD) | AdEMAMix with decaying weight decay |

## Available architectures

| Flag | Architecture | Notes |
|------|-------------|-------|
| `enoki` | Enoki (GPT-3-like) | RoPE, QK-LayerNorm, pre-LN, residual scaling |
| `qwen3` | Qwen3 | SwiGLU, RMSNorm, elementwise gating |

## Key hyperparameters

ADana's hyperparameters are designed to be scale-invariant (no retuning across model sizes):

| Parameter | Default | Flag | Description |
|-----------|---------|------|-------------|
| kappa | 0.85 | `--kappa` | Spectral dimension; controls Nesterov damping |
| delta | 8.0 | `--delta` | EMA coefficient for log-time momentum |
| omega | 4.0 | `--omega` | Weight decay constant |

## Multi-GPU (FSDP)

For distributed training, pass `--distributed_backend fsdp` and set `--nproc` to match your GPU count. Adjust `--batch_size` (per-GPU micro-batch) and `--acc_steps` to hit your target global batch size:

```
global_batch = batch_size × nproc × acc_steps
```

## SLURM clusters

For SLURM-managed clusters, use the restart scripts which handle job preemption and auto-resume:

```bash
sbatch --account=YOUR_ACCOUNT --time=3:00:00 \
  --nodes=1 --gpus-per-node=4 --mem=0 \
  scripts/restart_enoki.sh \
  --opt adana --heads 8 --nproc 4 \
  --batch_size 8 --acc_steps 16 \
  --distributed_backend fsdp
```

### Offline tokenizer cache

If compute nodes lack internet access, pre-cache the tiktoken GPT-2 tokenizer on the login node:

```bash
mkdir -p ~/tiktoken_cache
python -c "
import requests, hashlib, os
cache = os.path.expanduser('~/tiktoken_cache')
for url in [
    'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe',
    'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json',
]:
    key = hashlib.sha1(url.encode()).hexdigest()
    print(f'Downloading {url.split(\"/\")[-1]} -> {key}')
    r = requests.get(url)
    with open(os.path.join(cache, key), 'wb') as f:
        f.write(r.content)
print('Done.')
"
```

Then set `TIKTOKEN_CACHE_DIR=~/tiktoken_cache` in your `scripts/config.local`.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'yaml'`** — Run `pip install pyyaml`.
- **Tokenizer download errors on compute nodes** — See "Offline tokenizer cache" above.
- **FSDP hangs during eval** — Ensure you're on the latest commit; this was fixed so all ranks participate in eval forward passes.

## Project structure

```
src/
  main.py                 # Training entry point
  config/
    base.py               # CLI argument definitions
    scaling.py            # Auto-compute model dims from --heads
    scaling_rules.yaml    # LR, WD, iteration scaling rules
  optim/
    adana.py              # ADana optimizer
    dana_star_mk4.py      # Dana-Star-MK4 optimizer
  models/
    enoki.py              # Enoki (GPT-3-like) transformer
    qwen3.py              # Qwen3 transformer
  distributed/
    fsdp.py               # FSDP2 backend
scripts/
  launch.sh               # Universal launcher
  config.sh               # Cluster auto-detection + defaults
  config.local.example    # Template for user-local settings
jax/                      # JAX theoretical analysis (PLRF model)
visualization/            # WandB data → paper figures
```

## Citation

```bibtex
@article{paquette2025logarithmic,
  title={Logarithmic-time Schedules for Scaling Language Models with Momentum},
  author={Paquette, Elliot and Paquette, Courtney},
  journal={arXiv preprint arXiv:2602.05298},
  year={2026}
}
```
