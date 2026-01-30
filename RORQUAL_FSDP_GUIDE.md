# Rorqual FSDP Test Run Guide

Step-by-step guide to clone the refactored branch onto Rorqual (Alliance cluster) and launch an FSDP test experiment.

**Experiment:** 8-head Enoki (~18.9M non-emb / ~70M total params), FSDP, 4 GPUs, global batch 512, WandB group `fsdp-experiment`.

---

## Step 1: Clone and checkout

```bash
ssh rorqual   # or narval, whichever is the login node

cd ~
git clone https://github.com/LosVolterrosHermanos/danastar.git
cd danastar
git checkout refactor/cleanup
```

If the repo already exists:

```bash
cd ~/danastar
git fetch origin
git checkout refactor/cleanup
git pull origin refactor/cleanup
```

**Verify:** After pulling, the `scripts/` directory should only contain 5 files: `config.sh`, `config.local.example`, `launch.sh`, `restart_enoki.sh`, `restart_qwen3.sh`. If you still see folders like `scripts/narval/`, `scripts/fir/`, `scripts/124m/`, etc., you're not on the latest commit — re-run `git pull`.

---

## Step 2: Set up Python environment

```bash
module load gcc
module load arrow/21.0.0
module load python/3.13

python -m venv ~/danastarenv
source ~/danastarenv/bin/activate
pip install -r rorqualrequirements.txt
```

If the venv already exists, just activate and update:

```bash
module load gcc
module load arrow/21.0.0
module load python/3.13
source ~/danastarenv/bin/activate
pip install -r rorqualrequirements.txt --upgrade
```

---

## Step 3: Set up tiktoken cache (offline tokenizer)

Rorqual compute nodes don't have internet. The tiktoken GPT-2 tokenizer files must be cached on the login node first.

```bash
mkdir -p ~/tiktoken_cache
source ~/danastarenv/bin/activate

python -c "
import requests, hashlib

vocab_bpe_url = 'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe'
encoder_json_url = 'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json'

vocab_key = hashlib.sha1(vocab_bpe_url.encode()).hexdigest()
encoder_key = hashlib.sha1(encoder_json_url.encode()).hexdigest()

import os
cache_dir = os.path.expanduser('~/tiktoken_cache')

print(f'Downloading vocab.bpe -> {vocab_key}')
r = requests.get(vocab_bpe_url)
with open(os.path.join(cache_dir, vocab_key), 'wb') as f:
    f.write(r.content)

print(f'Downloading encoder.json -> {encoder_key}')
r = requests.get(encoder_json_url)
with open(os.path.join(cache_dir, encoder_key), 'wb') as f:
    f.write(r.content)

print('Done.')
"
```

---

## Step 4: Verify data exists

The experiment needs tokenized FineWeb data. Check that it exists at whatever path you'll set as `DATASETS_DIR` in Step 5:

```bash
# Example — adjust path to match your setup
ls -lh ~/links/projects/def-epoch-datasets/fineweb-10BT/
# Should have: train.bin, val.bin

# OR if using fineweb-100BT:
ls -lh ~/links/projects/def-epoch-datasets/fineweb-100BT/
```

If neither exists, you need to tokenize first (this is a separate long-running job, see README.md).

---

## Step 5: Create your user config file

Rorqual is already configured in `config.sh` (auto-detected from hostname). What you need to set up is your **user-level config** with paths and credentials.

```bash
cp scripts/config.local.example scripts/config.local
vim scripts/config.local
```

Edit `scripts/config.local` with your settings:

```bash
# Weights & Biases
WANDB_API_KEY="your-api-key-here"
WANDB_ENTITY="ep-rmt-ml-opt"
WANDB_PROJECT="danastar"

# Data and checkpoint paths (these are user-specific)
DATASETS_DIR="$HOME/links/projects/def-epoch-datasets"
RESULTS_BASE_FOLDER="$HOME/links/projects/def-epoch-datasets/exps"
```

This file is sourced by `config.sh` before every run. It lives in the repo's `scripts/` directory but is gitignored, so each user maintains their own copy per cluster.

---

## Step 6: Create logs directory

```bash
cd ~/danastar
mkdir -p logs
```

---

## Step 7: Verify scaling rules locally (sanity check)

Before submitting, verify the Python scaling logic works:

```bash
source ~/danastarenv/bin/activate
cd ~/danastar

python3 -c "
import importlib.util, os
spec = importlib.util.spec_from_file_location('scaling', 'src/config/scaling.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

dims = mod.compute_dimensions('enoki', 8)
print('Model dimensions:')
print(f'  n_head={dims[\"n_head\"]}, n_layer={dims[\"n_layer\"]}, n_embd={dims[\"n_embd\"]}')
print(f'  mlp_hidden_dim={dims[\"mlp_hidden_dim\"]}, qkv_dim={dims[\"qkv_dim\"]}')
print(f'  non_emb_params={dims[\"non_emb_params\"]:,} ({dims[\"non_emb_params\"]/1e6:.1f}M)')
print(f'  total_params={dims[\"total_params\"]:,} ({dims[\"total_params\"]/1e6:.1f}M)')
"
```

Expected output:

```
n_head=8, n_layer=6, n_embd=512
mlp_hidden_dim=2048, qkv_dim=64
non_emb_params=18,874,368 (18.9M)
total_params=70,385,664 (70.4M)
```

---

## Step 8: Submit the job

```bash
cd ~/danastar

sbatch --account=rrg-bengioy-ad --time=3:00:00 \
  --nodes=1 --gpus-per-node=h100:4 --mem=0 \
  scripts/restart_enoki.sh \
  --opt adana --heads 8 --nproc 4 \
  --batch_size 8 --acc_steps 16 \
  --distributed_backend fsdp \
  --wandb_group fsdp-experiment \
  --no_auto_resume
```

### What this does

| Parameter | Value | Why |
|-----------|-------|-----|
| `--heads 8` | 8 attention heads | 18.9M non-emb params (~70M total) |
| `--nproc 4` | 4 GPUs | Matches `--gpus-per-node=h100:4` |
| `--batch_size 8` | 8 sequences/GPU/microstep | Per-GPU micro-batch |
| `--acc_steps 16` | 16 accumulation steps | Global batch = 8 * 4 * 16 = 512 sequences |
| `--distributed_backend fsdp` | FSDP2 sharding | Test FSDP (not needed at this model size, but that's the point) |
| `--wandb_group fsdp-experiment` | WandB group name | Groups runs together in WandB |
| `--no_auto_resume` | Fresh start | No checkpoint loading, clean test |

### Computed values (by launch.sh)

| Value | Formula | Result |
|-------|---------|--------|
| LR | `22.5 * (122000 + 18874368)^(-0.618)` | ~7.14e-4 (adana, kappa=0.85) |
| WD_TS | `iterations / 10` | 134 |
| Weight decay | `omega / wd_ts` = `4.0 / 134` | ~0.0299 (independent WD, not divided by LR) |
| Tokens/step | `8 * 16 * 2048 * 4` = 1,048,576 | ~1.0M tokens/step |
| Iterations | `20 * 70385664 / 1048576` (20 * total_params) | 1342 |
| Warmup | `iterations / 50` | 26 |
| Scheduler | cos_inf (cosine decay to 10% of peak LR) | |

### Important caveat: LR formulas are for batch-32

The LR scaling formulas in `scaling_rules.yaml` were fit for global batch size 32 (the paper's standard). For global batch 512, the optimal LR may differ. For this test run it should be fine — we're testing FSDP mechanics, not optimizing loss. For production batch-512 runs, consider overriding with `--lr <value>`.

---

## Step 9: Monitor

```bash
# Check job status
squeue -u $USER

# Watch logs
tail -f logs/restart_enoki*-<JOBID>.out

# Check WandB
# Look for runs in project "danastar", group "fsdp-experiment"
```

### What to look for

1. **Startup prints:** Should see `[config.sh] Cluster: rorqual` and the scaling rule output with correct dimensions
2. **FSDP initialization:** Should see FSDP wrapping messages (no errors about parameter sharding)
3. **Training loop:** Loss should decrease from ~10-11 initially. With 1342 iterations, should finish well within 3 hours at this model size
4. **Checkpointing:** Latest checkpoint saves every 1000 steps, so one checkpoint write around step 1000
5. **Eval:** Eval runs every 200 steps by default. Watch for any FSDP eval deadlocks (should be fixed, but worth confirming)

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'yaml'`**
→ pyyaml not installed. Run `pip install pyyaml` in the venv. (It's included via `rorqualrequirements.txt` as a transitive dependency of other packages, but add it explicitly if missing.)

**`TIKTOKEN_CACHE_DIR` errors / tokenizer download attempts**
→ Repeat Step 3 on the login node. The cache must exist before the compute job runs.

**FSDP hangs during eval**
→ This bug was fixed in the refactor (all ranks now participate in eval forward passes). If it still hangs, check that you're on the latest `refactor/cleanup` commit.

**`--distributed_backend fsdp` not recognized**
→ Check `src/distributed/__init__.py` has the `"fsdp"` entry. The refactored branch should have it.

**WandB offline mode**
→ If compute nodes have no internet, add `WANDB_MODE=offline` to your `scripts/config.local`, or pass `--no_wandb` for the first test. The old Rorqual scripts used `wandb offline`.
