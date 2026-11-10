#!/bin/bash
#SBATCH --account=rrg-epaq_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/home/epaq/scratch/adana-runs/slurm/%x_%j.out
# =============================================================================
# rorqual sbatch wrapper around scripts/launch.sh (mlexpos/adana)
#
# usage: sbatch -J <name> [--time=H:MM:SS] [--gpus-per-node=h100:4 --cpus-per-task=32 --mem=256G] \
#            ~/scratch/adana/scripts/rorqual_job.sh --arch enoki --opt adamw --heads 6 [launch.sh args...]
#
# - Number of torchrun procs defaults to the number of GPUs allocated (override: --nproc N).
# - --batch_size * --acc_steps is the GLOBAL batch (split across GPUs by DDP).
# - Module loads + venv come from scripts/config.sh; paths/wandb from scripts/config.local.
# =============================================================================
set -eo pipefail

ADANA_DIR="${ADANA_DIR:-$HOME/scratch/adana}"
cd "$ADANA_DIR"   # launch.sh uses $(pwd)/src/...

# Modules + venv + user config (config.sh sources config.local)
source scripts/config.sh

export WANDB_MODE=offline
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$HOME/tiktoken_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
export NPROC_PER_NODE="${NPROC_PER_NODE:-${NGPU:-1}}"

echo "[$(date +%F\ %T)] job=${SLURM_JOB_ID:-none} name=${SLURM_JOB_NAME:-} host=$(hostname -s) gpus=$NGPU nproc=$NPROC_PER_NODE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "python: $(which python) ($(python -V 2>&1)); torch $(python -c "import torch; print(torch.__version__)")"
echo "launch.sh $*"

set +e
bash scripts/launch.sh "$@"
rc=$?
echo "[$(date +%F\ %T)] exit $rc"
exit $rc
