#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/epaq/adana-runs/slurm/%x_%j.out
# =============================================================================
# trillium (SciNet) sbatch wrapper around scripts/launch.sh (mlexpos/adana): whole 4 x H100 nodes (the compute partition does
# not oversubscribe).  usage: sbatch -J <name> [--time=H:MM:SS] scripts/trillium_job.sh --arch enoki --opt ... [launch.sh args]
# --batch_size * --acc_steps is the GLOBAL batch (split across GPUs by DDP).
# =============================================================================
set -eo pipefail
ADANA_DIR="${ADANA_DIR:-$SCRATCH/adana}"
cd "$ADANA_DIR"
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
