#!/bin/bash
#SBATCH --account=aip-cypaquet
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/e/epaq/adana-runs/slurm/%x_%j.out
# =============================================================================
# tamia (Mila/Alliance) sbatch wrapper around scripts/launch.sh (mlexpos/adana): whole 4 x H100 nodes (gpubase_bynode_*; the
# partition follows from --time: <=3h b1, <=12h b2, <=24h b3).
# usage: sbatch -J <name> [--time=H:MM:SS] scripts/tamia_job.sh --arch enoki --opt ... [launch.sh args]
# --batch_size * --acc_steps is the GLOBAL batch (split across GPUs by DDP).
# =============================================================================
set -eo pipefail
ADANA_DIR="${ADANA_DIR:-/scratch/e/epaq/adana}"
cd "$ADANA_DIR"
export DSTAR_CLUSTER=tamia
source scripts/config.sh

export WANDB_MODE=offline
# every cache off $HOME (25 GiB quota)
export XDG_CACHE_HOME="/scratch/e/epaq/.cache"
export MPLCONFIGDIR="/scratch/e/epaq/.cache/matplotlib"
export WANDB_CACHE_DIR="/scratch/e/epaq/.cache/wandb" WANDB_CONFIG_DIR="/scratch/e/epaq/.config/wandb"
mkdir -p "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$HOME/tiktoken_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Runs longer than one job (partitions stop at 24 h) are chains of jobs (sbatch --dependency=afterany:<previous>, same arguments,
# auto-resume).  ADANA_DEADLINE: ADANA_MARGIN seconds (default 45 min: a refresh in progress, a checkpoint copy in flight, the
# final copy) before this job's end, the trainer writes the latest checkpoint and stops; the next job resumes from it, and a
# job whose run is already complete exits at once (src/main.py).
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${ADANA_DEADLINE:-}" ]; then
    END=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)
    if END_S=$(date -d "$END" +%s 2>/dev/null); then
        export ADANA_DEADLINE=$((END_S - ${ADANA_MARGIN:-2700}))
        echo "job ends $END; checkpoint and stop at $(date -d @$ADANA_DEADLINE '+%F %T')"
    fi
fi

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
