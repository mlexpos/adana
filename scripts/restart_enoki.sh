#!/bin/bash
# =============================================================================
# restart_enoki.sh - SLURM restart wrapper for Enoki architecture
# =============================================================================
# Submits a training job via launch.sh and auto-requeues on completion
# for long training runs that exceed SLURM time limits.
#
# Usage:
#   sbatch --time=4:00:00 --nodes=1 --gpus-per-node=h100:1 --mem=80GB \
#     scripts/restart_enoki.sh --opt dana-star-mk4 --heads 6 [launch.sh args...]
#
# Environment:
#   RESTART_COUNT (auto-incremented, default 0)
#   ITERATIONS_TO_RUN (if set, enables restart after each segment)
# =============================================================================
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-gpu=8

set -euo pipefail

# When SLURM executes a batch script, it copies it to a local spool directory,
# so BASH_SOURCE[0] resolves to the spool path (not the repo). Use
# SLURM_SUBMIT_DIR (the directory where sbatch was invoked) instead.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/scripts"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
RESTART_COUNT="${RESTART_COUNT:-0}"

echo "============================================================"
echo "Enoki Restart Wrapper (restart #$RESTART_COUNT)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-N/A}"
echo "============================================================"

# Create logs directory
mkdir -p logs

# Run training via launch.sh
bash "$SCRIPT_DIR/launch.sh" --arch enoki --auto_resume "$@"
TRAINING_EXIT_CODE=$?

echo "Training exited with code: $TRAINING_EXIT_CODE"

# Auto-requeue if training succeeded and iterations_to_run was used
if [ $TRAINING_EXIT_CODE -eq 0 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    # Check if iterations_to_run was passed (implies we want restart)
    HAS_ITR=0
    for arg in "$@"; do
        if [ "$arg" = "--iterations_to_run" ]; then
            HAS_ITR=1
            break
        fi
    done

    if [ $HAS_ITR -eq 1 ]; then
        NEW_RESTART_COUNT=$((RESTART_COUNT + 1))
        echo "Requeuing job (restart #$NEW_RESTART_COUNT)..."

        # Extract SLURM resource allocation from current job
        SLURM_INFO=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null || true)
        ACCOUNT=$(echo "$SLURM_INFO" | grep -oP 'Account=\K[^ ]+' || echo "")
        TIME_LIMIT=$(echo "$SLURM_INFO" | grep -oP 'TimeLimit=\K[^ ]+' || echo "4:00:00")
        NUM_NODES=$(echo "$SLURM_INFO" | grep -oP 'NumNodes=\K[^ ]+' || echo "1")

        SLURM_ARGS=()
        [ -n "$ACCOUNT" ] && SLURM_ARGS+=(--account="$ACCOUNT")
        SLURM_ARGS+=(--time="$TIME_LIMIT")
        SLURM_ARGS+=(--nodes="$NUM_NODES")

        # Preserve GPU allocation
        if [ -n "${SLURM_GPUS_PER_NODE:-}" ]; then
            SLURM_ARGS+=(--gpus-per-node="$SLURM_GPUS_PER_NODE")
        fi

        # Preserve memory allocation
        MEM_INFO=$(echo "$SLURM_INFO" | grep -oP 'MinMemoryNode=\K[^ ]+' || echo "")
        if [ -n "$MEM_INFO" ] && [ "$MEM_INFO" != "0" ]; then
            SLURM_ARGS+=(--mem="${MEM_INFO}")
        fi

        sbatch --export=ALL,RESTART_COUNT=$NEW_RESTART_COUNT \
            "${SLURM_ARGS[@]}" \
            "$0" "$@"

        echo "Requeue submitted."
    fi
fi

exit $TRAINING_EXIT_CODE
