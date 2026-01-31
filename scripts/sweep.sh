#!/bin/bash
# =============================================================================
# sweep.sh - Template sweep script for submitting multiple training runs
# =============================================================================
# Edit the configuration section below, then run:
#
#   bash scripts/sweep.sh          # submit all jobs
#   bash scripts/sweep.sh --dry    # print commands without submitting
#
# Each combination of (HEADS x LR_MULTIPLIERS) becomes one sbatch job.
# All jobs land in the same WANDB_GROUP for easy comparison.
# =============================================================================

set -euo pipefail

# =============================================================================
# ===== EDIT THIS SECTION FOR YOUR EXPERIMENT ================================
# =============================================================================

# --- Experiment identity ---
WANDB_GROUP="fsdp-experiment"
ACCOUNT="rrg-bengioy-ad"

# --- Architecture and optimizer ---
ARCH="enoki"                    # enoki or qwen3
OPT="adana"                    # adana, dana-star-mk4, adamw, ademamix, d-muon, ...
KAPPA="0.85"                   # kappa for DANA variants

# --- Model sizes to sweep (number of attention heads) ---
HEADS_LIST=( 8 )

# --- LR multipliers (applied to the auto-computed LR from scaling rules) ---
# 1.0 = use formula LR as-is. Set to ( 1.0 ) to skip LR sweep.
LR_MULTIPLIERS=( 1.0 )

# --- Batch configuration ---
BATCH_SIZE=8                   # per-GPU micro-batch size
ACC_STEPS=16                   # gradient accumulation steps
                               # global_batch = BATCH_SIZE * ACC_STEPS * NPROC

# --- GPU / SLURM resources ---
NPROC=4                        # GPUs per node
TIME="3:00:00"                 # wall time per job
MEM="0"                        # memory (0 = allocate as needed on Alliance)

# --- Distributed backend ---
DISTRIBUTED_BACKEND="fsdp"     # nccl (DDP) or fsdp

# --- Training options ---
SCHEDULER="cos_inf"            # cos_inf (decays to 10% of peak LR), cos, wsd, linear
NO_AUTO_RESUME=1               # 1 = fresh start, 0 = resume from checkpoint
ITERATIONS_OVERRIDE=""         # override auto-computed iterations (leave empty for Chinchilla-optimal)
ITERATIONS_TO_RUN=""           # max iterations per SLURM job (for auto-restart, leave empty to disable)

# --- WandB ---
WANDB_OFFLINE=0                # 1 = offline mode (for nodes without internet), sync later

# --- Other ---
EXTRA_ARGS=""                  # any additional args passed to launch.sh
                               # e.g. "--no_wandb" "--no_compile" "--eval_interval 100"

# =============================================================================
# ===== END OF CONFIGURATION =================================================
# =============================================================================

# Check for --dry flag
DRY_RUN=0
if [[ "${1:-}" == "--dry" ]]; then
    DRY_RUN=1
    echo "=== DRY RUN (no jobs will be submitted) ==="
    echo ""
fi

# Use restart script based on architecture
case "$ARCH" in
    enoki)  RESTART_SCRIPT="scripts/restart_enoki.sh" ;;
    qwen3)  RESTART_SCRIPT="scripts/restart_qwen3.sh" ;;
    *)      echo "ERROR: Unknown architecture '$ARCH'"; exit 1 ;;
esac

# Source config to get access to scaling rules for LR preview
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Print sweep summary
TOTAL_JOBS=$(( ${#HEADS_LIST[@]} * ${#LR_MULTIPLIERS[@]} ))
GLOBAL_BATCH=$(( BATCH_SIZE * ACC_STEPS * NPROC ))

echo "============================================================"
echo "Sweep: $WANDB_GROUP"
echo "============================================================"
echo "Architecture:  $ARCH"
echo "Optimizer:     $OPT (kappa=$KAPPA)"
echo "Backend:       $DISTRIBUTED_BACKEND"
echo "Heads:         ${HEADS_LIST[*]}"
echo "LR multipliers: ${LR_MULTIPLIERS[*]}"
echo "Batch:         ${BATCH_SIZE} x ${ACC_STEPS} x ${NPROC} GPUs = ${GLOBAL_BATCH} global"
echo "SLURM:         ${NPROC} GPUs, ${TIME}, mem=${MEM}"
echo "WandB:         $([ "$WANDB_OFFLINE" -eq 1 ] && echo "offline" || echo "online")"
echo "Total jobs:    $TOTAL_JOBS"
echo "============================================================"
echo ""

job_count=0

for HEADS in "${HEADS_LIST[@]}"; do
    # Compute base LR from scaling rules
    BASE_LR=$(python3 -c "
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location('scaling', os.path.join('$SCRIPT_DIR', '..', 'src', 'config', 'scaling.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
dims = mod.compute_dimensions('$ARCH', $HEADS)
lr = mod.compute_lr('$ARCH', '$OPT', dims['non_emb_params'], kappa=$KAPPA)
if lr is None:
    print('ERROR', file=sys.stderr)
    sys.exit(1)
print(f'{lr:.6e}')
print(f'{dims[\"non_emb_params\"]}', file=sys.stderr)
print(f'{dims[\"total_params\"]}', file=sys.stderr)
" 2>/dev/null)

    # Get dimensions for display
    DIMS=$(python3 -c "
import importlib.util, os
spec = importlib.util.spec_from_file_location('scaling', os.path.join('$SCRIPT_DIR', '..', 'src', 'config', 'scaling.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
dims = mod.compute_dimensions('$ARCH', $HEADS)
tokens_per_step = $GLOBAL_BATCH * 2048
iters = int(20 * dims['total_params'] / tokens_per_step)
print(f'non_emb={dims[\"non_emb_params\"]/1e6:.1f}M total={dims[\"total_params\"]/1e6:.1f}M iters={iters}')
")

    echo "--- heads=$HEADS ($DIMS, base_lr=$BASE_LR) ---"

    for MULT in "${LR_MULTIPLIERS[@]}"; do
        LR=$(python3 -c "print(f'{$MULT * $BASE_LR:.6e}')")
        job_count=$((job_count + 1))

        # Build launch args
        LAUNCH_ARGS=(
            --opt "$OPT"
            --heads "$HEADS"
            --nproc "$NPROC"
            --batch_size "$BATCH_SIZE"
            --acc_steps "$ACC_STEPS"
            --kappa "$KAPPA"
            --lr "$LR"
            --scheduler "$SCHEDULER"
            --distributed_backend "$DISTRIBUTED_BACKEND"
            --wandb_group "$WANDB_GROUP"
        )

        if [ "$NO_AUTO_RESUME" -eq 1 ]; then
            LAUNCH_ARGS+=(--no_auto_resume)
        fi

        if [ -n "$ITERATIONS_OVERRIDE" ]; then
            LAUNCH_ARGS+=(--iterations "$ITERATIONS_OVERRIDE")
        fi

        if [ -n "$ITERATIONS_TO_RUN" ]; then
            LAUNCH_ARGS+=(--iterations_to_run "$ITERATIONS_TO_RUN")
        fi

        if [ "$WANDB_OFFLINE" -eq 1 ]; then
            LAUNCH_ARGS+=(--wandb_offline)
        fi

        if [ -n "$EXTRA_ARGS" ]; then
            LAUNCH_ARGS+=($EXTRA_ARGS)
        fi

        JOB_NAME="${ARCH}_${OPT}_h${HEADS}_lr${MULT}"

        echo "  [$job_count/$TOTAL_JOBS] $JOB_NAME  lr=$LR (${MULT}x)"

        if [ "$DRY_RUN" -eq 1 ]; then
            echo "    sbatch --account=$ACCOUNT --time=$TIME --nodes=1 --gpus-per-node=h100:$NPROC --mem=$MEM --job-name=$JOB_NAME $RESTART_SCRIPT ${LAUNCH_ARGS[*]}"
        else
            sbatch --account="$ACCOUNT" \
                   --time="$TIME" \
                   --nodes=1 \
                   --gpus-per-node="h100:$NPROC" \
                   --mem="$MEM" \
                   --job-name="$JOB_NAME" \
                   "$RESTART_SCRIPT" \
                   "${LAUNCH_ARGS[@]}"

            if [ $? -eq 0 ]; then
                echo "    -> submitted"
            else
                echo "    -> FAILED (exit code $?)"
            fi
        fi
    done
    echo ""
done

echo "============================================================"
echo "Submitted $job_count / $TOTAL_JOBS jobs"
echo "WandB group: $WANDB_GROUP"
echo "============================================================"
