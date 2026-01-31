#!/bin/bash
# =============================================================================
# benchmark_batch.sh - Find max batch size for FSDP and FSDP+TP4
# =============================================================================
# Single SLURM job that tests each (heads, tp_size, batch_size) combo
# sequentially to find the largest feasible batch size per configuration.
#
# Submits as one job:
#   sbatch --account=rrg-bengioy-ad --time=3:00:00 --nodes=1 \
#          --gpus-per-node=h100:4 --mem=0 scripts/benchmark_batch.sh
#
# Results are printed to stdout and collected in RESULTS_FILE.
# =============================================================================
#SBATCH --output=logs/benchmark_batch-%j.out
#SBATCH --error=logs/benchmark_batch-%j.err
#SBATCH --cpus-per-gpu=8

set -uo pipefail

# --- Source cluster config ---
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/scripts"
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
source "$SCRIPT_DIR/config.sh"

mkdir -p logs

# =============================================================================
# Configuration
# =============================================================================
HEADS_LIST=(28 32 36 44 48)
TP_SIZES=(1 4)
BATCH_SIZES=(4 8 16 32)         # tested in order; we record which ones pass

ARCH="enoki"
OPT="dana-star-mk4"
KAPPA="0.85"
NPROC=4                          # GPUs per node (must be >= max(TP_SIZES))
ITERATIONS=5
WARMUP_STEPS=3
LOG_INTERVAL=1
EVAL_INTERVAL=9999               # skip eval
ACC_STEPS=1
SEQ_LEN=2048
SCHEDULER="cos"
DISTRIBUTED_BACKEND="fsdp"

RESULTS_FILE="logs/benchmark_batch_results.txt"

# =============================================================================
# Helper: run one configuration, return 0 on success, 1 on failure
# =============================================================================
run_one() {
    local heads=$1
    local tp_size=$2
    local batch_size=$3

    # Compute dimensions via scaling rules
    local scaling_out
    scaling_out=$(python3 -c "
import importlib.util, os
spec = importlib.util.spec_from_file_location('scaling', os.path.join('$REPO_DIR', 'src', 'config', 'scaling.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
dims = mod.compute_dimensions('$ARCH', $heads)
lr = mod.compute_lr('$ARCH', '$OPT', dims['non_emb_params'], kappa=$KAPPA)
if lr is None:
    lr = 1e-4  # fallback for benchmark
# WD setup for dana-star-mk4
iterations = $ITERATIONS
wd_ts = max(iterations // 10, 1)
omega = 4.0
wd = omega / wd_ts
print(f'N_HEAD={dims[\"n_head\"]}')
print(f'N_LAYER={dims[\"n_layer\"]}')
print(f'N_EMBD={dims[\"n_embd\"]}')
print(f'MLP_HIDDEN={dims[\"mlp_hidden_dim\"]}')
print(f'QKV_DIM={dims[\"head_dim\"]}')
print(f'TOTAL_PARAMS={dims[\"total_params\"]}')
print(f'LR={lr}')
print(f'WD={wd}')
print(f'WD_TS={wd_ts}')
")
    eval "$scaling_out"

    echo ""
    echo "============================================================"
    echo "  heads=$heads  tp=$tp_size  batch=$batch_size"
    echo "  n_embd=$N_EMBD  n_layer=$N_LAYER  params=$(python3 -c "print(f'{$TOTAL_PARAMS/1e6:.0f}M')")"
    echo "============================================================"

    # TP flags
    local tp_flags=""
    if [ "$tp_size" -gt 1 ]; then
        tp_flags="--tp_size $tp_size"
    fi

    # Run training; capture output, detect OOM
    local output
    output=$(torchrun --standalone --nproc_per_node=$NPROC "$REPO_DIR/src/main.py" \
        --config_format base \
        --model enoki \
        --distributed_backend $DISTRIBUTED_BACKEND \
        $tp_flags \
        --n_embd $N_EMBD --n_head $N_HEAD --n_layer $N_LAYER \
        --qkv_dim $QKV_DIM --mlp_hidden_dim $MLP_HIDDEN \
        --batch_size $batch_size --sequence_length $SEQ_LEN --acc_steps $ACC_STEPS \
        --datasets_dir $DATASETS_DIR \
        --dataset fineweb_100 \
        --iterations $ITERATIONS \
        --lr $LR --weight_decay $WD \
        --scheduler $SCHEDULER \
        --warmup_steps $WARMUP_STEPS \
        --grad_clip 0.5 \
        --init-scheme ScaledGPT \
        --dropout 0.0 --seed 0 \
        --eval_interval $EVAL_INTERVAL \
        --eval_batches 4 \
        --log_interval $LOG_INTERVAL \
        --latest_ckpt_interval 0 \
        --permanent_ckpt_interval 0 \
        --weight_tying False \
        --opt dana-star-mk4 --delta 8.0 --kappa $KAPPA --clipsnr 1.0 \
        --wd_decaying --wd_ts $WD_TS \
        2>&1) || true

    # Check for OOM
    if echo "$output" | grep -qi "OutOfMemoryError\|CUDA out of memory\|out of memory"; then
        echo "  => OOM"
        return 1
    fi

    # Check for other fatal errors (but not OOM)
    if echo "$output" | grep -qi "Error\|Traceback" | grep -qvi "memory"; then
        # Parse whether it actually ran any iterations
        if ! echo "$output" | grep -q "Train: Iter="; then
            echo "  => FAILED (non-OOM error)"
            echo "$output" | grep -E "Error|Traceback" | head -3
            return 1
        fi
    fi

    # Extract peak memory and iter_dt from output
    local peak_mem iter_dt
    peak_mem=$(echo "$output" | grep -o "peak_allocated=[0-9.]*" | head -1 | cut -d= -f2)
    # Get iter_dt from the last training iteration (steady state)
    iter_dt=$(echo "$output" | grep "Train: Iter=$ITERATIONS " | grep -o "iter_dt=[0-9.e+-]*s" | head -1 | sed 's/iter_dt=//;s/s$//')

    echo "  => OK  peak_mem=${peak_mem:-?} GiB  iter_dt=${iter_dt:-?}s"
    echo "$output" | grep -E "Train: Iter=" | tail -3

    # Record result
    echo "heads=$heads  tp=$tp_size  batch=$batch_size  peak_mem=${peak_mem:-?}  iter_dt=${iter_dt:-?}  STATUS=OK" >> "$RESULTS_FILE"
    return 0
}

# =============================================================================
# Main benchmark loop
# =============================================================================
echo "============================================================"
echo "Batch Size Benchmark"
echo "============================================================"
echo "Heads:      ${HEADS_LIST[*]}"
echo "TP sizes:   ${TP_SIZES[*]}"
echo "Batch sizes to try: ${BATCH_SIZES[*]}"
echo "GPUs:       $NPROC"
echo "Optimizer:  $OPT (kappa=$KAPPA)"
echo "Iterations: $ITERATIONS"
echo "Results:    $RESULTS_FILE"
echo "============================================================"
echo ""

# Clear results file
echo "# Batch Size Benchmark Results - $(date)" > "$RESULTS_FILE"
echo "# heads  tp  batch  peak_mem(GiB)  iter_dt(s)  status" >> "$RESULTS_FILE"

for HEADS in "${HEADS_LIST[@]}"; do
    for TP in "${TP_SIZES[@]}"; do
        echo ""
        echo "########## heads=$HEADS  tp_size=$TP ##########"

        max_batch=0
        for BS in "${BATCH_SIZES[@]}"; do
            if run_one "$HEADS" "$TP" "$BS"; then
                max_batch=$BS
            else
                echo "heads=$HEADS  tp=$TP  batch=$BS  STATUS=OOM" >> "$RESULTS_FILE"
                # Once we OOM, no point trying larger batch sizes
                break
            fi
        done

        echo ""
        echo ">>> heads=$HEADS  tp=$TP  max_batch=$max_batch"
        echo "heads=$HEADS  tp=$TP  MAX_BATCH=$max_batch" >> "$RESULTS_FILE"
    done
done

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
cat "$RESULTS_FILE"
echo "============================================================"
