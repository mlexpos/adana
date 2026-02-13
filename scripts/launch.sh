#!/bin/bash
# =============================================================================
# launch.sh - Universal training launch script for all optimizers and architectures
# =============================================================================
# Usage:
#   bash scripts/launch.sh --arch enoki --opt adana --heads 3
#   bash scripts/launch.sh --arch qwen3 --opt dana-star-mk4 --heads 6 --wandb_group my_sweep
#   bash scripts/launch.sh --arch enoki --opt adamw --heads 12 --nproc 4
#
# This script:
# 1. Sources config.sh for cluster/user settings
# 2. Uses Python scaling rules to compute model dimensions, LR, iterations
# 3. Sets optimizer-specific flags (WD_TS, beta values, etc.)
# 4. Launches training via torchrun
# =============================================================================

set -euo pipefail

# --- Source cluster/user config ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# =============================================================================
# Parse arguments
# =============================================================================
ARCH=""
OPT=""
HEADS=""
WANDB_GROUP=""
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACC_STEPS="${ACC_STEPS:-1}"
SEQ_LEN="${SEQ_LEN:-2048}"
DATASET="${DATASET:-fineweb_100}"
SCHEDULER="${SCHEDULER:-cos_inf}"
KAPPA="${KAPPA:-0.85}"
CLIPSNR="${CLIPSNR:-2.0}"
OMEGA="${OMEGA:-4.0}"
DELTA="${DELTA:-8.0}"
GAMMA_3_FACTOR="${GAMMA_3_FACTOR:-1.0}"
WD_TS_DIVISOR="${WD_TS_DIVISOR:-}"
WD_TS_CONST="${WD_TS_CONST:-}"
WD_TS_DIVISOR_SET=""  # Track if user explicitly set divisor
LR_OVERRIDE=""
ITERATIONS_OVERRIDE=""
WARMUP_STEPS=""
GRAD_CLIP="${GRAD_CLIP:-0.5}"
INIT_SCHEME="${INIT_SCHEME:-ScaledGPT}"
COMPILE="${COMPILE:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-115}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
LATEST_CKPT_INTERVAL="${LATEST_CKPT_INTERVAL:-1000}"
PERMANENT_CKPT_INTERVAL="${PERMANENT_CKPT_INTERVAL:-0}"
AUTO_RESUME="${AUTO_RESUME:-1}"
ITERATIONS_TO_RUN=""
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
Z_LOSS_COEFF="${Z_LOSS_COEFF:-1e-4}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)             ARCH="$2"; shift 2 ;;
        --opt)              OPT="$2"; shift 2 ;;
        --heads)            HEADS="$2"; shift 2 ;;
        --wandb_group)      WANDB_GROUP="$2"; shift 2 ;;
        --nproc)            NPROC_PER_NODE="$2"; shift 2 ;;
        --batch_size)       BATCH_SIZE="$2"; shift 2 ;;
        --acc_steps)        ACC_STEPS="$2"; shift 2 ;;
        --seq_len)          SEQ_LEN="$2"; shift 2 ;;
        --dataset)          DATASET="$2"; shift 2 ;;
        --scheduler)        SCHEDULER="$2"; shift 2 ;;
        --kappa)            KAPPA="$2"; shift 2 ;;
        --clipsnr)          CLIPSNR="$2"; shift 2 ;;
        --omega)            OMEGA="$2"; shift 2 ;;
        --delta)            DELTA="$2"; shift 2 ;;
        --gamma_3_factor)   GAMMA_3_FACTOR="$2"; shift 2 ;;
        --wd_ts_divisor)    WD_TS_DIVISOR="$2"; WD_TS_DIVISOR_SET=1; shift 2 ;;
        --wd_ts_const)      WD_TS_CONST="$2"; shift 2 ;;
        --lr)               LR_OVERRIDE="$2"; shift 2 ;;
        --iterations)       ITERATIONS_OVERRIDE="$2"; shift 2 ;;
        --warmup_steps)     WARMUP_STEPS="$2"; shift 2 ;;
        --grad_clip)        GRAD_CLIP="$2"; shift 2 ;;
        --init_scheme)      INIT_SCHEME="$2"; shift 2 ;;
        --no_compile)       COMPILE=0; shift ;;
        --no_wandb)         WANDB_ENABLED=0; shift ;;
        --wandb_offline)    WANDB_OFFLINE=1; shift ;;
        --eval_interval)    EVAL_INTERVAL="$2"; shift 2 ;;
        --log_interval)     LOG_INTERVAL="$2"; shift 2 ;;
        --latest_ckpt_interval)   LATEST_CKPT_INTERVAL="$2"; shift 2 ;;
        --permanent_ckpt_interval) PERMANENT_CKPT_INTERVAL="$2"; shift 2 ;;
        --distributed_backend) DISTRIBUTED_BACKEND="$2"; shift 2 ;;
        --no_auto_resume)   AUTO_RESUME=0; shift ;;
        --iterations_to_run) ITERATIONS_TO_RUN="$2"; shift 2 ;;
        --z_loss_coeff)     Z_LOSS_COEFF="$2"; shift 2 ;;
        *)                  EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Validate required args
if [ -z "$ARCH" ] || [ -z "$OPT" ] || [ -z "$HEADS" ]; then
    echo "Usage: bash scripts/launch.sh --arch {enoki,qwen3} --opt <optimizer> --heads <N> [options]"
    echo ""
    echo "Required:"
    echo "  --arch {enoki,qwen3}    Model architecture"
    echo "  --opt <optimizer>       Optimizer name (adana, dana-mk4, dana-star, dana-star-mk4, adamw, ...)"
    echo "  --heads <N>             Number of attention heads"
    echo ""
    echo "Optional:"
    echo "  --wandb_group <name>    WandB run group"
    echo "  --nproc <N>             GPUs per node (default: 1)"
    echo "  --batch_size <N>        Batch size per GPU (default: 32)"
    echo "  --acc_steps <N>         Gradient accumulation steps (default: 1)"
    echo "  --lr <float>            Override auto-computed learning rate"
    echo "  --kappa <float>         Kappa for DANA variants (default: 0.85)"
    echo "  --clipsnr <float>       SNR clipping for MK4 variants (default: 2.0)"
    echo "  --omega <float>         Weight decay omega (default: 4.0)"
    echo "  --wd_ts_divisor <N>     WD_TS = ITERATIONS/N (default: 10)"
    echo "  --wd_ts_const <N>       WD_TS = N (raw constant, mutually exclusive with divisor)"
    echo "  --iterations <N>        Override auto-computed iterations"
    echo "  --iterations_to_run <N> Max iterations per SLURM job (for restart)"
    echo "  --no_compile            Disable torch.compile"
    echo "  --no_wandb              Disable wandb logging"
    exit 1
fi

# =============================================================================
# Compute model dimensions and LR via Python scaling rules
# =============================================================================
# Compute global batch size for LR formula selection
# NOTE: batch_size * acc_steps is the global effective batch; the distributed
# backend divides this across GPUs internally (see get_adjusted_args_for_process).
GLOBAL_BATCH=$((BATCH_SIZE * ACC_STEPS))

SCALING_OUTPUT=$(python3 -c "
import importlib.util, os
spec = importlib.util.spec_from_file_location('scaling', os.path.join('$(pwd)', 'src', 'config', 'scaling.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

dims = mod.compute_dimensions('$ARCH', $HEADS)
lr = mod.compute_lr('$ARCH', '$OPT', dims['non_emb_params'], kappa=$KAPPA, batch_size=$GLOBAL_BATCH)

print(f\"N_HEAD={dims['n_head']}\")
print(f\"N_LAYER={dims['n_layer']}\")
print(f\"N_EMBD={dims['n_embd']}\")
print(f\"MLP_HIDDEN={dims['mlp_hidden_dim']}\")
print(f\"QKV_DIM={dims['head_dim']}\")
print(f\"NON_EMB={dims['non_emb_params']}\")
print(f\"TOTAL_PARAMS={dims['total_params']}\")
if lr is not None:
    print(f\"LR={lr}\")
")
eval "$SCALING_OUTPUT"

# Use LR override if provided, otherwise use computed LR
if [ -n "$LR_OVERRIDE" ]; then
    LR="$LR_OVERRIDE"
elif [ -z "${LR:-}" ]; then
    echo "ERROR: No LR formula found for optimizer '$OPT' in scaling rules."
    echo "Please provide --lr explicitly."
    exit 1
fi

# Compute iterations (Chinchilla-optimal: tokens = 20 * total_params)
TOKENS_PER_STEP=$((BATCH_SIZE * ACC_STEPS * SEQ_LEN))
if [ -n "$ITERATIONS_OVERRIDE" ]; then
    ITERATIONS="$ITERATIONS_OVERRIDE"
else
    ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / $TOKENS_PER_STEP))")
fi

# Compute warmup steps (default: 2% of iterations = iterations / 50)
if [ -z "$WARMUP_STEPS" ]; then
    WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
fi

# =============================================================================
# Compute WD_TS from divisor or constant (mutually exclusive)
# =============================================================================
# Check for mutual exclusion: error if user explicitly set both
if [ -n "$WD_TS_CONST" ] && [ -n "$WD_TS_DIVISOR_SET" ]; then
    echo "ERROR: Cannot specify both --wd_ts_divisor and --wd_ts_const"
    exit 1
fi

if [ -n "$WD_TS_CONST" ]; then
    WD_TS="$WD_TS_CONST"
else
    # Use divisor (default to 10 if not set)
    WD_TS_DIVISOR="${WD_TS_DIVISOR:-10}"
    WD_TS=$(python3 -c "print(int($ITERATIONS / $WD_TS_DIVISOR))")
fi

# =============================================================================
# Set optimizer-specific flags and weight decay
# =============================================================================
# Weight decay convention:
#   - DANA variants & decaying-WD (independent WD, paper convention):
#     WD_TS = ITERATIONS/N (default N=10) or constant, WD = OMEGA / WD_TS
#     The optimizer multiplies by schedule γ(t) but NOT by peak LR γ*
#   - AdamW/Ademamix (PyTorch convention, coupled WD):
#     WD = OMEGA / (LR * ITERATIONS)
#     PyTorch multiplies by lr internally, so we divide by LR to compensate
OPT_FLAGS=""
case "$OPT" in
    adana|dana-mk4|dana-star|dana-star-mk4)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / $WD_TS)")
        case "$OPT" in
            adana)
                OPT_FLAGS="--opt adana --delta $DELTA --kappa $KAPPA --gamma_3_factor $GAMMA_3_FACTOR --wd_decaying --wd_ts $WD_TS"
                ;;
            dana-mk4)
                OPT_FLAGS="--opt dana-mk4 --delta $DELTA --kappa $KAPPA --clipsnr $CLIPSNR --gamma_3_factor $GAMMA_3_FACTOR --wd_decaying --wd_ts $WD_TS"
                ;;
            dana-star)
                OPT_FLAGS="--opt dana-star --delta $DELTA --kappa $KAPPA --gamma_3_factor $GAMMA_3_FACTOR --wd_decaying --wd_ts $WD_TS"
                ;;
            dana-star-mk4)
                OPT_FLAGS="--opt dana-star-mk4 --delta $DELTA --kappa $KAPPA --clipsnr $CLIPSNR --gamma_3_factor $GAMMA_3_FACTOR --wd_decaying --wd_ts $WD_TS"
                ;;
        esac
        ;;
    adamw)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        OPT_FLAGS="--opt adamw --beta1 0.9 --beta2 0.999"
        ;;
    ademamix)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        OPT_FLAGS="--opt ademamix --beta1 0.9 --beta2 0.999 --delta $DELTA --kappa $KAPPA --gamma_3_factor $GAMMA_3_FACTOR --adema_beta3_warmup $ITERATIONS --adema_alpha_warmup $ITERATIONS"
        ;;
    adamw-decaying-wd)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / $WD_TS)")
        OPT_FLAGS="--opt adamw-decaying-wd --beta1 0.9 --beta2 0.999 --wd_decaying --wd_ts $WD_TS"
        ;;
    ademamix-decaying-wd)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / $WD_TS)")
        OPT_FLAGS="--opt ademamix-decaying-wd --beta1 0.9 --beta2 0.999 --delta $DELTA --kappa $KAPPA --gamma_3_factor $GAMMA_3_FACTOR --adema_beta3_warmup $ITERATIONS --adema_alpha_warmup $ITERATIONS --wd_decaying --wd_ts $WD_TS"
        ;;
    d-muon)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        OPT_FLAGS="--opt d-muon --beta1 0.8 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5"
        ;;
    manau)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        WD_TS=$(python3 -c "print(int($ITERATIONS))")
        OPT_FLAGS="--opt manau --delta $DELTA --kappa $KAPPA --clipsnr $CLIPSNR --momentum 0.95 --nesterov True --muon_ns_steps 5 --matched_adamw_rms 0.2 --dana_momentum False --mk4A 0.0 --mk4B 0.0 --wd_decaying --wd_ts $WD_TS"
        ;;
    *)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        OPT_FLAGS="--opt $OPT"
        ;;
esac

# =============================================================================
# Set architecture-specific flags
# =============================================================================
ARCH_FLAGS=""
case "$ARCH" in
    enoki)
        MODEL_NAME="enoki"
        ARCH_FLAGS="--weight_tying False"
        ;;
    qwen3)
        MODEL_NAME="qwen3"
        ARCH_FLAGS="--weight_tying False --elementwise_attn_output_gate --normalization_layer_type rmsnorm"
        ;;
    *)
        echo "ERROR: Unknown architecture '$ARCH'. Use 'enoki' or 'qwen3'."
        exit 1
        ;;
esac

# Build compile flag
COMPILE_FLAG=""
if [ "$COMPILE" -eq 1 ]; then
    COMPILE_FLAG="--compile"
fi

# Build wandb flags
WANDB_FLAGS=""
if [ "$WANDB_ENABLED" -eq 1 ]; then
    WANDB_FLAGS="--wandb --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_ENTITY" ]; then
        WANDB_FLAGS="$WANDB_FLAGS --wandb_entity $WANDB_ENTITY"
    fi
fi

# Build auto-resume flag
RESUME_FLAGS=""
if [ "$AUTO_RESUME" -eq 1 ]; then
    RESUME_FLAGS="--auto_resume"
fi

# Build iterations_to_run flag
ITR_FLAGS=""
if [ -n "$ITERATIONS_TO_RUN" ]; then
    ITR_FLAGS="--iterations_to_run $ITERATIONS_TO_RUN"
fi

# Build wandb group and run prefix flags
PREFIX_FLAGS=""
if [ -n "$WANDB_GROUP" ]; then
    PREFIX_FLAGS="--wandb_group $WANDB_GROUP --run_prefix $WANDB_GROUP"
fi

# Build scheduler flags
SCHEDULER_FLAGS="--scheduler $SCHEDULER"
if [ "$SCHEDULER" = "cos_inf" ]; then
    SCHEDULER_FLAGS="$SCHEDULER_FLAGS --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1"
fi

# =============================================================================
# Print configuration
# =============================================================================
echo "============================================================"
echo "DSTAR Training Launch"
echo "============================================================"
echo "Architecture: $ARCH ($MODEL_NAME)"
echo "Optimizer:    $OPT"
echo "Heads:        $HEADS"
echo "Dimensions:   n_embd=$N_EMBD, n_head=$N_HEAD, n_layer=$N_LAYER"
echo "MLP hidden:   $MLP_HIDDEN"
echo "QKV dim:      $QKV_DIM"
echo "Parameters:   non_emb=${NON_EMB} ($(python3 -c "print(f'{$NON_EMB/1e6:.1f}M')")), total=${TOTAL_PARAMS} ($(python3 -c "print(f'{$TOTAL_PARAMS/1e6:.1f}M')"))"
echo "LR:           $LR"
if [ -n "$WD_TS_CONST" ]; then
    echo "Weight Decay: $WEIGHT_DECAY (omega=$OMEGA, wd_ts_const=$WD_TS_CONST)"
else
    echo "Weight Decay: $WEIGHT_DECAY (omega=$OMEGA, wd_ts_divisor=$WD_TS_DIVISOR, wd_ts=$WD_TS)"
fi
echo "Iterations:   $ITERATIONS (from 20 * total_params / tokens_per_step)"
echo "Warmup:       $WARMUP_STEPS (iterations/50)"
echo "Scheduler:    $SCHEDULER"
echo "Grad clip:    $GRAD_CLIP"
echo "Batch:        ${BATCH_SIZE}x${ACC_STEPS}x${SEQ_LEN} (${NPROC_PER_NODE} GPUs, global=${BATCH_SIZE}*${ACC_STEPS}=$((BATCH_SIZE * ACC_STEPS)))"
echo "Dataset:      $DATASET"
echo "Backend:      $DISTRIBUTED_BACKEND"
echo "============================================================"

# =============================================================================
# Launch training
# =============================================================================

# Set wandb offline mode (must be env var so torchrun children inherit it)
if [ "$WANDB_OFFLINE" -eq 1 ]; then
    export WANDB_MODE=offline
fi

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE ./src/main.py \
    --config_format base \
    --model $MODEL_NAME \
    --distributed_backend $DISTRIBUTED_BACKEND \
    $COMPILE_FLAG \
    --n_embd $N_EMBD --n_head $N_HEAD --n_layer $N_LAYER \
    --qkv_dim $QKV_DIM --mlp_hidden_dim $MLP_HIDDEN \
    --batch_size $BATCH_SIZE --sequence_length $SEQ_LEN --acc_steps $ACC_STEPS \
    --datasets_dir $DATASETS_DIR \
    --dataset $DATASET \
    --iterations $ITERATIONS \
    --lr $LR --weight_decay $WEIGHT_DECAY \
    $SCHEDULER_FLAGS \
    --warmup_steps $WARMUP_STEPS \
    --grad_clip $GRAD_CLIP \
    --init-scheme $INIT_SCHEME \
    --dropout 0.0 --seed 0 \
    --z_loss_coeff $Z_LOSS_COEFF \
    --eval_interval $EVAL_INTERVAL \
    --log_interval $LOG_INTERVAL \
    --latest_ckpt_interval $LATEST_CKPT_INTERVAL \
    --permanent_ckpt_interval $PERMANENT_CKPT_INTERVAL \
    --results_base_folder $RESULTS_BASE_FOLDER \
    $ARCH_FLAGS \
    $OPT_FLAGS \
    $WANDB_FLAGS \
    $RESUME_FLAGS \
    $ITR_FLAGS \
    $PREFIX_FLAGS \
    "${EXTRA_ARGS[@]}"
