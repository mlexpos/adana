#!/bin/bash

# Generic Qwen3Next restart script - MoE with expert parallelism
# This script contains all the architecture and training logic
# User-specific settings (account, API keys, etc.) should be set in wrapper scripts

# Default values
LR=15e-4
OMEGA=4.0
CLIPSNR=2.0
BATCH_SIZE=32
ACC_STEPS=1
NPROC_PER_NODE=4  # Default to 4 GPUs for expert parallelism
HEADS=""
OPTIMIZER="dana-star-mk4"
INIT_SCHEME="KarpathyGPT2"
DEPTH_SCALAR_EXPONENT=0.0
ITERATIONS_TO_RUN=none
RESULTS_BASE_FOLDER="./exps"
RESTART_COUNT=${RESTART_COUNT:-0}
RESTART_WRAPPER_SCRIPT=${RESTART_WRAPPER_SCRIPT:-""}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Qwen3Next MoE configuration (fixed)
MOE_NUM_EXPERTS=64  # Always 64 experts
MOE_NUM_EXPERTS_PER_TOK=4  # Always 4 active per token
USE_GATED_SHARED_EXPERT="--use_gated_shared_expert"  # Gated shared expert enabled
SHARED_EXPERT_SIZE=4096  # Smaller shared expert to save memory
DECODER_SPARSE_STEP=1  # All layers use MoE

# Parse command line arguments
ORIG_ARGS=( "$@" )
while [[ $# -gt 0 ]]; do
    case $1 in
        --heads)
            HEADS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --omega)
            OMEGA="$2"
            shift 2
            ;;
        --clipsnr)
            CLIPSNR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --acc_steps)
            ACC_STEPS="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --init-scheme)
            INIT_SCHEME="$2"
            shift 2
            ;;
        --depth-scalar-exponent)
            DEPTH_SCALAR_EXPONENT="$2"
            shift 2
            ;;
        --iterations_to_run)
            ITERATIONS_TO_RUN="$2"
            shift 2
            ;;
        --results_base_folder)
            RESULTS_BASE_FOLDER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$HEADS" ]; then
    echo "Error: --heads argument is required"
    echo "Usage: ./Qwen3Next_generic_restart.sh --heads <heads> [options]"
    echo ""
    echo "Qwen3Next uses the following scaling (based on Qwen3):"
    echo "  - head_dim = 128 (fixed)"
    echo "  - n_head = heads"
    echo "  - n_layer = 2 * heads"
    echo "  - n_embd = heads * 128"
    echo "  - mlp_hidden = 3 * n_embd"
    echo "  - MoE: 64 experts, 4 active per token (6.25% sparsity)"
    echo "  - Gated shared expert enabled (learned sigmoid gating)"
    echo ""
    echo "Options:"
    echo "  --lr <value>              Learning rate (default: 15e-4)"
    echo "  --omega <value>           Weight decay strength parameter (default: 4.0)"
    echo "  --clipsnr <value>         Clip SNR for dana-star-mk4 (default: 2.0)"
    echo "  --batch_size <value>      Batch size (default: 32)"
    echo "  --acc_steps <value>       Accumulation steps (default: 1)"
    echo "  --nproc_per_node <value>  Processes per node (default: 4 for multi-GPU)"
    echo "  --optimizer <type>        Optimizer type (default: dana-star-mk4)"
    echo "  --init-scheme <type>      Initialization scheme (default: KarpathyGPT2)"
    echo "  --depth-scalar-exponent <value>  Depth-based residual scaling exponent (default: 0.0)"
    echo "  --iterations_to_run <value>  Iterations to run (default: none)"
    echo "  --results_base_folder <path>  Base folder for checkpoints (default: ./exps)"
    exit 1
fi

if [ "$HEADS" -lt 4 ]; then
    echo "Error: heads must be at least 4"
    exit 1
fi

# Validate that num_experts is divisible by nproc_per_node
if [ $((MOE_NUM_EXPERTS % NPROC_PER_NODE)) -ne 0 ]; then
    echo "Error: MOE_NUM_EXPERTS ($MOE_NUM_EXPERTS) must be divisible by nproc_per_node ($NPROC_PER_NODE)"
    echo "With 64 experts, valid GPU counts are: 1, 2, 4, 8, 16, 32, 64"
    echo "Recommended: 4 GPUs (16 experts per GPU)"
    exit 1
fi

# Calculate model parameters based on Qwen3 scaling
HEAD_DIM=128
N_HEAD=$(python3 -c "print(int($HEADS))")
N_LAYER=$(python3 -c "print(int(2 * $HEADS))")
N_EMBD=$(python3 -c "print(int($HEADS * 128))")
MLP_HIDDEN=$(python3 -c "print(int(3 * $N_EMBD))")

# Calculate residual stream scalar based on depth and exponent
RESIDUAL_STREAM_SCALAR=$(python3 -c "print($N_LAYER ** $DEPTH_SCALAR_EXPONENT)")

# Calculate non-embedding parameters for Qwen3Next with MoE
# Base Qwen3 non_emb + MoE expansion
TOTAL_QKV_DIM=$(python3 -c "print(int($N_HEAD * $HEAD_DIM))")

# Calculate base non-emb (attention + layer norms)
BASE_NON_EMB=$(python3 -c "
n_layer = $N_LAYER
n_embd = $N_EMBD
head_dim = $HEAD_DIM
total_qkv_dim = $TOTAL_QKV_DIM

# Per layer (without MLP)
attn = 5 * n_embd * total_qkv_dim  # q_proj (2x with gating) + k_proj + v_proj + o_proj
qk_norm = 2 * head_dim
layer_norms = 2 * n_embd

per_layer = attn + qk_norm + layer_norms

# Total
base_non_emb = n_layer * per_layer + n_embd  # +n_embd for final norm

print(int(base_non_emb))
")

# Calculate MoE expansion
# Router: n_embd * num_experts per layer
# Routed experts: 63 experts (64 - 1 shared) × 3 × n_embd × mlp_hidden per layer
# Gated shared expert: 1 × 3 × n_embd × shared_expert_size + gate projection per layer
MOE_NON_EMB=$(python3 -c "
n_layer = $N_LAYER
n_embd = $N_EMBD
mlp_hidden = $MLP_HIDDEN
num_routed_experts = $MOE_NUM_EXPERTS - 1  # 63 routed experts
shared_expert_size = $SHARED_EXPERT_SIZE

# Per MoE layer
router = n_embd * $MOE_NUM_EXPERTS
routed_experts_mlp = num_routed_experts * 3 * n_embd * mlp_hidden  # gate_proj + up_proj + down_proj
shared_expert_mlp = 3 * n_embd * shared_expert_size  # smaller shared expert
shared_expert_gate = n_embd  # gate projection

per_moe_layer = router + routed_experts_mlp + shared_expert_mlp + shared_expert_gate

# Total (all layers use MoE)
moe_expansion = n_layer * per_moe_layer

print(int(moe_expansion))
")

NON_EMB=$(python3 -c "print(int($BASE_NON_EMB + $MOE_NON_EMB))")
TOTAL_PARAMS=$(python3 -c "print(int($NON_EMB + 2 * $N_EMBD * 50304))")
ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / 65536))")

# Configure optimizer-specific parameters and compute weight decay from OMEGA
case $OPTIMIZER in
    dana-star-mk4)
        WD_TS=$(python3 -c "print(int($ITERATIONS / 10))")
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $WD_TS))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt dana-star-mk4 --lr $LR --delta 8 --kappa 0.75 --clipsnr $CLIPSNR --weight_decay $WEIGHT_DECAY --wd_decaying --wd_ts $WD_TS"
        ;;
    adamw)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        WD_TS="N/A"
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt adamw --lr $LR --weight_decay $WEIGHT_DECAY --beta1 0.9 --beta2 0.999"
        ;;
    dana)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        WD_TS="N/A"
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt dana --lr $LR --delta 8 --kappa 0.75 --weight_decay $WEIGHT_DECAY --beta1 0.9 --use_v_ema --v_ema_beta 0.999"
        ;;
    ademamix)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        WD_TS="N/A"
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt ademamix --lr $LR --weight_decay $WEIGHT_DECAY --beta1 0.9 --beta2 0.999 --delta 8 --kappa 0.75 --gamma_3_factor 1.0 --adema_beta3_warmup $ITERATIONS --adema_alpha_warmup $ITERATIONS"
        ;;
    d-muon)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        WD_TS="N/A"
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt d-muon --lr $LR --weight_decay $WEIGHT_DECAY --beta1 0.8 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5"
        ;;
    manau)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        WD_TS=$(python3 -c "print(int($ITERATIONS / 1))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt manau --lr $LR --weight_decay $WEIGHT_DECAY --momentum 0.95 --nesterov True --muon_ns_steps 5 --matched_adamw_rms 0.2 --dana_momentum False --delta 8 --kappa 0.75 --mk4A 0.0 --mk4B 0.0 --clipsnr $CLIPSNR --wd_decaying --wd_ts $WD_TS"
        ;;
    manau-hard)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        WD_TS=$(python3 -c "print(int($ITERATIONS / 1))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt manau --lr $LR --weight_decay $WEIGHT_DECAY --momentum 0.95 --nesterov True --muon_ns_steps 5 --matched_adamw_rms 0.2 --dana_momentum True --delta 8 --kappa 0.75 --mk4A 0.0 --mk4B 0.0 --clipsnr $CLIPSNR --wd_decaying --wd_ts $WD_TS"
        ;;
    adamw-decaying-wd)
        WD_TS=$(python3 -c "print(int($ITERATIONS / 10))")
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $WD_TS))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt adamw-decaying-wd --lr $LR --weight_decay $WEIGHT_DECAY --beta1 0.9 --beta2 0.999 --wd_decaying --wd_ts $WD_TS"
        ;;
    ademamix-decaying-wd)
        WD_TS=$(python3 -c "print(int($ITERATIONS / 10))")
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $WD_TS))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt ademamix-decaying-wd --lr $LR --weight_decay $WEIGHT_DECAY --beta1 0.9 --beta2 0.999 --delta 8 --kappa 0.75 --gamma_3_factor 1.0 --adema_beta3_warmup $ITERATIONS --adema_alpha_warmup $ITERATIONS --wd_decaying --wd_ts $WD_TS"
        ;;
    dana-star-no-tau)
        WD_TS=$(python3 -c "print(int($ITERATIONS / 10))")
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $WD_TS))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt dana-star-no-tau --lr $LR --delta 8 --kappa 0.75 --clipsnr $CLIPSNR --weight_decay $WEIGHT_DECAY --wd_decaying --wd_ts $WD_TS"
        ;;
    dana-star)
        WD_TS=$(python3 -c "print(int($ITERATIONS / 10))")
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $WD_TS))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt dana-star --lr $LR --delta 8 --kappa 0.75 --clipsnr $CLIPSNR --weight_decay $WEIGHT_DECAY --wd_decaying --wd_ts $WD_TS"
        ;;
    dana-mk4)
        WD_TS=$(python3 -c "print(int($ITERATIONS / 10))")
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $WD_TS))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt dana-mk4 --lr $LR --delta 8 --kappa 0.75 --clipsnr $CLIPSNR --weight_decay $WEIGHT_DECAY --wd_decaying --wd_ts $WD_TS"
        ;;
    *)
        echo "Error: Unknown optimizer $OPTIMIZER"
        echo "Available optimizers: dana-star-mk4, adamw, dana, ademamix, d-muon, manau, manau-hard, adamw-decaying-wd, ademamix-decaying-wd, dana-star-no-tau, dana-star, dana-mk4"
        exit 1
        ;;
esac

# Conditionally add iterations_to_run and auto-resume flags if provided
if [ "$ITERATIONS_TO_RUN" != "none" ]; then
    EXTRA_RUN_FLAGS="--iterations_to_run $ITERATIONS_TO_RUN --latest_ckpt_interval 10000 --auto_resume"
else
    EXTRA_RUN_FLAGS=""
fi

echo "=== Qwen3Next Configuration: $HEADS heads with MoE ==="
echo "n_layer: $N_LAYER (= 2 * $HEADS)"
echo "n_head: $N_HEAD"
echo "qkv_dim (head_dim): $HEAD_DIM (fixed)"
echo "n_embd: $N_EMBD (= $HEADS * 128)"
echo "mlp_hidden_dim: $MLP_HIDDEN (= 3 * $N_EMBD)"
echo "Elementwise gating: enabled"
echo ""
echo "MoE Configuration:"
echo "  Total experts: $MOE_NUM_EXPERTS"
echo "  Active per token: $MOE_NUM_EXPERTS_PER_TOK ($(python3 -c "print(100 * $MOE_NUM_EXPERTS_PER_TOK / $MOE_NUM_EXPERTS)")% sparsity)"
echo "  Routed experts: $((MOE_NUM_EXPERTS - 1))"
echo "  Gated shared expert: 1 (size: $SHARED_EXPERT_SIZE)"
echo "  All layers use MoE: yes"
echo "  Expert parallelism: $NPROC_PER_NODE GPUs ($((MOE_NUM_EXPERTS / NPROC_PER_NODE)) experts/GPU)"
echo ""
echo "Total parameters: $TOTAL_PARAMS"
echo "Iterations: $ITERATIONS"
echo "Iterations to run: $ITERATIONS_TO_RUN"
echo "Learning rate: $LR"
echo "Omega: $OMEGA"
echo "Weight decay: $WEIGHT_DECAY"
echo "Weight decay timestep: $WD_TS"
echo "Clip SNR: $CLIPSNR"
echo "Batch size: $BATCH_SIZE"
echo "Accumulation steps: $ACC_STEPS"
echo "Processes per node: $NPROC_PER_NODE"
echo "Optimizer: $OPTIMIZER"
echo "Init scheme: $INIT_SCHEME"
echo "Depth scalar exponent: $DEPTH_SCALAR_EXPONENT"
echo "Residual stream scalar: $RESIDUAL_STREAM_SCALAR"
echo "Results base folder: $RESULTS_BASE_FOLDER"
echo "=========================================="

EVAL_INTERVAL=$(python3 -c "print(115)")

# Run training with Qwen3Next and expert parallelism
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE ./src/main.py --config_format base --model qwen3next \
        --distributed_backend nccl --compile \
        --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
        --n_embd $N_EMBD --qkv_dim $HEAD_DIM --n_head $N_HEAD --n_layer $N_LAYER \
        --mlp_hidden_dim $MLP_HIDDEN \
        --moe --moe_num_experts $MOE_NUM_EXPERTS --moe_num_experts_per_tok $MOE_NUM_EXPERTS_PER_TOK \
        $USE_GATED_SHARED_EXPERT --shared_expert_intermediate_size $SHARED_EXPERT_SIZE \
        --decoder_sparse_step $DECODER_SPARSE_STEP \
        --expert_parallel \
        --batch_size $BATCH_SIZE --sequence_length 2048 --acc_steps $ACC_STEPS \
        --iterations $ITERATIONS \
        --dropout 0.0 --warmup_steps $WARMUP_STEPS --grad_clip 0.5 --seed 0 \
        --init-scheme $INIT_SCHEME --residual-stream-scalar $RESIDUAL_STREAM_SCALAR \
        --z_loss_coeff 0 \
        --normalization_layer_type rmsnorm \
        --weight_tying False \
        --elementwise_attn_output_gate \
        $OPT_PARAMS $EXTRA_RUN_FLAGS \
        --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
        --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
        --results_base_folder "$RESULTS_BASE_FOLDER" \
        --eval_interval $EVAL_INTERVAL --log_interval 50

# Capture the exit code
TRAINING_EXIT_CODE=$?

# Restart logic (same as Qwen3)
if [ "$ITERATIONS_TO_RUN" != "none" ] && [[ "$ITERATIONS_TO_RUN" =~ ^[0-9]+$ ]]; then
    MAX_RESTARTS=$(python3 -c "print(int($ITERATIONS // $ITERATIONS_TO_RUN))")

    # Only increment restart count if training succeeded
    if [ $TRAINING_EXIT_CODE -eq 0 ]; then
        if [[ $RESTART_COUNT -lt $MAX_RESTARTS ]]; then
            NEW_RESTART_COUNT=$((RESTART_COUNT + 1))
            echo "Training completed successfully, requeuing for next chunk (restart $NEW_RESTART_COUNT / $MAX_RESTARTS)"

        # Check if a restart wrapper script was specified
        if [ -z "$RESTART_WRAPPER_SCRIPT" ]; then
            echo "Error: RESTART_WRAPPER_SCRIPT environment variable not set"
            echo "Cannot requeue job without knowing which wrapper script to use"
            exit 1
        fi

        # Extract original SLURM args to preserve resources
        scontext=$(scontrol show job $SLURM_JOB_ID 2>/dev/null || echo "")
        account=$(echo "$scontext" | grep -o 'Account=[^ ]*' | cut -d= -f2)
        nodes=$(echo "$scontext" | grep -o 'NumNodes=[^ ]*' | cut -d= -f2); nodes=${nodes:-"1"}
        timelimit=$(echo "$scontext" | grep -o 'TimeLimit=[^ ]*' | cut -d= -f2)
        gres_raw=$(echo "$scontext" | grep -o 'Gres=[^ ]*' | cut -d= -f2)
        if [ -z "$gres_raw" ]; then
            if [ -n "$SLURM_GPUS_PER_NODE" ]; then
                gpus_per_node="$SLURM_GPUS_PER_NODE"
            else
                gpus_per_node="a100:${NPROC_PER_NODE}"
            fi
        else
            gpus_per_node=$(echo "$gres_raw" | sed 's/^gpu://')
        fi
        mem=$(echo "$scontext" | grep -o 'MinMemoryNode=[^ ]*' | cut -d= -f2); mem=${mem:-"0"}
        job_name=$(echo "$scontext" | grep -o 'JobName=[^ ]*' | cut -d= -f2); job_name=${job_name:-""}

        SLURM_ARGS=(
            --time="$timelimit"
            --nodes="$nodes"
            --gpus-per-node="$gpus_per_node"
            --cpus-per-gpu="${SLURM_CPUS_PER_GPU:-8}"
            --mem="$mem"
            --job-name="$job_name"
        )

        # Only add account if it was extracted successfully
        if [ -n "$account" ]; then
            SLURM_ARGS+=(--account="$account")
        fi

            echo "Using original SLURM args for requeue: ${SLURM_ARGS[@]}"
            sbatch --export=ALL,RESTART_COUNT=$NEW_RESTART_COUNT "${SLURM_ARGS[@]}" "$RESTART_WRAPPER_SCRIPT" "${ORIG_ARGS[@]}"
        else
            echo "Max restarts ($MAX_RESTARTS) reached, not requeuing"
        fi
    else
        # Training failed, requeue without incrementing restart count
        echo "Training failed with exit code $TRAINING_EXIT_CODE"
        echo "Requeuing job without incrementing restart count (keeping RESTART_COUNT=$RESTART_COUNT)"

        # Check if a restart wrapper script was specified
        if [ -z "$RESTART_WRAPPER_SCRIPT" ]; then
            echo "Error: RESTART_WRAPPER_SCRIPT environment variable not set"
            echo "Cannot requeue job without knowing which wrapper script to use"
            exit 1
        fi

        # Extract original SLURM args to preserve resources
        scontext=$(scontrol show job $SLURM_JOB_ID 2>/dev/null || echo "")
        account=$(echo "$scontext" | grep -o 'Account=[^ ]*' | cut -d= -f2)
        nodes=$(echo "$scontext" | grep -o 'NumNodes=[^ ]*' | cut -d= -f2); nodes=${nodes:-"1"}
        timelimit=$(echo "$scontext" | grep -o 'TimeLimit=[^ ]*' | cut -d= -f2)
        gres_raw=$(echo "$scontext" | grep -o 'Gres=[^ ]*' | cut -d= -f2)
        if [ -z "$gres_raw" ]; then
            if [ -n "$SLURM_GPUS_PER_NODE" ]; then
                gpus_per_node="$SLURM_GPUS_PER_NODE"
            else
                gpus_per_node="a100:${NPROC_PER_NODE}"
            fi
        else
            gpus_per_node=$(echo "$gres_raw" | sed 's/^gpu://')
        fi
        mem=$(echo "$scontext" | grep -o 'MinMemoryNode=[^ ]*' | cut -d= -f2); mem=${mem:-"0"}
        job_name=$(echo "$scontext" | grep -o 'JobName=[^ ]*' | cut -d= -f2); job_name=${job_name:-""}

        SLURM_ARGS=(
            --time="$timelimit"
            --nodes="$nodes"
            --gpus-per-node="$gpus_per_node"
            --cpus-per-gpu="${SLURM_CPUS_PER_GPU:-8}"
            --mem="$mem"
            --job-name="$job_name"
        )

        # Only add account if it was extracted successfully
        if [ -n "$account" ]; then
            SLURM_ARGS+=(--account="$account")
        fi

        echo "Using original SLURM args for requeue: ${SLURM_ARGS[@]}"
        sbatch --export=ALL,RESTART_COUNT=$RESTART_COUNT "${SLURM_ARGS[@]}" "$RESTART_WRAPPER_SCRIPT" "${ORIG_ARGS[@]}"
    fi
fi
