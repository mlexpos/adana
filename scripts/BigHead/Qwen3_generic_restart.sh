#!/bin/bash

# Generic Qwen3 restart script - can be used by all users
# This script contains all the architecture and training logic
# User-specific settings (account, API keys, etc.) should be set in wrapper scripts

# Default values
LR=15e-4
OMEGA=4.0
CLIPSNR=2.0
BATCH_SIZE=32
ACC_STEPS=1
NPROC_PER_NODE=1
HEADS=""
OPTIMIZER="dana-star-mk4"
INIT_SCHEME="KarpathyGPT2"
DEPTH_SCALAR_EXPONENT=0.0
ITERATIONS_TO_RUN=none
RESULTS_BASE_FOLDER="./exps"
RESTART_COUNT=${RESTART_COUNT:-0}
RESTART_WRAPPER_SCRIPT=${RESTART_WRAPPER_SCRIPT:-""}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    echo "Usage: ./Qwen3_generic_restart.sh --heads <heads> [options]"
    echo ""
    echo "Qwen3 uses the following scaling (from Qwen3 paper):"
    echo "  - head_dim = 128 (fixed)"
    echo "  - n_head = heads"
    echo "  - n_layer = 2 * heads"
    echo "  - n_embd = heads * 128"
    echo "  - mlp_hidden = 3 * n_embd"
    echo ""
    echo "Options:"
    echo "  --lr <value>              Learning rate (default: 15e-4)"
    echo "  --omega <value>           Weight decay strength parameter (default: 4.0)"
    echo "  --clipsnr <value>         Clip SNR for dana-star-mk4 (default: 2.0)"
    echo "  --batch_size <value>      Batch size (default: 32)"
    echo "  --acc_steps <value>       Accumulation steps (default: 1)"
    echo "  --nproc_per_node <value>  Processes per node (default: 1)"
    echo "  --optimizer <type>        Optimizer type: dana-star-mk4, adamw, dana, ademamix, d-muon, manau, manau-hard, adamw-decaying-wd, ademamix-decaying-wd, dana-star-no-tau, dana-star, dana-mk4 (default: dana-star-mk4)"
    echo "  --init-scheme <type>      Initialization scheme: KarpathyGPT2, Standard, ScaledGPT (default: KarpathyGPT2)"
    echo "  --depth-scalar-exponent <value>  Exponent for depth-based residual scaling: scalar = n_layer^exp (default: 0.0)"
    echo "  --iterations_to_run <value>  Iterations to run (default: none)"
    echo "  --results_base_folder <path>  Base folder for checkpoints (default: ./exps)"
    exit 1
fi

if [ "$HEADS" -lt 4 ]; then
    echo "Error: heads must be at least 4"
    exit 1
fi

# Check that heads is divisible by 4 for clean n_layer calculation
if [ $((HEADS % 4)) -ne 0 ]; then
    echo "Warning: heads=$HEADS is not divisible by 4. n_layer will be rounded down."
fi

# Calculate model parameters based on Qwen3 scaling
# Formula: head_dim = 128 (fixed), n_layer = 2*heads, n_embd = heads * 128, mlp = 3 * n_embd
HEAD_DIM=128
N_HEAD=$(python3 -c "print(int($HEADS))")
N_LAYER=$(python3 -c "print(int(2 * $HEADS))")
N_EMBD=$(python3 -c "print(int($HEADS * 128))")
MLP_HIDDEN=$(python3 -c "print(int(3 * $N_EMBD))")

# Calculate residual stream scalar based on depth and exponent
RESIDUAL_STREAM_SCALAR=$(python3 -c "print($N_LAYER ** $DEPTH_SCALAR_EXPONENT)")

# Calculate non-embedding parameters for Qwen3
# With gating: non_emb = n_layer * (5 * n_embd * total_qkv_dim + 2 * head_dim + 9 * n_embd^2 + 2 * n_embd) + n_embd
TOTAL_QKV_DIM=$(python3 -c "print(int($N_HEAD * $HEAD_DIM))")
NON_EMB=$(python3 -c "
n_layer = $N_LAYER
n_embd = $N_EMBD
head_dim = $HEAD_DIM
total_qkv_dim = $TOTAL_QKV_DIM

# Per layer
attn = 5 * n_embd * total_qkv_dim  # q_proj (2x with gating) + k_proj + v_proj + o_proj
qk_norm = 2 * head_dim
mlp = 9 * n_embd * n_embd  # gate_proj + up_proj + down_proj (mlp_hidden = 3 * n_embd)
layer_norms = 2 * n_embd

per_layer = attn + qk_norm + mlp + layer_norms

# Total
non_emb = n_layer * per_layer + n_embd  # +n_embd for final norm

print(int(non_emb))
")
TOTAL_PARAMS=$(python3 -c "print(int($NON_EMB + 2 * $N_EMBD * 50304))")
ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / 65536))")

# Configure optimizer-specific parameters and compute weight decay from OMEGA
case $OPTIMIZER in
    dana-star-mk4)
        # For dana-star-mk4: WD_TS = ITERATIONS/10, WEIGHT_DECAY = OMEGA / (LR * WD_TS)
        WD_TS=$(python3 -c "print(int($ITERATIONS / 10))")
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $WD_TS))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt dana-star-mk4 --lr $LR --delta 8 --kappa 0.75 --clipsnr $CLIPSNR --weight_decay $WEIGHT_DECAY --wd_decaying --wd_ts $WD_TS"
        ;;
    adamw)
        # For adamw: WEIGHT_DECAY = OMEGA / (LR * ITERATIONS)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        WD_TS="N/A"
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt adamw --lr $LR --weight_decay $WEIGHT_DECAY --beta1 0.9 --beta2 0.999"
        ;;
    dana)
        # For dana: WEIGHT_DECAY = OMEGA / (LR * ITERATIONS)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        WD_TS="N/A"
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt dana --lr $LR --delta 8 --kappa 0.75 --weight_decay $WEIGHT_DECAY --beta1 0.9 --use_v_ema --v_ema_beta 0.999"
        ;;
    ademamix)
        # For ademamix: WEIGHT_DECAY = OMEGA / (LR * ITERATIONS)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        WD_TS="N/A"
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt ademamix --lr $LR --weight_decay $WEIGHT_DECAY --beta1 0.9 --beta2 0.999 --delta 8 --kappa 0.75 --gamma_3_factor 1.0 --adema_beta3_warmup $ITERATIONS --adema_alpha_warmup $ITERATIONS"
        ;;
    d-muon)
        # For d-muon: WEIGHT_DECAY = OMEGA / (LR * ITERATIONS)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        WD_TS="N/A"
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt d-muon --lr $LR --weight_decay $WEIGHT_DECAY --beta1 0.8 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5"
        ;;
    manau)
        # For manau (standard momentum): WEIGHT_DECAY = OMEGA / (LR * ITERATIONS)
        # Muon parameters use standard fixed momentum (dana_momentum=False)
        # DANA-STAR-MK4 parameters use adaptive updates
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        WD_TS=$(python3 -c "print(int($ITERATIONS / 1))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt manau --lr $LR --weight_decay $WEIGHT_DECAY --momentum 0.95 --nesterov True --muon_ns_steps 5 --matched_adamw_rms 0.2 --dana_momentum False --delta 8 --kappa 0.75 --mk4A 0.0 --mk4B 0.0 --clipsnr $CLIPSNR --wd_decaying --wd_ts $WD_TS"
        ;;
    manau-hard)
        # For manau-hard (DANA momentum): WEIGHT_DECAY = OMEGA / (LR * ITERATIONS)
        # Both Muon and DANA-STAR-MK4 parameters use DANA-style adaptive EMA (dana_momentum=True)
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $ITERATIONS))")
        WD_TS=$(python3 -c "print(int($ITERATIONS / 1))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt manau --lr $LR --weight_decay $WEIGHT_DECAY --momentum 0.95 --nesterov True --muon_ns_steps 5 --matched_adamw_rms 0.2 --dana_momentum True --delta 8 --kappa 0.75 --mk4A 0.0 --mk4B 0.0 --clipsnr $CLIPSNR --wd_decaying --wd_ts $WD_TS"
        ;;
    adamw-decaying-wd)
        # For adamw-decaying-wd: Same as dana-star-mk4
        WD_TS=$(python3 -c "print(int($ITERATIONS / 10))")
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $WD_TS))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt adamw-decaying-wd --lr $LR --weight_decay $WEIGHT_DECAY --beta1 0.9 --beta2 0.999 --wd_decaying --wd_ts $WD_TS"
        ;;
    ademamix-decaying-wd)
        # For ademamix-decaying-wd: Same as dana-star-mk4
        WD_TS=$(python3 -c "print(int($ITERATIONS / 10))")
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $WD_TS))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt ademamix-decaying-wd --lr $LR --weight_decay $WEIGHT_DECAY --beta1 0.9 --beta2 0.999 --delta 8 --kappa 0.75 --gamma_3_factor 1.0 --adema_beta3_warmup $ITERATIONS --adema_alpha_warmup $ITERATIONS --wd_decaying --wd_ts $WD_TS"
        ;;
    dana-star-no-tau)
        # For dana-star-no-tau: Same as dana-star-mk4
        WD_TS=$(python3 -c "print(int($ITERATIONS / 10))")
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $WD_TS))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt dana-star-no-tau --lr $LR --delta 8 --kappa 0.75 --clipsnr $CLIPSNR --weight_decay $WEIGHT_DECAY --wd_decaying --wd_ts $WD_TS"
        ;;
    dana-star)
        # For dana-star: Same as dana-star-mk4
        WD_TS=$(python3 -c "print(int($ITERATIONS / 10))")
        WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $WD_TS))")
        WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")
        OPT_PARAMS="--opt dana-star --lr $LR --delta 8 --kappa 0.75 --clipsnr $CLIPSNR --weight_decay $WEIGHT_DECAY --wd_decaying --wd_ts $WD_TS"
        ;;
    dana-mk4)
        # For dana-mk4: Same as dana-star-mk4
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
    EXTRA_RUN_FLAGS="--iterations_to_run $ITERATIONS_TO_RUN --latest_ckpt_interval 1000 --auto_resume"
else
    EXTRA_RUN_FLAGS=""
fi

echo "=== Qwen3 Configuration: $HEADS heads ==="
echo "n_layer: $N_LAYER (= 2 * $HEADS)"
echo "n_head: $N_HEAD"
echo "qkv_dim (head_dim): $HEAD_DIM (fixed)"
echo "n_embd: $N_EMBD (= $HEADS * 128)"
echo "mlp_hidden_dim: $MLP_HIDDEN (= 3 * $N_EMBD)"
echo "Elementwise gating: enabled"
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

# Run training
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE ./src/main.py --config_format base --model qwen3 \
        --distributed_backend nccl --compile \
        --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
        --n_embd $N_EMBD --qkv_dim $HEAD_DIM --n_head $N_HEAD --n_layer $N_LAYER \
        --mlp_hidden_dim $MLP_HIDDEN \
        --batch_size $BATCH_SIZE --sequence_length 2048 --acc_steps $ACC_STEPS \
        --iterations $ITERATIONS \
        --dropout 0.0 --warmup_steps $WARMUP_STEPS --grad_clip 0.5 --seed 0 \
        --init-scheme $INIT_SCHEME --residual-stream-scalar $RESIDUAL_STREAM_SCALAR \
        --z_loss_coeff 0 \
        --norm_type rmsnorm \
        --weight_tying False \
        --elementwise_attn_output_gate \
        $OPT_PARAMS $EXTRA_RUN_FLAGS \
        --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
        --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
        --results_base_folder "$RESULTS_BASE_FOLDER" \
        --eval_interval $EVAL_INTERVAL --log_interval 50

# Capture the exit code
TRAINING_EXIT_CODE=$?

# Restart logic (modeled after tamia_test_steps-to-run.sh)
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
                if [[ "$SLURM_GPUS_PER_NODE" == h100:* ]]; then
                    gpus_per_node="$SLURM_GPUS_PER_NODE"
                else
                    gpus_per_node="h100:${SLURM_GPUS_PER_NODE}"
                fi
            else
                gpus_per_node="h100:4"
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
                if [[ "$SLURM_GPUS_PER_NODE" == h100:* ]]; then
                    gpus_per_node="$SLURM_GPUS_PER_NODE"
                else
                    gpus_per_node="h100:${SLURM_GPUS_PER_NODE}"
                fi
            else
                gpus_per_node="h100:4"
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
