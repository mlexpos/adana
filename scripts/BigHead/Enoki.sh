#!/bin/bash

# Default values
LR=15e-4
OMEGA=4.0
CLIPSNR=2.0
BATCH_SIZE=32
ACC_STEPS=1
NPROC_PER_NODE=1
HEADS=""
OPTIMIZER="dana-star-mk4"

# Parse command line arguments
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
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$HEADS" ]; then
    echo "Error: --heads argument is required"
    echo "Usage: ./Enoki.sh --heads <heads> [options]"
    echo ""
    echo "Enoki uses DiLoco scaling with fixed aspect ratio:"
    echo "  - head_dim = 64 (fixed)"
    echo "  - n_head = heads"
    echo "  - n_layer = 3 * heads / 4"
    echo "  - n_embd = heads * 64"
    echo "  - mlp_hidden = 4 * n_embd"
    echo ""
    echo "Options:"
    echo "  --lr <value>              Learning rate (default: 15e-4)"
    echo "  --omega <value>           Weight decay strength parameter (default: 4.0)"
    echo "  --clipsnr <value>         Clip SNR for dana-star-mk4 (default: 2.0)"
    echo "  --batch_size <value>      Batch size (default: 32)"
    echo "  --acc_steps <value>       Accumulation steps (default: 1)"
    echo "  --nproc_per_node <value>  Processes per node (default: 1)"
    echo "  --optimizer <type>        Optimizer type: dana-star-mk4, adamw, dana, ademamix, d-muon (default: dana-star-mk4)"
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

# Calculate model parameters based on DiLoco/Enoki scaling
# Formula: head_dim = 64 (fixed), n_layer = 3*heads/4, n_embd = heads * 64, mlp = 4 * n_embd
HEAD_DIM=64
N_HEAD=$(python3 -c "print(int($HEADS))")
N_LAYER=$(python3 -c "print(int(3 * $HEADS // 4))")
N_EMBD=$(python3 -c "print(int($HEADS * 64))")
MLP_HIDDEN=$(python3 -c "print(int(4 * $N_EMBD))")

# Calculate iterations based on total parameters
# Total params = non_emb + 2 * n_embd * 50304
# Non-emb = 12 * n_embd^2 * n_layer (standard DiLoco formula)
NON_EMB=$(python3 -c "print(int(12 * $N_EMBD * $N_EMBD * $N_LAYER))")
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
    *)
        echo "Error: Unknown optimizer $OPTIMIZER"
        echo "Available optimizers: dana-star-mk4, adamw, dana, ademamix, d-muon"
        exit 1
        ;;
esac

echo "=== Enoki Configuration: $HEADS heads ==="
echo "n_layer: $N_LAYER (= 3 * $HEADS / 4)"
echo "n_head: $N_HEAD"
echo "qkv_dim (head_dim): $HEAD_DIM (fixed)"
echo "n_embd: $N_EMBD (= $HEADS * 64)"
echo "mlp_hidden_dim: $MLP_HIDDEN (= 4 * $N_EMBD)"
echo "Total parameters: $TOTAL_PARAMS"
echo "Iterations: $ITERATIONS"
echo "Learning rate: $LR"
echo "Omega: $OMEGA"
echo "Weight decay: $WEIGHT_DECAY"
echo "Weight decay timestep: $WD_TS"
echo "Clip SNR: $CLIPSNR"
echo "Batch size: $BATCH_SIZE"
echo "Accumulation steps: $ACC_STEPS"
echo "Processes per node: $NPROC_PER_NODE"
echo "Optimizer: $OPTIMIZER"
echo "=========================================="

EVAL_INTERVAL=$(python3 -c "print(115)")

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
    --n_embd $N_EMBD --qkv_dim $HEAD_DIM --n_head $N_HEAD --n_layer $N_LAYER \
    --mlp_hidden_dim $MLP_HIDDEN \
    --batch_size $BATCH_SIZE --sequence_length 2048 --acc_steps $ACC_STEPS \
    --iterations $ITERATIONS \
    --dropout 0.0 --warmup_steps $WARMUP_STEPS --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    $OPT_PARAMS \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval $EVAL_INTERVAL --log_interval 50
