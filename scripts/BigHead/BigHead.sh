#!/bin/bash

# Default values
LR=15e-4
OMEGA=4.0
CLIPSNR=2.0
BATCH_SIZE=32
ACC_STEPS=1
NPROC_PER_NODE=1
DEPTH=""
OPTIMIZER="dana-star-mk4"
INIT_SCHEME="KarpathyGPT2"
DEPTH_SCALAR_EXPONENT=0.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --depth)
            DEPTH="$2"
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
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate depth argument
if [ -z "$DEPTH" ]; then
    echo "Error: --depth argument is required"
    echo "Usage: ./BigHead.sh --depth <depth>=4> [options]"
    echo "Options:"
    echo "  --lr <value>              Learning rate (default: 15e-4)"
    echo "  --omega <value>           Weight decay strength parameter (default: 4.0)"
    echo "  --clipsnr <value>         Clip SNR for dana-star-mk4 (default: 2.0)"
    echo "  --batch_size <value>      Batch size (default: 32)"
    echo "  --acc_steps <value>       Accumulation steps (default: 1)"
    echo "  --nproc_per_node <value>  Processes per node (default: 1)"
    echo "  --optimizer <type>        Optimizer type: dana-star-mk4, adamw, dana, ademamix, d-muon, manau, manau-hard (default: dana-star-mk4)"
    echo "  --init-scheme <type>      Initialization scheme: KarpathyGPT2, Standard (default: KarpathyGPT2)"
    echo "  --depth-scalar-exponent <value>  Exponent for depth-based residual scaling: scalar = n_layer^exp (default: 0.0)"
    exit 1
fi

if [ "$DEPTH" -lt 4 ]; then
    echo "Error: depth must be at least 4"
    exit 1
fi

# Calculate model parameters based on depth using Python for correct integer types
# Formula: head_dim = 16 * depth, n_embd = 16 * depth^2, mlp = 32 * depth^2
HEAD_DIM=$(python3 -c "print(int(16 * $DEPTH))")
N_EMBD=$(python3 -c "print(int(16 * $DEPTH * $DEPTH))")
MLP_HIDDEN=$(python3 -c "print(int(32 * $DEPTH * $DEPTH))")
N_HEAD=$(python3 -c "print(int($DEPTH))")
N_LAYER=$(python3 -c "print(int($DEPTH))")

# Calculate residual stream scalar based on depth and exponent
RESIDUAL_STREAM_SCALAR=$(python3 -c "print($N_LAYER ** $DEPTH_SCALAR_EXPONENT)")

# Calculate iterations based on total parameters
# Total params = non_emb + 2 * n_embd * 50304
# Non-emb = depth * (3 * head_dim * n_embd * n_head + n_embd^2 + 2 * n_embd * mlp + 8 * n_embd) + 2 * n_embd
NON_EMB=$(python3 -c "print(int($DEPTH * (3 * $HEAD_DIM * $N_EMBD * $N_HEAD + $N_EMBD * $N_EMBD + 2 * $N_EMBD * $MLP_HIDDEN + 8 * $N_EMBD) + 2 * $N_EMBD))")
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
        OPT_PARAMS="--opt ademamix --lr $LR --weight_decay $WEIGHT_DECAY --beta1 0.9 --beta2 0.999 --delta 8 --kappa  0.75 --gamma_3_factor 1.0 --adema_beta3_warmup $ITERATIONS --adema_alpha_warmup $ITERATIONS"
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
    *)
        echo "Error: Unknown optimizer $OPTIMIZER"
        echo "Available optimizers: dana-star-mk4, adamw, dana, ademamix, d-muon, manau, manau-hard"
        exit 1
        ;;
esac

echo "=== BigHead Configuration for Depth $DEPTH ==="
echo "n_layer: $N_LAYER"
echo "n_head: $N_HEAD"
echo "qkv_dim (head_dim): $HEAD_DIM"
echo "n_embd: $N_EMBD"
echo "mlp_hidden_dim: $MLP_HIDDEN"
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
echo "Init scheme: $INIT_SCHEME"
echo "Depth scalar exponent: $DEPTH_SCALAR_EXPONENT"
echo "Residual stream scalar: $RESIDUAL_STREAM_SCALAR"
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
    --init-scheme $INIT_SCHEME --residual-stream-scalar $RESIDUAL_STREAM_SCALAR \
    --z_loss_coeff 0.0 \
    $OPT_PARAMS \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval $EVAL_INTERVAL --log_interval 50