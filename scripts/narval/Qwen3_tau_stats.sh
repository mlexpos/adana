#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=0

# Qwen3 Tau Statistics Collection Script
# Wrapper script that adds --collect-tau-stats to the standard Qwen3 training

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=03c99521910548176ebfa4f418db1c9602e2afa3
export WANDB_PROJECT=danastar
export WANDB_RUN_GROUP=tau_stats
export WANDB_ENTITY=ep-rmt-ml-opt

export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

module load arrow/21.0.0
module load python/3.13

echo "Loaded modules"

source $HOME/danastarenv/bin/activate
echo "Activated virtual environment"

# Set up directories
export DATASETS_DIR="$HOME/scratch/datasets"
export RESULTS_BASE_FOLDER="$HOME/scratch/checkpoints"

echo "Using FineWeb 100BT dataset from: $DATASETS_DIR"
echo "Using checkpoint directory: $RESULTS_BASE_FOLDER"

wandb offline

# Parse command line arguments for this wrapper
HEADS=""
LR=""
OMEGA=4.0
KAPPA=0.75
CLIPSNR=2.0
BATCH_SIZE=32
ACC_STEPS=1
NPROC_PER_NODE=1
INIT_SCHEME="ScaledGPT"
DEPTH_SCALAR_EXPONENT=0.0
ITERATIONS_TO_RUN=""

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
        --kappa)
            KAPPA="$2"
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
        --depth-scalar-exponent)
            DEPTH_SCALAR_EXPONENT="$2"
            shift 2
            ;;
        --iterations_to_run)
            ITERATIONS_TO_RUN="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$HEADS" ] || [ -z "$LR" ]; then
    echo "Error: --heads and --lr arguments are required"
    exit 1
fi

# Calculate Qwen3 model parameters
HEAD_DIM=128
N_HEAD=$HEADS
N_LAYER=$(python3 -c "print(int(2 * $HEADS))")
N_EMBD=$(python3 -c "print(int($HEADS * 128))")
MLP_HIDDEN=$(python3 -c "print(int(3 * $N_EMBD))")

# Calculate non-embedding parameters
TOTAL_QKV_DIM=$(python3 -c "print(int($N_HEAD * $HEAD_DIM))")
NON_EMB=$(python3 -c "
n_layer = $N_LAYER
n_embd = $N_EMBD
head_dim = $HEAD_DIM
total_qkv_dim = $TOTAL_QKV_DIM

# Per layer
attn = 5 * n_embd * total_qkv_dim
qk_norm = 2 * head_dim
mlp = 9 * n_embd * n_embd
layer_norms = 2 * n_embd

per_layer = attn + qk_norm + mlp + layer_norms

# Total
non_emb = n_layer * per_layer + n_embd

print(int(non_emb))
")

# Calculate iterations and parameters
TOTAL_PARAMS=$(python3 -c "print(int($NON_EMB + 2 * $N_EMBD * 50304))")
ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / 65536))")

# Calculate residual stream scalar
RESIDUAL_STREAM_SCALAR=$(python3 -c "print($N_LAYER ** $DEPTH_SCALAR_EXPONENT)")

# Calculate weight decay
WD_TS=$(python3 -c "print(int($ITERATIONS / 10))")
WEIGHT_DECAY=$(python3 -c "print($OMEGA / ($LR * $WD_TS))")
WARMUP_STEPS=$(python3 -c "print(int($ITERATIONS / 50))")

# Evaluation interval
EVAL_INTERVAL=115

echo "=== Qwen3 Tau Stats Configuration: $HEADS heads ==="
echo "n_layer: $N_LAYER"
echo "n_head: $N_HEAD"
echo "head_dim: $HEAD_DIM"
echo "n_embd: $N_EMBD"
echo "mlp_hidden: $MLP_HIDDEN"
echo "Total parameters: $TOTAL_PARAMS"
echo "Iterations: $ITERATIONS"
if [ -n "$ITERATIONS_TO_RUN" ]; then
    echo "Iterations to run (override): $ITERATIONS_TO_RUN"
fi
echo "Learning rate: $LR"
echo "Omega: $OMEGA"
echo "Kappa: $KAPPA"
echo "Weight decay: $WEIGHT_DECAY"
echo "Weight decay timestep: $WD_TS"
echo "Clip SNR: $CLIPSNR"
echo "Tau stats collection: ENABLED"
echo "=========================================="

# Build iterations_to_run flag if provided
ITERATIONS_TO_RUN_FLAG=""
if [ -n "$ITERATIONS_TO_RUN" ]; then
    ITERATIONS_TO_RUN_FLAG="--iterations_to_run $ITERATIONS_TO_RUN"
fi

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE ./src/main.py --config_format base --model qwen3 \
    --distributed_backend nccl --compile \
    --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
    --n_embd $N_EMBD --qkv_dim $HEAD_DIM --n_head $N_HEAD --n_layer $N_LAYER \
    --mlp_hidden_dim $MLP_HIDDEN \
    --batch_size $BATCH_SIZE --sequence_length 2048 --acc_steps $ACC_STEPS \
    --iterations $ITERATIONS \
    $ITERATIONS_TO_RUN_FLAG \
    --dropout 0.0 --warmup_steps $WARMUP_STEPS --grad_clip 0.5 --seed 0 \
    --init-scheme $INIT_SCHEME --residual-stream-scalar $RESIDUAL_STREAM_SCALAR \
    --z_loss_coeff 0.0 \
    --opt dana-star-mk4 --lr $LR --delta 8 --kappa $KAPPA --clipsnr $CLIPSNR \
    --weight_decay $WEIGHT_DECAY --wd_decaying --wd_ts $WD_TS \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
    --eval_interval $EVAL_INTERVAL --log_interval 50 \
    --results_base_folder "$RESULTS_BASE_FOLDER" \
    --weight_tying false \
    --collect-tau-stats
