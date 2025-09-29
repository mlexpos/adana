#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=200GB                    # "alloc as needed" on Alliance

# Hugging Face caches
export HF_HOME="$HOME/hf"
export WANDB_API_KEY=d2f72ec36001f518a4ecf4fe12149a8267e526b0
export WANDB_PROJECT=danastar
export WANDB_RUN_GROUP=AdamW_180M_lr_weight_decay_sweeps
export WANDB_ENTITY=ep-rmt-ml-opt

export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

module load arrow/21.0.0
module load python/3.13

echo "Loaded modules"

source $HOME/danastarenv/bin/activate
echo "Activated virtual environment"

# Set up directories
FINEWEB_DIR="$HOME/links/scratch/fineweb"
DATASETS_DIR="$HOME/scratch/datasets"

echo "Using FineWeb 100BT dataset from: $DATASETS_DIR"

wandb offline

# Modified version that accepts --lr and --weight_decay as command line arguments
# Usage: ./narval-cypaq-adamw-nozloss-sweep.sh --lr <learning_rate> --weight_decay <weight_decay>

# Default values
LR=1e-3
WEIGHT_DECAY=1e-3

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr)
            LR="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "Using lr=$LR and weight_decay=$WEIGHT_DECAY"

torchrun --standalone --nproc_per_node=4 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
    --n_embd 1024 --qkv_dim 64 --n_head 16 --n_layer 12 \
    --mlp_hidden_dim 4096 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --iterations 77527 \
    --dropout 0.0 --warmup_steps 1551 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt adamw --lr $LR --weight_decay $WEIGHT_DECAY \
    --beta1 0.9 --beta2 0.999 \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115
