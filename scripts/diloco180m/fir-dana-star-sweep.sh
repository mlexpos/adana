#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=256GB                    # "alloc as needed" on Alliance

# Hugging Face caches
export HF_HOME="$HOME/scratch/hf"
export WANDB_API_KEY=03c99521910548176ebfa4f418db1c9602e2afa3
export WANDB_RUN_GROUP=DanaStar_180M_lr_weight_decay_sweeps
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt

export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

# Function to load modules with retry logic
load_module_with_retry() {
    local module_name="$1"
    local max_attempts=3
    local attempt=1
    local wait_time=5
    
    echo "Attempting to load module: $module_name"
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt/$max_attempts: Loading $module_name"
        
        if module load "$module_name" 2>/dev/null; then
            echo "Successfully loaded $module_name"
            return 0
        else
            echo "Failed to load $module_name (attempt $attempt/$max_attempts)"
            if [ $attempt -lt $max_attempts ]; then
                echo "Waiting ${wait_time}s before retry..."
                sleep $wait_time
                wait_time=$((wait_time * 2))  # Exponential backoff
            fi
        fi
        
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: Failed to load $module_name after $max_attempts attempts"
    echo "This may indicate CVMFS/filesystem issues on this compute node"
    exit 1
}

# Load modules with retry logic
load_module_with_retry "arrow/21.0.0"
load_module_with_retry "python/3.13"

echo "Successfully loaded all modules"

source $HOME/scratch/danastarenv/bin/activate
echo "Activated virtual environment"

# Set up directories
FINEWEB_DIR="$HOME/links/scratch/fineweb"
DATASETS_DIR="$HOME/scratch/datasets"

echo "Using FineWeb 100BT dataset from: $DATASETS_DIR"

wandb offline

# Modified version that accepts --lr and --wd_ts as command line arguments
# Usage: ./narval-dana-star-sweep.sh --lr <learning_rate> --wd_ts <weight_decay_timestep>

# Default values
LR=5e-4
WD_TS=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr)
            LR="$2"
            shift 2
            ;;
        --wd_ts)
            WD_TS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "Using lr=$LR and wd_ts=$WD_TS"

torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
    --n_embd 1024 --qkv_dim 64 --n_head 16 --n_layer 12 \
    --mlp_hidden_dim 4096 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --iterations 77527 \
    --dropout 0.0 --warmup_steps 1551 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt dana-star --lr $LR --delta 8 --kappa 0.75 --clipsnr 2.0 \
    --weight_decay 1.0 --wd_decaying --wd_ts $WD_TS \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115 --log_interval 50
