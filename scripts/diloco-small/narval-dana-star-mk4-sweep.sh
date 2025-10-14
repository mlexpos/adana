#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=80GB                    # "alloc as needed" on Alliance

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=03c99521910548176ebfa4f418db1c9602e2afa3
export WANDB_PROJECT=danastar
export WANDB_RUN_GROUP=DanaStar_MK4_Small_Sweep_formula9
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

# Modified version of dana-star-mk4.sh that accepts --lr, --wd_ts, and --clipsnr as command line arguments
# Usage: ./narval-dana-star-mk4-sweep.sh --lr <learning_rate> --wd_ts <weight_decay_timestep> --clipsnr <clip_snr>

# Default values
LR=16e-4
WD_TS=1000
CLIPSNR=1.0

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
        --clipsnr)
            CLIPSNR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "Using lr=$LR, wd_ts=$WD_TS, and clipsnr=$CLIPSNR"

torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
    --n_embd 384 --qkv_dim 64 --n_head 6 --n_layer 4 \
    --mlp_hidden_dim 1536 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --iterations 13953 \
    --dropout 0.0 --warmup_steps 280 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt dana-star-mk4 --lr $LR --delta 8 --clipsnr $CLIPSNR \
    --weight_decay 1.0 --wd_decaying --wd_ts $WD_TS \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115 --log_interval 50
