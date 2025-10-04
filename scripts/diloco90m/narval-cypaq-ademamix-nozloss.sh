#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=80GB                    # "alloc as needed" on Alliance

# Hugging Face caches
export HF_HOME="$HOME/hf"
export WANDB_API_KEY=d2f72ec36001f518a4ecf4fe12149a8267e526b0
export WANDB_PROJECT=danastar
export WANDB_RUN_GROUP=Courtney_testing
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

# Choices for good scaling: beta_3 has already been set to 1/ iterations and adema_alpha = iterations * (1 + iterations)**(-kappa) for kappa=0.75. There is only to set lr = lr_adam / 2 and wd = wd_adam after having swept adam_w

torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --n_embd 768 --qkv_dim 64 --n_head 12 --n_layer 9 \
    --mlp_hidden_dim 3072 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
    --iterations 43024 \
    --dropout 0.0 --warmup_steps 860 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt ademamix --lr 1e-3 --weight_decay 1e-3 \
    --beta1 0.9 --beta2 0.999 \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --delta 8 --kappa 0.75 --gamma_3_factor 1.0 \
    --adema_beta3_warmup 43024 --adema_alpha_warmup 43024 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115