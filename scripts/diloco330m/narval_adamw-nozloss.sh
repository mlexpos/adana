#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-gpu=12
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=0                    # "alloc as needed" on Alliance

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=03c99521910548176ebfa4f418db1c9602e2afa3
export WANDB_PROJECT=danastar
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

# Run the diloco330m experiment with FineWeb 100BT dataset
# torchrun --standalone --nproc_per_node=4 ./src/main.py --config_format base --model diloco \
#     --distributed_backend nccl --compile \
#     --datasets_dir "$DATASETS_DIR" \
#     --n_embd 1280 --qkv_dim 64 --n_head 20 --n_layer 15 \
#     --mlp_hidden_dim 5120 \
#     --batch_size 32 --sequence_length 2048 --acc_steps 1 \
#     --dataset fineweb_100 --iterations 100708 \
#     --dropout 0.0 --grad_clip 2.5 --seed 0 \
#     --opt dana-star --lr 5e-4 --delta 8 --kappa 0.75 --clipsnr 1.6 \
#     --scheduler cos --warmup_steps 2000 \
#     --weight_decay 0.001 --wd_decaying --wd_ts 100 \
#     --beta1 0.9 --beta2 0.999 \
#     --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
#     --eval_interval 115 --log_interval 50 --latest_ckpt_interval 1000

torchrun --standalone --nproc_per_node=4 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
    --n_embd 1280 --qkv_dim 64 --n_head 20 --n_layer 15 \
    --mlp_hidden_dim 5120 \
    --batch_size 4 --sequence_length 2048 --acc_steps 8 \
    --iterations 129312 \
    --dropout 0.0 --warmup_steps 2586 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt adamw --lr 1e-3 --weight_decay 1e-3 \
    --beta1 0.9 --beta2 0.999 \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115