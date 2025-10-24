#! /bin/bash
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:80GB
#SBATCH --partition=main
#SBATCH --cpus-per-task=4

source ~/projects/llm-optimizer-benchmark/llm/bin/activate

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt
export WANDB_RUN_GROUP=test_checkpointing
export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

DATASETS_DIR="$HOME/scratch/fineweb/"

# Get values
lr=0.001
w=4.0

echo "GPU $i: lr=$lr, w=$w"

wd=$(awk "BEGIN {print $w / $lr / 43024}")

uv run torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --n_embd 768 --qkv_dim 64 --n_head 12 --n_layer 9 \
    --mlp_hidden_dim 3072 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
    --iterations 43024 \
    --dropout 0.0 --warmup_steps 860 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt adamw --lr 1e-3 --weight_decay 1e-3 \
    --beta1 0.9 --beta2 0.999 \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115 --latest_ckpt_interval 10000 --permanent_ckpt_interval 10000 --run_prefix $(date +%s)