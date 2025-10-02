#! /bin/bash
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:80GB
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=4

source ~/projects/llm-optimizer-benchmark/llm/bin/activate

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt
export WANDB_RUN_GROUP=Ademamix_small_comparison_new_dataset
export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

# uv run torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
#     --distributed_backend nccl --compile \
#     --n_embd 1024 --qkv_dim 64 --n_head 16 --n_layer 12 \
#     --mlp_hidden_dim 4096 \
#     --batch_size 32 --sequence_length 2048 --acc_steps 1 \
#     --dataset fineweb --iterations 77527 \
#     --dropout 0.0 --warmup_steps 1551 --grad_clip 0.5 --seed 0 \
#     --z_loss_coeff 0.0 \
#     --opt dana --lr 1.75e-4 --delta 8 --kappa 0.75 --weight_decay 0.34286 \
#     --use_grad_ema_for_g2 --grad_ema_beta 0.0 --use_v_ema --v_ema_beta 0.999 \
#     --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
#     --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
#     --eval_interval 115

# uv run torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
#     --distributed_backend nccl --compile \
#     --n_embd 768 --qkv_dim 64 --n_head 12 --n_layer 9 \
#     --mlp_hidden_dim 3072 \
#     --batch_size 32 --sequence_length 2048 --acc_steps 1 \
#     --dataset fineweb --iterations 43024 \
#     --dropout 0.0 --warmup_steps 860 --grad_clip 0.5 --seed 0 \
#     --z_loss_coeff 0.0 \
#     --opt dana --lr 3.5e-4 --delta 8 --kappa 0.75 --weight_decay 0.51429 \
#     --use_grad_ema_for_g2 --grad_ema_beta 0.0 --use_v_ema --v_ema_beta 0.999 \
#     --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
#     --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
#     --eval_interval 115

DATASETS_DIR="$HOME/scratch/fineweb/"

for lr in 7.5e-4 1.5e-3 3e-3; do
    for wd in 0.13333 0.066667 0.26667; do
        uv run torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
            --distributed_backend nccl --compile \
            --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
            --n_embd 384 --qkv_dim 64 --n_head 6 --n_layer 4 \
            --mlp_hidden_dim 1536 \
            --batch_size 32 --sequence_length 2048 --acc_steps 1 \
            --iterations 13953 \
            --dropout 0.0 --warmup_steps 279 --grad_clip 0.5 --seed 0 \
            --z_loss_coeff 0.0 \
            --opt ademamix --lr $lr --weight_decay $wd \
            --beta1 0.9 --beta2 0.999 \
            --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
            --adema_beta3 0.99943 --adema_alpha 10.9 \
            --adema_beta3_warmup 13953 --adema_alpha_warmup 13953 \
            --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
            --eval_interval 115
    done
done

# for wd in 0.53333 0.26667; do
#     for lr in 7.5e-4 1.5e-3; do
#         uv run torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
#             --distributed_backend nccl --compile \
#             --n_embd 384 --qkv_dim 64 --n_head 6 --n_layer 4 \
#             --mlp_hidden_dim 1536 \
#             --batch_size 32 --sequence_length 2048 --acc_steps 1 \
#             --dataset fineweb --iterations 13953 \
#             --dropout 0.0 --warmup_steps 279 --grad_clip 0.5 --seed 0 \
#             --z_loss_coeff 0.0 \
#             --opt dana --lr $lr --delta 8 --kappa 0.75 --weight_decay $wd \
#             --use_grad_ema_for_g2 --grad_ema_beta 0.0 --use_v_ema --v_ema_beta 0.999 \
#             --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
#             --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
#             --eval_interval 115
#     done
# done