#!/bin/bash
#SBATCH --account=aip-gidelgau
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                    # “alloc as needed” on Alliance

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt
export WANDB_RUN_GROUP=auto_dana_small
export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

module load arrow/21.0.0
module load python/3.10.13
module load httpproxy
echo "Loaded modules"

source ~/links/projects/aip-gidelgau/dferbach/benchmarking_optimizers/llm/bin/activate
echo "Activated virtual environment"

DATASETS_DIR="$HOME/links/scratch/fineweb"

lr=0.0005
iterations=13953
wd=$(echo "4.0 / $lr / $iterations" | bc -l)
gamma_3_factor=1.
printf "Weight decay: %f\n" $wd
uv run torchrun --standalone --nproc_per_node=4 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --n_embd 384 --qkv_dim 64 --n_head 6 --n_layer 4 \
    --mlp_hidden_dim 1536 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
    --iterations $iterations \
    --dropout 0.0 --warmup_steps 279 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt auto-dana --lr $lr --delta 8 --kappa 0.75 --weight_decay $wd \
    --beta1 0.9 --use_v_ema --v_ema_beta 0.999 --gamma_3_factor $gamma_3_factor \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115