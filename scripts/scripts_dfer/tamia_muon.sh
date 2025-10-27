#!/bin/bash
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=aip-gidelgau
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                    # “alloc as needed” on Alliance

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt
export WANDB_RUN_GROUP=180M_muon_sweep
export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

module load arrow/21.0.0
module load python/3.10.13
module load httpproxy
echo "Loaded modules"

source ~/links/projects/aip-gidelgau/dferbach/benchmarking_optimizers/llm/bin/activate
echo "Activated virtual environment"

DATASETS_DIR="$HOME/links/scratch/fineweb"

momentum=0.95
for lr in 3e-4 1e-3 3e-3 6e-3 1e-2
do
for w in 3.0 0.5 1.0
do
weight_decay=$(awk "BEGIN {printf \"%.10f\", $w / $lr / 21481}")
for muon_lr_factor in 1e-2 5e-3 2e-2 
do
uv run torchrun --standalone --nproc_per_node=4 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
     --n_embd 1024 --qkv_dim 64 --n_head 16 --n_layer 12 \
    --mlp_hidden_dim 4096 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
    --iterations 77527 \
    --dropout 0.0 --warmup_steps 1551 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt muon --lr $lr --muon_lr_factor $muon_lr_factor --weight_decay $weight_decay \
    --beta1 0.9 --beta2 0.999 --momentum $momentum --nesterov True --muon_ns_steps 5 \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115
done
done
done