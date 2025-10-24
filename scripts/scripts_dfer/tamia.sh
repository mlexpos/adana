#!/bin/bash
#SBATCH --account=aip-gidelgau
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                    # “alloc as needed” on Alliance

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt
export WANDB_RUN_GROUP=Ademamix_small_plrf_behavior
export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

module load arrow/21.0.0
module load python/3.10.13
module load httpproxy
echo "Loaded modules"

source ~/links/projects/aip-gidelgau/dferbach/benchmarking_optimizers/llm/bin/activate
echo "Activated virtual environment"
# GRID SEARCH: Multi-dimensional tuple approach
# GRID_BATCH format: "opt_idx,beta1_idx,gamma_3_idx,wd_idx"
# Each index selects from predefined arrays
# Each node uses 4 predefined learning rates (one per GPU)
# Usage: GRID_BATCH="opt_idx,beta1_idx,gamma_3_idx,wd_idx" sbatch tamia.sh

# Default to first combination if not set
if [ -z "$GRID_BATCH" ]; then
    GRID_BATCH="0,0"
    echo "GRID_BATCH not set, defaulting to $GRID_BATCH"
fi

echo "Running grid search with parameters: $GRID_BATCH"

# Launch four copies in parallel; each sees one GPU
srun --ntasks=4 --cpus-per-task=$SLURM_CPUS_PER_GPU \
     --gpus-per-task=h100:1 --gpu-bind=single:1 --output=logs/%x-%j_%t.out --error=logs/%x-%j_%t.err --exclusive \
     bash -c '
        i=$SLURM_LOCALID                 # 0..3
        # Define learning rates for each GPU (4 values)
        lrs=(3e-3)
        
        # Define grid parameters
        opts=("ademamix")
        w=(3.0)
        gamma_3_factors=(1.0)
        #gamma_3_factors=(1.0 0.037 0.111 0.333 )
        #gamma_3_factors=(3 9 27 81)
        beta1s=(0.9)
        #beta1s=(0.9 0.0 0.5 0.99)
        adema_beta_warmups=(13953 100 1000 4000)
        
        # Get values
        lr=${lrs[0]}                    # Each GPU gets different LR
        opt=${opts[0]}
        w=${w[0]}
        gamma_3_factor=${gamma_3_factors[0]}
        beta1=${beta1s[0]}
        adema_beta_warmup=${adema_beta_warmups[$i]}
        adema_beta3=$(awk "BEGIN {print 1 - 8 / $adema_beta_warmup}")
        wd=$(awk "BEGIN {print $w / $lr / 13953}")

        echo "GRID_BATCH=$GRID_BATCH, GPU $i: opt=$opt, wd=$wd, lr=$lr, gamma_3_factor=$gamma_3_factor, beta1=$beta1, adema_beta_warmup=$adema_beta_warmup"

        DATASETS_DIR="$HOME/links/scratch/fineweb"

        uv run torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
                --distributed_backend nccl --compile \
                --n_embd 384 --qkv_dim 64 --n_head 6 --n_layer 4 \
                --mlp_hidden_dim 1536 \
                --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
                --batch_size 32 --sequence_length 2048 --acc_steps 1 \
                --iterations 13953 \
                --dropout 0.0 --warmup_steps 279 --grad_clip 0.5 --seed 0 \
                --z_loss_coeff 0.0 \
                --opt $opt --lr $lr --weight_decay $wd \
                --beta1 $beta1 --beta2 0.999 --adema_beta3 $adema_beta3 \
                --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
                --delta 8 --kappa 0.75 --gamma_3_factor $gamma_3_factor \
                --adema_beta3_warmup $adema_beta_warmup --adema_alpha_warmup 13953 \
                --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
                --eval_interval 115
        '