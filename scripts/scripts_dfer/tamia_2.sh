#!/bin/bash
#SBATCH --account=aip-gidelgau
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                    # “alloc as needed” on Alliance

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt
<<<<<<< Updated upstream
export WANDB_RUN_GROUP=Ademamix_small_lr_wd_delta_gamma3factor_sweeps_new
=======
export WANDB_RUN_GROUP=test_checkpointing
>>>>>>> Stashed changes
export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

module load arrow/21.0.0
module load python/3.10.13
module load httpproxy
echo "Loaded modules"

source ~/links/projects/aip-gidelgau/dferbach/benchmarking_optimizers/llm/bin/activate
echo "Activated virtual environment"

<<<<<<< Updated upstream
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
        lr_base=0.003
        w_base=3.0
        delta_base=8.0
        gamma3factor_base=1.0

        # Define grid parameters
        lrs=( 0.01 0.1 1 10 )
        ws=( 0.1 1 10 )
        deltas=( 0.01 0.1 1 10 100 )
        gamma3factors=( 0.01 0.1 1 10 100 )

        # Parse GRID_BATCH tuple
        IFS=',' read -r wd_idx delta_idx gamma3factor_idx <<< "$GRID_BATCH"
        
        # Get values
        lr_factor=${lrs[$i]}                    # Each GPU gets different LR
        w_factor=${ws[$wd_idx]}
        delta_factor=${deltas[$delta_idx]}
        gamma3factor_factor=${gamma3factors[$gamma3factor_idx]}
        
        echo "GRID_BATCH=$GRID_BATCH, GPU $i: wd=$wd_factor, lr=$lr_factor, delta=$delta_factor, gamma3factor=$gamma3factor_factor"
        
        # change 
        lr=$(awk "BEGIN {print $lr_base * 10**(.5) * $lr_factor}")
        wd=$(awk "BEGIN {print $w_base * $w_factor / $lr / 13953}")
        delta=$(awk "BEGIN {print $delta_base * $delta_factor}")
        gamma3factor=$(awk "BEGIN {print $gamma3factor_base * $gamma3factor_factor}")

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
                --opt ademamix --lr $lr --weight_decay $wd \
                --beta1 0.9 --beta2 0.999 \
                --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
                --delta $delta --kappa 0.75 --gamma_3_factor $gamma3factor \
                --adema_beta3_warmup 13953 --adema_alpha_warmup 13953 \
                --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
                --eval_interval 115
        '
=======
# Get values
lr=0.004
w=3.0

echo "GPU $i: lr=$lr, w=$w"

wd=$(awk "BEGIN {print $w / $lr / 13953}")

DATASETS_DIR="$HOME/links/scratch/fineweb"

uv run torchrun --standalone --nproc_per_node=4 ./src/main.py --config_format base --model diloco \
        --distributed_backend nccl --compile \
        --n_embd 384 --qkv_dim 64 --n_head 6 --n_layer 4 \
        --mlp_hidden_dim 1536 \
        --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
        --batch_size 32 --sequence_length 2048 --acc_steps 1 \
        --iterations 13953 \
        --dropout 0.0 --warmup_steps 279 --grad_clip 0.5 --seed 0 \
        --z_loss_coeff 0.0 \
        --opt adamw --lr $lr --weight_decay $wd \
        --beta1 0.9 --beta2 0.999 \
        --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
        --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
        --eval_interval 115 --latest_ckpt_interval 115 --permanent_ckpt_interval 0 --auto_resume True
>>>>>>> Stashed changes
