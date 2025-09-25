#!/bin/bash
#SBATCH --account=aip-gidelgau
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                    # “alloc as needed” on Alliance

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt

module load arrow/21.0.0
module load python/3.10.13
module load httpproxy
echo "Loaded modules"

source ~/links/projects/aip-gidelgau/dferbach/benchmarking_optimizers/llm/bin/activate
echo "Activated virtual environment"
# GRID SEARCH: 4 learning rates × 3 weight decay values = 12 combinations
# Run with different GRID_BATCH values (0, 1, 2) to cover all combinations
# Usage: GRID_BATCH=0 sbatch tamia.sh  (for wd=1e-7)
#        GRID_BATCH=1 sbatch tamia.sh  (for wd=1e-5)  
#        GRID_BATCH=2 sbatch tamia.sh  (for wd=1e-3)

# Default to batch 0 if not set
if [ -z "$GRID_BATCH" ]; then
    GRID_BATCH=0
    echo "GRID_BATCH not set, defaulting to 0 (w=1e-7)"
fi

echo "Running grid search batch $GRID_BATCH"

# Launch four copies in parallel; each sees one GPU
srun --ntasks=4 --cpus-per-task=$SLURM_CPUS_PER_GPU \
     --gpus-per-task=h100:1 --gpu-bind=single:1 --output=logs/%x-%j_%t.out --error=logs/%x-%j_%t.err --exclusive \
     bash -c '
        i=$SLURM_LOCALID                 # 0..3
        # Define learning rates for each GPU (4 values)
        lrs=(3e-4 3e-3 3e-2 3e-1)
        
        # Define weight decay values (3 values)
        ws=(1e-7 1e-4 1e-1)
        
        # Calculate which lr and wd to use based on task ID and batch
        lr_idx=$i  # 0-3 (each GPU gets a different learning rate)
        ws_idx=$GRID_BATCH  # 0-2 (weight decay changes per batch)
        
        lr=${lrs[$lr_idx]}
        w=${ws[$ws_idx]}

        wd=$(awk "BEGIN {print $w / $lr}")
        
        echo "GRID_BATCH=$GRID_BATCH, GPU $i: lr=$lr, weight_decay=$wd"
        
        # Clean up previous experiments
        # rm -rf exps/*adamw*

        uv run torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
            --distributed_backend nccl --compile \
            --n_embd 768 --qkv_dim 64 --n_head 12 --n_layer 9 \
            --mlp_hidden_dim 3072 \
            --batch_size 32 --sequence_length 2048 --acc_steps 1 \
            --dataset fineweb --iterations 43024 \
            --dropout 0.0 --warmup_steps 860 --grad_clip 0.5 --seed 0 \
            --z_loss_coeff 0.0 \
            --opt adamw --lr $lr --weight_decay $wd \
            --beta1 0.9 --beta2 0.999 \
            --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
            --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
            --eval_interval 115
        '