#!/bin/bash
#SBATCH --time=00:01:30
#SBATCH --gres=gpu:80GB:1
#SBATCH --partition=unkillable
#SBATCH --mem=32GB
#SBATCH --signal=USR1@60

source ~/projects/llm-optimizer-benchmark/llm/bin/activate

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt
export WANDB_RUN_GROUP=test_checkpointing
export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

DATASETS_DIR="$HOME/scratch/fineweb/"

handle_timeout() {
    echo "Received timeout signal (USR1), requeuing job $SLURM_JOBID at $(date)"
    trap '' USR1
    # Forward signal to torchrun process if it exists
    if [ ! -z "$TORCHRUN_PID" ]; then
        kill -USR1 $TORCHRUN_PID 2>/dev/null || true
        # Give it a moment to save checkpoint
        sleep 5
    fi
    sbatch $0
    exit 0
}

trap 'handle_timeout' USR1

(torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --n_embd 384 --qkv_dim 64 --n_head 6 --n_layer 4 \
    --mlp_hidden_dim 1536 \
    --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --iterations 100000 \
    --dropout 0.0 --warmup_steps 279 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt adamw --lr 1e-3 --weight_decay 1e-3 \
    --beta1 0.9 --beta2 0.999 \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 100 --latest_ckpt_interval 1000 --auto_resume) &
TORCHRUN_PID=$!
wait $TORCHRUN_PID
