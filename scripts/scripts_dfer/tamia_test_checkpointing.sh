#!/bin/bash
#SBATCH --account=aip-gidelgau
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                    # "alloc as needed" on Alliance

START_TIME=$(date +%s)

# Get job information
scontext=$(scontrol show job $SLURM_JOB_ID)
restarts=$(echo "$scontext" | grep -o 'Restarts=.' | cut -d= -f2)
outfile=$(echo "$scontext"  | grep 'StdOut='       | cut -d= -f2)
errfile=$(echo "$scontext"  | grep 'StdErr='       | cut -d= -f2)
timelimit=$(echo "$scontext" | grep -o 'TimeLimit=.*' | awk '{print $1}' | cut -d= -f2)

# Convert time limit to seconds
timelimit_seconds=$(echo "$timelimit" | awk -F'[-:]' '{if(NF==4)print $1*86400+$2*3600+$3*60+$4; else if(NF==3)print $1*3600+$2*60+$3; else print 0}')

echo "Time limit: ${timelimit_seconds}s"

term_handler()
{
    ELAPSED=$(($(date +%s) - START_TIME))
    TIME_UNTIL_LIMIT=$((timelimit_seconds - ELAPSED))
    
    echo "executing term_handler at $(date)"
    echo "Elapsed: ${ELAPSED}s, Remaining: ${TIME_UNTIL_LIMIT}s (limit: ${timelimit_seconds}s)"
    
    # Kill torchrun and its children if it's still running
    if [[ -n "$TORCHRUN_PID" ]] && kill -0 $TORCHRUN_PID 2>/dev/null; then
        echo "Killing torchrun (PID: $TORCHRUN_PID) and its children"
        pkill -P $TORCHRUN_PID 2>/dev/null || true
        kill $TORCHRUN_PID 2>/dev/null || true
    fi
    
    if [[ $TIME_UNTIL_LIMIT -lt 20 ]]; then
        echo "Requeuing job at $(date)"
        echo time until limit: $TIME_UNTIL_LIMIT
        sbatch $0
    fi
}

# declare the function handling the TERM signal
trap 'term_handler' TERM

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt
export WANDB_RUN_GROUP=test_checkpointing
export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

module load arrow/21.0.0
module load python/3.10.13
module load httpproxy
echo "Loaded modules"

source ~/links/projects/aip-gidelgau/dferbach/benchmarking_optimizers/llm/bin/activate
echo "Activated virtual environment"

DATASETS_DIR="$HOME/links/scratch/fineweb"

# Run torchrun in background so signals can interrupt wait
torchrun --standalone --nproc_per_node=4 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --n_embd 384 --qkv_dim 64 --n_head 6 --n_layer 4 \
    --mlp_hidden_dim 1536 \
    --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --iterations 200000 \
    --dropout 0.0 --warmup_steps 279 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt adamw --lr 1e-3 --weight_decay 1e-3 \
    --beta1 0.9 --beta2 0.999 \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 100 --latest_ckpt_interval 1000 --auto_resume &

TORCHRUN_PID=$!
echo "Started torchrun with PID: $TORCHRUN_PID"

# Wait for torchrun - wait is interruptible by signals
wait $TORCHRUN_PID
TORCHRUN_EXIT=$?

echo "torchrun exited with code: $TORCHRUN_EXIT"
