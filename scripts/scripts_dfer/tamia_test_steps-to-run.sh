#!/bin/bash
#SBATCH --account=aip-gidelgau
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                    # "alloc as needed" on Alliance

# Parse --restart_steps_to_run argument
RESTART_STEPS_TO_RUN="none"
RESTART_COUNT=${RESTART_COUNT:-0}
FILTERED_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --restart_steps_to_run=*)
            RESTART_STEPS_TO_RUN="${1#*=}"
            shift
            ;;
        --restart_steps_to_run)
            RESTART_STEPS_TO_RUN="$2"
            shift 2
            ;;
        *)
            FILTERED_ARGS+=("$1")
            shift
            ;;
    esac
done

export RESTART_STEPS_TO_RUN RESTART_COUNT

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

# Set up restart logic if enabled
if [[ "$RESTART_STEPS_TO_RUN" != "none" ]] && [[ "$RESTART_STEPS_TO_RUN" =~ ^[0-9]+$ ]]; then
    MAX_RESTARTS=$RESTART_STEPS_TO_RUN
    
    # Get job information for restart logic
    START_TIME=$(date +%s)
    scontext=$(scontrol show job $SLURM_JOB_ID 2>/dev/null || echo "")
    timelimit=$(echo "$scontext" | grep -o 'TimeLimit=.*' | awk '{print $1}' | cut -d= -f2)
    timelimit_seconds=$(echo "$timelimit" | awk -F'[-:]' '{if(NF==4)print $1*86400+$2*3600+$3*60+$4; else if(NF==3)print $1*3600+$2*60+$3; else print 0}')
    
    # Extract SLURM job parameters for restart from scontrol
    # Extract values with fallbacks and track which were found
    account=$(echo "$scontext" | grep -o 'Account=[^ ]*' | cut -d= -f2)
    account=${account:-"aip-gidelgau"}
    [ -z "$(echo "$scontext" | grep -o 'Account=[^ ]*' | cut -d= -f2)" ] && echo "WARNING: Account not found in scontrol, using default: $account" || echo "Found Account: $account"
    
    nodes=$(echo "$scontext" | grep -o 'NumNodes=[^ ]*' | cut -d= -f2)
    nodes=${nodes:-"1"}
    [ -z "$(echo "$scontext" | grep -o 'NumNodes=[^ ]*' | cut -d= -f2)" ] && echo "WARNING: NumNodes not found in scontrol, using default: $nodes" || echo "Found NumNodes: $nodes"
    
    gres_raw=$(echo "$scontext" | grep -o 'Gres=[^ ]*' | cut -d= -f2)
    if [ -z "$gres_raw" ]; then
        # Use SLURM_GPUS_PER_NODE if set, otherwise default to 4
        # If SLURM_GPUS_PER_NODE already includes "h100:", use it as-is, otherwise add "h100:" prefix
        if [ -n "$SLURM_GPUS_PER_NODE" ]; then
            if [[ "$SLURM_GPUS_PER_NODE" == h100:* ]]; then
                gpus_per_node="$SLURM_GPUS_PER_NODE"
            else
                gpus_per_node="h100:${SLURM_GPUS_PER_NODE}"
            fi
        else
            gpus_per_node="h100:4"
        fi
        echo "WARNING: Gres not found in scontrol, using default: $gpus_per_node"
    else
        # Remove gpu: prefix if present, then the value should be in format like "h100:4"
        gpus_per_node=$(echo "$gres_raw" | sed 's/^gpu://')
        echo "Found Gres: $gres_raw -> $gpus_per_node"
    fi
    
    mem=$(echo "$scontext" | grep -o 'MinMemoryCPU=[^ ]*' | cut -d= -f2)
    mem=${mem:-"0"}
    [ -z "$(echo "$scontext" | grep -o 'MinMemoryCPU=[^ ]*' | cut -d= -f2)" ] && echo "WARNING: MinMemoryCPU not found in scontrol, using default: $mem" || echo "Found MinMemoryCPU: $mem"
    
    job_name=$(echo "$scontext" | grep -o 'JobName=[^ ]*' | cut -d= -f2)
    job_name=${job_name:-""}
    [ -z "$job_name" ] && echo "WARNING: JobName not found in scontrol, using empty string" || echo "Found JobName: $job_name"
    
    SLURM_ARGS=(
        --account="$account"
        --time="$timelimit"
        --nodes="$nodes"
        --gpus-per-node="$gpus_per_node"
        --cpus-per-gpu="${SLURM_CPUS_PER_GPU:-8}"
        --mem="$mem"
        --job-name="$job_name"
    )
    echo "Using original SLURM args: ${SLURM_ARGS[@]}"
    
    # Run torchrun in background so signals can interrupt wait
    torchrun --standalone --nproc_per_node=4 ./src/main.py --config_format base --model diloco \
        --distributed_backend nccl --compile \
        --n_embd 384 --qkv_dim 64 --n_head 6 --n_layer 4 \
        --mlp_hidden_dim 1536 \
        --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
        --batch_size 32 --sequence_length 2048 --acc_steps 1 \
        --iterations 1234 --iterations_to_run 200\
        --dropout 0.0 --warmup_steps 279 --grad_clip 0.5 --seed 0 \
        --z_loss_coeff 0.0 \
        --opt adamw --lr 1e-3 --weight_decay 1e-3 \
        --beta1 0.9 --beta2 0.999 \
        --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
        --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
        --eval_interval 100 --latest_ckpt_interval 1000 --auto_resume "${FILTERED_ARGS[@]}"

    echo "torchrun exited"
    
    # If process completed successfully (exit code 0) and we haven't reached max restarts, requeue
    if [[ $RESTART_COUNT -lt $MAX_RESTARTS ]]; then
        NEW_RESTART_COUNT=$((RESTART_COUNT + 1))
        echo "Process completed successfully, requeuing for next batch (restart $NEW_RESTART_COUNT / $MAX_RESTARTS)"
        echo "Using original SLURM args: ${SLURM_ARGS[@]}"
        sbatch --export=RESTART_COUNT=$NEW_RESTART_COUNT "${SLURM_ARGS[@]}" "$0" --restart_steps_to_run "$RESTART_STEPS_TO_RUN" "${FILTERED_ARGS[@]}"
    elif [[ $RESTART_COUNT -ge $MAX_RESTARTS ]]; then
        echo "Max restarts ($MAX_RESTARTS) reached, not requeuing"
    fi
else
    # No restart logic - run torchrun normally
    torchrun --standalone --nproc_per_node=4 ./src/main.py --config_format base --model diloco \
        --distributed_backend nccl --compile \
        --n_embd 384 --qkv_dim 64 --n_head 6 --n_layer 4 \
        --mlp_hidden_dim 1536 \
        --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
        --batch_size 32 --sequence_length 2048 --acc_steps 1 \
        --iterations 1233 --iterations_to_run 200\
        --dropout 0.0 --warmup_steps 279 --grad_clip 0.5 --seed 0 \
        --z_loss_coeff 0.0 \
        --opt adamw --lr 1e-3 --weight_decay 1e-3 \
        --beta1 0.9 --beta2 0.999 \
        --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
        --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
        --eval_interval 100 --latest_ckpt_interval 1000 --auto_resume "${FILTERED_ARGS[@]}"
    
    echo "torchrun exited"
fi
