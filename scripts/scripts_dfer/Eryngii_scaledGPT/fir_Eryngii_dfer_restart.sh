#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                    # "alloc as needed" on Alliance

# Parse --restart_on_time_limit argument
RESTART_ON_TIME_LIMIT="none"
RESTART_COUNT=${RESTART_COUNT:-0}
FILTERED_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --restart_on_time_limit)
            RESTART_ON_TIME_LIMIT="$2"
            shift 2
            ;;
        *)
            FILTERED_ARGS+=("$1")
            shift
            ;;
    esac
done

export RESTART_ON_TIME_LIMIT RESTART_COUNT

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt
export WANDB_RUN_GROUP=test_checkpointing_restart
export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

module load arrow/21.0.0
module load python/3.10.13
#module load httpproxy
echo "Loaded modules"

source ~/projects/rrg-bengioy-ad/dferbach/danastar/llm/bin/activate
echo "Activated virtual environment"

export DATASETS_DIR="$HOME/scratch/"

echo "Using FineWeb 100BT dataset from: $DATASETS_DIR"

# Set up restart logic if enabled
if [[ "$RESTART_ON_TIME_LIMIT" != "none" ]] && [[ "$RESTART_ON_TIME_LIMIT" =~ ^[0-9]+$ ]]; then
    MAX_RESTARTS=$RESTART_ON_TIME_LIMIT
    
    # Get job information for restart logic
    START_TIME=$(date +%s)
    scontext=$(scontrol show job $SLURM_JOB_ID 2>/dev/null || echo "")
    timelimit=$(echo "$scontext" | grep -o 'TimeLimit=.*' | awk '{print $1}' | cut -d= -f2)
    timelimit_seconds=$(echo "$timelimit" | awk -F'[-:]' '{if(NF==4)print $1*86400+$2*3600+$3*60+$4; else if(NF==3)print $1*3600+$2*60+$3; else print 0}')
    
    # Extract SLURM job parameters for restart from scontrol
    # Extract values with fallbacks and track which were found
    account=$(echo "$scontext" | grep -o 'Account=[^ ]*' | cut -d= -f2)
    account=${account:-"rrg-bengioy-ad"}
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
    
    term_handler()
    {
        ELAPSED=$(($(date +%s) - START_TIME))
        TIME_UNTIL_LIMIT=$((timelimit_seconds - ELAPSED))
        
        echo "executing term_handler at $(date)"
        echo "Elapsed: ${ELAPSED}s, Remaining: ${TIME_UNTIL_LIMIT}s (limit: ${timelimit_seconds}s)"
        echo "Current restart count: $RESTART_COUNT / $MAX_RESTARTS"
        
        if [[ $TIME_UNTIL_LIMIT -lt 60 ]] && [[ $RESTART_COUNT -lt $MAX_RESTARTS ]]; then
            NEW_RESTART_COUNT=$((RESTART_COUNT + 1))
            echo "Requeuing job at $(date) (restart $NEW_RESTART_COUNT / $MAX_RESTARTS)"
            echo "Using original SLURM args: ${SLURM_ARGS[@]}"
            sbatch --export=RESTART_COUNT=$NEW_RESTART_COUNT "${SLURM_ARGS[@]}" "$0" --restart_on_time_limit "$RESTART_ON_TIME_LIMIT" "${FILTERED_ARGS[@]}"
        elif [[ $RESTART_COUNT -ge $MAX_RESTARTS ]]; then
            echo "Max restarts ($MAX_RESTARTS) reached, not requeuing"
        fi
    }
    
    # Set up signal handler for TERM signal
    trap 'term_handler' TERM
    
    # Call the main BigHead.sh script with filtered arguments in background
    bash scripts/BigHead/Eryngii.sh --init-scheme ScaledGPT "${FILTERED_ARGS[@]}" &
    MAIN_PID=$!
    echo "Started main process with PID: $MAIN_PID"
    
    # Wait for main process - wait is interruptible by signals
    wait $MAIN_PID
    MAIN_EXIT=$?
    
    echo "Main process exited with code: $MAIN_EXIT"
else
    # No restart logic - call script normally
    bash scripts/BigHead/Eryngii.sh --init-scheme ScaledGPT "${FILTERED_ARGS[@]}"
fi
