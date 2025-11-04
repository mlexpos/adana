#!/bin/bash
#SBATCH --account=aip-gidelgau
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                    # "alloc as needed" on Alliance

# Parse --num_restart and --iterations_to_run arguments
NUM_RESTART="none"
ITERATIONS_TO_RUN=""
RESTART_COUNT=${RESTART_COUNT:-0}
FILTERED_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_restart)
            NUM_RESTART="$2"
            shift 2
            ;;
        --iterations_to_run)
            ITERATIONS_TO_RUN="$2"
            shift 2
            ;;
        *)
            FILTERED_ARGS+=("$1")
            shift
            ;;
    esac
done

export NUM_RESTART RESTART_COUNT

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt
export WANDB_RUN_GROUP=eryngii_scaledGPT
export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

module load arrow/21.0.0
module load python/3.10.13
module load httpproxy
echo "Loaded modules"

source ~/links/projects/aip-gidelgau/dferbach/benchmarking_optimizers/llm/bin/activate
echo "Activated virtual environment"

export DATASETS_DIR="$HOME/links/scratch/fineweb"

echo "Using FineWeb 100BT dataset from: $DATASETS_DIR"

# Set up restart logic if enabled
if [[ "$NUM_RESTART" != "none" ]] && [[ "$NUM_RESTART" =~ ^[0-9]+$ ]] && [[ -n "$ITERATIONS_TO_RUN" ]]; then
    MAX_RESTARTS=$NUM_RESTART
    
    # Get job information for restart logic
    scontext=$(scontrol show job $SLURM_JOB_ID 2>/dev/null || echo "")
    timelimit=$(echo "$scontext" | grep -o 'TimeLimit=.*' | awk '{print $1}' | cut -d= -f2)
    
    # Extract SLURM job parameters for restart from scontrol
    account=$(echo "$scontext" | grep -o 'Account=[^ ]*' | cut -d= -f2)
    account=${account:-"aip-gidelgau"}
    
    nodes=$(echo "$scontext" | grep -o 'NumNodes=[^ ]*' | cut -d= -f2)
    nodes=${nodes:-"1"}
    
    gres_raw=$(echo "$scontext" | grep -o 'Gres=[^ ]*' | cut -d= -f2)
    if [ -z "$gres_raw" ]; then
        if [ -n "$SLURM_GPUS_PER_NODE" ]; then
            if [[ "$SLURM_GPUS_PER_NODE" == h100:* ]]; then
                gpus_per_node="$SLURM_GPUS_PER_NODE"
            else
                gpus_per_node="h100:${SLURM_GPUS_PER_NODE}"
            fi
        else
            gpus_per_node="h100:4"
        fi
    else
        gpus_per_node=$(echo "$gres_raw" | sed 's/^gpu://')
    fi
    
    mem=$(echo "$scontext" | grep -o 'MinMemoryCPU=[^ ]*' | cut -d= -f2)
    mem=${mem:-"0"}
    
    job_name=$(echo "$scontext" | grep -o 'JobName=[^ ]*' | cut -d= -f2)
    job_name=${job_name:-""}
    
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
    
    # Call the main BigHead.sh script with filtered arguments (runs in foreground)
    bash scripts/BigHead/Eryngii.sh --iterations-to-run "$ITERATIONS_TO_RUN" "${FILTERED_ARGS[@]}"
    MAIN_EXIT=$?
    
    echo "Main process exited with code: $MAIN_EXIT"
    
    # If process completed successfully (exit code 0) and we haven't reached max restarts, requeue
    if [[ $MAIN_EXIT -eq 0 ]] && [[ $RESTART_COUNT -lt $MAX_RESTARTS ]]; then
        NEW_RESTART_COUNT=$((RESTART_COUNT + 1))
        echo "Process completed successfully, requeuing for next batch (restart $NEW_RESTART_COUNT / $MAX_RESTARTS)"
        sbatch --export=RESTART_COUNT=$NEW_RESTART_COUNT "${SLURM_ARGS[@]}" "$0" --num_restart "$NUM_RESTART" --iterations_to_run "$ITERATIONS_TO_RUN" "${FILTERED_ARGS[@]}"
    elif [[ $RESTART_COUNT -ge $MAX_RESTARTS ]]; then
        echo "Max restarts ($MAX_RESTARTS) reached, not requeuing"
    fi
else
    # No restart logic - call script normally
    if [[ -n "$ITERATIONS_TO_RUN" ]]; then
        bash scripts/BigHead/Eryngii.sh --iterations-to-run "$ITERATIONS_TO_RUN" "${FILTERED_ARGS[@]}"
    else
        bash scripts/BigHead/Eryngii.sh "${FILTERED_ARGS[@]}"
    fi
fi
