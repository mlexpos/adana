#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=0                    # allocate as needed

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=03c99521910548176ebfa4f418db1c9602e2afa3
export WANDB_PROJECT=danastar
export WANDB_RUN_GROUP=Qwen3Next_MoE_Expert_Parallel
export WANDB_ENTITY=ep-rmt-ml-opt

export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

module load arrow/21.0.0
module load python/3.13

echo "Loaded modules"

source $HOME/danastarenv/bin/activate
echo "Activated virtual environment"

# Set up directories
export DATASETS_DIR="$HOME/scratch/datasets"
export RESULTS_BASE_FOLDER="$HOME/scratch/checkpoints"

echo "Using FineWeb 100BT dataset from: $DATASETS_DIR"
echo "Using checkpoint directory: $RESULTS_BASE_FOLDER"

# Allow wandb to resume runs
# Note: WANDB_DIR is set in main.py to exp_dir to avoid multiple offline-run folders
# export WANDB_RESUME=allow

wandb offline

# Set the restart wrapper script path for the generic restart logic
export RESTART_WRAPPER_SCRIPT="scripts/narval/Qwen3Next_epaq.sh"

# Call the generic Qwen3Next restart script with all arguments
bash scripts/BigHead/Qwen3Next_generic_restart.sh --init-scheme ScaledGPT --results_base_folder "$RESULTS_BASE_FOLDER" "$@"
