#!/bin/bash

# Enoki Narval wrapper script with no QK normalization
# This script sets environment variables (including WANDB_RUN_GROUP) and calls the generic restart script
# SLURM directives are passed via sbatch command in the launch script

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=03c99521910548176ebfa4f418db1c9602e2afa3
export WANDB_PROJECT=danastar
export WANDB_RUN_GROUP=Enoki_ScaledGPT_noqk
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

wandb offline

# Set the restart wrapper script path for the generic restart logic
export RESTART_WRAPPER_SCRIPT="scripts/narval/narval_Enoki_noqk.sh"

# Call the generic Enoki restart script with all arguments, forcing ScaledGPT init and --no-qknorm
bash scripts/BigHead/Enoki_generic_restart.sh --init-scheme ScaledGPT --results_base_folder "$RESULTS_BASE_FOLDER" --no-qknorm "$@"
