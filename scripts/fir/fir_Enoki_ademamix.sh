#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=0                    # "alloc as needed" on Alliance

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=03c99521910548176ebfa4f418db1c9602e2afa3
export WANDB_PROJECT=danastar
export WANDB_RUN_GROUP=Enoki_ScaledGPT_AdeMaMix
export WANDB_ENTITY=ep-rmt-ml-opt

export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

module load arrow/21.0.0
module load python/3.13

echo "Loaded modules"

source $HOME/danastarenv/bin/activate
echo "Activated virtual environment"

# Set up directories
export DATASETS_DIR="$HOME/links/scratch/datasets"

echo "Using FineWeb 100BT dataset from: $DATASETS_DIR"

wandb offline

# Set the restart wrapper script path for the generic restart logic
export RESTART_WRAPPER_SCRIPT="scripts/fir/fir_Enoki_ademamix.sh"

# Call the generic Enoki restart script with all arguments and force ScaledGPT init scheme
bash scripts/BigHead/Enoki_generic_restart.sh --init-scheme ScaledGPT "$@"
