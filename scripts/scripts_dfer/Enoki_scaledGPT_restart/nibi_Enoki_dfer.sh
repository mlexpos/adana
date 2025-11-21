#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                 # "alloc as needed" on Alliance

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt
export WANDB_RUN_GROUP=Enoki_ScaledGPT
export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

module load arrow/21.0.0
module load python/3.10.13
#module load httpproxy
echo "Loaded modules"

source ~/danastar/llm/bin/activate
echo "Activated virtual environment"

export DATASETS_DIR="$HOME/scratch/"

echo "Using FineWeb 100BT dataset from: $DATASETS_DIR"

# Call the main Enoki.sh script with all arguments and force ScaledGPT init scheme
bash scripts/BigHead/Enoki.sh --init-scheme ScaledGPT "$@"
