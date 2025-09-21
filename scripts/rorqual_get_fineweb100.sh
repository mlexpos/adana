#!/bin/bash
#SBATCH --job-name=rorqual_get_fineweb100
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=650GB                # "alloc as needed" on Alliance

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=03c99521910548176ebfa4f418db1c9602e2afa3
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt

export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

module load arrow/21.0.0
module load python/3.13

echo "Loaded modules"

source $HOME/danastarenv/bin/activate
echo "Activated virtual environment"

# Set up directories
FINEWEB_DIR="$HOME/links/projects/def-epaq/fineweb"
DATASETS_DIR="$HOME/links/scratch/datasets"

echo "Processing FineWeb 100BT sample data..."
echo "Source directory: $FINEWEB_DIR"
echo "Output directory: $DATASETS_DIR"

# Run the tokenization script
# Using 40 processes (default) and larger batch size for efficiency
python src/data/fineweb_100.py \
    --datasets-dir "$DATASETS_DIR" \
    --fineweb-dir "$FINEWEB_DIR" \
    --num-proc 192

echo "Tokenization completed!" 