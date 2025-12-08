#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=40GB
#SBATCH --job-name=hummingbird_dana_star

# Load modules
module load arrow/21.0.0
module load python/3.13

echo "Loaded modules"

# Activate virtual environment
source $HOME/jaxenv/bin/activate
echo "Activated virtual environment"

# Change to the JAX directory
cd $HOME/danastar/jax

# Create results directory if it doesn't exist
mkdir -p results

# Run hummingbird plot script
python hummingbird_plot.py \
    --clipsnr 1.0 \
    --delta 6.0 \
    --steps 10000 \
    --tanea_lr_scalar 0.01 \
    --adam_lr 0.001 \
    --batch_size 1000 \
    --optimizer dana-star \
    --results_dir results \
    --output_prefix hummingbird

echo "Hummingbird plot completed"
