#! /bin/bash
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:80GB
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=4

module load anaconda/3
conda activate llm-benchmark

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=llm-optimizer-benchmark
export WANDB_ENTITY=team_damien_frb

# Clean up previous dana-star experiments
rm -rf exps/*dana-star*

chmod +x scripts/124m/dana-star.sh
scripts/124m/dana-star.sh

#test