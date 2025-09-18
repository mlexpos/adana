#! /bin/bash
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:05:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:80GB
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=4

source ~/projects/llm-optimizer-benchmark/llm/bin/activate

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=llm-optimizer-benchmark
export WANDB_ENTITY=team_damien_frb

opt=dana
# Clean up previous experiments
rm -rf exps/*$opt*

chmod +x scripts/124m/$opt.sh
scripts/124m/$opt.sh