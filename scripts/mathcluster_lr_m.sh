#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH --gpus=1
#SBATCH --propagate=NONE
#SBATCH --account=paquettec-2025
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --signal=B:TERM@60

#module load cuda/cuda-12.6
module load miniconda/miniconda-winter2025

#virtualenv --no-download $HOME/jaxenv
#source $HOME/jaxenv/bin/activate
#pip install --no-index jax[cuda12] flax pandas optax matplotlib tiktoken huggingface-hub
#pip install --upgrade "jax[cuda12]"
#pip install flax pandas optax matplotlib tiktoken huggingface-hub

cd $HOME/danastar/jax

# Configuration - modify these values as needed
M_VALUES="1,100,10000"  # Comma-separated list of m values (number of classes)
ZETA_VALUE="1.2"  # Single zeta value (class frequency decay exponent)

cd $HOME/danastar/jax

python lr_m_sweeps.py \
    --m_range "$M_VALUES" \
    --zeta "$ZETA_VALUE" \
    --alpha 1.3 \
    --beta 1.3 \
    --v 2000 \
    --d 500 \
    --steps 1000000 \
    --batch_size 100 \
    --g2_scale 0.001875 \
    --g3_over_g2 1.0 \
    --tanea_lr_scalar 1.0 \
    --tanea_global_exponent 0.0 \
    --tanea_kappa 0.75 \
    --adam_beta2 0.95 \
    --adam_beta1 0.9 \
    --adam_lr 0.0035 \
    --adam_star_lr 0.0035 \
    --long_adam_lr 0.0035 \
    --enable_adam \
    --enable_long_adam \
    --enable_adam_star \
    --enable_dana_star \
    --disable_long_adam_nesterov \
    --disable_adam_nesterov_star \
    --random_seed 42 \
    --results_dir $HOME/danastarresults