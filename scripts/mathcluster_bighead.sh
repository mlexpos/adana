#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro 
#SBATCH --gpus=1
#SBATCH --propagate=NONE 
#SBATCH --account=paquettee-2025
#SBATCH -o logs/slurm.%N.%j.out # STDOUT
#SBATCH -e logs/slurm.%N.%j.err # STDERR
#SBATCH --signal=B:TERM@60

#export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=03c99521910548176ebfa4f418db1c9602e2afa3
export WANDB_PROJECT=danastar
export WANDB_ENTITY="ep-rmt-ml-opt"
export DATASETS_DIR="/home/math/elliot.paquette@MCGILL.CA/danastar/src/data/datasets"

#module load cuda/cuda-12.6
module load miniconda/miniconda-winter2025

#virtualenv --no-download /home/c/cypaquet/jaxenv
#source /home/c/cypaquet/jaxenv/bin/activate
#pip install --no-index jax[cuda12] flax pandas optax matplotlib tiktoken huggingface-hub
#pip install --upgrade "jax[cuda12]"
#pip install flax pandas optax matplotlib tiktoken huggingface-hub

cd /home/math/elliot.paquette@MCGILL.CA/danastar/


cd /home/math/elliot.paquette@MCGILL.CA/danastar/

bash ./scripts/BigHead/BigHead.sh --depth 4 --lr 1e-4 --omega 4.0 --clipsnr 2.0 --optimizer dana-star-mk4