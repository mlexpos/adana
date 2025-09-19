#!/bin/bash
#SBATCH --account=aip-gidelgau
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                    # “alloc as needed” on Alliance

# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=llm-optimizer-benchmark
export WANDB_ENTITY=team_damien_frb

module load arrow/21.0.0
module load python/3.10.13
module load httpproxy
echo "Loaded modules"

source ~/links/projects/aip-gidelgau/dferbach/benchmarking_optimizers/llm/bin/activate
echo "Activated virtual environment"
# MANUAL SWEEP
# Launch four copies in parallel; each sees one GPU
srun --ntasks=4 --cpus-per-task=$SLURM_CPUS_PER_GPU \
     --gpus-per-task=h100:1 --gpu-bind=single:1 --output=logs/%x-%j_%t.out --error=logs/%x-%j_%t.err --exclusive \
     bash -c '
        i=$SLURM_LOCALID                 # 0..3
        case $i in
          0) opt=dana kappa=0.75 delta=8 lr=1e-3 use_grad_ema_for_g2_flag="--use_grad_ema_for_g2" grad_ema_beta=0.9 use_v_ema_flag="--use_v_ema" v_ema_beta=0.99 iterations=64000;;
          1) opt=dana kappa=0.75 delta=8 lr=1e-3 use_grad_ema_for_g2_flag="--use_grad_ema_for_g2" grad_ema_beta=0.9 use_v_ema_flag="--use_v_ema" v_ema_beta=0.999 iterations=64000;;
          2) opt=dana kappa=0.75 delta=8 lr=1e-3 use_grad_ema_for_g2_flag="--use_grad_ema_for_g2" grad_ema_beta=0.9 use_v_ema_flag="--use_v_ema" v_ema_beta=0.9999 iterations=64000;;
          3) opt=dana kappa=0.75 delta=8 lr=1e-3 use_grad_ema_for_g2_flag="" grad_ema_beta=0.9 use_v_ema_flag="--use_v_ema" v_ema_beta=0.999 iterations=64000;;
        esac
        
        # Clean up previous dana-star experiments
        
        rm -rf exps/*$opt*

        uv run torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model llama \
                  --distributed_backend nccl --compile \
                  --n_embd 768 --n_head 12 --n_layer 12 \
                  --batch_size 64 --sequence_length 512 --acc_steps 4 \
                  --dataset fineweb --iterations $iterations \
                  --dropout 0.0  --grad_clip 0.5 --seed 0 \
                  --opt $opt --lr $lr --delta $delta --kappa $kappa \
                  --scheduler cos --warmup_steps 2000 \
                  --weight_decay 0.1 $use_grad_ema_for_g2_flag --grad_ema_beta $grad_ema_beta \
                  $use_v_ema_flag --v_ema_beta $v_ema_beta \
                  --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
                  --eval_interval 115 --log_interval 50 --latest_ckpt_interval 1000
        '