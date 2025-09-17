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
          0) opt=adamw ;;
          1) opt=dana-star kappa=0.5 ;;
          2) opt=dana-star kappa=0.75 ;;
          3) opt=dana-star kappa=1.0 ;;
        esac
        # Clean up previous dana-star experiments
        rm -rf exps/*$opt*
        
        if [ $opt = "adamw" ]; then
        uv run torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model llama \
            --distributed_backend nccl --compile \
            --n_embd 768 --n_head 12 --n_layer 12 \
            --batch_size 64 --sequence_length 512 --acc_steps 4 \
            --dataset fineweb --iterations 64000 \
            --dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
            --opt adamw --lr 5e-4 --weight_decay 0.1 --scheduler wsd --wsd_fract_decay 0.2 \
            --delta 8 --kappa 0.0 --clipsnr 1.6 \
            --beta1 0.8 --beta2 0.999 --wsd_final_lr_scale 1e-2 \
            --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
            --eval_interval 115 --latest_ckpt_interval 1000 
        fi
        if [ $opt = "dana-star" ]; then
        uv run torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model llama \
            --distributed_backend nccl --compile \
            --n_embd 768 --n_head 12 --n_layer 12 \
            --batch_size 64 --sequence_length 512 --acc_steps 4 \
            --dataset fineweb --iterations 64000 \
            --dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
            --opt dana-star --lr 5e-4 --weight_decay 0.1 --scheduler wsd --wsd_fract_decay 0.2 \
            --delta 8 --kappa $kappa --clipsnr 1.6 \
            --beta1 0.8 --beta2 0.999 --wsd_final_lr_scale 1e-2 \
            --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
            --eval_interval 115 --latest_ckpt_interval 1000 
        fi
        '