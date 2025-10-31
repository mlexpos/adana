# Hugging Face caches
export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
export WANDB_PROJECT=danastar
export WANDB_ENTITY=ep-rmt-ml-opt
export WANDB_RUN_GROUP=test_checkpointing
export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

module load arrow/21.0.0
module load python/3.10.13
module load httpproxy
echo "Loaded modules"

source ~/links/projects/aip-gidelgau/dferbach/benchmarking_optimizers/llm/bin/activate
echo "Activated virtual environment"

DATASETS_DIR="$HOME/links/scratch/fineweb"

torchrun --standalone --nproc_per_node=4 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --n_embd 384 --qkv_dim 64 --n_head 6 --n_layer 4 \
    --mlp_hidden_dim 1536 \
    --datasets_dir "$DATASETS_DIR" --dataset fineweb_100 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --iterations 2000 \
    --dropout 0.0 --warmup_steps 279 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt adamw --lr 1e-3 --weight_decay 1e-3 \
    --beta1 0.9 --beta2 0.999 \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 100 --latest_ckpt_interval 100 --auto_resume
