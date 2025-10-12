#!/bin/bash

# Choices for good scaling: there is only to set lr = lr_adam / 2

torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --n_embd 384 --qkv_dim 64 --n_head 6 --n_layer 4 \
    --mlp_hidden_dim 1536 \
    --datasets_dir "$DATASETS_DIR" --dataset fineweb \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --iterations 13953 \
    --dropout 0.0 --warmup_steps 280 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt dana-star-mk4 --lr 16e-4 --delta 8 --clipsnr 1.0 \
    --weight_decay 1.0 --wd_decaying --wd_ts 1000 \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115 --log_interval 50