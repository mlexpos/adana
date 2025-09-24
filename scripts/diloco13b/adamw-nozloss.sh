#!/bin/bash

torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --n_embd 2048 --qkv_dim 64 --n_head 32 --n_layer 24 \
    --mlp_hidden_dim 8192 \
    --batch_size 4 --sequence_length 2048 --acc_steps 8 \
    --dataset fineweb --iterations 400800 \
    --dropout 0.0 --warmup_steps 8016 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt adamw --lr 1e-3 --weight_decay 0.001 \
    --beta1 0.9 --beta2 0.999 \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115