#!/bin/bash

torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --n_embd 512 --qkv_dim 64 --n_head 8 --n_layer 6 \
    --mlp_hidden_dim 2048 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --dataset fineweb --iterations 10681 \
    --dropout 0.0 --warmup_steps 214 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt adamw --lr 1e-3 --weight_decay 1e-3 \
    --beta1 0.9 --beta2 0.999 \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115 --latest_ckpt_interval 1000