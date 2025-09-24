#!/bin/bash

# Choices for good scaling: there is only to set lr = lr_adam / 2 and wd = wd_adam after having swept adam_w

torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --n_embd 1024 --qkv_dim 64 --n_head 16 --n_layer 12 \
    --mlp_hidden_dim 4096 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --dataset fineweb --iterations 77527 \
    --dropout 0.0 --warmup_steps 1551 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt dana --lr 1e-3 --delta 8 --kappa 0.75 --weight_decay 1e-3 \
    --use_grad_ema_for_g2 --grad_ema_beta 0.9 --use_v_ema --v_ema_beta 0.999 \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115 \