#!/bin/bash

# Choices for good scaling: beta_3 has already been set to 1/ iterations and adema_alpha = iterations * (1 + iterations)**(-kappa) for kappa=0.75. There is only to set lr = lr_adam / 2 and wd = wd_adam after having swept adam_w

torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --n_embd 1280 --qkv_dim 64 --n_head 20 --n_layer 15 \
    --mlp_hidden_dim 5120 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --dataset fineweb --iterations 100708 \
    --dropout 0.0 --warmup_steps 2014 --grad_clip 0.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt ademamix --lr 1e-3 --weight_decay 1e-3 \
    --beta1 0.9 --beta2 0.999 \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --adema_beta3 0.9999901 --adema_alpha 17.81 \
    --adema_beta3_warmup 100708 --adema_alpha_warmup 100708 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115 --latest_ckpt_interval 1000 \