#!/bin/bash

torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --n_embd 512 --qkv_dim 64 --n_head 8 --n_layer 6 \
    --mlp_hidden_dim 2048 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --dataset fineweb --iterations 13732 \
    --dropout 0.0 --warmup_steps 200 --grad_clip 2.5 --seed 0 \
    --z_loss_coeff 0.0 \
    --opt adamw --lr 1e-3 --weight_decay 0.001 --scheduler cos \
    --beta1 0.9 --beta2 0.99 --wsd_final_lr_scale 1e2 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115 --latest_ckpt_interval 1000