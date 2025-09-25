#!/bin/bash

torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --n_embd 1536 --qkv_dim 64 --n_head 24 --n_layer 18 \
    --mlp_hidden_dim 6144 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --dataset fineweb --iterations 202698 \
    --dropout 0.0 --grad_clip 0.5 --seed 0 \
    --opt dana-star --lr 5e-4 --delta 8 --kappa 0.75 --clipsnr 2.0 \
    --warmup_steps 4054 \
    --weight_decay 0.001 --wd_decaying --wd_ts 100 \
    --scheduler cos_inf --cos_inf_steps 0 --div_factor 1e2 --final_div_factor 1e-1 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115 --log_interval 50