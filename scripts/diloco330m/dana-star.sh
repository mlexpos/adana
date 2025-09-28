#!/bin/bash

torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model diloco \
    --distributed_backend nccl --compile \
    --n_embd 1280 --qkv_dim 64 --n_head 20 --n_layer 15 \
    --mlp_hidden_dim 5120 \
    --batch_size 32 --sequence_length 2048 --acc_steps 1 \
    --dataset fineweb --iterations 129312 \
    --dropout 0.0 --grad_clip 2.5 --seed 0 \
    --opt dana-star --lr 5e-4 --delta 8 --kappa 0.75 --clipsnr 1.6 \
    --scheduler cos --warmup_steps 2586 \
    --weight_decay 0.001 --wd_decaying --wd_ts 100 \
    --beta1 0.9 --beta2 0.999 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115 --log_interval 50 --latest_ckpt_interval 1000