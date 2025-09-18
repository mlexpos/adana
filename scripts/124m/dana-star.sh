#!/bin/bash

uv run torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model llama \
    --distributed_backend nccl --compile \
    --n_embd 768 --n_head 12 --n_layer 12 \
    --batch_size 64 --sequence_length 512 --acc_steps 4 \
    --dataset fineweb --iterations 64000 \
    --dropout 0.0  --grad_clip 0.5 --seed 0 \
    --opt dana-star --lr 1e-3 --delta 8 --kappa 0.75 --clipsnr 1.6 \
    --scheduler cos --warmup_steps 2000 \
    --weight_decay 0.1 --wd_decaying --wd_ts 100 \
    --beta1 0.9 --beta2 0.999 \
    --wandb --wandb_project $WANDB_PROJECT  --wandb_entity $WANDB_ENTITY \
    --eval_interval 115 --log_interval 50 --latest_ckpt_interval 1000 \
    


