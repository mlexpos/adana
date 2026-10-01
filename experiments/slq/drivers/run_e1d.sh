#!/bin/bash
cd ~/dana-exp && source ~/jax-blackwell-env/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,WHILE,CONDITIONAL --xla_gpu_graph_min_graph_size=1"
python drivers/e1_plrf.py --arms lowg2 --out runs/e1d --alpha 0.4 --beta 0.5 --d 4096 --v 16384 --B 2 32 512 32768 --nmax 1000000
python drivers/e1_plrf.py --arms lowg2 --out runs/e1d --alpha 0.3 --beta 0.6 --d 4096 --v 16384 --B 2 128 512 32768 --nmax 1000000
echo E1D_ALL_DONE
