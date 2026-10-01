#!/bin/bash
cd ~/dana-exp && source ~/jax-blackwell-env/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,WHILE,CONDITIONAL --xla_gpu_graph_min_graph_size=1"
python drivers/e3.py --variant v1 --phase sgd --B 16 --nmax 30000
python drivers/e3.py --variant v1 --phase sgd --B 128 --nmax 15000
python drivers/e3.py --variant v1 --phase sgd --B 512 --nmax 4000
echo E3_SGD_DONE
