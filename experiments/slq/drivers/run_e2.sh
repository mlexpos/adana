#!/bin/bash
cd ~/dana-exp && source ~/jax-blackwell-env/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 JAX_DEFAULT_MATMUL_PRECISION=float32
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,WHILE,CONDITIONAL --xla_gpu_graph_min_graph_size=1"
python drivers/e2.py --task nrf --alpha 1.0 --beta 0.7 --d 1024 --v 2048 --B 4 64 1024 8192 --nmax 100000
python drivers/e2.py --task nrf --alpha 0.4 --beta 0.5 --d 1024 --v 4096 --B 4 64 1024 8192 --nmax 100000
python drivers/e2.py --task 2layer --alpha 1.0 --d 256 --v 512 --B 4 64 1024 --nmax 50000
python drivers/e2.py --task 2layer --alpha 0.4 --d 256 --v 512 --B 4 64 1024 --nmax 50000
echo E2_ALL_DONE
