#!/bin/bash
# E1 full sweep. 2a>1: B_c = trK/lam1 = O(1): regimes small=2, medium=64, large(<d)=1024, huge(>>d)=32768.
# 2a<1: B_c ~ d^{1-2a}: regimes small=2, critical~B_c, medium=512, huge=32768.
cd ~/dana-exp && source ~/jax-blackwell-env/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,WHILE,CONDITIONAL --xla_gpu_graph_min_graph_size=1"
python drivers/e1_plrf.py --arms shadow --out runs/e1s --alpha 1.0 --beta 0.7 --d 4096 --B 2 --nmax 2000000
python drivers/e1_plrf.py --arms shadow --out runs/e1s --alpha 1.0 --beta 0.7 --d 4096 --B 64 1024 32768 --nmax 1000000
python drivers/e1_plrf.py --arms shadow --out runs/e1s --alpha 0.8 --beta 0.7 --d 4096 --B 2 --nmax 2000000
python drivers/e1_plrf.py --arms shadow --out runs/e1s --alpha 0.8 --beta 0.7 --d 4096 --B 64 1024 32768 --nmax 1000000
python drivers/e1_plrf.py --arms shadow --out runs/e1s --alpha 0.4 --beta 0.5 --d 4096 --v 16384 --B 2 32 512 32768 --nmax 1000000
python drivers/e1_plrf.py --arms shadow --out runs/e1s --alpha 0.3 --beta 0.6 --d 4096 --v 16384 --B 2 128 512 32768 --nmax 1000000
echo E1S_ALL_DONE
