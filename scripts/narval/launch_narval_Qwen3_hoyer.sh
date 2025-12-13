#!/bin/bash

# Qwen3 Hoyer Loss Learning Rate Sweep for Narval
# Uses ScaledGPT initialization with fixed hoyer_loss_coeff=1e-4
# Sweeps learning rate with tau statistics collection enabled
# QK normalization: DISABLED
# For each head count, runs multiple learning rates: multipliers of the formula prediction
# Learning rate formula: lr = 3.72e+03 × (3.70e+03 + P)^-0.894 where P = NON_EMB
# Qwen3 scaling: head_dim=128 (fixed), n_layer=2*heads, n_embd=128*heads, mlp=3*n_embd


OMEGA_ARRAY=( 4.0 )
HEADS_ARRAY=( 4 5 )
LR_MULTIPLIERS=( 1.0 2.0 0.5 4.0 0.25 0.125 8.0 )
CLIPSNR=2.0
BATCH_SIZE=32
ACC_STEPS=1
HOYER_LOSS_COEFF=1e-4  # Fixed Hoyer loss coefficient

# SLURM configuration for Narval (1 A100 GPU)
GPUS_PER_NODE=1
CPUS_PER_GPU=8
TOTAL_CPUS=8             # 1 GPU × 8 CPUs/GPU
MEM=0                    # 0 = allocate as needed
TIME_HOURS=12

# ScaledGPT initialization parameters
INIT_SCHEME="ScaledGPT"
DEPTH_SCALAR_EXPONENT=0.0
ITERATIONS_TO_RUN=100000

# QK normalization (disabled for this sweep)
NO_QKNORM=true

echo "Starting Qwen3 Hoyer Loss LR Sweep (Narval)"
echo "Head counts: ${HEADS_ARRAY[@]}"
echo "Omega values: ${OMEGA_ARRAY[@]}"
echo "LR multipliers: ${LR_MULTIPLIERS[@]}"
echo "Hoyer loss coefficient: $HOYER_LOSS_COEFF (fixed)"
echo "Clip SNR: $CLIPSNR"
echo "QK Normalization: DISABLED"
echo "Tau stats collection: ENABLED"
echo "GPUs per node: $GPUS_PER_NODE"
echo "CPUs per GPU: $CPUS_PER_GPU"
echo "Total CPUs: $TOTAL_CPUS"
echo "Memory: ${MEM} (allocate as needed)"
echo "Time allocation: ${TIME_HOURS}h"
echo "Init scheme: $INIT_SCHEME"
echo "Depth scalar exponent: $DEPTH_SCALAR_EXPONENT"
echo "Iterations to run: $ITERATIONS_TO_RUN"
echo ""

# Function to calculate model parameters for a given head count
calculate_params() {
    local HEADS=$1

    # Qwen3 architecture parameters
    local HEAD_DIM=128
    local N_HEAD=$(python3 -c "print(int($HEADS))")
    local N_LAYER=$(python3 -c "print(int(2 * $HEADS))")
    local N_EMBD=$(python3 -c "print(int($HEADS * 128))")
    local MLP_HIDDEN=$(python3 -c "print(int(3 * $N_EMBD))")

    # Calculate non-embedding parameters for Qwen3
    # With gating: non_emb = n_layer * (5 * n_embd * total_qkv_dim + 2 * head_dim + 9 * n_embd^2 + 2 * n_embd) + n_embd
    local TOTAL_QKV_DIM=$(python3 -c "print(int($N_HEAD * $HEAD_DIM))")

    local NON_EMB=$(python3 -c "
n_layer = $N_LAYER
n_embd = $N_EMBD
head_dim = $HEAD_DIM
total_qkv_dim = $TOTAL_QKV_DIM

# Per layer
attn = 5 * n_embd * total_qkv_dim  # q_proj (2x with gating) + k_proj + v_proj + o_proj
qk_norm = 2 * head_dim
mlp = 9 * n_embd * n_embd  # gate_proj + up_proj + down_proj (mlp_hidden = 3 * n_embd)
layer_norms = 2 * n_embd

per_layer = attn + qk_norm + mlp + layer_norms

# Total
non_emb = n_layer * per_layer + n_embd  # +n_embd for final norm

print(int(non_emb))
")

    # Calculate total parameters and iterations
    local TOTAL_PARAMS=$(python3 -c "print(int($NON_EMB + 2 * $N_EMBD * 50304))")
    local ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / 65536))")

    echo "$NON_EMB $ITERATIONS $TOTAL_PARAMS"
}

# Calculate reference for heads=4
read NON_EMB_4 ITERATIONS_4 TOTAL_PARAMS_4 <<< $(calculate_params 4)
C_4=$(python3 -c "print($NON_EMB_4 * $ITERATIONS_4)")

echo "Reference (heads=4):"
echo "  NON_EMB = $NON_EMB_4"
echo "  ITERATIONS = $ITERATIONS_4"
echo "  TOTAL_PARAMS = $TOTAL_PARAMS_4"
echo "  C(4) = $C_4"
echo ""

# Counter for job tracking
job_count=0
total_jobs=$((${#OMEGA_ARRAY[@]} * ${#HEADS_ARRAY[@]} * ${#LR_MULTIPLIERS[@]}))
echo "Total jobs to run: $total_jobs"
echo ""

# Loop over omega values
for OMEGA in "${OMEGA_ARRAY[@]}"; do
    echo "Processing omega=$OMEGA"
    echo ""

    # Loop over head counts
    for HEADS in "${HEADS_ARRAY[@]}"; do
        echo "  Processing heads=$HEADS (omega=$OMEGA)"

        # Calculate parameters for this head count
        read NON_EMB ITERATIONS TOTAL_PARAMS <<< $(calculate_params $HEADS)

        # Calculate computational cost C = NON_EMB * ITERATIONS
        C=$(python3 -c "print($NON_EMB * $ITERATIONS)")

        # Calculate base learning rate using formula: lr = 3.72e+03 × (3.70e+03 + P)^-0.894
        BASE_LR=$(python3 -c "print(3.72e3 * ((3.70e3 + $NON_EMB) ** -0.894))")

        # Calculate n_layer for this head count
        N_LAYER=$(python3 -c "print(int(2 * $HEADS))")

        echo "    HEADS = $HEADS"
        echo "    N_LAYER = $N_LAYER (= 2 * $HEADS)"
        echo "    NON_EMB = $NON_EMB"
        echo "    ITERATIONS = $ITERATIONS"
        echo "    C = $C"
        echo "    TOTAL_PARAMS = $TOTAL_PARAMS"
        echo "    Time allocation: ${TIME_HOURS}h"
        echo "    Base LR (formula): $BASE_LR"
        echo "    Hoyer loss coeff: $HOYER_LOSS_COEFF"
        echo ""

        echo "    BATCH_SIZE = $BATCH_SIZE"
        echo "    ACC_STEPS = $ACC_STEPS"
        echo "    Effective batch size = $((BATCH_SIZE * ACC_STEPS))"
        echo ""

        # Loop over learning rate multipliers
        for MULT in "${LR_MULTIPLIERS[@]}"; do
            # Calculate actual learning rate
            LR=$(python3 -c "print($MULT * $BASE_LR)")

            job_count=$((job_count + 1))
            echo "    Job $job_count/$total_jobs: omega=$OMEGA, heads=$HEADS, lr=$LR (${MULT}x base), hoyer=$HOYER_LOSS_COEFF"

            # Submit the job with ScaledGPT initialization, Hoyer loss, and tau stats
            sbatch --time=${TIME_HOURS}:00:00 \
                   --nodes=1 \
                   --gpus-per-node=a100:${GPUS_PER_NODE} \
                   --cpus-per-gpu=${CPUS_PER_GPU} \
                   --mem=${MEM} \
                   --job-name=Q3_Hoyer_om${OMEGA}_h${HEADS}_lr${MULT} \
                   scripts/narval/Qwen3_hoyer.sh \
                   --heads $HEADS \
                   --lr $LR \
                   --omega $OMEGA \
                   --optimizer dana-star-mk4 \
                   --kappa 0.75 \
                   --batch_size $BATCH_SIZE \
                   --acc_steps $ACC_STEPS \
                   --clipsnr $CLIPSNR \
                   --hoyer_loss_coeff $HOYER_LOSS_COEFF \
                   --nproc_per_node ${GPUS_PER_NODE} \
                   --depth-scalar-exponent $DEPTH_SCALAR_EXPONENT \
                   --iterations_to_run $ITERATIONS_TO_RUN \
                   --no-qknorm

            # Check if the job was successful
            if [ $? -eq 0 ]; then
                echo "      ✓ Job submitted successfully"
            else
                echo "      ✗ Job failed with exit code $?"
            fi

            echo ""
        done

        echo "  ----------------------------------------"
    done

    echo "========================================"
done

echo "Sweep completed. Total jobs submitted: $job_count"
echo ""
echo "Sweep configuration:"
echo "  Optimizer: Dana-star-mk4"
echo "  Model: Qwen3"
echo "  Omega values: ${OMEGA_ARRAY[@]}"
echo "  Head counts: ${HEADS_ARRAY[@]}"
echo "  LR multipliers: ${LR_MULTIPLIERS[@]}"
echo "  LR formula: lr = 3.72e+03 × (3.70e+03 + NON_EMB)^-0.894"
echo "  Hoyer loss coefficient: $HOYER_LOSS_COEFF (fixed)"
echo "  Clip SNR: $CLIPSNR"
echo "  QK Normalization: DISABLED"
echo "  Tau stats collection: ENABLED"
echo ""
echo "Resource allocation per job:"
echo "  Nodes: 1"
echo "  GPUs: ${GPUS_PER_NODE} × A100"
echo "  CPUs: ${TOTAL_CPUS} (${CPUS_PER_GPU} per GPU)"
echo "  Memory: ${MEM} (allocate as needed)"
echo "  Time: ${TIME_HOURS} hours"
echo "  Init scheme: ${INIT_SCHEME}"
echo "  Depth scalar exponent: ${DEPTH_SCALAR_EXPONENT}"
echo ""
echo "Qwen3 architecture scaling:"
echo "  head_dim = 128 (fixed)"
echo "  n_layer = 2 × heads"
echo "  n_embd = 128 × heads"
echo "  mlp_hidden = 3 × n_embd"
echo "  Elementwise attention gating: enabled"
echo ""
