#!/bin/bash

# Enoki Dana-Star-No-Tau-Kappa-0-8 ScaledGPT Initialization Single-GPU Sweep for Narval
# Uses ScaledGPT initialization scheme with 1 GPU
# For each head count, runs multiple learning rates: multipliers of the formula prediction
# Learning rate formula: lr = 1.34e+01 \times (7.04e03 + P)^{-0.627} where P = NON_EMB
# Enoki scaling: head_dim=64 (fixed), n_layer=3*heads/4, n_embd=64*heads, mlp=4*n_embd

OMEGA_ARRAY=( 4.0 )
HEADS_ARRAY=( 20 22 24 )
LR_MULTIPLIERS=( 1.0 1.25 1.5 0.75 0.5 0.3 3.0)

# SLURM configuration for Narval and batch size/accumulation steps: they are erased later accounting for the number of heads
BATCH_SIZE=32
ACC_STEPS=1
GPUS_PER_NODE=1
CPUS_PER_GPU=4
TOTAL_CPUS=4
MEM=64GB          # 0 = allocate as needed
TIME_HOURS=24

# ScaledGPT initialization parameters
INIT_SCHEME="ScaledGPT"
DEPTH_SCALAR_EXPONENT=0.0

echo "Starting Enoki Dana-Star-No-Tau-Kappa-0-8 ScaledGPT Initialization sweep (Narval)"
echo "Head counts: ${HEADS_ARRAY[@]}"
echo "Omega values: ${OMEGA_ARRAY[@]}"
echo "LR multipliers: ${LR_MULTIPLIERS[@]}"
echo "Init scheme: $INIT_SCHEME"
echo "Depth scalar exponent: $DEPTH_SCALAR_EXPONENT"
echo ""

# Function to calculate model parameters for a given head count
calculate_params() {
    local HEADS=$1

    # Enoki architecture parameters (DiLoco scaling)
    local HEAD_DIM=64
    local N_HEAD=$(python3 -c "print(int($HEADS))")
    local N_LAYER=$(python3 -c "print(int(3 * $HEADS // 4))")
    local N_EMBD=$(python3 -c "print(int($HEADS * 64))")
    local MLP_HIDDEN=$(python3 -c "print(int(4 * $N_EMBD))")

    # Calculate non-embedding parameters (DiLoco formula)
    # Non-emb = 12 * n_embd^2 * n_layer
    local NON_EMB=$(python3 -c "print(int(12 * $N_EMBD * $N_EMBD * $N_LAYER))")

    # Calculate total parameters and iterations
    local TOTAL_PARAMS=$(python3 -c "print(int($NON_EMB + 2 * $N_EMBD * 50304))")
    local ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / 65536))")

    echo "$NON_EMB $ITERATIONS"
}

# Calculate reference for heads=16
read NON_EMB_16 ITERATIONS_16 <<< $(calculate_params 16)
C_16=$(python3 -c "print($NON_EMB_16 * $ITERATIONS_16)")

echo "Reference (heads=16):"
echo "  NON_EMB = $NON_EMB_16"
echo "  ITERATIONS = $ITERATIONS_16"
echo "  C(16) = $C_16"
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
        read NON_EMB ITERATIONS <<< $(calculate_params $HEADS)

        # Calculate computational cost C = NON_EMB * ITERATIONS
        C=$(python3 -c "print($NON_EMB * $ITERATIONS)")

        # Calculate base learning rate using formula: lr = 1.34e+01 \times (7.04e03 + P)^{-0.627}
        BASE_LR=$(python3 -c "print(1.34e+01 * (7.04e03 + $NON_EMB) ** -0.627)")

        # Calculate n_layer for this head count
        N_LAYER=$(python3 -c "print(int(3 * $HEADS // 4))")

        echo "    HEADS = $HEADS"
        echo "    N_LAYER = $N_LAYER (= 3 * $HEADS / 4)"
        echo "    NON_EMB = $NON_EMB"
        echo "    ITERATIONS = $ITERATIONS"
        echo "    C = $C"
        echo "    Time allocation: ${TIME_HOURS}h"
        echo "    Base LR (formula): $BASE_LR"
        echo ""

        # Loop over learning rate multipliers
        for MULT in "${LR_MULTIPLIERS[@]}"; do
            # Calculate actual learning rate
            LR=$(python3 -c "print($MULT * $BASE_LR)")

            job_count=$((job_count + 1))
            echo "    Job $job_count/$total_jobs: omega=$OMEGA, heads=$HEADS, lr=$LR (${MULT}x base)"

            if [ $HEADS -le 14 ]; then
                TIME_HOURS=24
                BATCH_SIZE=32
                ACC_STEPS=1
                GPUS_PER_NODE=1
                CPUS_PER_GPU=4
                TOTAL_CPUS=4
                MEM=64GB
            elif [ $HEADS -le 18 ]; then
                TIME_HOURS=24
                BATCH_SIZE=16
                ACC_STEPS=2
                GPUS_PER_NODE=1
                CPUS_PER_GPU=4
                TOTAL_CPUS=4
                MEM=64GB
            elif [ $HEADS -le 20 ]; then
                TIME_HOURS=24
                BATCH_SIZE=1
                ACC_STEPS=32
                GPUS_PER_NODE=4
                CPUS_PER_GPU=4
                TOTAL_CPUS=16
                MEM=128GB
            else
                TIME_HOURS=48
                BATCH_SIZE=1
                ACC_STEPS=32
                GPUS_PER_NODE=4
                CPUS_PER_GPU=4
                TOTAL_CPUS=16
                MEM=128GB
            fi

            echo "Batch size: $BATCH_SIZE"
            echo "Accumulation steps: $ACC_STEPS"
            echo "Effective batch size: $((BATCH_SIZE * ACC_STEPS))"
            echo "GPUs per node: $GPUS_PER_NODE"
            echo "CPUs per GPU: $CPUS_PER_GPU"
            echo "Total CPUs: $TOTAL_CPUS"
            echo "Memory: ${MEM} (allocate as needed)"
            echo "Time allocation: ${TIME_HOURS}h"

            # Submit the job with single-GPU configuration and ScaledGPT initialization
            sbatch --time=${TIME_HOURS}:00:00 \
                   --gpus-per-node=a100:${GPUS_PER_NODE} \
                   --cpus-per-gpu=${CPUS_PER_GPU} \
                   --mem=${MEM} \
                   --job-name=EN_Dana-Star-No-Tau-Kappa-0-8_SGPT_om${OMEGA}_h${HEADS}_lr${MULT} \
                   scripts/scripts_dfer/Enoki_scaledGPT/narval_Enoki_dfer.sh \
                   --heads $HEADS \
                   --lr $LR \
                   --omega $OMEGA \
                   --batch_size $BATCH_SIZE \
                   --acc_steps $ACC_STEPS \
                   --optimizer dana-star-no-tau-kappa-0-8 \
                   --nproc_per_node ${GPUS_PER_NODE} \
                   --depth-scalar-exponent $DEPTH_SCALAR_EXPONENT

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
echo "  Omega values: ${OMEGA_ARRAY[@]}"
echo "  Head counts: ${HEADS_ARRAY[@]}"
echo "  LR multipliers: ${LR_MULTIPLIERS[@]}"
echo ""
echo "Resource allocation per job:"
echo "  Nodes: 1"
echo "  GPUs: ${GPUS_PER_NODE} × A100"
echo "  CPUs: ${TOTAL_CPUS} (${CPUS_PER_GPU} per GPU)"
echo "  Memory: ${MEM} (allocate as needed)"
echo "  Time: ${TIME_HOURS} hours"
echo "  Init scheme: ${INIT_SCHEME}"
echo "  Depth scalar exponent: ${DEPTH_SCALAR_EXPONENT}"
