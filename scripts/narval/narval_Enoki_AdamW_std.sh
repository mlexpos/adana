#!/bin/bash

# Enoki AdamW Standard Initialization Single-GPU Sweep for Narval
# Uses Standard initialization scheme with 1 GPU
# For each head count, runs multiple learning rates: multipliers of the formula prediction
# Learning rate formula: lr = 2.07e-06 + 1.35e+03 * P^{-0.779} where P = NON_EMB
# Enoki scaling: head_dim=64 (fixed), n_layer=3*heads/4, n_embd=64*heads, mlp=4*n_embd

OMEGA_ARRAY=( 2.0 1.0 0.5 0.25 0.125 0.0625 )
HEADS_ARRAY=( 8 )
LR_MULTIPLIERS=( 4.0 6.0 8.0 10.0 12.0 20.0)

BATCH_SIZE=32
ACC_STEPS=1

# SLURM configuration for Narval (1 GPU)
GPUS_PER_NODE=1
CPUS_PER_GPU=8
TOTAL_CPUS=8
MEM=80GB          # 0 = allocate as needed
TIME_HOURS=2

# Standard initialization parameters
INIT_SCHEME="Standard"
DEPTH_SCALAR_EXPONENT=-0.5

echo "Starting Enoki AdamW Standard Initialization sweep (Narval)"
echo "Head counts: ${HEADS_ARRAY[@]}"
echo "Omega values: ${OMEGA_ARRAY[@]}"
echo "LR multipliers: ${LR_MULTIPLIERS[@]}"
echo "Batch size: $BATCH_SIZE"
echo "Accumulation steps: $ACC_STEPS"
echo "Effective batch size: $((BATCH_SIZE * ACC_STEPS))"
echo "GPUs per node: $GPUS_PER_NODE"
echo "CPUs per GPU: $CPUS_PER_GPU"
echo "Total CPUs: $TOTAL_CPUS"
echo "Memory: ${MEM} (allocate as needed)"
echo "Time allocation: ${TIME_HOURS}h"
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

        # Calculate base learning rate using formula: lr = 2.07e-06 + 1.35e+03 * P^{-0.779}
        BASE_LR=$(python3 -c "print(2.07e-06 + 1.35e+03 * ($NON_EMB ** -0.779))")

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

            # Submit the job with single-GPU configuration and Standard initialization
            sbatch --time=${TIME_HOURS}:00:00 \
                   --nodes=1 \
                   --gpus-per-node=a100:${GPUS_PER_NODE} \
                   --cpus-per-gpu=${CPUS_PER_GPU} \
                   --mem=${MEM} \
                   --job-name=EN_AW_STD_om${OMEGA}_h${HEADS}_lr${MULT} \
                   scripts/narval/Enoki_cypaq_std.sh \
                   --heads $HEADS \
                   --lr $LR \
                   --omega $OMEGA \
                   --batch_size $BATCH_SIZE \
                   --acc_steps $ACC_STEPS \
                   --optimizer adamw \
                   --nproc_per_node ${GPUS_PER_NODE} \
                   --init-scheme $INIT_SCHEME \
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
