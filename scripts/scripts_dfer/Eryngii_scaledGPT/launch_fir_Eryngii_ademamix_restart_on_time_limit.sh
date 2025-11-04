#!/bin/bash

# Eryngii with ScaledGPT initialization for AdamW Multi-GPU Sweep on Fir
# Uses 4 GPUs per node for larger models
# For each n_head value, runs multiple learning rates: multipliers of base LR
# Base learning rate: lr =  6.48e+02 × (5.67e + 06 + P)^ -0.776

OMEGA=4.0
HEADS=(16)
LR_MULTIPLIERS=(1.0)

# SLURM configuration for Fir
GPUS_PER_NODE=4
CPUS_PER_GPU=12
TOTAL_CPUS=48  # 4 GPUs × 12 CPUs/GPU
MEM=0          # 0 = allocate as needed
TIME_HOURS=24
RESTART_ON_TIME_LIMIT= 7

echo "Starting Eryngii Ademamix Multi-GPU sweep (Fir) with restart on time limit"
echo "Heads: ${HEADS[@]}"
echo "Omega: $OMEGA"
echo "LR multipliers: ${LR_MULTIPLIERS[@]}"
echo "GPUs per node: $GPUS_PER_NODE"
echo "CPUs per GPU: $CPUS_PER_GPU"
echo "Total CPUs: $TOTAL_CPUS"
echo "Memory: ${MEM} (allocate as needed)"
echo "Time allocation: ${TIME_HOURS}h"
echo "Restart on time limit: ${RESTART_ON_TIME_LIMIT}"
echo ""

# Function to calculate model parameters for Eryngii
calculate_params() {
    local HEADS=$1

    # Eryngii architecture parameters
    local HEAD_DIM=$(python3 -c "print(int(round(32 * $HEADS / 3 / 8) * 8))")
    local N_HEAD=$(python3 -c "print(int($HEADS))")
    local N_LAYER=$(python3 -c "print(int($HEADS**2 // 8))")
    local N_EMBD=$(python3 -c "print(int($N_HEAD * $HEAD_DIM))")
    local MLP_HIDDEN=$(python3 -c "print(int(4 * $N_EMBD))")

    # Calculate non-embedding parameters
    # Non-emb = 12 * n_embd^2 * n_layer (DiLoco formula)
    local NON_EMB=$(python3 -c "print(int(12 * $N_EMBD * $N_EMBD * $N_LAYER))")

    # Calculate total parameters and iterations
    local TOTAL_PARAMS=$(python3 -c "print(int($NON_EMB + 2 * $N_EMBD * 50304))")
    local ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / 65536))")

    echo "$NON_EMB $ITERATIONS"
}

# Calculate reference for heads=4
read NON_EMB_4 ITERATIONS_4 <<< $(calculate_params 4)

# Counter for job tracking
job_count=0
total_jobs=$((${#HEADS[@]} * ${#LR_MULTIPLIERS[@]}))
echo "Total jobs to run: $total_jobs"
echo ""

# Loop over heads
for HEADS in "${HEADS[@]}"; do
    echo "Processing heads=$HEADS"

    # Set time allocation based on heads
    if [ $HEADS -le 6 ]; then
        TIME_SPEC="00:30:00"
    elif [ $HEADS -le 9 ]; then
        TIME_SPEC="03:00:00"
    elif [ $HEADS -le 11 ]; then
        TIME_SPEC="12:00:00"
    elif [ $HEADS -le 12 ]; then
        TIME_SPEC="24:00:00"
    else
        TIME_SPEC="24:00:00"
    fi

    # Calculate parameters for this heads
    read NON_EMB ITERATIONS <<< $(calculate_params $HEADS)

    # Calculate computational cost C = NON_EMB * ITERATIONS
    C=$(python3 -c "print($NON_EMB * $ITERATIONS * 6 * 2048 * 32 / (8.64e19))")

    # Base learning rate
    BASE_LR=$(python3 -c "print(6.48e+02 * (5.67e+06 + $NON_EMB)**(-0.776))")

    echo "  NON_EMB = $NON_EMB"
    echo "  ITERATIONS = $ITERATIONS"
    echo "  C = $(python3 -c "print($C)") PetaFLOPS Days"
    echo "  Time allocation: ${TIME_SPEC}"
    echo "  Base LR: $BASE_LR (Formula: lr = 6.48e+02 × (5.67e + 06 + P)^ -0.776)"
    echo ""

    # Loop over learning rate multipliers
    for MULT in "${LR_MULTIPLIERS[@]}"; do
        # Calculate actual learning rate
        LR=$(python3 -c "print($MULT * $BASE_LR)")

        job_count=$((job_count + 1))
        echo "  Job $job_count/$total_jobs: heads=$HEADS, lr=$LR (${MULT}x base)"

        # Submit the job with multi-GPU configuration
        sbatch --time=${TIME_SPEC} \
               --nodes=1 \
               --gpus-per-node=h100:${GPUS_PER_NODE} \
               --cpus-per-gpu=${CPUS_PER_GPU} \
               --mem=${MEM} \
               --job-name=Eryngii_ScaledGPT_ademamix_h${HEADS}_lr${MULT} \
               scripts/scripts_dfer/Eryngii_scaledGPT/fir_Eryngii_dfer_restart.sh \
               --heads $HEADS \
               --lr $LR \
               --omega $OMEGA \
               --optimizer ademamix \
               --nproc_per_node ${GPUS_PER_NODE} \
               --restart_on_time_limit ${RESTART_ON_TIME_LIMIT} \
               --latest_ckpt_interval 1000 \
               --auto_resume

        # Check if the job was successful
        if [ $? -eq 0 ]; then
            echo "    ✓ Job submitted successfully"
        else
            echo "    ✗ Job failed with exit code $?"
        fi

        echo ""
    done

    echo "----------------------------------------"
done

echo "Sweep completed. Total jobs submitted: $job_count"
echo ""
echo "Resource allocation per job:"
echo "  Nodes: 1"
echo "  GPUs: ${GPUS_PER_NODE} × H100"
echo "  CPUs: ${TOTAL_CPUS} (${CPUS_PER_GPU} per GPU)"
echo "  Memory: ${MEM} (allocate as needed)"
echo "  Time: ${TIME_SPEC}"

