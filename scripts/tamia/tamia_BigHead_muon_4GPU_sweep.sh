#!/bin/bash

# BigHead D-Muon Multi-GPU Sweep for Tamia (depths {10,11,12})
# Uses 4 GPUs per node for larger models
# For each depth, runs 3 learning rates: 0.75x, 1.0x, and 1.25x the formula prediction
# Learning rate formula: lr = 1.99e-04 + 3.29e+00 × P^{-0.452} where P = NON_EMB

OMEGA=4.0
DEPTHS=(10 11 12)
LR_MULTIPLIERS=(1.00 0.75 1.25 1.50 0.50)

# SLURM configuration for Tamia
GPUS_PER_NODE=4
CPUS_PER_GPU=12
TOTAL_CPUS=48  # 4 GPUs × 12 CPUs/GPU
MEM=0          # 0 = allocate as needed
TIME_HOURS=24

echo "Starting BigHead D-Muon Multi-GPU sweep (Tamia)"
echo "Depths: ${DEPTHS[@]}"
echo "Omega: $OMEGA"
echo "LR multipliers: ${LR_MULTIPLIERS[@]}"
echo "GPUs per node: $GPUS_PER_NODE"
echo "CPUs per GPU: $CPUS_PER_GPU"
echo "Total CPUs: $TOTAL_CPUS"
echo "Memory: ${MEM} (allocate as needed)"
echo "Time allocation: ${TIME_HOURS}h"
echo ""

# Function to calculate model parameters for a given depth
calculate_params() {
    local DEPTH=$1

    # Model architecture parameters
    local HEAD_DIM=$(python3 -c "print(int(16 * $DEPTH))")
    local N_EMBD=$(python3 -c "print(int(16 * $DEPTH * $DEPTH))")
    local MLP_HIDDEN=$(python3 -c "print(int(32 * $DEPTH * $DEPTH))")
    local N_HEAD=$(python3 -c "print(int($DEPTH))")
    local N_LAYER=$(python3 -c "print(int($DEPTH))")

    # Calculate non-embedding parameters
    # Non-emb = depth * (3 * head_dim * n_embd * n_head + n_embd^2 + 2 * n_embd * mlp + 8 * n_embd) + 2 * n_embd
    local NON_EMB=$(python3 -c "print(int($DEPTH * (3 * $HEAD_DIM * $N_EMBD * $N_HEAD + $N_EMBD * $N_EMBD + 2 * $N_EMBD * $MLP_HIDDEN + 8 * $N_EMBD) + 2 * $N_EMBD))")

    # Calculate total parameters and iterations
    local TOTAL_PARAMS=$(python3 -c "print(int($NON_EMB + 2 * $N_EMBD * 50304))")
    local ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / 65536))")

    echo "$NON_EMB $ITERATIONS"
}

# Calculate reference for depth=9
read NON_EMB_9 ITERATIONS_9 <<< $(calculate_params 9)
C_9=$(python3 -c "print($NON_EMB_9 * $ITERATIONS_9)")

echo "Reference (depth=9):"
echo "  NON_EMB = $NON_EMB_9"
echo "  ITERATIONS = $ITERATIONS_9"
echo "  C(9) = $C_9"
echo ""

# Counter for job tracking
job_count=0
total_jobs=$((${#DEPTHS[@]} * ${#LR_MULTIPLIERS[@]}))
echo "Total jobs to run: $total_jobs"
echo ""

# Loop over depths
for DEPTH in "${DEPTHS[@]}"; do
    echo "Processing depth=$DEPTH"

    # Calculate parameters for this depth
    read NON_EMB ITERATIONS <<< $(calculate_params $DEPTH)

    # Calculate computational cost C = NON_EMB * ITERATIONS
    C=$(python3 -c "print($NON_EMB * $ITERATIONS)")

    # Calculate base learning rate using formula: lr = 1.99e-04 + 3.29e+00 * P^{-0.452}
    BASE_LR=$(python3 -c "print(1.99e-04 + 3.29e+00 * ($NON_EMB ** -0.452))")

    echo "  NON_EMB = $NON_EMB"
    echo "  ITERATIONS = $ITERATIONS"
    echo "  C = $C"
    echo "  Time allocation: ${TIME_HOURS}h"
    echo "  Base LR (formula): $BASE_LR"
    echo ""

    # Loop over learning rate multipliers
    for MULT in "${LR_MULTIPLIERS[@]}"; do
        # Calculate actual learning rate
        LR=$(python3 -c "print($MULT * $BASE_LR)")

        job_count=$((job_count + 1))
        echo "  Job $job_count/$total_jobs: depth=$DEPTH, lr=$LR (${MULT}x base)"

        # Submit the job with multi-GPU configuration
        sbatch --time=${TIME_HOURS}:00:00 \
               --nodes=1 \
               --gpus-per-node=h100:${GPUS_PER_NODE} \
               --cpus-per-gpu=${CPUS_PER_GPU} \
               --mem=${MEM} \
               --job-name=BH_dmuon_4G_d${DEPTH}_lr${MULT} \
               scripts/narval/BigHead_cypaq.sh \
               --depth $DEPTH \
               --lr $LR \
               --omega $OMEGA \
               --optimizer d-muon \
               --nproc_per_node ${GPUS_PER_NODE}

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
echo "  Time: ${TIME_HOURS} hours"
