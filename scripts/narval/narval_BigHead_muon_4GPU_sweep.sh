#!/bin/bash

# BigHead D-Muon Multi-GPU Sweep for Narval (depths {8,9,10,11})
# Uses 4 GPUs per node for larger models
# For each depth, runs multiple learning rates: multipliers of the formula prediction
# Learning rate formula: lr = 8.47e-04 + 6.03e+01 × P^{-0.658} where P = NON_EMB

OMEGA=4.0
DEPTHS=( 8 9 )
LR_MULTIPLIERS=(1.0 0.75 1.25 1.5 0.5)

# SLURM configuration for Narval (4 GPUs)
GPUS_PER_NODE=4
CPUS_PER_GPU=8
TOTAL_CPUS=32  # 4 GPUs × 8 CPUs/GPU
MEM=0          # 0 = allocate as needed
TIME_HOURS=12

echo "Starting BigHead D-Muon Multi-GPU sweep (Narval)"
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

    # BigHead architecture parameters
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

# Calculate reference for depth=8
read NON_EMB_8 ITERATIONS_8 <<< $(calculate_params 8)
C_8=$(python3 -c "print($NON_EMB_8 * $ITERATIONS_8)")

echo "Reference (depth=8):"
echo "  NON_EMB = $NON_EMB_8"
echo "  ITERATIONS = $ITERATIONS_8"
echo "  C(8) = $C_8"
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

    # Calculate base learning rate using formula: lr = 8.47e-04 + 6.03e+01 * P^{-0.658}
    BASE_LR=$(python3 -c "print(8.47e-04 + 6.03e+01 * ($NON_EMB ** -0.658))")

    echo "  DEPTH = $DEPTH"
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
               --gpus-per-node=a100:${GPUS_PER_NODE} \
               --cpus-per-gpu=${CPUS_PER_GPU} \
               --mem=${MEM} \
               --job-name=BH_dmuon_4G_d${DEPTH}_lr${MULT} \
               scripts/narval/BigHead_epaq.sh \
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
echo "  GPUs: ${GPUS_PER_NODE} × A100"
echo "  CPUs: ${TOTAL_CPUS} (${CPUS_PER_GPU} per GPU)"
echo "  Memory: ${MEM} (allocate as needed)"
echo "  Time: ${TIME_HOURS} hours"
