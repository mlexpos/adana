#!/bin/bash

# Enoki D-Muon Multi-GPU Sweep for Narval
# Uses 4 GPUs per node for larger models
# For each head count, runs multiple learning rates: multipliers of the formula prediction
# Learning rate formula: lr =  2.65e-05 + 1.41e+01 * P^{-0.531} where P = NON_EMB
# Enoki scaling: head_dim=64 (fixed), n_layer=3*heads/4, n_embd=64*heads, mlp=4*n_embd

OMEGA=4.0
HEADS_ARRAY=( 24 )
LR_MULTIPLIERS=(1.5 1.75)

BATCH_SIZE=8  # Need to be divisible by 4
ACC_STEPS=4   # BATCH_SIZE * ACC_STEPS = 32

# SLURM configuration for Narval (4 GPUs)
GPUS_PER_NODE=4
CPUS_PER_GPU=12
TOTAL_CPUS=48  # 4 GPUs × 12 CPUs/GPU
MEM=0          # 0 = allocate as needed
TIME_HOURS=44

echo "Starting Enoki D-Muon Multi-GPU sweep (Narval)"
echo "Head counts: ${HEADS_ARRAY[@]}"
echo "Omega: $OMEGA"
echo "LR multipliers: ${LR_MULTIPLIERS[@]}"
echo "Batch size: $BATCH_SIZE"
echo "Accumulation steps: $ACC_STEPS"
echo "Effective batch size: $((BATCH_SIZE * ACC_STEPS))"
echo "GPUs per node: $GPUS_PER_NODE"
echo "CPUs per GPU: $CPUS_PER_GPU"
echo "Total CPUs: $TOTAL_CPUS"
echo "Memory: ${MEM} (allocate as needed)"
echo "Time allocation: ${TIME_HOURS}h"
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
total_jobs=$((${#HEADS_ARRAY[@]} * ${#LR_MULTIPLIERS[@]}))
echo "Total jobs to run: $total_jobs"
echo ""

# Loop over head counts
for HEADS in "${HEADS_ARRAY[@]}"; do
    echo "Processing heads=$HEADS"

    # Calculate parameters for this head count
    read NON_EMB ITERATIONS <<< $(calculate_params $HEADS)

    # Calculate computational cost C = NON_EMB * ITERATIONS
    C=$(python3 -c "print($NON_EMB * $ITERATIONS)")

    # Calculate base learning rate using formula: lr = 2.65e-05 + 1.41e+01 * P^{-0.531}
    BASE_LR=$(python3 -c "print(2.65e-05 + 1.41e+01 * ($NON_EMB ** -0.531))")

    # Calculate n_layer for this head count
    N_LAYER=$(python3 -c "print(int(3 * $HEADS // 4))")

    echo "  HEADS = $HEADS"
    echo "  N_LAYER = $N_LAYER (= 3 * $HEADS / 4)"
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
        echo "  Job $job_count/$total_jobs: heads=$HEADS, lr=$LR (${MULT}x base)"

        # Submit the job with multi-GPU configuration
        sbatch --time=${TIME_HOURS}:00:00 \
               --nodes=1 \
               --gpus-per-node=a100:${GPUS_PER_NODE} \
               --cpus-per-gpu=${CPUS_PER_GPU} \
               --mem=${MEM} \
               --job-name=EN_dmuon_4G_h${HEADS}_lr${MULT} \
               scripts/narval/Enoki_epaq.sh \
               --heads $HEADS \
               --lr $LR \
               --omega $OMEGA \
               --batch_size $BATCH_SIZE \
               --acc_steps $ACC_STEPS \
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

