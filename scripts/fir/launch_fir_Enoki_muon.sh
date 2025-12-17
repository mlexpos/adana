#!/bin/bash

# Enoki Muon ScaledGPT Initialization Sweep for Fir
# Uses ScaledGPT initialization scheme with 4 GPUs
# For each head count, runs multiple learning rates: multipliers of the formula prediction
# Learning rate formula: lr = 2.19 × (5.64e+04 + P)^-0.417 where P = NON_EMB
# Enoki scaling: head_dim=64 (fixed), n_layer=3*heads/4, n_embd=64*heads, mlp=4*n_embd
# iterations to run formula = 24*3600 / (5.83 × 10^-4 * (TOTAL_PARAMS/1e6)^0.91) / 2


OMEGA_ARRAY=( 4.0 )
HEADS_ARRAY=( 34 36 )
LR_MULTIPLIERS=( 1.0 )
BATCH_SIZE=2
ACC_STEPS=16

# SLURM configuration for Fir (4 GPUs)
GPUS_PER_NODE=4
CPUS_PER_GPU=8
TOTAL_CPUS=32            # 4 GPUs × 8 CPUs/GPU
MEM=0                   # 0 = allocate as needed
TIME_HOURS=24

# ScaledGPT initialization parameters
INIT_SCHEME="ScaledGPT"
DEPTH_SCALAR_EXPONENT=0.0
ITERATIONS_TO_RUN=80000

# QK normalization and tau stats flags
NO_QKNORM=false
COLLECT_TAU_STATS=true

echo "Starting Enoki Muon ScaledGPT Initialization sweep (Fir)"
echo "Head counts: ${HEADS_ARRAY[@]}"
echo "Omega values: ${OMEGA_ARRAY[@]}"
echo "LR multipliers: ${LR_MULTIPLIERS[@]}"
echo "GPUs per node: $GPUS_PER_NODE"
echo "CPUs per GPU: $CPUS_PER_GPU"
echo "Total CPUs: $TOTAL_CPUS"
echo "Memory: ${MEM} (allocate as needed)"
echo "Time allocation: ${TIME_HOURS}h"
echo "Init scheme: $INIT_SCHEME"
echo "Depth scalar exponent: $DEPTH_SCALAR_EXPONENT"
echo "Iterations to run: $ITERATIONS_TO_RUN"
echo "QK Normalization: $([ "$NO_QKNORM" = true ] && echo "DISABLED" || echo "ENABLED")"
echo "Tau stats collection: $([ "$COLLECT_TAU_STATS" = true ] && echo "ENABLED" || echo "DISABLED")"
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

    echo "$NON_EMB $ITERATIONS $TOTAL_PARAMS"
}

# Calculate reference for heads=16
read NON_EMB_16 ITERATIONS_16 TOTAL_PARAMS_16 <<< $(calculate_params 16)
C_16=$(python3 -c "print($NON_EMB_16 * $ITERATIONS_16)")

echo "Reference (heads=16):"
echo "  NON_EMB = $NON_EMB_16"
echo "  ITERATIONS = $ITERATIONS_16"
echo "  TOTAL_PARAMS = $TOTAL_PARAMS_16"
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
        read NON_EMB ITERATIONS TOTAL_PARAMS <<< $(calculate_params $HEADS)

        # Calculate computational cost C = NON_EMB * ITERATIONS
        C=$(python3 -c "print($NON_EMB * $ITERATIONS)")

        # Calculate base learning rate using formula: lr = 2.19 × (5.64e+04 + P)^-0.417
        BASE_LR=$(python3 -c "print(2.19 * ((5.64e04 + $NON_EMB) ** -0.417))")

        # Calculate n_layer for this head count
        N_LAYER=$(python3 -c "print(int(3 * $HEADS // 4))")

        echo "    HEADS = $HEADS"
        echo "    N_LAYER = $N_LAYER (= 3 * $HEADS / 4)"
        echo "    NON_EMB = $NON_EMB"
        echo "    ITERATIONS = $ITERATIONS"
        echo "    C = $C"
        echo "    TOTAL_PARAMS = $TOTAL_PARAMS"
        echo "    Time allocation: ${TIME_HOURS}h"
        echo "    Base LR (formula): $BASE_LR"
        echo ""

        echo "    BATCH_SIZE = $BATCH_SIZE"
        echo "    ACC_STEPS = $ACC_STEPS"
        echo "    Effective batch size = $((BATCH_SIZE * ACC_STEPS))"
        echo ""

        # Loop over learning rate multipliers
        for MULT in "${LR_MULTIPLIERS[@]}"; do
            # Calculate actual learning rate
            LR=$(python3 -c "print($MULT * $BASE_LR)")

            # Calculate iterations to run
            #ITERATIONS_TO_RUN=$(python3 -c "print(int(24 * 3600 / (5.83e-04 * ($TOTAL_PARAMS/1e6) ** 0.91) / 2))")

            job_count=$((job_count + 1))
            echo "    Job $job_count/$total_jobs: omega=$OMEGA, heads=$HEADS, lr=$LR (${MULT}x base)"

            # Build optional flags
            OPTIONAL_FLAGS=""
            if [ "$NO_QKNORM" = true ]; then
                OPTIONAL_FLAGS="$OPTIONAL_FLAGS --no-qknorm"
            fi
            if [ "$COLLECT_TAU_STATS" = true ]; then
                OPTIONAL_FLAGS="$OPTIONAL_FLAGS --collect-tau-stats"
            fi

            # Submit the job with ScaledGPT initialization
            sbatch --account=rrg-bengioy-ad \
                   --time=${TIME_HOURS}:00:00 \
                   --nodes=1 \
                   --gpus-per-node=h100:${GPUS_PER_NODE} \
                   --cpus-per-gpu=${CPUS_PER_GPU} \
                   --mem=${MEM} \
                   --job-name=EN_Muon_SGPT_om${OMEGA}_h${HEADS}_lr${MULT} \
                   scripts/fir/fir_Enoki_muon.sh \
                   --heads $HEADS \
                   --lr $LR \
                   --omega $OMEGA \
                   --batch_size $BATCH_SIZE \
                   --acc_steps $ACC_STEPS \
                   --optimizer d-muon \
                   --nproc_per_node ${GPUS_PER_NODE} \
                   --depth-scalar-exponent $DEPTH_SCALAR_EXPONENT \
                   --iterations_to_run $ITERATIONS_TO_RUN \
                   $OPTIONAL_FLAGS

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
echo "  Optimizer: Muon (d-muon)"
echo "  Omega values: ${OMEGA_ARRAY[@]}"
echo "  Head counts: ${HEADS_ARRAY[@]}"
echo "  LR multipliers: ${LR_MULTIPLIERS[@]}"
echo "  LR formula: lr = 2.19 × (5.64e+04 + NON_EMB)^-0.417"
echo ""
echo "Resource allocation per job:"
echo "  Nodes: 1"
echo "  GPUs: ${GPUS_PER_NODE} × H100"
echo "  CPUs: ${TOTAL_CPUS} (${CPUS_PER_GPU} per GPU)"
echo "  Memory: ${MEM} (allocate as needed)"
echo "  Time: ${TIME_HOURS} hours"
echo "  Init scheme: ${INIT_SCHEME}"
echo "  Depth scalar exponent: ${DEPTH_SCALAR_EXPONENT}"
