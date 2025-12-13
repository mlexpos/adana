#!/bin/bash

# Enoki Ademamix Sweep across depths {8,9,10,11}
# For each depth, runs multiple learning rates: multipliers of the formula prediction
# Learning rate formula: lr = 1.50e-05 + 9.15e+01 × P^{-0.660} where P = NON_EMB

OMEGA=4.0
HEADS_ARRAY=( 6 8 10 )
LR_MULTIPLIERS=(1.0 0.75 1.25 1.5 0.5 1.75 0.25)

echo "Starting Enoki Ademamix sweep"
echo "Head counts: ${HEADS_ARRAY[@]}"
echo "Omega: $OMEGA"
echo "LR multipliers: ${LR_MULTIPLIERS[@]}"
echo ""

# Function to calculate model parameters for a given depth
calculate_params() {
    local HEADS=$1

    # Enoki architecture parameters (DiLoco scaling with fixed aspect ratio)
    local HEAD_DIM=64
    local N_HEAD=$(python3 -c "print(int($HEADS))")
    local N_LAYER=$(python3 -c "print(int(3 * $HEADS // 4))")
    local N_EMBD=$(python3 -c "print(int($HEADS * 64))")
    local MLP_HIDDEN=$(python3 -c "print(int(4 * $N_EMBD))")

    # Calculate non-embedding parameters
    # Non-emb = 12 * n_embd^2 * n_layer (standard DiLoco formula)
    local NON_EMB=$(python3 -c "print(int(12 * $N_EMBD * $N_EMBD * $N_LAYER))")

    # Calculate total parameters and iterations
    local TOTAL_PARAMS=$(python3 -c "print(int($NON_EMB + 2 * $N_EMBD * 50304))")
    local ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / 65536))")

    echo "$NON_EMB $ITERATIONS"
}

# Calculate C(4) for reference
read NON_EMB_4 ITERATIONS_4 <<< $(calculate_params 4)
C_4=$(python3 -c "print($NON_EMB_4 * $ITERATIONS_4)")

echo "Reference (depth=4):"
echo "  NON_EMB = $NON_EMB_4"
echo "  ITERATIONS = $ITERATIONS_4"
echo "  C(4) = $C_4"
echo ""

# Counter for job tracking
job_count=0
total_jobs=$((${#HEADS_ARRAY[@]} * ${#LR_MULTIPLIERS[@]}))
echo "Total jobs to run: $total_jobs"
echo ""

# Loop over depths
for HEADS in "${HEADS_ARRAY[@]}"; do
    echo "Processing depth=$HEADS"

    # Calculate parameters for this depth
    read NON_EMB ITERATIONS <<< $(calculate_params $HEADS)

    # Calculate computational cost C = NON_EMB * ITERATIONS
    C=$(python3 -c "print($NON_EMB * $ITERATIONS)")

    # Calculate time in hours based on compute
    TIME_HOURS=8

    # Calculate base learning rate using formula: lr = 1.50e-05 + 9.15e+01 * P^{-0.660}
    BASE_LR=$(python3 -c "print(1.50e-05 + 9.15e+01 * ($NON_EMB ** -0.660))")

    echo "  NON_EMB = $NON_EMB"
    echo "  ITERATIONS = $ITERATIONS"
    echo "  C = $C"
    echo "  Estimated time: ${TIME_HOURS}h"
    echo "  Base LR (formula): $BASE_LR"
    echo ""

    # Loop over learning rate multipliers
    for MULT in "${LR_MULTIPLIERS[@]}"; do
        # Calculate actual learning rate
        LR=$(python3 -c "print($MULT * $BASE_LR)")

        job_count=$((job_count + 1))
        echo "  Job $job_count/$total_jobs: heads=$HEADS, lr=$LR (${MULT}x base)"

        # Submit the job with calculated parameters
        sbatch --time=${TIME_HOURS}:00:00 \
               --job-name=EN_ademamix_d${HEADS}_lr${MULT} \
               scripts/narval/Enoki_epaq.sh \
               --heads $HEADS \
               --lr $LR \
               --omega $OMEGA \
               --optimizer ademamix

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
