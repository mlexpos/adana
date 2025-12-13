#!/bin/bash

# EggHead AdamW Sweep across different head counts
# For each head count, runs multiple learning rates: multipliers of the formula prediction
# Learning rate formula: lr = 3.20e+01 × P^(-0.600) where P = NON_EMB

OMEGA=4.0
HEADS_ARRAY=( 4 5 6 7 )
LR_MULTIPLIERS=(1.0 1.5 2.0 0.5 0.25 1.25 1.75)

echo "Starting EggHead AdamW sweep"
echo "Head counts: ${HEADS_ARRAY[@]}"
echo "Omega: $OMEGA"
echo "LR multipliers: ${LR_MULTIPLIERS[@]}"
echo ""

# Function to calculate model parameters for a given number of heads
calculate_params() {
    local HEADS=$1

    # Model architecture parameters (functions of HEADS)
    local HEAD_DIM=$(python3 -c "print(int(16 * $HEADS))")
    local N_EMBD=$(python3 -c "print(int(16 * $HEADS * $HEADS))")
    local MLP_HIDDEN=$(python3 -c "print(int(32 * $HEADS * $HEADS))")
    local N_HEAD=$(python3 -c "print(int($HEADS))")
    local N_LAYER=$(python3 -c "print(int($HEADS * ($HEADS - 1) / 2))")

    # Calculate non-embedding parameters
    # Non-emb = n_layer * (3 * head_dim * n_embd * n_head + n_embd^2 + 2 * n_embd * mlp + 8 * n_embd) + 2 * n_embd
    local NON_EMB=$(python3 -c "print(int($N_LAYER * (3 * $HEAD_DIM * $N_EMBD * $N_HEAD + $N_EMBD * $N_EMBD + 2 * $N_EMBD * $MLP_HIDDEN + 8 * $N_EMBD) + 2 * $N_EMBD))")

    # Calculate total parameters and iterations
    local TOTAL_PARAMS=$(python3 -c "print(int($NON_EMB + 2 * $N_EMBD * 50304))")
    local ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / 65536))")

    echo "$NON_EMB $ITERATIONS"
}

# Calculate C(4) for time normalization (4 heads)
read NON_EMB_4 ITERATIONS_4 <<< $(calculate_params 4)
C_4=$(python3 -c "print($NON_EMB_4 * $ITERATIONS_4)")

echo "Reference (heads=4):"
echo "  NON_EMB = $NON_EMB_4"
echo "  ITERATIONS = $ITERATIONS_4"
echo "  C(4) = $C_4"
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

    # Calculate time in hours: C(heads) / C(4) scaled appropriately
    # We'll use this to estimate SLURM time
    #TIME_HOURS=$(python3 -c "import math; print(max(1, int(math.ceil($C / $C_4))))")
    TIME_HOURS=12

    # Calculate base learning rate using formula: lr = 4.03e+01 * P^(-0.590)
    BASE_LR=$(python3 -c "print(3.20e+01 * ($NON_EMB ** -0.600))")

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
               --job-name=EH_AdamW_h${HEADS}_lr${MULT} \
               scripts/narval/EggHead_epaq.sh \
               --heads $HEADS \
               --lr $LR \
               --omega $OMEGA \
               --optimizer adamw

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
