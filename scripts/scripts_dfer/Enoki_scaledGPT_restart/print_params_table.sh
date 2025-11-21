#!/bin/bash

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

# Print table header
printf "%-8s %-20s %-20s\n" "Heads" "Non_Emb" "Total_Params"
printf "%-8s %-20s %-20s\n" "------" "-------------------" "-------------------"

# Loop over heads 35 to 44
for HEADS in {35..44}; do
    read NON_EMB ITERATIONS TOTAL_PARAMS <<< $(calculate_params $HEADS)
    printf "%-8d %-20d %-20d\n" $HEADS $NON_EMB $TOTAL_PARAMS
done

