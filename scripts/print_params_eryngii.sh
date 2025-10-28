#!/bin/bash

# Print model parameters for Eryngii architecture across different n_heads values

echo "Eryngii Model Parameters (n_head = 4 to 10)"
echo "=============================================="
echo ""
echo "Eryngii scaling:"
echo "  - head_dim = 32 * heads / 3 (rounded to multiple of 8)"
echo "  - n_head = heads"
echo "  - n_layer = heads^2 / 8"
echo "  - n_embd = n_head * head_dim"
echo "  - mlp_hidden = 4 * n_embd"
echo ""

# Function to calculate model parameters for Eryngii
calculate_eryngii_params() {
    local HEADS=$1

    # Eryngii architecture parameters
    local HEAD_DIM=$(python3 -c "print(int(round(32 * $HEADS / 3 / 8) * 8))")
    local N_HEAD=$(python3 -c "print(int($HEADS))")
    local N_LAYER=$(python3 -c "print(int($HEADS**2 // 8))")
    local N_EMBD=$(python3 -c "print(int($N_HEAD * $HEAD_DIM))")
    local MLP_HIDDEN=$(python3 -c "print(int(4 * $N_EMBD))")

    # Calculate non-embedding parameters
    # Non-emb = 12 * n_embd^2 * n_layer (standard DiLoco formula)
    local NON_EMB=$(python3 -c "print(int(12 * $N_EMBD * $N_EMBD * $N_LAYER))")

    # Calculate embedding parameters (vocabulary)
    local EMB_PARAMS=$(python3 -c "print(int(2 * $N_EMBD * 50304))")
    
    # Calculate total parameters
    local TOTAL_PARAMS=$(python3 -c "print(int($NON_EMB + $EMB_PARAMS))")
    
    # Calculate iterations (20 tokens per parameter)
    local ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / 65536))")
    
    # Calculate total tokens
    local TOTAL_TOKENS=$(python3 -c "print(int(65536 * $ITERATIONS))")
    
    echo "$N_HEAD $HEAD_DIM $N_LAYER $N_EMBD $MLP_HIDDEN $NON_EMB $EMB_PARAMS $TOTAL_PARAMS $ITERATIONS $TOTAL_TOKENS"
}

# Print table header
printf "%-7s %-9s %-8s %-8s %-12s %-14s %-14s %-14s %-12s %-14s\n" \
    "n_head" "head_dim" "n_layer" "n_embd" "mlp_hidden" "non_emb" "emb_params" "total_params" "iterations" "total_tokens"
printf "%-7s %-9s %-8s %-8s %-12s %-14s %-14s %-14s %-12s %-14s\n" \
    "------" "---------" "--------" "--------" "------------" "--------------" "--------------" "--------------" "------------" "--------------"

# Loop through n_heads from 4 to 10
for HEADS in {4..10}; do
    read N_HEAD HEAD_DIM N_LAYER N_EMBD MLP_HIDDEN NON_EMB EMB_PARAMS TOTAL_PARAMS ITERATIONS TOTAL_TOKENS <<< $(calculate_eryngii_params $HEADS)
    
    # Format with commas for readability
    NON_EMB_FMT=$(python3 -c "print(f'{$NON_EMB:,}')")
    EMB_PARAMS_FMT=$(python3 -c "print(f'{$EMB_PARAMS:,}')")
    TOTAL_PARAMS_FMT=$(python3 -c "print(f'{$TOTAL_PARAMS:,}')")
    ITERATIONS_FMT=$(python3 -c "print(f'{$ITERATIONS:,}')")
    TOTAL_TOKENS_FMT=$(python3 -c "print(f'{$TOTAL_TOKENS:,}')")
    
    printf "%-7s %-9s %-8s %-8s %-12s %-14s %-14s %-14s %-12s %-14s\n" \
        "$N_HEAD" "$HEAD_DIM" "$N_LAYER" "$N_EMBD" "$MLP_HIDDEN" \
        "$NON_EMB_FMT" "$EMB_PARAMS_FMT" "$TOTAL_PARAMS_FMT" "$ITERATIONS_FMT" "$TOTAL_TOKENS_FMT"
done

echo ""
echo "Notes:"
echo "  - Iterations = 20 × total_params / 65536"
echo "  - Total tokens = 65536 × iterations"
echo "  - Non-emb params use DiLoco formula: 12 × n_embd² × n_layer"

