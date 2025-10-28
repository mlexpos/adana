#!/bin/bash

# Print model parameters for BigHead architecture across different depths

echo "BigHead Model Parameters"
echo "========================"
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

    # Calculate embedding parameters (vocabulary)
    local EMB_PARAMS=$(python3 -c "print(int(2 * $N_EMBD * 50304))")
    
    # Calculate total parameters
    local TOTAL_PARAMS=$(python3 -c "print(int($NON_EMB + $EMB_PARAMS))")
    
    # Calculate iterations (20 tokens per parameter)
    local ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / 65536))")
    
    # Calculate total tokens
    local TOTAL_TOKENS=$(python3 -c "print(int(65536 * $ITERATIONS))")
    
    # Format numbers with commas
    local NON_EMB_FMT=$(python3 -c "print(f'{$NON_EMB:,}')")
    local EMB_PARAMS_FMT=$(python3 -c "print(f'{$EMB_PARAMS:,}')")
    local TOTAL_PARAMS_FMT=$(python3 -c "print(f'{$TOTAL_PARAMS:,}')")
    local ITERATIONS_FMT=$(python3 -c "print(f'{$ITERATIONS:,}')")
    local TOTAL_TOKENS_FMT=$(python3 -c "print(f'{$TOTAL_TOKENS:,}')")
    
    echo "Depth $DEPTH:"
    echo "  n_layer:       $N_LAYER"
    echo "  n_head:        $N_HEAD"
    echo "  head_dim:      $HEAD_DIM"
    echo "  n_embd:        $N_EMBD"
    echo "  mlp_hidden:    $MLP_HIDDEN"
    echo "  Non-emb params: $NON_EMB_FMT"
    echo "  Emb params:     $EMB_PARAMS_FMT"
    echo "  Total params:   $TOTAL_PARAMS_FMT"
    echo "  Iterations:     $ITERATIONS_FMT"
    echo "  Total tokens:   $TOTAL_TOKENS_FMT"
    echo ""
}

# Loop through depths 4 to 12
for DEPTH in {4..12}; do
    calculate_params $DEPTH
done

echo "========================"
echo "Architecture formula:"
echo "  head_dim = 16 × depth"
echo "  n_embd = 16 × depth²"
echo "  mlp_hidden = 32 × depth²"
echo "  n_head = depth"
echo "  n_layer = depth"
echo ""
echo "Training budget:"
echo "  iterations = 20 × total_params / 65536"
echo "  total_tokens = 65536 × iterations"

