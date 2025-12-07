#!/bin/bash

# Qwen3Next AdamW ScaledGPT Initialization Sweep for Narval (4 A100 GPUs)
# Uses ScaledGPT initialization scheme with expert parallelism
# 64 experts distributed across 4 GPUs (16 experts per GPU)
# Learning rate formula: lr = 1.27e+04 × (9.23e+04 + P)^-0.848 where P = NON_EMB
# Qwen3Next scaling: head_dim=128, n_layer=2*heads, n_embd=128*heads, mlp=3*n_embd
# MoE: 64 experts, 4 active per token, 1 gated shared expert

OMEGA_ARRAY=( 4.0 )
HEADS_ARRAY=( 14 16 18 )
LR_MULTIPLIERS=( 1.0 )
CLIPSNR=2.0
BATCH_SIZE=8  # Per GPU, effective = 32 with 4 GPUs
ACC_STEPS=1

# SLURM configuration for Narval (4 A100 GPUs)
GPUS_PER_NODE=4
CPUS_PER_GPU=8
TOTAL_CPUS=32
MEM=0
TIME_HOURS=24

# ScaledGPT initialization parameters
INIT_SCHEME="ScaledGPT"
DEPTH_SCALAR_EXPONENT=0.0
ITERATIONS_TO_RUN=100000

echo "Starting Qwen3Next AdamW Expert Parallel sweep (Narval)"
echo "Head counts: ${HEADS_ARRAY[@]}"
echo "Omega values: ${OMEGA_ARRAY[@]}"
echo "LR multipliers: ${LR_MULTIPLIERS[@]}"
echo "Clip SNR: $CLIPSNR"
echo "GPUs per node: $GPUS_PER_NODE (A100)"
echo "Experts per GPU: 16 (64 total experts)"
echo "Active per token: 4 (6.25% sparsity)"
echo "CPUs per GPU: $CPUS_PER_GPU"
echo "Total CPUs: $TOTAL_CPUS"
echo "Memory: ${MEM} (allocate as needed)"
echo "Time allocation: ${TIME_HOURS}h"
echo "Init scheme: $INIT_SCHEME"
echo "Depth scalar exponent: $DEPTH_SCALAR_EXPONENT"
echo "Iterations to run: $ITERATIONS_TO_RUN"
echo ""

# Function to calculate model parameters for Qwen3Next with MoE
calculate_params() {
    local HEADS=$1

    # Qwen3 architecture parameters
    local HEAD_DIM=128
    local N_HEAD=$(python3 -c "print(int($HEADS))")
    local N_LAYER=$(python3 -c "print(int(2 * $HEADS))")
    local N_EMBD=$(python3 -c "print(int($HEADS * 128))")
    local MLP_HIDDEN=$(python3 -c "print(int(3 * $N_EMBD))")

    # Calculate base non-emb (attention + layer norms)
    local TOTAL_QKV_DIM=$(python3 -c "print(int($N_HEAD * $HEAD_DIM))")
    local BASE_NON_EMB=$(python3 -c "
n_layer = $N_LAYER
n_embd = $N_EMBD
head_dim = $HEAD_DIM
total_qkv_dim = $TOTAL_QKV_DIM

# Per layer (without MLP)
attn = 5 * n_embd * total_qkv_dim
qk_norm = 2 * head_dim
layer_norms = 2 * n_embd

per_layer = attn + qk_norm + layer_norms

# Total
base_non_emb = n_layer * per_layer + n_embd

print(int(base_non_emb))
")

    # Calculate ACTIVE parameters for MoE (accounting for sparsity)
    # With 64 experts, 4 active per token:
    #   - Only 3 active routed experts (4 total - 1 shared = 3 routed)
    #   - Plus 1 gated shared expert (always active)
    #   - Plus router (always active)
    local ACTIVE_MOE_PARAMS=$(python3 -c "
n_layer = $N_LAYER
n_embd = $N_EMBD
mlp_hidden = $MLP_HIDDEN
num_active_routed = 3  # 4 active - 1 shared = 3 routed experts active
shared_expert_size = 4096

# Per MoE layer (ACTIVE parameters only)
router = n_embd * 64  # Router always active
active_routed_experts_mlp = num_active_routed * 3 * n_embd * mlp_hidden  # Only 3 active
shared_expert_mlp = 3 * n_embd * shared_expert_size  # Shared expert always active
shared_expert_gate = n_embd  # Gate always active

per_moe_layer = router + active_routed_experts_mlp + shared_expert_mlp + shared_expert_gate

# Total
active_moe_params = n_layer * per_moe_layer

print(int(active_moe_params))
")

    # Calculate total ACTIVE non-embedding parameters
    local ACTIVE_NON_EMB=$(python3 -c "print(int($BASE_NON_EMB + $ACTIVE_MOE_PARAMS))")

    # Calculate total parameters (for reference, but use ACTIVE for compute)
    local TOTAL_PARAMS=$(python3 -c "print(int($ACTIVE_NON_EMB + 2 * $N_EMBD * 50304))")

    # Use ACTIVE parameters for iteration count (compute scales with active params)
    local ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / 65536))")

    echo "$ACTIVE_NON_EMB $ITERATIONS $TOTAL_PARAMS"
}

# Calculate reference for heads=4
read ACTIVE_NON_EMB_4 ITERATIONS_4 TOTAL_PARAMS_4 <<< $(calculate_params 4)
C_4=$(python3 -c "print($ACTIVE_NON_EMB_4 * $ITERATIONS_4)")

echo "Reference (heads=4):"
echo "  ACTIVE_NON_EMB = $ACTIVE_NON_EMB_4 (accounting for 4/64 expert sparsity)"
echo "  ITERATIONS = $ITERATIONS_4"
echo "  TOTAL_PARAMS = $TOTAL_PARAMS_4"
echo "  C(4) = $C_4"
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
        read ACTIVE_NON_EMB ITERATIONS TOTAL_PARAMS <<< $(calculate_params $HEADS)

        # Calculate computational cost C = ACTIVE_NON_EMB * ITERATIONS
        C=$(python3 -c "print($ACTIVE_NON_EMB * $ITERATIONS)")

        # Calculate base learning rate using ACTIVE parameters
        BASE_LR=$(python3 -c "print(1.27e+04 * ((9.23e04 + $ACTIVE_NON_EMB) ** -0.848))")

        # Calculate n_layer
        N_LAYER=$(python3 -c "print(int(2 * $HEADS))")

        echo "    HEADS = $HEADS"
        echo "    N_LAYER = $N_LAYER (= 2 * $HEADS)"
        echo "    ACTIVE_NON_EMB = $ACTIVE_NON_EMB (3 routed + 1 shared expert active per token)"
        echo "    ITERATIONS = $ITERATIONS"
        echo "    C = $C"
        echo "    TOTAL_PARAMS = $TOTAL_PARAMS"
        echo "    Time allocation: ${TIME_HOURS}h"
        echo "    Base LR (formula): $BASE_LR"
        echo ""

        echo "    BATCH_SIZE = $BATCH_SIZE (per GPU)"
        echo "    ACC_STEPS = $ACC_STEPS"
        echo "    Effective batch size = $((BATCH_SIZE * GPUS_PER_NODE * ACC_STEPS))"
        echo ""

        # Loop over learning rate multipliers
        for MULT in "${LR_MULTIPLIERS[@]}"; do
            # Calculate actual learning rate
            LR=$(python3 -c "print($MULT * $BASE_LR)")

            job_count=$((job_count + 1))
            echo "    Job $job_count/$total_jobs: omega=$OMEGA, heads=$HEADS, lr=$LR (${MULT}x base)"

            # Submit the job with expert parallelism
            sbatch --time=${TIME_HOURS}:00:00 \
                   --nodes=1 \
                   --gpus-per-node=a100:${GPUS_PER_NODE} \
                   --cpus-per-gpu=${CPUS_PER_GPU} \
                   --mem=${MEM} \
                   --job-name=Q3N_AdamW_SGPT_om${OMEGA}_h${HEADS}_lr${MULT} \
                   scripts/narval/Qwen3Next_epaq.sh \
                   --heads $HEADS \
                   --lr $LR \
                   --omega $OMEGA \
                   --batch_size $BATCH_SIZE \
                   --acc_steps $ACC_STEPS \
                   --optimizer adamw \
                   --nproc_per_node ${GPUS_PER_NODE} \
                   --depth-scalar-exponent $DEPTH_SCALAR_EXPONENT \
                   --iterations_to_run $ITERATIONS_TO_RUN

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
echo "  Optimizer: AdamW"
echo "  Model: Qwen3Next (64 experts, expert parallelism)"
echo "  Omega values: ${OMEGA_ARRAY[@]}"
echo "  Head counts: ${HEADS_ARRAY[@]}"
echo "  LR multipliers: ${LR_MULTIPLIERS[@]}"
echo "  LR formula: lr = 1.27e+04 × (9.23e+04 + ACTIVE_NON_EMB)^-0.848"
echo "  Note: ACTIVE_NON_EMB accounts for 3 active routed + 1 shared expert"
echo "  Clip SNR: $CLIPSNR"
echo ""
echo "Resource allocation per job:"
echo "  Nodes: 1"
echo "  GPUs: ${GPUS_PER_NODE} × A100"
echo "  Experts per GPU: 16 (64 total)"
echo "  Active per token: 4 (6.25% sparsity)"
echo "  CPUs: ${TOTAL_CPUS} (${CPUS_PER_GPU} per GPU)"
echo "  Memory: ${MEM} (allocate as needed)"
echo "  Time: ${TIME_HOURS} hours"
echo "  Init scheme: ${INIT_SCHEME}"
echo "  Depth scalar exponent: ${DEPTH_SCALAR_EXPONENT}"
echo ""
echo "Qwen3Next architecture scaling:"
echo "  head_dim = 128 (fixed)"
echo "  n_layer = 2 × heads"
echo "  n_embd = 128 × heads"
echo "  mlp_hidden = 3 × n_embd"
echo "  MoE: 64 experts, 4 active/token, 1 gated shared"
echo "  Expert parallelism: enabled"
