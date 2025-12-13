#!/bin/bash

# Tau Statistics Collection Sweep for Narval
# Collects tau statistics from dana-star-mk4 optimizer during transformer training
# Runs on Enoki and Qwen3 architectures with fixed hyperparameters

# Architecture configurations (small sizes for debugging)
ENOKI_HEADS=16
QWEN3_HEADS=8

# Fixed hyperparameters
KAPPA=0.75
OMEGA=4.0
CLIPSNR=2.0
DELTA=8.0
BATCH_SIZE=32
ACC_STEPS=1
SEQUENCE_LENGTH=2048

# SLURM configuration for Narval (1 A100 GPU for debugging)
GPUS_PER_NODE=4
CPUS_PER_GPU=8
TOTAL_CPUS=32             # 4 GPU × 8 CPUs/GPU
MEM=0
TIME_HOURS=12

# Initialization scheme
INIT_SCHEME="ScaledGPT"
DEPTH_SCALAR_EXPONENT=0.0

# QK normalization flag (set to true to disable QK norm)
NO_QKNORM=false

echo "Starting Tau Statistics Collection Sweep (Narval)"
echo "=================================================="
echo ""
echo "Architectures:"
echo "  Enoki: heads=$ENOKI_HEADS"
echo "  Qwen3: heads=$QWEN3_HEADS"
echo ""
echo "Fixed hyperparameters:"
echo "  kappa=$KAPPA"
echo "  omega=$OMEGA"
echo "  clipsnr=$CLIPSNR"
echo "  delta=$DELTA"
echo "  batch_size=$BATCH_SIZE"
echo "  acc_steps=$ACC_STEPS"
echo ""
echo "SLURM configuration:"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  CPUs per GPU: $CPUS_PER_GPU"
echo "  Memory: $MEM"
echo "  Time allocation: ${TIME_HOURS}h"
echo ""
echo "QK Normalization: $([ "$NO_QKNORM" = true ] && echo "DISABLED" || echo "ENABLED")"
echo ""

# Function to calculate Enoki parameters
calculate_enoki_params() {
    local HEADS=$1

    # Enoki architecture (DiLoCo scaling)
    local HEAD_DIM=64
    local N_HEAD=$HEADS
    local N_LAYER=$(python3 -c "print(int(3 * $HEADS // 4))")
    local N_EMBD=$(python3 -c "print(int($HEADS * 64))")
    local MLP_HIDDEN=$(python3 -c "print(int(4 * $N_EMBD))")

    # Calculate non-embedding parameters (DiLoCo formula)
    # Non-emb = 12 * n_embd^2 * n_layer
    local NON_EMB=$(python3 -c "print(int(12 * $N_EMBD * $N_EMBD * $N_LAYER))")

    # Calculate total parameters: non_emb + embeddings (vocab=50304)
    local TOTAL_PARAMS=$(python3 -c "print(int($NON_EMB + 2 * $N_EMBD * 50304))")

    # Chinchilla scaling: iterations = 20 * total_params / (batch_size * sequence_length)
    local ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / ($BATCH_SIZE * $SEQUENCE_LENGTH)))")

    # Learning rate formula: lr = 4.40e+01 × (8.35e+03 + NON_EMB)^-0.664
    local LR=$(python3 -c "print(4.40e+01 * ((8.35e+03 + $NON_EMB) ** -0.664))")

    echo "$NON_EMB $TOTAL_PARAMS $ITERATIONS $LR $N_LAYER $N_EMBD"
}

# Function to calculate Qwen3 parameters
calculate_qwen3_params() {
    local HEADS=$1

    # Qwen3 architecture parameters
    local HEAD_DIM=128
    local N_HEAD=$HEADS
    local N_LAYER=$(python3 -c "print(int(2 * $HEADS))")
    local N_EMBD=$(python3 -c "print(int($HEADS * 128))")
    local MLP_HIDDEN=$(python3 -c "print(int(3 * $N_EMBD))")

    # Calculate non-embedding parameters for Qwen3
    # With gating: non_emb = n_layer * (5 * n_embd * total_qkv_dim + 2 * head_dim + 9 * n_embd^2 + 2 * n_embd) + n_embd
    local TOTAL_QKV_DIM=$(python3 -c "print(int($N_HEAD * $HEAD_DIM))")

    local NON_EMB=$(python3 -c "
n_layer = $N_LAYER
n_embd = $N_EMBD
head_dim = $HEAD_DIM
total_qkv_dim = $TOTAL_QKV_DIM

# Per layer
attn = 5 * n_embd * total_qkv_dim  # q_proj (2x with gating) + k_proj + v_proj + o_proj
qk_norm = 2 * head_dim
mlp = 9 * n_embd * n_embd  # gate_proj + up_proj + down_proj (mlp_hidden = 3 * n_embd)
layer_norms = 2 * n_embd

per_layer = attn + qk_norm + mlp + layer_norms

# Total
non_emb = n_layer * per_layer + n_embd  # +n_embd for final norm

print(int(non_emb))
")

    # Calculate total parameters
    local TOTAL_PARAMS=$(python3 -c "print(int($NON_EMB + 2 * $N_EMBD * 50304))")

    # Chinchilla scaling: iterations = 20 * total_params / (batch_size * sequence_length)
    local ITERATIONS=$(python3 -c "print(int(20 * $TOTAL_PARAMS / ($BATCH_SIZE * $SEQUENCE_LENGTH)))")

    # Learning rate formula: lr = 3.72e+03 × (3.70e+03 + NON_EMB)^-0.894
    local LR=$(python3 -c "print(3.72e+03 * ((3.70e+03 + $NON_EMB) ** -0.894))")

    echo "$NON_EMB $TOTAL_PARAMS $ITERATIONS $LR $N_LAYER $N_EMBD"
}

# Counter for job tracking
job_count=0

echo "=========================================="
echo "ENOKI Architecture"
echo "=========================================="
echo ""

# Calculate Enoki parameters
read ENOKI_NON_EMB ENOKI_TOTAL_PARAMS ENOKI_ITERATIONS ENOKI_LR ENOKI_N_LAYER ENOKI_N_EMBD <<< $(calculate_enoki_params $ENOKI_HEADS)

echo "Enoki configuration (heads=$ENOKI_HEADS):"
echo "  n_layer = $ENOKI_N_LAYER (= 3 * $ENOKI_HEADS / 4)"
echo "  n_embd = $ENOKI_N_EMBD (= 64 * $ENOKI_HEADS)"
echo "  NON_EMB = $ENOKI_NON_EMB"
echo "  TOTAL_PARAMS = $ENOKI_TOTAL_PARAMS"
echo "  ITERATIONS (Chinchilla) = $ENOKI_ITERATIONS"
echo "  LR (formula) = $ENOKI_LR"
echo ""

job_count=$((job_count + 1))
echo "Job $job_count: Enoki tau stats collection"

# Build no-qknorm flag if requested
NO_QKNORM_FLAG=""
if [ "$NO_QKNORM" = true ]; then
    NO_QKNORM_FLAG="--no-qknorm"
fi

sbatch --time=${TIME_HOURS}:00:00 \
       --nodes=1 \
       --gpus-per-node=a100:${GPUS_PER_NODE} \
       --cpus-per-gpu=${CPUS_PER_GPU} \
       --mem=${MEM} \
       --job-name=TauStats_Enoki_h${ENOKI_HEADS} \
       scripts/narval/Enoki_tau_stats.sh \
       --heads $ENOKI_HEADS \
       --lr $ENOKI_LR \
       --omega $OMEGA \
       --kappa $KAPPA \
       --batch_size $BATCH_SIZE \
       --acc_steps $ACC_STEPS \
       --clipsnr $CLIPSNR \
       --nproc_per_node ${GPUS_PER_NODE} \
       --depth-scalar-exponent $DEPTH_SCALAR_EXPONENT \
       $NO_QKNORM_FLAG

if [ $? -eq 0 ]; then
    echo "  ✓ Job submitted successfully"
else
    echo "  ✗ Job failed with exit code $?"
fi

echo ""
echo "=========================================="
echo "QWEN3 Architecture"
echo "=========================================="
echo ""

# Calculate Qwen3 parameters
read QWEN3_NON_EMB QWEN3_TOTAL_PARAMS QWEN3_ITERATIONS QWEN3_LR QWEN3_N_LAYER QWEN3_N_EMBD <<< $(calculate_qwen3_params $QWEN3_HEADS)

echo "Qwen3 configuration (heads=$QWEN3_HEADS):"
echo "  n_layer = $QWEN3_N_LAYER (= 2 * $QWEN3_HEADS)"
echo "  n_embd = $QWEN3_N_EMBD (= 128 * $QWEN3_HEADS)"
echo "  NON_EMB = $QWEN3_NON_EMB"
echo "  TOTAL_PARAMS = $QWEN3_TOTAL_PARAMS"
echo "  ITERATIONS (Chinchilla) = $QWEN3_ITERATIONS"
echo "  LR (formula) = $QWEN3_LR"
echo ""

job_count=$((job_count + 1))
echo "Job $job_count: Qwen3 tau stats collection"

sbatch --time=${TIME_HOURS}:00:00 \
       --nodes=1 \
       --gpus-per-node=a100:${GPUS_PER_NODE} \
       --cpus-per-gpu=${CPUS_PER_GPU} \
       --mem=${MEM} \
       --job-name=TauStats_Qwen3_h${QWEN3_HEADS} \
       scripts/narval/Qwen3_tau_stats.sh \
       --heads $QWEN3_HEADS \
       --lr $QWEN3_LR \
       --omega $OMEGA \
       --kappa $KAPPA \
       --batch_size $BATCH_SIZE \
       --acc_steps $ACC_STEPS \
       --clipsnr $CLIPSNR \
       --nproc_per_node ${GPUS_PER_NODE} \
       --depth-scalar-exponent $DEPTH_SCALAR_EXPONENT \
       --iterations_to_run $QWEN3_ITERATIONS \
       $NO_QKNORM_FLAG

if [ $? -eq 0 ]; then
    echo "  ✓ Job submitted successfully"
else
    echo "  ✗ Job failed with exit code $?"
fi

echo ""
echo "=========================================="
echo "Sweep Summary"
echo "=========================================="
echo ""
echo "Total jobs submitted: $job_count"
echo ""
echo "Configuration:"
echo "  Optimizer: dana-star-mk4"
echo "  Fixed hyperparameters:"
echo "    kappa = $KAPPA"
echo "    omega = $OMEGA"
echo "    clipsnr = $CLIPSNR"
echo "    delta = $DELTA"
echo "  Tau stats collection: ENABLED"
echo "  QK Normalization: $([ "$NO_QKNORM" = true ] && echo "DISABLED" || echo "ENABLED")"
echo "  WandB group: tau_stats"
echo ""
echo "Architectures:"
echo "  1. Enoki (heads=$ENOKI_HEADS): $ENOKI_TOTAL_PARAMS params, $ENOKI_ITERATIONS iters, lr=$ENOKI_LR"
echo "  2. Qwen3 (heads=$QWEN3_HEADS): $QWEN3_TOTAL_PARAMS params, $QWEN3_ITERATIONS iters, lr=$QWEN3_LR"
echo ""
echo "Learning rate formulas:"
echo "  Enoki: lr = 4.40e+01 × (8.35e+03 + NON_EMB)^-0.664"
echo "  Qwen3: lr = 3.72e+03 × (3.70e+03 + NON_EMB)^-0.894"
echo ""
echo "Training duration: Chinchilla scaling (20 × params / batch_tokens)"
echo ""
