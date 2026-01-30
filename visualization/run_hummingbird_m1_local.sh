#!/bin/bash
#
# Local script to generate hummingbird data with m=1
# Based on hyperparameters from the 2026 experiments
#
# Runs:
# - dana-mk4 with clipsnr=1000, 4, 1, 0.25 (kappa sweep 0-1)
# - dana-star-mk4 with kappa=0.25 only
# For batch sizes B=1 and B=10

# Activate virtual environment
source /opt/e-py/bin/activate
echo "Activated virtual environment"

# Change to the JAX directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAX_DIR="${SCRIPT_DIR}/../jax"
cd "$JAX_DIR"
echo "Working directory: $(pwd)"

# Create results directory
RESULTS_DIR="${SCRIPT_DIR}/hummingbird_data"
mkdir -p "$RESULTS_DIR"
echo "Results directory: $RESULTS_DIR"

# Common parameters from 2026 experiments (except m=1)
ALPHA=1.0
BETA=-0.3
V=10000
D=5000
M=1
ZETA=1.0
STEPS=1000000
G2_SCALE=0.001
G3_OVER_G2=1.0
TANEA_LR_SCALAR=1.0
DELTA=6.0
ADAM_LR=0.001
ADAM_BETA1=0.99
ADAM_BETA2=0.999
STUDENT_T_DOF=3.0
SIGMA=0.0
RANDOM_SEED=42

# Batch sizes to run
BATCH_SIZES=(1 10)

# ClipSNR values for dana-mk4
CLIPSNR_VALUES=(1000 4 1 0.25)

echo "=============================================="
echo "Hummingbird m=1 Data Generation"
echo "=============================================="
echo "Parameters:"
echo "  m=$M (reduced from 1000)"
echo "  alpha=$ALPHA, beta=$BETA"
echo "  v=$V, d=$D, zeta=$ZETA"
echo "  steps=$STEPS"
echo "  g2_scale=$G2_SCALE, tanea_lr_scalar=$TANEA_LR_SCALAR"
echo "  delta=$DELTA"
echo "  Batch sizes: ${BATCH_SIZES[*]}"
echo "  ClipSNR values for dana-mk4: ${CLIPSNR_VALUES[*]}"
echo "=============================================="
echo ""

# Run dana-mk4 experiments with different clipsnr values
for BATCH in "${BATCH_SIZES[@]}"; do
    for CLIPSNR in "${CLIPSNR_VALUES[@]}"; do
        echo "----------------------------------------------"
        echo "Running dana-mk4: batch=$BATCH, clipsnr=$CLIPSNR"
        echo "----------------------------------------------"

        PREFIX="hummingbird_dana-mk4_m${M}_clipsnr${CLIPSNR}_batch${BATCH}"

        python hummingbird_plot.py \
            --alpha $ALPHA \
            --beta $BETA \
            --v $V \
            --d $D \
            --m $M \
            --zeta $ZETA \
            --steps $STEPS \
            --batch_size $BATCH \
            --g2_scale $G2_SCALE \
            --g3_over_g2 $G3_OVER_G2 \
            --tanea_lr_scalar $TANEA_LR_SCALAR \
            --optimizer dana-mk4 \
            --clipsnr $CLIPSNR \
            --delta $DELTA \
            --adam_lr $ADAM_LR \
            --adam_beta1 $ADAM_BETA1 \
            --adam_beta2 $ADAM_BETA2 \
            --student_t_dof $STUDENT_T_DOF \
            --sigma $SIGMA \
            --random_seed $RANDOM_SEED \
            --results_dir "$RESULTS_DIR" \
            --output_prefix "$PREFIX"

        echo ""
    done
done

# Run dana-star-mk4 experiments with kappa=0.25 only
for BATCH in "${BATCH_SIZES[@]}"; do
    echo "----------------------------------------------"
    echo "Running dana-star-mk4: batch=$BATCH, kappa=0.25 only"
    echo "----------------------------------------------"

    PREFIX="hummingbird_dana-star-mk4_m${M}_kappa0.25_batch${BATCH}"

    python hummingbird_plot.py \
        --alpha $ALPHA \
        --beta $BETA \
        --v $V \
        --d $D \
        --m $M \
        --zeta $ZETA \
        --steps $STEPS \
        --batch_size $BATCH \
        --g2_scale $G2_SCALE \
        --g3_over_g2 $G3_OVER_G2 \
        --tanea_lr_scalar $TANEA_LR_SCALAR \
        --optimizer dana-star-mk4 \
        --kappa_min 0.0 \
        --kappa_max 0.25 \
        --kappa_step 0.05 \
        --clipsnr 0.5 \
        --delta $DELTA \
        --adam_lr $ADAM_LR \
        --adam_beta1 $ADAM_BETA1 \
        --adam_beta2 $ADAM_BETA2 \
        --student_t_dof $STUDENT_T_DOF \
        --sigma $SIGMA \
        --random_seed $RANDOM_SEED \
        --results_dir "$RESULTS_DIR" \
        --output_prefix "$PREFIX"

    echo ""
done

echo "=============================================="
echo "Done! Data files saved to: $RESULTS_DIR"
echo "=============================================="
ls -la "$RESULTS_DIR"/*m${M}*.pkl 2>/dev/null || echo "No m=$M files generated yet"
