#!/bin/bash
#
# Wrapper script to run hummingbird_plot_from_data_multiopt.py
# with all optimizers for different batch sizes (m1 model)
#
# Data files:
# - dana-mk4 with clipsnr=0.25, 1, 4
# - adana (clipsnr=1000)
# - dana-star-mk4 (kappa 0-0.25)
#
# Usage: ./run_hummingbird_multiopt_m1.sh [--kappa_min VALUE] [--batch BATCH_SIZE]
#
# If --batch is not specified, runs both batch sizes (1, 10)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/hummingbird_data"
OUTPUT_DIR="${SCRIPT_DIR}/results"

# Default kappa_min
KAPPA_MIN=0.0

# Parse arguments
BATCH_SIZES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --kappa_min)
            KAPPA_MIN="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZES+=("$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If no batch sizes specified, run both
if [ ${#BATCH_SIZES[@]} -eq 0 ]; then
    BATCH_SIZES=(1 10)
fi

# Data file mappings by batch size
# Format: OPTIMIZER_BATCHSIZE_FILE

# Batch size 1
ADANA_1="${DATA_DIR}/hummingbird_dana-mk4_m1_clipsnr1000_batch1_dana-mk4_alpha1.0_m1_zeta1.0_beta-0.3_sigma0.0_steps1000000_20260115_161700.pkl"
DANA_MK4_CLIP4_1="${DATA_DIR}/hummingbird_dana-mk4_m1_clipsnr4_batch1_dana-mk4_alpha1.0_m1_zeta1.0_beta-0.3_sigma0.0_steps1000000_20260115_161849.pkl"
DANA_MK4_CLIP1_1="${DATA_DIR}/hummingbird_dana-mk4_m1_clipsnr1_batch1_dana-mk4_alpha1.0_m1_zeta1.0_beta-0.3_sigma0.0_steps1000000_20260115_161824.pkl"
DANA_MK4_CLIP025_1="${DATA_DIR}/hummingbird_dana-mk4_m1_clipsnr0.25_batch1_dana-mk4_alpha1.0_m1_zeta1.0_beta-0.3_sigma0.0_steps1000000_20260115_161734.pkl"
DANA_STAR_MK4_1="${DATA_DIR}/hummingbird_dana-star-mk4_m1_kappa0.25_batch1_dana-star-mk4_alpha1.0_m1_zeta1.0_beta-0.3_sigma0.0_steps1000000_20260115_161431.pkl"

# Batch size 10
ADANA_10="${DATA_DIR}/hummingbird_dana-mk4_m1_clipsnr1000_batch10_dana-mk4_alpha1.0_m1_zeta1.0_beta-0.3_sigma0.0_steps1000000_20260115_161817.pkl"
DANA_MK4_CLIP4_10="${DATA_DIR}/hummingbird_dana-mk4_m1_clipsnr4_batch10_dana-mk4_alpha1.0_m1_zeta1.0_beta-0.3_sigma0.0_steps1000000_20260115_161800.pkl"
DANA_MK4_CLIP1_10="${DATA_DIR}/hummingbird_dana-mk4_m1_clipsnr1_batch10_dana-mk4_alpha1.0_m1_zeta1.0_beta-0.3_sigma0.0_steps1000000_20260115_161731.pkl"
DANA_MK4_CLIP025_10="${DATA_DIR}/hummingbird_dana-mk4_m1_clipsnr0.25_batch10_dana-mk4_alpha1.0_m1_zeta1.0_beta-0.3_sigma0.0_steps1000000_20260115_161749.pkl"
DANA_STAR_MK4_10="${DATA_DIR}/hummingbird_dana-star-mk4_m1_kappa0.25_batch10_dana-star-mk4_alpha1.0_m1_zeta1.0_beta-0.3_sigma0.0_steps1000000_20260115_161515.pkl"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Hummingbird Multi-Optimizer Plot Generator (m=1)"
echo "=============================================="
echo "Kappa min: $KAPPA_MIN"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Output dir: $OUTPUT_DIR"
echo ""

for BATCH in "${BATCH_SIZES[@]}"; do
    echo "----------------------------------------------"
    echo "Running batch_size=$BATCH"
    echo "----------------------------------------------"

    # Get the data files for this batch size
    ADANA_VAR="ADANA_${BATCH}"
    DANA_MK4_CLIP4_VAR="DANA_MK4_CLIP4_${BATCH}"
    DANA_MK4_CLIP1_VAR="DANA_MK4_CLIP1_${BATCH}"
    DANA_MK4_CLIP025_VAR="DANA_MK4_CLIP025_${BATCH}"
    DANA_STAR_MK4_VAR="DANA_STAR_MK4_${BATCH}"

    ADANA_FILE="${!ADANA_VAR}"
    DANA_MK4_CLIP4_FILE="${!DANA_MK4_CLIP4_VAR}"
    DANA_MK4_CLIP1_FILE="${!DANA_MK4_CLIP1_VAR}"
    DANA_MK4_CLIP025_FILE="${!DANA_MK4_CLIP025_VAR}"
    DANA_STAR_MK4_FILE="${!DANA_STAR_MK4_VAR}"

    OUTPUT_PREFIX="hummingbird_m1_multiopt_batch${BATCH}_kappa${KAPPA_MIN}"

    python3 "${SCRIPT_DIR}/hummingbird_plot_from_data_multiopt.py" \
        --data_files \
            "$ADANA_FILE" \
            "$DANA_MK4_CLIP4_FILE" \
            "$DANA_MK4_CLIP1_FILE" \
            "$DANA_MK4_CLIP025_FILE" \
            "$DANA_STAR_MK4_FILE" \
        --kappa_min "$KAPPA_MIN" \
        --output_dir "$OUTPUT_DIR" \
        --output_prefix "$OUTPUT_PREFIX" \
        --title 'PLRF $\kappa$ sweep' \
        --ylim 0.05 3.0 \
        --color_by_clipsnr

    echo ""
done

echo "=============================================="
echo "Done! Output files in: $OUTPUT_DIR"
echo "=============================================="
