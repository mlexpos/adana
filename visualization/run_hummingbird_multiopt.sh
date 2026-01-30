#!/bin/bash
#
# Wrapper script to run hummingbird_plot_from_data_multiopt.py
# with all optimizers for different batch sizes (m1000 model)
#
# Usage: ./run_hummingbird_multiopt.sh [--kappa_min VALUE] [--batch BATCH_SIZE]
#
# If --batch is not specified, runs all three batch sizes (10, 100, 1000)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/hummingbird_data"
OUTPUT_DIR="${SCRIPT_DIR}/results"

# Default kappa_min
KAPPA_MIN=0.5

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

# If no batch sizes specified, run all three
if [ ${#BATCH_SIZES[@]} -eq 0 ]; then
    BATCH_SIZES=(10 100 1000)
fi

# Data file mappings by batch size
# Format: OPTIMIZER_BATCHSIZE_FILE

# Batch size 10
ADEMAMIX_10="${DATA_DIR}/hummingbird_ademamix_alpha1.0_m1000_zeta1.0_beta-0.3_sigma0.0_steps1000000_20251212_081847.pkl"
DANA_MK4_10="${DATA_DIR}/hummingbird_dana-mk4_alpha1.0_m1000_zeta1.0_beta-0.3_sigma0.0_steps1000000_20251212_080835.pkl"
DANA_STAR_MK4_10="${DATA_DIR}/hummingbird_dana-star-mk4_alpha1.0_m1000_zeta1.0_beta-0.3_sigma0.0_steps1000000_20251212_081811.pkl"
ADANA_10="${DATA_DIR}/hummingbird_dana-mk4_alpha1.0_m1000_zeta1.0_beta-0.3_sigma0.0_steps1000000_20260115_080042.pkl"

# Batch size 100
ADEMAMIX_100="${DATA_DIR}/hummingbird_ademamix_alpha1.0_m1000_zeta1.0_beta-0.3_sigma0.0_steps1000000_20251212_081714.pkl"
DANA_MK4_100="${DATA_DIR}/hummingbird_dana-mk4_alpha1.0_m1000_zeta1.0_beta-0.3_sigma0.0_steps1000000_20251212_081319.pkl"
DANA_STAR_MK4_100="${DATA_DIR}/hummingbird_dana-star-mk4_alpha1.0_m1000_zeta1.0_beta-0.3_sigma0.0_steps1000000_20251212_081326.pkl"
ADANA_100="${DATA_DIR}/hummingbird_dana-mk4_alpha1.0_m1000_zeta1.0_beta-0.3_sigma0.0_steps1000000_20260115_082644.pkl"

# Batch size 1000
ADEMAMIX_1000="${DATA_DIR}/hummingbird_ademamix_alpha1.0_m1000_zeta1.0_beta-0.3_sigma0.0_steps1000000_20251212_010116.pkl"
DANA_MK4_1000="${DATA_DIR}/hummingbird_dana-mk4_alpha1.0_m1000_zeta1.0_beta-0.3_sigma0.0_steps1000000_20251211_191402.pkl"
DANA_STAR_MK4_1000="${DATA_DIR}/hummingbird_dana-star-mk4_alpha1.0_m1000_zeta1.0_beta-0.3_sigma0.0_steps1000000_20251211_191857.pkl"
ADANA_1000="${DATA_DIR}/hummingbird_dana-mk4_alpha1.0_m1000_zeta1.0_beta-0.3_sigma0.0_steps1000000_20260115_084128.pkl"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Hummingbird Multi-Optimizer Plot Generator"
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
    ADEMAMIX_VAR="ADEMAMIX_${BATCH}"
    DANA_MK4_VAR="DANA_MK4_${BATCH}"
    DANA_STAR_MK4_VAR="DANA_STAR_MK4_${BATCH}"
    ADANA_VAR="ADANA_${BATCH}"

    ADEMAMIX_FILE="${!ADEMAMIX_VAR}"
    DANA_MK4_FILE="${!DANA_MK4_VAR}"
    DANA_STAR_MK4_FILE="${!DANA_STAR_MK4_VAR}"
    ADANA_FILE="${!ADANA_VAR}"

    OUTPUT_PREFIX="hummingbird_multiopt_batch${BATCH}_kappa${KAPPA_MIN}"

    python3 "${SCRIPT_DIR}/hummingbird_plot_from_data_multiopt.py" \
        --data_files \
            "$ADEMAMIX_FILE" \
            "$DANA_MK4_FILE" \
            "$DANA_STAR_MK4_FILE" \
            "$ADANA_FILE" \
        --kappa_min "$KAPPA_MIN" \
        --output_dir "$OUTPUT_DIR" \
        --output_prefix "$OUTPUT_PREFIX" \
        --title 'MOE-PLRF $\kappa$ sweep'

    echo ""
done

echo "=============================================="
echo "Done! Output files in: $OUTPUT_DIR"
echo "=============================================="
