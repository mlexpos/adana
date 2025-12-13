#!/bin/bash

# Script to generate a new narval_35m_sweep.sh with failed run parameters
# Usage: ./generate_sweep_script_cypaq.sh

PARAMS_FILE="/home/cypaquet/danastar/scripts/narval/failed_runs_params.txt"
SWEEP_SCRIPT="/home/cypaquet/danastar/scripts/narval/narval_90m_sweep_adamw_reruns.sh"

if [ ! -f "$PARAMS_FILE" ]; then
    echo "Error: Parameters file not found: $PARAMS_FILE"
    echo "Please run analyze_failed_runs_cypaq.sh first."
    exit 1
fi

echo "Generating sweep script with failed run parameters..."
echo "Output: $SWEEP_SCRIPT"

# Create the sweep script header
cat > "$SWEEP_SCRIPT" << 'EOF'
#!/bin/bash

# Generated sweep script for failed runs
# This script launches jobs with the lr and weight_decay values from failed runs

# Array of (lr, weight_decay) pairs from failed runs
declare -a PARAMS=(
EOF

# Add parameters to the script
while read -r lr weight_decay; do
    echo "    \"$lr $weight_decay\"" >> "$SWEEP_SCRIPT"
done < "$PARAMS_FILE"

# Add the rest of the script
cat >> "$SWEEP_SCRIPT" << 'EOF'
)

echo "Launching ${#PARAMS[@]} jobs with failed run parameters..."

# Launch jobs for each parameter set
for param_set in "${PARAMS[@]}"; do
    # Parse lr and weight_decay from the parameter set
    lr=$(echo $param_set | cut -d' ' -f1)
    weight_decay=$(echo $param_set | cut -d' ' -f2)

    echo "Launching job with lr=$lr, weight_decay=$weight_decay"

    # Submit the job
    sbatch --job-name="narval-adamw-retry-lr${lr}-wd${weight_decay}" \
           /home/cypaquet/danastar/scripts/diloco90m/narval-cypaq-adamw-nozloss-sweep.sh \
           --lr "$lr" --weight_decay "$weight_decay"

    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo "All jobs submitted!"
EOF

# Make the script executable
chmod +x "$SWEEP_SCRIPT"

echo "Sweep script generated: $SWEEP_SCRIPT"
echo "To run: $SWEEP_SCRIPT"