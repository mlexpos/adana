#!/bin/bash

# Script to generate a new narval_35m_sweep.sh with failed run parameters
# Usage: ./generate_sweep_script.sh

PARAMS_FILE="/home/epaq/danastar/scripts/narval/failed_runs_params.txt"
SWEEP_SCRIPT="/home/epaq/danastar/scripts/narval/narval_35m_sweep.sh"

if [ ! -f "$PARAMS_FILE" ]; then
    echo "Error: Parameters file not found: $PARAMS_FILE"
    echo "Please run analyze_failed_runs.sh first."
    exit 1
fi

echo "Generating sweep script with failed run parameters..."
echo "Output: $SWEEP_SCRIPT"

# Create the sweep script header
cat > "$SWEEP_SCRIPT" << 'EOF'
#!/bin/bash

# Generated sweep script for failed runs
# This script launches jobs with the lr and wd_ts values from failed runs

# Array of (lr, wd_ts) pairs from failed runs
declare -a PARAMS=(
EOF

# Add parameters to the script
while read -r lr wd_ts; do
    echo "    \"$lr $wd_ts\"" >> "$SWEEP_SCRIPT"
done < "$PARAMS_FILE"

# Add the rest of the script
cat >> "$SWEEP_SCRIPT" << 'EOF'
)

echo "Launching ${#PARAMS[@]} jobs with failed run parameters..."

# Launch jobs for each parameter set
for param_set in "${PARAMS[@]}"; do
    # Parse lr and wd_ts from the parameter set
    lr=$(echo $param_set | cut -d' ' -f1)
    wd_ts=$(echo $param_set | cut -d' ' -f2)
    
    echo "Launching job with lr=$lr, wd_ts=$wd_ts"
    
    # Submit the job
    sbatch --job-name="narval-dana-star-retry-lr${lr}-wd${wd_ts}" \
           /home/epaq/danastar/scripts/diloco35m/narval-dana-star-sweep.sh \
           --lr "$lr" --wd_ts "$wd_ts"
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo "All jobs submitted!"
EOF

# Make the script executable
chmod +x "$SWEEP_SCRIPT"

echo "Sweep script generated: $SWEEP_SCRIPT"
echo "To run: $SWEEP_SCRIPT"
