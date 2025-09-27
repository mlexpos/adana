#!/bin/bash

# Generated sweep script for failed runs
# This script launches jobs with the lr and wd_ts values from failed runs

# Array of (lr, wd_ts) pairs from failed runs
# declare -a PARAMS=(
#     "16e-4 312.5000000000"
#     "32e-4 312.5000000000"
#     "32e-4 625.0000000000"
#     "64e-4 78.1250000000"
#     "64e-4 312.5000000000"
# )
declare -a PARAMS=(
    "16e-4 312.5000000000"
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
           /home/epaq/danastar/scripts/diloco90m/narval-dana-star-sweep-4node.sh \
           --lr "$lr" --wd_ts "$wd_ts"
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo "All jobs submitted!"
