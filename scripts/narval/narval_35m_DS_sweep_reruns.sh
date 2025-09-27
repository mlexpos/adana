#!/bin/bash

# Generated sweep script for failed runs
# This script launches jobs with the lr and wd_ts values from failed runs

# Array of (lr, wd_ts) pairs from failed runs
declare -a PARAMS=(
    "6e-4 416.6666666666667"
    "6e-4 3333.3333333333335"
    "7e-4 178.57142857142858"
    "7e-4 2857.1428571428573"
    "9e-4 277.77777777777777"
    "9e-4 555.5555555555555"
    "9e-4 2222.222222222222"
    "9e-4 4444.444444444444"
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
