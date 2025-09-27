#!/bin/bash

# Script to cancel SLURM jobs from 50194541 through 50194542

echo "Cancelling SLURM jobs from 50194541 to 50194542..."

# Cancel jobs in the range
for job_id in {50194531..50194566}; do
    echo "Cancelling job $job_id..."
    scancel $job_id
    if [ $? -eq 0 ]; then
        echo "Successfully cancelled job $job_id"
    else
        echo "Failed to cancel job $job_id (may not exist or already completed)"
    fi
done

echo "Job cancellation completed."
