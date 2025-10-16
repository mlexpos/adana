#!/bin/bash

# Script to kill SLURM jobs with IDs from 93955 to 93970
# Usage: ./kill_jobs_93955_93970.sh

START_ID=93955
END_ID=93970

echo "Killing SLURM jobs with IDs from $START_ID to $END_ID..."

# Count total jobs in range
TOTAL_JOBS=$((END_ID - START_ID + 1))
echo "Total jobs to cancel: $TOTAL_JOBS"

# Check which jobs actually exist
echo "Checking which jobs exist..."
EXISTING_JOBS=""
for job_id in $(seq $START_ID $END_ID); do
    if squeue -j $job_id -h >/dev/null 2>&1; then
        EXISTING_JOBS="$EXISTING_JOBS $job_id"
    fi
done

if [ -z "$EXISTING_JOBS" ]; then
    echo "No jobs found in the range $START_ID-$END_ID"
    exit 0
fi

EXISTING_COUNT=$(echo $EXISTING_JOBS | wc -w)
echo "Found $EXISTING_COUNT existing jobs: $EXISTING_JOBS"

# Ask for confirmation
read -p "Are you sure you want to cancel these $EXISTING_COUNT jobs? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelling jobs..."
    
    # Cancel each existing job
    for job_id in $EXISTING_JOBS; do
        echo "Cancelling job $job_id..."
        scancel $job_id
        if [ $? -eq 0 ]; then
            echo "Successfully cancelled job $job_id"
        else
            echo "Failed to cancel job $job_id"
        fi
    done
    
    echo "Done! All jobs in range $START_ID-$END_ID have been processed."
else
    echo "Operation cancelled. No jobs were cancelled."
fi
