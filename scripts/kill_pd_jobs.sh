#!/bin/bash

# Script to kill all SLURM jobs with PD (Pending) status
# Usage: ./kill_pd_jobs.sh

echo "Checking for pending SLURM jobs..."

# Get all job IDs with PD status
PD_JOBS=$(squeue -u $USER -t PD -h -o "%i" 2>/dev/null)

if [ -z "$PD_JOBS" ]; then
    echo "No pending jobs found for user $USER"
    exit 0
fi

echo "Found pending jobs: $PD_JOBS"

# Count the number of pending jobs
JOB_COUNT=$(echo "$PD_JOBS" | wc -l)
echo "Number of pending jobs: $JOB_COUNT"

# Ask for confirmation
read -p "Are you sure you want to cancel all $JOB_COUNT pending jobs? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelling pending jobs..."
    
    # Cancel each job
    for job_id in $PD_JOBS; do
        echo "Cancelling job $job_id..."
        scancel $job_id
        if [ $? -eq 0 ]; then
            echo "Successfully cancelled job $job_id"
        else
            echo "Failed to cancel job $job_id"
        fi
    done
    
    echo "Done! All pending jobs have been cancelled."
else
    echo "Operation cancelled. No jobs were cancelled."
fi








