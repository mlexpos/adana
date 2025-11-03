#!/bin/bash -l
#SBATCH --time=0-00:01:00
#SBATCH --partition=unkillable
unset LD_PRELOAD

# the sleep-loop at the end is running for max_iteration*30s
max_iteration=30

# only allow a single restart of the job. 
max_restarts=1

START_TIME=$(date +%s)

# Get job information
scontext=$(scontrol show job $SLURM_JOB_ID)
restarts=$(echo "$scontext" | grep -o 'Restarts=.' | cut -d= -f2)
outfile=$(echo "$scontext"  | grep 'StdOut='       | cut -d= -f2)
errfile=$(echo "$scontext"  | grep 'StdErr='       | cut -d= -f2)
timelimit=$(echo "$scontext" | grep -o 'TimeLimit=.*' | awk '{print $1}' | cut -d= -f2)

# Convert time limit to seconds
timelimit_seconds=$(echo "$timelimit" | awk -F'[-:]' '{if(NF==4)print $1*86400+$2*3600+$3*60+$4; else if(NF==3)print $1*3600+$2*60+$3; else print 0}')

echo "Time limit: ${timelimit_seconds}s"

# term handler
# the function is executed once the job gets the TERM signal
term_handler()
{
    ELAPSED=$(($(date +%s) - START_TIME))
    TIME_UNTIL_LIMIT=$((timelimit_seconds - ELAPSED))
    
    echo "executing term_handler at $(date)"
    echo "Elapsed: ${ELAPSED}s, Remaining: ${TIME_UNTIL_LIMIT}s (limit: ${timelimit_seconds}s)"
    
    if [[ $restarts -lt $max_restarts && $TIME_UNTIL_LIMIT -lt 20 ]]; then
        echo "Requeuing job at $(date)"
        echo time until limit: $TIME_UNTIL_LIMIT
        sbatch $0
    fi
}

# declare the function handling the TERM signal
trap 'term_handler' TERM

# print some job-information
cat <<EOF
SLURM_JOB_ID:         $SLURM_JOB_ID
SLURM_JOB_NAME:       $SLURM_JOB_NAME
SLURM_JOB_PARTITION:  $SLURM_JOB_PARTITION
SLURM_SUBMIT_HOST:    $SLURM_SUBMIT_HOST
TimeLimit:            $timelimit
Restarts:             $restarts
EOF

# the actual "calculation"
echo "starting calculation at $(date)"
i=0
while [[ $i -lt $max_iteration ]]; do
    # Sleep in smaller intervals so signals can be caught
    for j in {1..30}; do
        sleep 1
    done
    i=$(($i+1))
    echo "$i out of $max_iteration done at $(date)"
done

echo "all done at $(date)"