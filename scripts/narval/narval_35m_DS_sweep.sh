#!/bin/bash

# Define lr values: 2^2*1e-4, 2^3*1e-4, 2^4*1e-4, 2^5*1e-4, 2^6*1e-4
#lr_values=(4e-4 8e-4 16e-4 32e-4 64e-4)
#lr_values=(5e-4 6e-4 7e-4 9e-4)
lr_values=( 1e-4 2e-4 10e-4 )

# Define r values for wd_ts calculation
r_values=(-3 -2 -1 0 1 2 3)

echo "Starting grid search over lr and wd_ts parameters"
echo "lr values: ${lr_values[@]}"
echo "r values: ${r_values[@]}"

# Counter for job tracking
job_count=0
total_jobs=$((${#lr_values[@]} * ${#r_values[@]}))
echo "Total jobs to run: $total_jobs"

# Loop over lr values
for lr in "${lr_values[@]}"; do
    echo "Processing lr=$lr"
    
    # Loop over r values to calculate wd_ts
    for r in "${r_values[@]}"; do
        # Calculate wd_ts = 2^r / lr
        # Convert scientific notation to decimal using Python
        lr_decimal=$(python3 -c "print(float('$lr'))")
        
        # Using Python for floating point arithmetic
        if [ $r -ge 0 ]; then
            wd_ts=$(python3 -c "print((2**$r) / $lr_decimal)")
        else
            # For negative exponents, use 1/2^(-r)
            neg_r=$((-$r))
            wd_ts=$(python3 -c "print(1 / (2**$neg_r) / $lr_decimal)")
        fi
        
        job_count=$((job_count + 1))
        echo "Job $job_count/$total_jobs: lr=$lr, r=$r, wd_ts=$wd_ts"
        
        # Run the dana-star script with current parameters
        sbatch ./scripts/diloco35m/narval-dana-star-sweep.sh --lr $lr --wd_ts $wd_ts
        
        # Check if the job was successful
        if [ $? -eq 0 ]; then
            echo "Job $job_count submitted successfully"
        else
            echo "Job $job_count failed with exit code $?"
        fi
        
        echo "----------------------------------------"
    done
done

echo "Grid search completed. Total jobs run: $job_count"
