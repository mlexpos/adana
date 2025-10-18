#!/bin/bash

# Define lr values
lr_values=(2e-4 3e-4 4e-4 5e-4 6e-4 7e-4)

# Define r values for wd_ts calculation
r_values=( 1 2 3 )

# Define clipsnr values
clipsnr_values=( 0.0625 )

echo "Starting grid search over lr, wd_ts, and clipsnr parameters"
echo "lr values: ${lr_values[@]}"
echo "r values: ${r_values[@]}"
echo "clipsnr values: ${clipsnr_values[@]}"

# Counter for job tracking
job_count=0
total_jobs=$((${#lr_values[@]} * ${#r_values[@]} * ${#clipsnr_values[@]}))
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

        # Loop over clipsnr values
        for clipsnr in "${clipsnr_values[@]}"; do
            job_count=$((job_count + 1))
            echo "Job $job_count/$total_jobs: lr=$lr, r=$r, wd_ts=$wd_ts, clipsnr=$clipsnr"

            # Run the dana-star-mk4 script with current parameters
            sbatch ./scripts/diloco90m/narval-cypaq-dana-star-mk4-sweep.sh --lr $lr --wd_ts $wd_ts --clipsnr $clipsnr

            # Check if the job was successful
            if [ $? -eq 0 ]; then
                echo "Job $job_count submitted successfully"
            else
                echo "Job $job_count failed with exit code $?"
            fi

            echo "----------------------------------------"
        done
    done
done

echo "Grid search completed. Total jobs run: $job_count"
