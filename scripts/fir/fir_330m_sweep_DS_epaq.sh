#!/bin/bash

# Define lr values: 2^2*1e-4, 2^3*1e-4, 2^4*1e-4, 2^5*1e-4, 2^6*1e-4
lr_values=(6e-5 12e-5 4e-5 2e-5 8e-5 10e-5)

# Define r values for weight_decay calculation

r_values=(2)

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

        # Run the DanaStar script with current parameters
        sbatch ./scripts/diloco330m/fir-danastar-epaq-nozloss-sweep.sh --lr $lr --wd_ts $wd_ts

        sleep 1

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
