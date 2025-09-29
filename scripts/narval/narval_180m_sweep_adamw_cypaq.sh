#!/bin/bash

# Define lr values: 2^2*1e-4, 2^3*1e-4, 2^4*1e-4, 2^5*1e-4, 2^6*1e-4
lr_values=(2e-4 8e-4 3e-4 12e-4 5e-4 10e-4)

# Define r values for weight_decay calculation

r_values=(-2 -1 0 1 2 3)

echo "Starting grid search over lr and weight_decay parameters"
echo "lr values: ${lr_values[@]}"
echo "r values: ${r_values[@]}"

# Counter for job tracking
job_count=0
total_jobs=$((${#lr_values[@]} * ${#r_values[@]}))
echo "Total jobs to run: $total_jobs"

# Loop over lr values
for lr in "${lr_values[@]}"; do
    echo "Processing lr=$lr"

    # Loop over r values to calculate weight_decay
    for r in "${r_values[@]}"; do
        # Calculate weight_decay = (2^r / iterations) / lr
        # Using 77527 iterations for 180M model
        iterations=77527
        lr_decimal=$(python -c "print(float('$lr'))")

        weight_decay=$(python -c "print((2**$r / $iterations) / $lr_decimal)")

        job_count=$((job_count + 1))
        echo "Job $job_count/$total_jobs: lr=$lr, r=$r, weight_decay=$weight_decay"

        # Run the adamw script with current parameters
        sbatch ./scripts/diloco180m/narval-cypaq-adamw-nozloss-sweep.sh --lr $lr --weight_decay $weight_decay

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
