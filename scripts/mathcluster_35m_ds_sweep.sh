#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro 
#SBATCH --gpus=1
#SBATCH --propagate=NONE 
#SBATCH --account=paquettee-2025
#SBATCH -o logs/slurm.%N.%j.out # STDOUT
#SBATCH -e logs/slurm.%N.%j.err # STDERR
#SBATCH --signal=B:TERM@60

#export HF_HOME="$SLURM_TMPDIR/hf"
export WANDB_API_KEY=03c99521910548176ebfa4f418db1c9602e2afa3
export WANDB_PROJECT=danastar
export WANDB_ENTITY="ep-rmt-ml-opt"

#module load cuda/cuda-12.6
module load miniconda/miniconda-winter2025

#virtualenv --no-download /home/c/cypaquet/jaxenv
#source /home/c/cypaquet/jaxenv/bin/activate
#pip install --no-index jax[cuda12] flax pandas optax matplotlib tiktoken huggingface-hub
#pip install --upgrade "jax[cuda12]"
#pip install flax pandas optax matplotlib tiktoken huggingface-hub

cd /home/math/elliot.paquette@MCGILL.CA/danastar/

# Grid search parameters
# lr ranges over {2^k * 1e-4} for k from 2 to 6
# wd_ts ranges over {2^r / lr} for r in {-3, -2, -1, 0, 1, 2, 3}

# Define lr values: 2^2*1e-4, 2^3*1e-4, 2^4*1e-4, 2^5*1e-4, 2^6*1e-4
lr_values=(4e-4 8e-4 16e-4 32e-4 64e-4)

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
        # Convert scientific notation to decimal for bc
        lr_decimal=$(echo "$lr" | sed 's/e-4/ * 0.0001/' | sed 's/e-3/ * 0.001/' | sed 's/e-2/ * 0.01/' | sed 's/e-1/ * 0.1/' | sed 's/e+0/ * 1/' | sed 's/e+1/ * 10/' | sed 's/e+2/ * 100/' | sed 's/e+3/ * 1000/')
        
        # Using bc for floating point arithmetic
        if [ $r -ge 0 ]; then
            wd_ts=$(echo "scale=10; 2^$r / ($lr_decimal)" | bc -l)
        else
            # For negative exponents, use 1/2^(-r)
            neg_r=$((-$r))
            wd_ts=$(echo "scale=10; 1 / (2^$neg_r) / ($lr_decimal)" | bc -l)
        fi
        
        job_count=$((job_count + 1))
        echo "Job $job_count/$total_jobs: lr=$lr, r=$r, wd_ts=$wd_ts"
        
        # Run the dana-star script with current parameters
        bash ./scripts/diloco35m/dana-star-nozloss_2_param.sh --lr $lr --wd_ts $wd_ts
        
        # Check if the job was successful
        if [ $? -eq 0 ]; then
            echo "Job $job_count completed successfully"
        else
            echo "Job $job_count failed with exit code $?"
        fi
        
        echo "----------------------------------------"
    done
done

echo "Grid search completed. Total jobs run: $job_count"
