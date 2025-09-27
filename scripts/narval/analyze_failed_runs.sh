#!/bin/bash

# Script to analyze failed runs and extract their parameters
# Usage: ./analyze_failed_runs.sh

LOGS_DIR="/home/epaq/danastar/logs"
OUTPUT_FILE="/home/epaq/danastar/scripts/narval/failed_runs_params.txt"

echo "Analyzing failed runs from narval*5021*err files..."
echo "Output will be saved to: $OUTPUT_FILE"
echo ""

# Clear the output file
> "$OUTPUT_FILE"

# Counter for failed runs
failed_count=0

# Find all error files matching the pattern
for err_file in "$LOGS_DIR"/narval*5021*err; do
    if [ -f "$err_file" ]; then
        # Extract job ID from filename
        job_id=$(basename "$err_file" | grep -o '[0-9]\{8\}')
        
        # Check if this is a failed run by looking for error indicators
        if grep -q "RuntimeError\|Traceback\|ChildFailedError\|exitcode.*1" "$err_file"; then
            echo "Found failed run: $job_id"
            
            # Get corresponding output file
            out_file="${err_file%.err}.out"
            
            if [ -f "$out_file" ]; then
                # Extract lr and wd_ts values from the output file
                lr_value=$(grep -o "Using lr=[0-9.e-]*" "$out_file" | sed 's/Using lr=//')
                wd_ts_value=$(grep -o "wd_ts=[0-9.e-]*" "$out_file" | sed 's/wd_ts=//')
                
                if [ -n "$lr_value" ] && [ -n "$wd_ts_value" ]; then
                    echo "Job $job_id: lr=$lr_value, wd_ts=$wd_ts_value"
                    echo "$lr_value $wd_ts_value" >> "$OUTPUT_FILE"
                    ((failed_count++))
                else
                    echo "  Warning: Could not extract parameters from $out_file"
                fi
            else
                echo "  Warning: No corresponding output file found for $err_file"
            fi
        fi
    fi
done

echo ""
echo "Analysis complete. Found $failed_count failed runs."
echo "Parameters saved to: $OUTPUT_FILE"

if [ $failed_count -gt 0 ]; then
    echo ""
    echo "Failed run parameters:"
    cat "$OUTPUT_FILE"
fi
