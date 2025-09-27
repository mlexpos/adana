#!/bin/bash

# Script to analyze failed runs and extract their parameters
# Usage: ./analyze_failed_runs_cypaq.sh

LOGS_DIR="/home/cypaquet/danastar/logs"
OUTPUT_FILE="/home/cypaquet/danastar/scripts/narval/failed_runs_params.txt"

echo "Analyzing failed runs from narval*502446* and narval*502447* err files..."
echo "Output will be saved to: $OUTPUT_FILE"
echo ""

# Clear the output file
> "$OUTPUT_FILE"

# Counter for failed runs
failed_count=0

# Find all error files matching the patterns
for pattern in 502446 502447; do
    for err_file in "$LOGS_DIR"/narval*${pattern}*err; do
        if [ -f "$err_file" ]; then
            # Extract job ID from filename
            job_id=$(basename "$err_file" | grep -o '[0-9]\{8\}')

            # Check if this is a failed run by looking for error indicators
            if grep -q "RuntimeError\|Traceback\|ChildFailedError\|exitcode.*1" "$err_file"; then
                echo "Found failed run: $job_id"

                # Get corresponding output file
                out_file="${err_file%.err}.out"

                if [ -f "$out_file" ]; then
                    # Extract lr and weight_decay values from the output file
                    lr_value=$(grep -o "Using lr=[0-9.e-]*" "$out_file" | sed 's/Using lr=//')
                    weight_decay_value=$(grep -o "weight_decay=[0-9.e-]*" "$out_file" | sed 's/weight_decay=//')

                    if [ -n "$lr_value" ] && [ -n "$weight_decay_value" ]; then
                        echo "Job $job_id: lr=$lr_value, weight_decay=$weight_decay_value"
                        echo "$lr_value $weight_decay_value" >> "$OUTPUT_FILE"
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
done

echo ""
echo "Analysis complete. Found $failed_count failed runs."
echo "Parameters saved to: $OUTPUT_FILE"

if [ $failed_count -gt 0 ]; then
    echo ""
    echo "Failed run parameters:"
    cat "$OUTPUT_FILE"
fi