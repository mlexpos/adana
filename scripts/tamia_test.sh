for opt in 1; do
        for w in 0 1 2 3 4; do
                echo "Running opt=$opt w=$w"
                GRID_BATCH="$w,$opt" sbatch scripts/tamia.sh
        done
done