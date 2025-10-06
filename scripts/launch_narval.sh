for opt in 0; do
        for w in 0 1 2; do
                for lr in 0 1 2 3; do
                        echo "Running opt=$opt w=$w lr=$lr"
                        GRID_BATCH="$w,$opt,$lr" sbatch scripts/narval.sh
                done
        done
done