for wd_idx in 0 1 2 ; do
        for delta_idx in 0 1 2 3 4; do
                for gamma3factor_idx in 0 1 2 3 4; do
                        echo "Running wd_idx=$wd_idx delta_idx=$delta_idx gamma3factor_idx=$gamma3factor_idx"
                        GRID_BATCH="$wd_idx,$delta_idx,$gamma3factor_idx" sbatch scripts/tamia_2.sh
                        sleep 0.1 # sleep 0.1 seconds to avoid overwhelming the scheduler
                done
        done
done