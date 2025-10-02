for opt in 0 1; do
        for beta1 in 0 1; do
                for gamma_3_factor in 0 1; do
                        for w in 0 1 2; do
                                echo "Running $opt $beta1 $gamma_3_factor $w"
                                GRID_BATCH="$opt,$beta1,$gamma_3_factor,$w" sbatch scripts/tamia.sh
                        done
                done
        done
done