#!/bin/bash

squeue -t R -S -Q -o "%.10Q %.9P %.20j %.8u %.8T %.10M %.6D %.15b" | grep -E "(JOBID|gpu)" | head -n 100