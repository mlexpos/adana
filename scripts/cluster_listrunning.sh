#!/bin/bash

squeue -t R -S -M -o "%.10Q %.9P %.10a %.20j %.8u %.8T %.10M %.6D %.15b" | grep -E "(JOBID|gpu)" 