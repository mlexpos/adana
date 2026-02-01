#!/bin/bash

squeue -t PD -S -Q -o "%.10Q %.9P %.10a %.20j %.8u %.8T %.10l %.6D %.15b" | grep -E "(JOBID|gpu)" 