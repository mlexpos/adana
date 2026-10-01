#!/bin/bash
# Fetch npz outputs of rorqual slurm jobs into results/ (logs stay remote: ~/dana-exp/slurm/*.out, runs/*/log.txt)
set -e
cd "$(dirname "$0")/.."
ssh -o BatchMode=yes rorqual "cd ~/dana-exp/runs && find . -name '*.npz' -newer ~/dana-exp/slurm/job.sh | tar czf /tmp/epaq_runs.tgz -T -"
scp -O -q rorqual:/tmp/epaq_runs.tgz /tmp/epaq_runs.tgz
mkdir -p results && tar xzf /tmp/epaq_runs.tgz -C results && echo "fetched rorqual outputs -> results/"
