#!/bin/bash
# Fetch run outputs (npz/json/log) from the remote node into results/.
# usage: scripts/fetch.sh <exp-subdir>   (e.g. e1)
set -e
cd "$(dirname "$0")/.."
sub=${1:-}
ssh -o BatchMode=yes math-slurm "cd ~/dana-exp/runs && tar czf /tmp/dana_runs.tgz ${sub:-.}"
scp -O -q math-slurm:/tmp/dana_runs.tgz /tmp/dana_runs.tgz
mkdir -p results && tar xzf /tmp/dana_runs.tgz -C results && echo "fetched ${sub:-all} -> results/"
