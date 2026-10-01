#!/bin/bash
# Push code (not results) to the remote node.
set -e
cd "$(dirname "$0")/.."
ssh -o BatchMode=yes math-slurm 'mkdir -p ~/dana-exp/runs ~/dana-exp/cache'
COPYFILE_DISABLE=1 tar --no-xattrs -czf /tmp/dana_code.tgz --exclude results --exclude report --exclude '*.pdf' --exclude '__pycache__' adana tests drivers scripts NOTES.md 2>/dev/null 
scp -O -q /tmp/dana_code.tgz math-slurm:~/dana-exp/code.tgz
ssh -o BatchMode=yes math-slurm 'cd ~/dana-exp && tar xzf code.tgz && rm code.tgz && echo pushed'
