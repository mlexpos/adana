#!/bin/bash
# usage: scripts/relaunch.sh e1 [e2 ...]  -- (re)start driver scripts in background on the remote node
cd ~/dana-exp
for e in "$@"; do
  nohup bash drivers/run_$e.sh > runs/${e}_driver.out 2>&1 < /dev/null &
  sleep 1
done
