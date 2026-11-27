#!/bin/bash
# Waits for tuning to finish, then runs every TEST2 experiment once (docs/PROTOCOL_v2.md).
cd "$(dirname "$0")/.."
while pgrep -f "experiments/tune.py" > /dev/null; do sleep 20; done
mkdir -p results/test2
for e in E1 E5 E4 E3 E8 E6 E7 E2; do
  python experiments/run_experiments.py $e --workers 2 >> results/test2/run.log 2>&1
done
echo "ALL DONE" >> results/test2/run.log
