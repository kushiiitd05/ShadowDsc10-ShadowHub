#!/bin/bash
# Run every (set, fold, seed) job, P at a time with T lightgbm threads each (P*T <= 24). First seed completes first.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../venv/bin/python"
PROJ="$HERE/../../proj"
OUT="$HERE/out"
P=${1:-4}; T=${2:-6}
mkdir -p "$OUT" "$HERE/logs"
JOBS="$HERE/logs/jobs.txt"; : > "$JOBS"
for seed in 20260807 20260808 20260809; do
  for set in base summer dense7 season; do
    for fold in 5 1 2 3 4; do
      [ "$set" = "summer" ] && [ "$fold" = "5" ] && continue
      echo "$set $fold $seed" >> "$JOBS"
    done
  done
done
run_one() {
  set_name=$1; fold=$2; seed=$3
  OMP_NUM_THREADS=$T "$PY" "$HERE/origins_runner.py" "$PROJ" "$HERE/data" "$OUT" --set "$set_name" --fold "$fold" --seed "$seed" --n-jobs "$T" \
    > "$HERE/logs/run_${set_name}_f${fold}_s${seed}.log" 2>&1
}
export -f run_one; export HERE PY PROJ OUT T
cat "$JOBS" | xargs -P "$P" -L 1 bash -c 'run_one $0 $1 $2'
echo ALL_DONE > "$HERE/logs/all_done.flag"
