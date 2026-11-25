#!/bin/bash
# Build summer + new7 origins and the fold-5 file with W parallel single-thread workers.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../venv/bin/python"
PROJ="$HERE/../../proj"
DATA="$HERE/data"
W=${1:-12}
mkdir -p "$HERE/logs"
ORIGINS=$("$PY" -c "
import pandas as pd
p=pd.read_csv('$DATA/plan.csv')
print(' '.join(p[p.kind.isin(['summer','new7'])].origin))")
i=0
declare -a LISTS
for o in $ORIGINS; do LISTS[$((i % W))]+="$o,"; i=$((i+1)); done
for ((k=0;k<W;k++)); do
  L="${LISTS[$k]:-}"; L="${L%,}"
  EXTRA=""; [ $k -eq 0 ] && EXTRA="--fold5"
  [ -z "$L" ] && [ -z "$EXTRA" ] && continue
  OMP_NUM_THREADS=1 nohup "$PY" "$HERE/origin_builder.py" build --root "$PROJ" --data "$DATA" --origins "$L" $EXTRA > "$HERE/logs/build_$k.log" 2>&1 &
done
wait
"$PY" "$HERE/origin_builder.py" manifest --root "$PROJ" --data "$DATA" > "$HERE/logs/manifest.log" 2>&1
echo BUILD_DONE > "$HERE/logs/build_done.flag"
