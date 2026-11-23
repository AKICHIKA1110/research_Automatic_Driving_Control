#!/bin/bash
# How many frames are worth it at 10 Hz: T = 2..24, early2d / c3d, seeds 0-1 (existing results are skipped).
# Usage: bash scripts/run_tsweep.sh data/v1 results [epochs=20]
set -e
DATA=${1:?data dir}
OUT=${2:?results dir}
EP=${3:-20}
PY=${PY:-$HOME/venv_train/bin/python}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
run() {   # model T seed
  local d="$OUT/$1_T$2_s$3"
  [ -f "$d/summary.json" ] && return
  $PY scripts/train.py "$DATA" --model "$1" --T "$2" --seed "$3" --epochs "$EP" --out "$d" > "$d.log" 2>&1
}
# two GPU queues balanced by cost (c3d cost grows with T)
queue1() { run c3d 24 0; run c3d 24 1; run c3d 16 1; run c3d 12 0; run c3d 12 1; }
queue2() {
  for s in 0 1; do for T in 2 3 4 6 8 12 16 24; do run early2d $T $s; done; done
  for a in "8 1" "6 1" "6 0" "4 1" "3 1" "3 0" "2 1" "2 0"; do run c3d $a; done
}
queue1 & queue2 & wait
echo all done
